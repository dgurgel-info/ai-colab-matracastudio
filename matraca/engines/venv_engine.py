"""Um venv dedicado por motor de voz, com o modelo rodando num worker (processo) próprio.

* Cada motor tem as próprias bibliotecas e versões (qwen-tts e omnivoice se contradizem em
  transformers/gradio): nada disso entra no ambiente do notebook.
* Trocar de motor encerra o processo do anterior; o sistema operacional devolve toda a VRAM.
* O venv fica no disco (``<raiz>/<motor>/.venv``) e é reaproveitado enquanto a VM do Colab existir; o cache
  de pacotes do ``uv`` é compartilhado, então o segundo venv reaproveita o que o primeiro baixou.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Sequence, Tuple

import requests

from matraca.engines import rpc

Log = Callable[[str], None]
# progress(percentual, texto, até): a barra está em ``percentual`` e, enquanto a etapa durar, pode avançar
# devagar até ``até`` (etapas longas sem medida real, como instalar pacotes).
Progress = Callable[..., None]


class EngineWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineSpec:
    key: str                           # nome do módulo de carga em matraca.engines.worker
    label: str
    packages: Tuple[str, ...]          # instalados no venv do motor
    python: str = "3.12"
    startup_timeout: float = 2400.0    # baixar e carregar o modelo na primeira vez pode demorar


SPECS = {
    "Qwen3-TTS": EngineSpec("qwen3", "Qwen3-TTS 1.7B", ("qwen-tts",)),
    "OmniVoice": EngineSpec("omnivoice", "OmniVoice", ("omnivoice",)),
}


def clean_env(extra: Optional[dict] = None) -> dict:
    """Ambiente sem variáveis que fariam o uv/Python usar o ambiente do Colab em vez do venv."""
    drop = ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX")
    env = {k: v for k, v in os.environ.items() if not k.startswith("UV_") and k not in drop}
    env.update(extra or {})
    return env


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def src_root() -> str:
    """Pasta que contém o pacote ``matraca`` (vai no PYTHONPATH do worker)."""
    import matraca
    return str(Path(matraca.__file__).resolve().parents[1])


class VenvWorker:
    """Instala o venv de um motor, sobe o worker e conversa com ele."""

    def __init__(self, spec: EngineSpec, root: str = "engines", port: Optional[int] = None,
                 command: Optional[Sequence[str]] = None, request_timeout: float = 900.0):
        self.spec = spec
        self.dir = os.path.abspath(os.path.join(root, spec.key))
        self.venv_dir = os.path.join(self.dir, ".venv")
        self.port = port or free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.command = list(command) if command else None            # injetável nos testes
        self.request_timeout = request_timeout
        self.proc: Optional[subprocess.Popen] = None
        self.log_path = os.path.join(self.dir, "worker.log")
        self.install_log_path = os.path.join(self.dir, "install.log")
        self.marker_path = os.path.join(self.dir, "installed.json")
        atexit.register(self.stop)

    # ------------------------------------------------------------------ instalação
    @property
    def python(self) -> str:
        sub = "Scripts" if os.name == "nt" else "bin"
        return os.path.join(self.venv_dir, sub, "python.exe" if os.name == "nt" else "python")

    def installed(self) -> bool:
        try:
            marker = json.loads(Path(self.marker_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return marker.get("packages") == list(self.spec.packages) and os.path.exists(self.python)

    def _run(self, cmd: List[str], log: Log, env: dict) -> str:
        log("$ " + " ".join(cmd))
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        os.makedirs(self.dir, exist_ok=True)
        with open(self.install_log_path, "a", encoding="utf-8") as fh:
            fh.write(f"$ {' '.join(cmd)}\n{res.stdout}\n[exit {res.returncode}]\n")
        if res.returncode != 0:
            raise EngineWorkerError(f"Comando falhou ({res.returncode}): {' '.join(cmd)}\n{res.stdout[-2500:]}\n"
                                    f"(log completo: {self.install_log_path})")
        return res.stdout

    def ensure_installed(self, log: Log = print, progress: Optional[Progress] = None) -> bool:
        """Cria o venv e instala os pacotes do motor. Devolve True se instalou, False se já estava pronto."""
        progress = progress or (lambda *a, **k: None)
        if self.command or self.installed():
            return False
        os.makedirs(self.dir, exist_ok=True)
        env = clean_env()
        if shutil.which("uv") is None:
            progress(8, "Instalando o gerenciador uv...", 10)
            self._run([sys.executable, "-m", "pip", "install", "-q", "uv"], log, env=dict(os.environ))
        uv = shutil.which("uv") or os.path.join(os.path.dirname(sys.executable), "uv")
        progress(10, f"Criando o ambiente isolado de {self.spec.label} (Python {self.spec.python})...", 15)
        if os.path.isdir(self.venv_dir):
            shutil.rmtree(self.venv_dir, ignore_errors=True)       # venv incompleto de uma tentativa anterior
        self._run([uv, "venv", self.venv_dir, "-p", self.spec.python], log, env=env)
        version = self._run([self.python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                            log, env=env).strip()
        if version != self.spec.python:
            raise EngineWorkerError(f"O venv de {self.spec.label} deveria usar Python {self.spec.python}, "
                                    f"mas usa {version}.")
        progress(15, f"Instalando as dependências de {self.spec.label} (só na primeira vez; vários minutos)...", 62)
        self._run([uv, "pip", "install", "--python", self.python, *self.spec.packages], log, env=env)
        Path(self.marker_path).write_text(json.dumps({"packages": list(self.spec.packages)}), encoding="utf-8")
        return True

    # ------------------------------------------------------------------ ciclo de vida
    def _command(self) -> List[str]:
        if self.command:
            return self.command
        return [self.python, "-u", "-m", "matraca.engines.worker", "--engine", self.spec.key,
                "--port", str(self.port)]

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, n: int = 1500) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return fh.read()[-n:]
        except OSError:
            return ""

    def _healthy(self) -> bool:
        try:
            return requests.get(self.base + "/health", timeout=3).status_code == 200
        except requests.RequestException:
            return False

    def start(self, log: Log = print, progress: Optional[Progress] = None) -> None:
        progress = progress or (lambda *a, **k: None)
        if self.alive() and self._healthy():
            return
        os.makedirs(self.dir, exist_ok=True)
        progress(64, f"Iniciando {self.spec.label} (baixa o modelo na primeira vez)...", 97)
        log(f"🚀 Iniciando o worker de {self.spec.label} na porta {self.port}...")
        logfile = open(self.log_path, "w", encoding="utf-8")
        popen_kw = {"start_new_session": True} if os.name != "nt" else {}
        env = clean_env({"PYTHONPATH": src_root(), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
        self.proc = subprocess.Popen(self._command(), cwd=self.dir, stdout=logfile, stderr=subprocess.STDOUT,
                                     env=env, **popen_kw)
        t0, last = time.time(), 0.0
        while time.time() - t0 < self.spec.startup_timeout:
            if not self.alive():
                raise EngineWorkerError(f"O worker de {self.spec.label} encerrou ao iniciar.\n{self.log_tail(2500)}")
            if self._healthy():
                log(f"✅ {self.spec.label} pronto ({time.time() - t0:.0f}s).")
                return
            if time.time() - last > 20:
                tail = [l for l in self.log_tail(600).replace("\r", "\n").splitlines() if l.strip()]
                log(f"   ...carregando ({time.time() - t0:.0f}s) {tail[-1][:140] if tail else ''}")
                last = time.time()
            time.sleep(2)
        self.stop()
        raise EngineWorkerError(f"Tempo esgotado ao iniciar {self.spec.label}.\n{self.log_tail(2500)}")

    def stop(self) -> None:
        """Encerra o processo do motor (devolve a VRAM). Seguro para chamar várias vezes."""
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass

    close = stop

    # ------------------------------------------------------------------ chamadas
    def _post(self, payload: dict) -> Any:
        if not self.alive() and not self.command:
            raise EngineWorkerError(f"O worker de {self.spec.label} não está em execução.")
        resp = requests.post(self.base + "/rpc", json=payload, timeout=self.request_timeout)
        try:
            data = resp.json()
        except ValueError:
            raise EngineWorkerError(f"Resposta inválida do worker ({resp.status_code}): {resp.text[:300]}")
        if resp.status_code != 200:
            raise EngineWorkerError(f"{data.get('error', 'Erro')}: {data.get('message', '')}\n"
                                    f"{data.get('traceback', '')}".strip())
        return rpc.decode(data["result"])

    def call(self, method: str, *args, **kwargs) -> Any:
        return self._post({"op": "call", "method": method, "seed": rpc.ACTIVE_SEED.get(),
                           "args": rpc.encode(list(args)), "kwargs": rpc.encode(kwargs)})

    def attr(self, path: str) -> Any:
        return self._post({"op": "attr", "method": path})


# ------------------------------------------------------------------ proxies: a mesma interface dos modelos reais
class _Closeable:
    def __init__(self, worker: VenvWorker):
        self.worker = worker

    def close(self) -> None:
        self.worker.stop()


class QwenProxy(_Closeable):
    def __init__(self, worker: VenvWorker):
        super().__init__(worker)
        self.model = SimpleNamespace(
            extract_speaker_embedding=lambda **kw: worker.call("model.extract_speaker_embedding", **kw))

    def create_voice_clone_prompt(self, **kwargs):
        return self.worker.call("create_voice_clone_prompt", **kwargs)

    def generate_voice_clone(self, **kwargs):
        return self.worker.call("generate_voice_clone", **kwargs)


class OmniVoiceProxy(_Closeable):
    def generate(self, **kwargs):
        return self.worker.call("generate", **kwargs)


PROXIES = {"Qwen3-TTS": QwenProxy, "OmniVoice": OmniVoiceProxy}


def make_remote_loader(name: str, root: str) -> Callable[..., Any]:
    """Loader para o EngineManager: instala o venv (se preciso), sobe o worker e devolve o proxy."""
    spec = SPECS[name]

    def load(log: Log = print, progress: Optional[Progress] = None):
        worker = VenvWorker(spec, root=root)
        try:
            worker.ensure_installed(log, progress)
            worker.start(log, progress)
            return PROXIES[name](worker)
        except BaseException:
            worker.stop()
            raise
    return load
