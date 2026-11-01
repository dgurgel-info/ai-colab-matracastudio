"""Higgs TTS 3 4B (Boson AI): cliente e gerenciador do servidor SGLang-Omni.

O modelo NÃO tem API Python local: o caminho documentado é subir o servidor
(``sgl-omni serve``) e chamar ``POST /v1/audio/speech``. Para não conflitar com
``transformers==4.57.3`` e com os outros motores, o servidor roda num ambiente
virtual próprio (uv + Python 3.12) e é encerrado ao trocar de motor, liberando a VRAM.

LICENÇA: Boson Higgs TTS 3 Research and Non-Commercial License. A *Creator Use Grant*
permite monetizar vídeos nos canais do próprio criador, com atribuição à Boson AI.
Hospedar como serviço/produto exige licença comercial. Clonar voz exige consentimento.
https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/LICENSE

Nada aqui foi executado numa GPU real: os comandos de instalação e os campos da API vêm
da ficha do modelo e são verificados contra um servidor falso nos testes.
"""

from __future__ import annotations

import atexit
import io
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import requests
import soundfile as sf

MODEL_ID = "bosonai/higgs-tts-3-4b"
SGLANG_OMNI_REPO = "https://github.com/sgl-project/sglang-omni.git"
ATTRIBUTION = "This audio was created with Boson AI's Higgs Audio — https://www.boson.ai/higgs-audio"
LICENSE_NOTE = (
    "Higgs TTS 3 (Boson AI) usa a licença *Research and Non-Commercial*. Vídeos em canais próprios podem ser "
    "monetizados (Creator Use Grant) desde que a atribuição seja incluída; não use para hospedar um serviço. "
    "Clone apenas vozes com consentimento."
)

Log = Callable[[str], None]


class HiggsError(RuntimeError):
    pass


class HiggsServer:
    """Sobe/derruba o servidor e sintetiza partes via HTTP."""

    def __init__(self, workdir: str = "higgs_server", model_id: str = MODEL_ID, port: int = 8000,
                 command: Optional[Sequence[str]] = None, startup_timeout: float = 2400.0,
                 request_timeout: float = 600.0):
        self.workdir = os.path.abspath(workdir)
        self.model_id = model_id
        self.port = int(port)
        self.base = f"http://127.0.0.1:{self.port}"
        self.command = list(command) if command else None      # injetável nos testes
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.proc: Optional[subprocess.Popen] = None
        self.supports_seed: Optional[bool] = None               # descoberto na primeira chamada
        self.log_path = os.path.join(self.workdir, "server.log")
        atexit.register(self.stop)

    # ------------------------------------------------------------------ instalação (ambiente isolado)
    @property
    def repo_dir(self) -> str:
        return os.path.join(self.workdir, "sglang-omni")

    @property
    def venv_dir(self) -> str:
        return os.path.join(self.repo_dir, ".venv")

    @property
    def serve_bin(self) -> str:
        return os.path.join(self.venv_dir, "bin", "sgl-omni")

    @property
    def venv_bin(self) -> str:
        return os.path.join(self.venv_dir, "bin")

    @property
    def venv_python(self) -> str:
        return os.path.join(self.venv_dir, "bin", "python")

    @property
    def install_log_path(self) -> str:
        return os.path.join(self.workdir, "install.log")

    @staticmethod
    def clean_env() -> dict:
        """Ambiente sem as variáveis do Colab que fariam o uv ignorar o venv (UV_SYSTEM_PYTHON, VIRTUAL_ENV...)."""
        drop = ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX")
        return {k: v for k, v in os.environ.items() if not k.startswith("UV_") and k not in drop}

    def _run(self, cmd: List[str], log: Log, cwd: Optional[str] = None, env: Optional[dict] = None) -> str:
        log("$ " + " ".join(cmd))
        # Sessão própria: o SIGINT de "parar célula" do Colab vai ao grupo do kernel e mataria o instalador
        # no meio (uma instalação do Higgs leva ~20 min e baixa ~12 GB).
        popen_kw = {"start_new_session": True} if os.name != "nt" else {}
        res = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             **popen_kw)
        try:        # log completo da instalação: a mensagem de erro mostra só o final
            os.makedirs(self.workdir, exist_ok=True)
            with open(self.install_log_path, "a", encoding="utf-8") as fh:
                fh.write(f"$ {' '.join(cmd)}\n{res.stdout}\n[exit {res.returncode}]\n")
        except OSError:
            pass
        if res.returncode != 0:
            raise HiggsError(f"Comando falhou ({res.returncode}): {' '.join(cmd)}\n{res.stdout[-2500:]}\n"
                             f"(log completo: {self.install_log_path})")
        return res.stdout

    def installed(self) -> bool:
        return os.path.exists(self.serve_bin)

    def ensure_installed(self, log: Log = print) -> None:
        """uv + sglang-omni num venv Python 3.12 separado (feito uma vez por sessão)."""
        if self.command:
            return
        if self.installed():
            self.ensure_build_tools(log)       # venv de uma versão anterior pode não ter o ninja no PATH
            return
        os.makedirs(self.workdir, exist_ok=True)
        log("📦 Instalando o servidor do Higgs TTS 3 em ambiente isolado (pode levar vários minutos)...")
        uv = self._uv(log)
        if not os.path.isdir(self.repo_dir):
            self._run(["git", "clone", "--depth", "1", SGLANG_OMNI_REPO, self.repo_dir], log)
        env = self.clean_env()
        if os.path.isdir(self.venv_dir):
            # Venv incompleto (instalação interrompida ou que falhou): o uv recusa recriar sobre ele
            # ("A virtual environment already exists"). Os wheels ficam no cache do uv, então refazer é rápido.
            log("♻️ Removendo o ambiente incompleto do Higgs antes de recriar...")
            shutil.rmtree(self.venv_dir, ignore_errors=True)
        self._run([uv, "venv", self.venv_dir, "-p", "3.12"], log, cwd=self.repo_dir, env=env)
        # O venv TEM de ser Python 3.12: as dependências (ex.: outlines-core) só têm wheel pré-compilada
        # até o 3.12; no 3.13 do Colab o pip tentaria compilar em Rust e falharia.
        version = self._run([self.venv_python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                            log, env=env).strip()
        if version != "3.12":
            raise HiggsError(f"O ambiente isolado do Higgs deveria usar Python 3.12, mas usa {version}.")
        # --python força o uv a instalar NO venv (nunca no Python do Colab); --prerelease=allow é exigido
        # pelo sglang-omni (flash-attn-4 em beta), como na documentação oficial.
        self._run([uv, "pip", "install", "--python", self.venv_python, "--prerelease=allow", "-e", "."],
                  log, cwd=self.repo_dir, env=env)
        if not self.installed():
            raise HiggsError("A instalação terminou, mas o executável sgl-omni não foi encontrado.")
        self.ensure_build_tools(log)

    def _uv(self, log: Log) -> str:
        if shutil.which("uv") is None:
            self._run([sys.executable, "-m", "pip", "install", "-q", "uv"], log)
        return shutil.which("uv") or os.path.join(os.path.dirname(sys.executable), "uv")

    def server_env(self) -> dict:
        """Ambiente do servidor = o que ``source .venv/bin/activate`` daria.

        O flashinfer compila kernels (JIT) ao iniciar e chama ``ninja``/``nvcc`` pelo PATH. Sem o bin do venv
        (ninja) e o do CUDA (nvcc) no PATH, o servidor morre com ``FileNotFoundError: 'ninja'``.
        """
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX")}
        prefix = [self.venv_bin]
        cuda_home = env.get("CUDA_HOME") or ("/usr/local/cuda" if os.path.isdir("/usr/local/cuda") else "")
        if cuda_home:
            env["CUDA_HOME"] = cuda_home
            prefix.append(os.path.join(cuda_home, "bin"))
        env["PATH"] = os.pathsep.join(prefix + [env.get("PATH", "")])
        env["VIRTUAL_ENV"] = self.venv_dir
        return env

    def ensure_build_tools(self, log: Log = print) -> None:
        """Garante ``ninja`` no PATH do servidor (instala no venv se faltar) e avisa se não houver ``nvcc``."""
        path = self.server_env()["PATH"]
        if shutil.which("ninja", path=path) is None:
            log("🔧 Instalando o ninja no ambiente do Higgs (o flashinfer compila kernels ao iniciar)...")
            self._run([self._uv(log), "pip", "install", "--python", self.venv_python, "ninja"], log,
                      env=self.clean_env())
        if shutil.which("nvcc", path=path) is None:
            log("⚠️ nvcc não encontrado no PATH: a compilação dos kernels pode falhar. No Colab ele fica em /usr/local/cuda/bin.")

    # ------------------------------------------------------------------ ciclo de vida
    def _server_command(self) -> List[str]:
        if self.command:
            return self.command
        return [self.serve_bin, "serve", "--model-path", self.model_id, "--port", str(self.port)]

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _ready(self) -> bool:
        for path in ("/health", "/v1/models", "/"):
            try:
                if requests.get(self.base + path, timeout=3).status_code < 500:
                    return True
            except requests.RequestException:
                continue
        return False

    def _tail_log(self, n: int = 3500) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return fh.read()[-n:]
        except OSError:
            return ""

    def start(self, log: Log = print) -> None:
        if self._alive() and self._ready():
            return
        os.makedirs(self.workdir, exist_ok=True)
        log(f"🚀 Iniciando o servidor Higgs TTS 3 na porta {self.port} (carrega ~4B de parâmetros em BF16)...")
        logfile = open(self.log_path, "w", encoding="utf-8")
        popen_kw = {"start_new_session": True} if os.name != "nt" else {}
        self.proc = subprocess.Popen(self._server_command(), cwd=self.workdir, stdout=logfile,
                                     stderr=subprocess.STDOUT, env=self.server_env(), **popen_kw)
        t0, last = time.time(), 0.0
        while time.time() - t0 < self.startup_timeout:
            if not self._alive():
                raise HiggsError("O servidor do Higgs encerrou ao iniciar.\n" + self._tail_log())
            if self._ready():
                log(f"✅ Higgs TTS 3 pronto ({time.time() - t0:.0f}s).")
                return
            if time.time() - last > 30:
                log(f"   ...aguardando o servidor ({time.time() - t0:.0f}s)")
                last = time.time()
            time.sleep(2)
        self.stop()
        raise HiggsError("Tempo esgotado ao iniciar o servidor do Higgs.\n" + self._tail_log())

    def stop(self) -> None:
        """Encerra o servidor (libera a VRAM). Seguro para chamar várias vezes."""
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
            except Exception:
                pass
        time.sleep(1)

    close = stop

    # ------------------------------------------------------------------ síntese
    @staticmethod
    def max_tokens_for(text: str) -> int:
        """Folga generosa: o servidor para no fim da fala; truncar seria pior que sobrar."""
        return int(min(4096, max(1024, len(text) * 6)))

    def synthesize(self, text: str, ref_audio: str, ref_text: str, seed: Optional[int] = 42,
                   temperature: float = 0.8, top_k: int = 50) -> Tuple[np.ndarray, int]:
        if not self._alive() and not self.command:
            raise HiggsError("O servidor do Higgs não está em execução.")
        body = {
            "input": text,
            "references": [{"audio_path": os.path.abspath(ref_audio), "text": (ref_text or "").strip()}],
            "temperature": temperature, "top_k": top_k, "max_new_tokens": self.max_tokens_for(text),
        }
        send_seed = seed is not None and self.supports_seed is not False
        if send_seed:
            body["seed"] = int(seed)
        resp = requests.post(self.base + "/v1/audio/speech", json=body, timeout=self.request_timeout)
        if send_seed and resp.status_code in (400, 422) and "seed" in resp.text.lower():
            # O servidor não conhece o campo: seguir sem semente (resultado não reprodutível).
            self.supports_seed = False
            body.pop("seed")
            resp = requests.post(self.base + "/v1/audio/speech", json=body, timeout=self.request_timeout)
        elif send_seed and resp.status_code == 200:
            self.supports_seed = True
        if resp.status_code != 200:
            raise HiggsError(f"O servidor do Higgs respondeu {resp.status_code}: {resp.text[:400]}")
        try:
            data, sr = sf.read(io.BytesIO(resp.content), dtype="float32", always_2d=False)
        except Exception as exc:
            raise HiggsError(f"Resposta do Higgs não é um WAV válido: {exc}; início={resp.content[:80]!r}") from exc
        if data.ndim == 2:
            data = data.mean(axis=1)
        if data.size == 0 or not np.all(np.isfinite(data)):
            raise HiggsError("O Higgs devolveu áudio vazio ou inválido.")
        return np.ascontiguousarray(data, dtype=np.float32), int(sr)
