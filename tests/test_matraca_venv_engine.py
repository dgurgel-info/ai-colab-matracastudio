"""Workers por venv: protocolo, semente, referências remotas, instalação e ciclo de vida (sem GPU, sem rede)."""
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matraca.engines import rpc, venv_engine  # noqa: E402
from matraca.engines.venv_engine import (  # noqa: E402
    EngineSpec, EngineWorkerError, OmniVoiceProxy, QwenProxy, VenvWorker, free_port)

FAKE = str(Path(__file__).resolve().parent / "fake_engine_worker.py")
SPEC = EngineSpec("fake", "Motor falso", ("pacote-falso",))


@pytest.fixture
def worker(tmp_path):
    port = free_port()
    w = VenvWorker(SPEC, root=str(tmp_path), port=port, command=[sys.executable, FAKE, str(port)])
    w.spec = EngineSpec("fake", "Motor falso", ("pacote-falso",), startup_timeout=60)
    w.start(log=lambda m: None)
    yield w
    w.stop()


# ------------------------------------------------------------------ protocolo
def test_rpc_roundtrip_arrays_tuples_and_scalars():
    payload = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": (1, "x", None, True),
               "t": torch.tensor([1.5, 2.5]), "n": np.float32(2.0)}
    out = rpc.decode(rpc.encode(payload))
    np.testing.assert_array_equal(out["a"], payload["a"])
    assert out["b"] == (1, "x", None, True) and isinstance(out["b"], tuple)
    assert isinstance(out["t"], torch.Tensor) and out["t"].tolist() == [1.5, 2.5]
    assert out["n"] == 2.0


def test_rpc_unserializable_objects_become_refs_only_on_the_worker_side():
    class Opaque:
        pass

    refs = {}
    wire = rpc.encode([Opaque()], refs)
    assert refs and wire[0]["__t__"] == "ref"
    client_side = rpc.decode(wire)
    assert isinstance(client_side[0], rpc.RemoteRef)
    assert rpc.decode(rpc.encode(client_side), refs)[0] is refs["1"]      # a referência volta ao objeto real
    with pytest.raises(TypeError):
        rpc.encode(Opaque())                                              # no cliente, nada é guardado em silêncio
    with pytest.raises(KeyError):
        rpc.decode({"__t__": "ref", "id": "99"}, refs)                    # worker reiniciado: erro claro


# ------------------------------------------------------------------ worker real (processo) com modelo falso
def test_proxy_reads_remote_attributes_and_returns_audio(worker):
    assert worker.attr("tts_model.sample_rate") == 48000                 # atributo lido no worker
    audio = OmniVoiceProxy(worker).generate(text="olá")
    assert isinstance(audio, np.ndarray) and audio.shape == (8,)


def test_seed_travels_to_the_worker_and_makes_generation_reproducible(worker):
    proxy = OmniVoiceProxy(worker)

    def gen(seed):
        token = rpc.ACTIVE_SEED.set(seed)
        try:
            return proxy.generate(text="mesmo texto")
        finally:
            rpc.ACTIVE_SEED.reset(token)

    a, b, c = gen(7), gen(7), gen(8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_qwen_prompt_stays_remote_and_is_reused(worker):
    proxy = QwenProxy(worker)
    prompt = proxy.create_voice_clone_prompt(ref_audio="r.wav", ref_text="abcde", x_vector_only_mode=False)
    assert isinstance(prompt[0], rpc.RemoteRef)
    wavs, sr = proxy.generate_voice_clone(text="t", language="Portuguese", voice_clone_prompt=prompt)
    assert sr == 24000 and float(wavs[0][0]) == 5.0                      # o worker usou o objeto real (len('abcde'))
    emb = proxy.model.extract_speaker_embedding(audio=np.ones(10, np.float32), sr=24000)
    assert isinstance(emb, torch.Tensor) and emb.tolist() == [1.0, 24000.0]


def test_worker_errors_reach_the_notebook_with_the_message(worker):
    with pytest.raises(EngineWorkerError, match="texto inválido"):
        OmniVoiceProxy(worker).generate(text="explode")
    assert OmniVoiceProxy(worker).generate(text="ok") is not None            # e o worker continua vivo


def test_stop_ends_the_process_and_later_calls_fail_clearly(worker):
    proc = worker.proc
    worker.stop()
    worker.stop()                                                         # seguro repetir
    assert proc.poll() is not None
    worker.command = None
    with pytest.raises(EngineWorkerError, match="não está em execução"):
        worker.call("generate", text="x")


def test_worker_that_dies_on_startup_reports_its_log(tmp_path):
    w = VenvWorker(SPEC, root=str(tmp_path), port=free_port(),
                   command=[sys.executable, "-c", "print('ImportError: modelo quebrado'); raise SystemExit(3)"])
    with pytest.raises(EngineWorkerError, match="modelo quebrado"):
        w.start(log=lambda m: None)


# ------------------------------------------------------------------ instalação do venv (comandos simulados)
def fake_run_factory(calls, py_version="3.12", fail_on=None):
    def fake_run(cmd, env=None, stdout=None, stderr=None, text=None, **kw):
        calls.append({"cmd": list(cmd), "env": env})
        if fail_on and fail_on in " ".join(cmd):
            return types.SimpleNamespace(returncode=1, stdout="error: can't find Rust compiler")
        out = py_version + "\n" if "-c" in cmd else ""
        if "venv" in cmd:
            py = Path(cmd[cmd.index("venv") + 1]) / ("Scripts" if sys.platform == "win32" else "bin")
            py.mkdir(parents=True, exist_ok=True)
            (py / ("python.exe" if sys.platform == "win32" else "python")).write_text("")
        return types.SimpleNamespace(returncode=0, stdout=out)
    return fake_run


def test_install_creates_the_venv_and_targets_it_explicitly(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(venv_engine.subprocess, "run", fake_run_factory(calls))
    monkeypatch.setattr(venv_engine.shutil, "which", lambda n: "/usr/bin/uv")
    monkeypatch.setenv("UV_SYSTEM_PYTHON", "1")
    monkeypatch.setenv("VIRTUAL_ENV", "/usr/local")
    w = VenvWorker(SPEC, root=str(tmp_path))
    steps = []
    assert w.ensure_installed(lambda m: None, lambda pct, label, until=None: steps.append(pct)) is True
    install = next(c["cmd"] for c in calls if "pip" in c["cmd"] and "install" in c["cmd"])
    assert install[install.index("--python") + 1] == w.python and install[-1] == "pacote-falso"
    assert all("UV_SYSTEM_PYTHON" not in c["env"] and "VIRTUAL_ENV" not in c["env"] for c in calls)
    assert steps == sorted(steps) and w.installed()
    calls.clear()
    assert w.ensure_installed(lambda m: None) is False and calls == []      # já instalado: não repete


def test_changed_package_list_triggers_a_reinstall(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(venv_engine.subprocess, "run", fake_run_factory(calls))
    monkeypatch.setattr(venv_engine.shutil, "which", lambda n: "/usr/bin/uv")
    VenvWorker(SPEC, root=str(tmp_path)).ensure_installed(lambda m: None)
    newer = VenvWorker(EngineSpec("fake", "Motor falso", ("pacote-falso", "outro")), root=str(tmp_path))
    assert newer.installed() is False


def test_wrong_python_version_and_pip_failure_are_clear_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(venv_engine.shutil, "which", lambda n: "/usr/bin/uv")
    monkeypatch.setattr(venv_engine.subprocess, "run", fake_run_factory([], py_version="3.13"))
    with pytest.raises(EngineWorkerError, match="Python 3.12"):
        VenvWorker(SPEC, root=str(tmp_path / "a")).ensure_installed(lambda m: None)
    monkeypatch.setattr(venv_engine.subprocess, "run", fake_run_factory([], fail_on="pip install"))
    w = VenvWorker(SPEC, root=str(tmp_path / "b"))
    with pytest.raises(EngineWorkerError, match="Rust compiler"):
        w.ensure_installed(lambda m: None)
    assert "Rust compiler" in Path(w.install_log_path).read_text(encoding="utf-8")
    assert not w.installed()                                              # falhou: nada de marcador


def test_every_remote_engine_has_a_spec_a_proxy_and_a_worker_loader():
    from matraca.engines import worker as worker_module
    assert set(venv_engine.SPECS) == set(venv_engine.PROXIES) == {"Qwen3-TTS", "OmniVoice"}
    for spec in venv_engine.SPECS.values():
        assert spec.key in worker_module.LOADERS and spec.packages
    assert len({s.key for s in venv_engine.SPECS.values()}) == 2          # um venv por motor
