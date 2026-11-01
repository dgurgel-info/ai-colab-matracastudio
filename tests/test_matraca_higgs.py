"""Cliente/gerenciador do Higgs TTS 3 contra um servidor falso (sem GPU, sem baixar nada)."""
import os
import socket
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matraca import dsp  # noqa: E402
from matraca.engines.higgs import ATTRIBUTION, HiggsError, HiggsServer  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "fake_higgs_server.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def ref(tmp_path):
    path = tmp_path / "ref.wav"
    sf.write(path, np.zeros(24000, np.float32) + 0.01, 24000)
    return str(path)


def make(tmp_path, mode="accept", **kw):
    port = free_port()
    os.environ["FAKE_SEED_MODE"] = mode
    return HiggsServer(workdir=str(tmp_path / "srv"), port=port, command=[sys.executable, FAKE, str(port)],
                       startup_timeout=20, **kw)


def test_start_synthesize_with_seed_and_stop(tmp_path, ref):
    s = make(tmp_path)
    logs = []
    s.start(logs.append)
    try:
        a, sr = s.synthesize("Texto de teste para a parte.", ref, "transcrição", seed=7)
        b, _ = s.synthesize("Texto de teste para a parte.", ref, "transcrição", seed=7)
        c, _ = s.synthesize("Texto de teste para a parte.", ref, "transcrição", seed=9)
        assert sr == 24000 and a.dtype == np.float32 and a.ndim == 1
        np.testing.assert_allclose(a, b)
        assert not np.allclose(a, c)
        assert s.supports_seed is True
        assert any("pronto" in m for m in logs)
    finally:
        s.stop()
    assert not s._alive()
    s.stop()                                              # idempotente


def test_server_without_seed_support_falls_back_and_remembers(tmp_path, ref):
    s = make(tmp_path, mode="reject")
    s.start(lambda m: None)
    try:
        x, sr = s.synthesize("Qualquer texto.", ref, "ref", seed=42)
        assert len(x) > 0 and s.supports_seed is False
        s.synthesize("Outro texto.", ref, "ref", seed=42)  # já não envia a semente
    finally:
        s.stop()


def test_token_budget_scales_with_text_and_is_bounded():
    assert HiggsServer.max_tokens_for("curto") == 1024
    assert HiggsServer.max_tokens_for("x" * 400) == 2400
    assert HiggsServer.max_tokens_for("x" * 5000) == 4096


def test_server_crash_surfaces_the_log(tmp_path):
    s = make(tmp_path, mode="crash")
    with pytest.raises(HiggsError, match="out of memory"):
        s.start(lambda m: None)


def test_bad_reference_and_dead_server_are_clear_errors(tmp_path, ref):
    s = make(tmp_path)
    s.start(lambda m: None)
    try:
        with pytest.raises(HiggsError, match="400"):
            s.synthesize("Texto.", str(tmp_path / "nao_existe.wav"), "x")
    finally:
        s.stop()
    s2 = HiggsServer(workdir=str(tmp_path / "x"))           # sem comando injetado e sem processo
    with pytest.raises(HiggsError, match="não está em execução"):
        s2.synthesize("Texto.", ref, "x")


def test_attribution_text_is_the_one_required_by_the_license():
    assert ATTRIBUTION == "This audio was created with Boson AI's Higgs Audio — https://www.boson.ai/higgs-audio"


def test_output_is_usable_by_the_part_session(tmp_path, ref):
    from matraca.session import PartSession
    s = make(tmp_path)
    s.start(lambda m: None)
    try:
        def synth(text, seed):
            return s.synthesize(text, ref, "ref", seed=seed)
        sess = PartSession(str(tmp_path / "sess"), "en", 24000, synth_fn=synth)
        sess.plan("Uma frase para o Higgs. " * 30, target_chars=200)
        summary = sess.synthesize_all()
        assert summary["counts"]["error"] == 0 and len(sess.parts) >= 3
        audio = sess.render().audio
        assert dsp.loudness_lufs(audio, 24000) > -60
    finally:
        s.stop()


# ------------------------------------------------------------------ instalação (comandos simulados, sem rede)
def fake_installer(monkeypatch, tmp_path, python_version="3.12"):
    calls = []

    def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None, text=None, **kw):
        calls.append({"cmd": list(cmd), "env": env})
        out = python_version + "\n" if "-c" in cmd else ""
        if "venv" in cmd:                                   # o uv cria o venv e o executável do servidor
            venv = Path(cmd[cmd.index("venv") + 1])
            (venv / "bin").mkdir(parents=True, exist_ok=True)
            (venv / "bin" / "sgl-omni").write_text("")
        return types.SimpleNamespace(returncode=0, stdout=out)

    monkeypatch.setattr("matraca.engines.higgs.subprocess.run", fake_run)
    monkeypatch.setattr("matraca.engines.higgs.shutil.which", lambda name, path=None: "/usr/bin/uv")
    (tmp_path / "srv" / "sglang-omni").mkdir(parents=True)             # repositório "já clonado"
    return calls


def test_install_targets_the_venv_python_and_ignores_colab_uv_env(monkeypatch, tmp_path):
    calls = fake_installer(monkeypatch, tmp_path)
    monkeypatch.setenv("UV_SYSTEM_PYTHON", "1")
    monkeypatch.setenv("VIRTUAL_ENV", "/usr/local")
    s = HiggsServer(workdir=str(tmp_path / "srv"))
    s.ensure_installed(lambda m: None)
    install = next(c for c in calls if "pip" in c["cmd"] and "install" in c["cmd"])
    cmd = install["cmd"]
    assert cmd[cmd.index("--python") + 1] == s.venv_python          # nunca o Python do sistema
    assert "--prerelease=allow" in cmd and cmd[-2:] == ["-e", "."]
    assert all(c["env"] is not None and "UV_SYSTEM_PYTHON" not in c["env"] and "VIRTUAL_ENV" not in c["env"]
               for c in calls)


def test_install_refuses_a_venv_that_is_not_python_312(monkeypatch, tmp_path):
    fake_installer(monkeypatch, tmp_path, python_version="3.13")
    with pytest.raises(HiggsError, match="3.12"):
        HiggsServer(workdir=str(tmp_path / "srv")).ensure_installed(lambda m: None)


def test_install_failure_keeps_the_full_log(monkeypatch, tmp_path):
    def failing(cmd, **kw):
        return types.SimpleNamespace(returncode=1, stdout="x" * 5000 + "error: can't find Rust compiler")

    monkeypatch.setattr("matraca.engines.higgs.subprocess.run", failing)
    monkeypatch.setattr("matraca.engines.higgs.shutil.which", lambda name, path=None: "/usr/bin/uv")
    (tmp_path / "srv" / "sglang-omni").mkdir(parents=True)
    s = HiggsServer(workdir=str(tmp_path / "srv"))
    with pytest.raises(HiggsError, match="Rust compiler"):
        s.ensure_installed(lambda m: None)
    assert "Rust compiler" in Path(s.install_log_path).read_text(encoding="utf-8")


# ------------------------------------------------------------------ ambiente do servidor (flashinfer compila com ninja/nvcc)
def test_server_env_activates_the_venv_and_exposes_cuda(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "cuda"))
    monkeypatch.setenv("PYTHONPATH", "/algum/lugar")
    s = HiggsServer(workdir=str(tmp_path / "srv"))
    env = s.server_env()
    path = env["PATH"].split(os.pathsep)
    assert path[0] == s.venv_bin and path[1] == os.path.join(str(tmp_path / "cuda"), "bin")
    assert env["VIRTUAL_ENV"] == s.venv_dir and env["CUDA_HOME"] == str(tmp_path / "cuda")
    assert "PYTHONPATH" not in env                                   # não vaza o Python do notebook para o venv


def test_start_uses_the_server_env_so_ninja_is_found(monkeypatch, tmp_path):
    captured = {}

    class Stop(Exception):
        pass

    def fake_popen(cmd, **kw):
        captured.update(kw)
        raise Stop

    monkeypatch.setattr("matraca.engines.higgs.subprocess.Popen", fake_popen)
    s = HiggsServer(workdir=str(tmp_path / "srv"))
    with pytest.raises(Stop):
        s.start(lambda m: None)
    assert captured["env"]["PATH"].split(os.pathsep)[0] == s.venv_bin


def test_missing_ninja_is_installed_into_the_venv_and_a_missing_nvcc_is_only_a_warning(monkeypatch, tmp_path):
    calls, logs = [], []
    monkeypatch.setattr("matraca.engines.higgs.subprocess.run",
                        lambda cmd, **kw: calls.append(list(cmd)) or types.SimpleNamespace(returncode=0, stdout=""))
    monkeypatch.setattr("matraca.engines.higgs.shutil.which",
                        lambda name, path=None: "/usr/bin/uv" if name == "uv" else None)     # sem ninja nem nvcc
    s = HiggsServer(workdir=str(tmp_path / "srv"))
    s.ensure_build_tools(logs.append)
    assert calls == [["/usr/bin/uv", "pip", "install", "--python", s.venv_python, "ninja"]]
    assert any("nvcc" in m for m in logs)


def test_already_installed_server_still_gets_its_build_tools(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("matraca.engines.higgs.subprocess.run",
                        lambda cmd, **kw: calls.append(list(cmd)) or types.SimpleNamespace(returncode=0, stdout=""))
    monkeypatch.setattr("matraca.engines.higgs.shutil.which",
                        lambda name, path=None: "/usr/bin/uv" if name == "uv" else None)
    s = HiggsServer(workdir=str(tmp_path / "srv"))
    os.makedirs(os.path.dirname(s.serve_bin))
    open(s.serve_bin, "w").close()                                   # venv de uma versão anterior, sem ninja
    s.ensure_installed(lambda m: None)
    assert calls and calls[0][-1] == "ninja" and "clone" not in " ".join(calls[0])


def test_stale_venv_from_an_interrupted_install_is_removed_before_recreating(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, cwd=None, env=None, stdout=None, stderr=None, text=None, **kw):
        if "venv" in cmd:
            seen["stale_present_at_uv_venv"] = (stale / "lib" / "restos").exists()
            venv = Path(cmd[cmd.index("venv") + 1])
            (venv / "bin").mkdir(parents=True, exist_ok=True)
            (venv / "bin" / "sgl-omni").write_text("")
        return types.SimpleNamespace(returncode=0, stdout="3.12\n" if "-c" in cmd else "")

    monkeypatch.setattr("matraca.engines.higgs.subprocess.run", fake_run)
    monkeypatch.setattr("matraca.engines.higgs.shutil.which", lambda name, path=None: "/usr/bin/uv")
    (tmp_path / "srv" / "sglang-omni").mkdir(parents=True)
    stale = tmp_path / "srv" / "sglang-omni" / ".venv"
    (stale / "lib").mkdir(parents=True)
    (stale / "lib" / "restos").write_text("instalação pela metade")        # sem bin/sgl-omni: não conta como instalado
    HiggsServer(workdir=str(tmp_path / "srv")).ensure_installed(lambda m: None)
    assert seen["stale_present_at_uv_venv"] is False                     # o uv não encontra o venv antigo


def test_installers_run_in_their_own_session_so_a_kernel_interrupt_does_not_kill_them(monkeypatch, tmp_path):
    captured = []

    def fake_run(cmd, **kw):
        captured.append(kw)
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr("matraca.engines.higgs.subprocess.run", fake_run)
    HiggsServer(workdir=str(tmp_path / "srv"))._run(["true"], lambda m: None)
    if os.name != "nt":
        assert captured[0].get("start_new_session") is True
