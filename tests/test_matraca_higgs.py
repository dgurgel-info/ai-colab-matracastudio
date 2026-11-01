"""Cliente/gerenciador do Higgs TTS 3 contra um servidor falso (sem GPU, sem baixar nada)."""
import os
import socket
import sys
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
