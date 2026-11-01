"""Linha do tempo final: intro/outro preservados, duração exata, masterização e remontagem offline."""
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from matraca import dsp  # noqa: E402
from matraca.session import PartSession  # noqa: E402
from matraca.timeline import TimelineContext, load_context, render_final, save_context  # noqa: E402
from test_matraca_session import FakeEngine, PART_CHARS, SR, sentences  # noqa: E402


def build(tmp_path, total=40.0, start=4.0, end=36.0):
    rng = np.random.default_rng(1)
    original = (rng.normal(0, 0.05, int(total * SR))).astype(np.float32)   # "música" de fundo
    orig_path = tmp_path / "orig.wav"
    sf.write(orig_path, original, SR, subtype="FLOAT")
    s = PartSession(str(tmp_path / "sess"), "pt-br", SR, synth_fn=FakeEngine())
    s.plan(sentences(9), target_chars=PART_CHARS)
    s.synthesize_all()
    ctx = TimelineContext(speech_start=start, speech_end=end, total_duration=total, sync=True)
    save_context(s, ctx, str(orig_path))
    return s, ctx, original, orig_path


def test_final_has_exact_duration_keeps_intro_outro_and_is_mastered(tmp_path):
    s, ctx, original, _ = build(tmp_path)
    out = tmp_path / "final.wav"
    info = render_final(s, str(out))
    y, sr = sf.read(out, dtype="float32")
    assert sr == 48000 and sf.info(out).subtype == "PCM_24"
    assert info["duration"] == pytest.approx(40.0, abs=0.01)
    assert dsp.loudness_lufs(y, sr) == pytest.approx(-14.0, abs=1.0)
    assert dsp.true_peak_db(y, sr) <= -1.0
    # a intro é a original (mesma forma, só ganho global): correlação altíssima
    a = dsp.resample(original[: int(3.5 * SR)], SR, sr)
    b = y[: int(3.5 * sr)]
    n = min(len(a), len(b))
    assert abs(np.corrcoef(a[:n], b[:n])[0, 1]) > 0.98


def test_without_master_keeps_native_rate_and_no_clipping(tmp_path):
    s, ctx, *_ = build(tmp_path)
    ctx.master = False
    out = tmp_path / "raw.wav"
    render_final(s, str(out), ctx)
    y, sr = sf.read(out, dtype="float32")
    assert sr == SR and np.max(np.abs(y)) < 1.0


def test_rebuild_after_fix_works_without_original_or_engine(tmp_path):
    s, ctx, _, orig_path = build(tmp_path)
    first = tmp_path / "v1.wav"
    render_final(s, str(first))
    s.set_text(2, "Texto corrigido depois da revisão do áudio gerado.")
    s.regenerate(2)

    orig_path.unlink()                          # o original não existe mais
    reloaded = PartSession.load(s.root)         # nem o motor
    assert load_context(reloaded).total_duration == 40.0
    second = tmp_path / "v2.wav"
    info = render_final(reloaded, str(second))
    assert info["duration"] == pytest.approx(40.0, abs=0.01)
    a, _ = sf.read(first, dtype="float32")
    b, _ = sf.read(second, dtype="float32")
    assert not np.allclose(a, b)                # a correção mudou o áudio
    # fora da parte corrigida a intro continua idêntica
    n = int(3.5 * 48000)
    assert np.corrcoef(a[:n], b[:n])[0, 1] > 0.999


def test_no_sync_returns_natural_length_and_skips_outro(tmp_path):
    s, ctx, *_ = build(tmp_path)
    ctx.sync = False
    out = tmp_path / "nosync.wav"
    info = render_final(s, str(out), ctx)
    natural = s.render().audio.shape[0] / SR
    assert info["duration"] == pytest.approx(ctx.speech_start + natural, abs=0.2)
    assert info["speed"] == 1.0
