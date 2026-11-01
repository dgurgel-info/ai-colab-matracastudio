"""Medições de DSP validadas contra o ffmpeg (ebur128) e sinais sintéticos conhecidos."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matraca import dsp  # noqa: E402

SR = 24000


def voiced_like(seconds=6.0, f0=120.0, sr=SR, seed=0, tilt=0.0, level=0.2):
    """Sinal vozeado sintético: harmônicas de f0 com envelope silábico e pausas."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros_like(t)
    for k in range(1, 30):
        x += np.sin(2 * np.pi * f0 * k * t) / (k ** (1.0 + tilt))
    env = np.clip(np.sin(2 * np.pi * 3.0 * t), 0, None) ** 0.7
    x = x * env
    x += rng.normal(0, 1e-4, x.size)
    return (x / np.max(np.abs(x)) * level).astype(np.float32)


def ffmpeg_lufs(x, sr):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.wav"
        sf.write(p, x, sr, subtype="FLOAT")
        r = subprocess.run([dsp.find_ffmpeg(), "-nostdin", "-hide_banner", "-i", str(p),
                            "-af", "ebur128=peak=true", "-f", "null", "-"],
                           stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out = r.stderr.decode(errors="ignore")
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", out)[-1])


@pytest.mark.parametrize("level", [0.05, 0.2, 0.6])
def test_loudness_matches_ffmpeg_ebur128(level):
    x = voiced_like(level=level)
    assert abs(dsp.loudness_lufs(x, SR) - ffmpeg_lufs(x, SR)) < 0.3


def test_loudness_is_sample_rate_independent():
    x = voiced_like(sr=48000, level=0.25)
    y = dsp.resample(x, 48000, 24000)
    assert abs(dsp.loudness_lufs(x, 48000) - dsp.loudness_lufs(y, 24000)) < 0.3


def test_gain_changes_loudness_by_the_same_amount_and_silence_is_floor():
    x = voiced_like()
    base = dsp.loudness_lufs(x, SR)
    assert dsp.loudness_lufs(dsp.apply_gain_db(x, 4.0), SR) == pytest.approx(base + 4.0, abs=0.05)
    assert dsp.loudness_lufs(np.zeros(SR, np.float32), SR) == dsp.SILENCE_LUFS
    assert dsp.loudness_lufs(np.zeros(0, np.float32), SR) == dsp.SILENCE_LUFS


def test_short_clip_without_a_full_block_still_measures():
    x = voiced_like(seconds=0.3)
    assert -60 < dsp.loudness_lufs(x, SR) < 0


def test_f0_tracks_pitch_of_the_voice():
    for f0 in (100.0, 140.0, 210.0):
        got = dsp.median_f0_hz(voiced_like(f0=f0), SR)
        assert got == pytest.approx(f0, rel=0.03)


def test_spectral_shape_distinguishes_dark_from_bright():
    dark = dsp.spectral_shape_db(voiced_like(tilt=1.2), SR)
    bright = dsp.spectral_shape_db(voiced_like(tilt=0.2), SR)
    ok = ~np.isnan(dark) & ~np.isnan(bright)
    assert ok.sum() > 10
    assert np.mean(np.abs(dark[ok] - bright[ok])) > 3.0
    same = dsp.spectral_shape_db(voiced_like(tilt=0.2, seed=5), SR)
    ok2 = ~np.isnan(same) & ~np.isnan(bright)
    assert np.mean(np.abs(same[ok2] - bright[ok2])) < 1.0


def test_spectral_shape_does_not_depend_on_level():
    a = dsp.spectral_shape_db(voiced_like(level=0.05), SR)
    b = dsp.spectral_shape_db(voiced_like(level=0.5), SR)
    ok = ~np.isnan(a)
    assert np.max(np.abs(a[ok] - b[ok])) < 0.5


def test_time_stretch_changes_duration_only():
    x = voiced_like(seconds=4.0, f0=130.0)
    fast = dsp.time_stretch(x, SR, 1.15)
    assert len(fast) / SR == pytest.approx(4.0 / 1.15, rel=0.02)
    assert dsp.median_f0_hz(fast, SR) == pytest.approx(130.0, rel=0.04)
    assert dsp.time_stretch(x, SR, 1.0) is x


def test_safe_gain_respects_ceiling():
    x = voiced_like(level=0.8)
    g = dsp.safe_gain_db(x, 12.0, ceiling_dbfs=-1.0)
    assert dsp.peak_db(dsp.apply_gain_db(x, g)) <= -0.99


def test_fade_edges_zeroes_boundaries_and_keeps_middle():
    x = np.ones(SR, np.float32)
    y = dsp.fade_edges(x, SR, 10)
    assert y[0] == 0 and y[-1] == 0 and y[SR // 2] == 1


def test_master_hits_target_loudness_and_peak(tmp_path):
    x = voiced_like(seconds=20.0, level=0.3)
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src, x, SR, subtype="FLOAT")
    info = dsp.master_for_streaming(str(src), str(dst), target_lufs=-14.0, true_peak_dbtp=-1.5)
    y, sr = sf.read(dst)
    assert sr == 48000
    assert sf.info(dst).subtype == "PCM_24"
    assert dsp.loudness_lufs(y.astype(np.float32), sr) == pytest.approx(-14.0, abs=0.7)
    assert dsp.true_peak_db(y.astype(np.float32), sr) <= -1.0
    assert info["input_lufs"] < -14.0
