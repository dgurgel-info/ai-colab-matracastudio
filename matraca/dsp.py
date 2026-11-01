"""Medição e operações de áudio sem efeitos espectrais sobre a voz.

Princípio do projeto: a coerência de timbre vem da geração (referência, semente e
prompt fixos). Aqui só se *mede* o áudio e se aplica **ganho** (volume) — nunca
equalização, redução de ruído ou filtros que alterem o timbre.

Todas as funções trabalham com ``numpy.ndarray`` mono ``float32`` em [-1, 1].
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

import numpy as np
from scipy import signal

# Coeficientes publicados da ITU-R BS.1770-4 para 48 kHz (estágios 1 e 2 do filtro K).
_K_PRE_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_K_PRE_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_K_HP_B = np.array([1.0, -2.0, 1.0])
_K_HP_A = np.array([1.0, -1.99004745483398, 0.99007225036621])
_LOUDNESS_RATE = 48000
SILENCE_LUFS = -70.0


def as_mono(audio) -> np.ndarray:
    """Converte tensores/listas em vetor mono float32 contíguo."""
    if hasattr(audio, "detach"):  # torch.Tensor sem importar torch
        audio = audio.detach().cpu().numpy()
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=0) if x.shape[0] <= x.shape[1] else x.mean(axis=1)
    return np.ascontiguousarray(x.reshape(-1))


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    g = np.gcd(int(src), int(dst))
    return signal.resample_poly(x, int(dst) // g, int(src) // g).astype(np.float32)


def peak_db(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return -120.0 if p <= 1e-9 else 20.0 * np.log10(p)


def true_peak_db(x: np.ndarray, sr: int) -> float:
    """Pico verdadeiro aproximado (sobreamostragem 4x), em dBTP."""
    if x.size == 0:
        return -120.0
    return peak_db(signal.resample_poly(x, 4, 1))


def loudness_lufs(x: np.ndarray, sr: int) -> float:
    """Loudness integrado (ITU-R BS.1770-4, mono) com gates absoluto e relativo.

    Para trechos com menos de 400 ms não há bloco completo: usa-se a energia do trecho
    inteiro, sem gate. Retorna ``SILENCE_LUFS`` para silêncio.
    """
    x = as_mono(x)
    if x.size == 0 or not np.any(np.abs(x) > 1e-7):
        return SILENCE_LUFS
    y = resample(x, sr, _LOUDNESS_RATE).astype(np.float64)
    y = signal.lfilter(_K_PRE_B, _K_PRE_A, y)
    y = signal.lfilter(_K_HP_B, _K_HP_A, y)
    block = int(0.4 * _LOUDNESS_RATE)
    step = int(0.1 * _LOUDNESS_RATE)
    if y.size < block:
        ms = float(np.mean(y * y))
        return SILENCE_LUFS if ms <= 0 else max(SILENCE_LUFS, -0.691 + 10 * np.log10(ms))
    n_blocks = 1 + (y.size - block) // step
    sq = y * y
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    starts = np.arange(n_blocks) * step
    ms = (csum[starts + block] - csum[starts]) / block
    with np.errstate(divide="ignore"):
        lk = -0.691 + 10 * np.log10(np.maximum(ms, 1e-12))
    keep = lk > -70.0
    if not np.any(keep):
        return SILENCE_LUFS
    rel = -0.691 + 10 * np.log10(np.mean(ms[keep])) - 10.0
    keep &= lk > rel
    if not np.any(keep):
        return SILENCE_LUFS
    return float(-0.691 + 10 * np.log10(np.mean(ms[keep])))


def apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    return (x * np.float32(10 ** (gain_db / 20.0))).astype(np.float32)


def safe_gain_db(x: np.ndarray, wanted_db: float, ceiling_dbfs: float = -1.0) -> float:
    """Limita o ganho para que o pico não passe do teto (sem compressão nem clipping)."""
    room = ceiling_dbfs - peak_db(x)
    return float(min(wanted_db, room))


def trim_dc(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)).astype(np.float32) if x.size else x


def fade_edges(x: np.ndarray, sr: int, fade_ms: float = 12.0) -> np.ndarray:
    """Rampa curta cosseno nas bordas, para evitar estalos nas emendas."""
    n = min(int(sr * fade_ms / 1000.0), x.size // 2)
    if n <= 0:
        return x
    y = x.copy()
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n, dtype=np.float32)))
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


# --------------------------------------------------------------------------- medidas de voz
def _frame_signal(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if x.size < frame:
        x = np.pad(x, (0, frame - x.size))
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def speech_mask(x: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0,
                rel_db: float = -35.0) -> tuple[np.ndarray, np.ndarray]:
    """Quadros com energia relevante (fala), relativos ao nível de pico de curto prazo."""
    frame = max(16, int(sr * frame_ms / 1000))
    hop = max(8, int(sr * hop_ms / 1000))
    frames = _frame_signal(x, frame, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    ref = np.percentile(rms, 95)
    mask = rms > ref * 10 ** (rel_db / 20.0)
    return frames, mask


def active_ratio(x: np.ndarray, sr: int) -> float:
    if x.size == 0:
        return 0.0
    _, mask = speech_mask(x, sr)
    return float(np.mean(mask))


# Bandas de 1/3 de oitava (Hz) usadas só para *medir* o formato espectral médio.
_BAND_CENTERS = np.array([125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                          2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000], dtype=np.float64)


def spectral_shape_db(x: np.ndarray, sr: int) -> np.ndarray:
    """Espectro médio de longo prazo da fala por banda, em dB, sem o nível global.

    Mede cor tonal/“estilo de microfone”. Bandas acima do Nyquist ficam como NaN.
    """
    if x.size < int(0.1 * sr):
        return np.full(_BAND_CENTERS.shape, np.nan)
    frames, mask = speech_mask(x, sr, frame_ms=64.0, hop_ms=32.0)
    if not np.any(mask):
        return np.full(_BAND_CENTERS.shape, np.nan)
    frames = frames[mask] * np.hanning(frames.shape[1])
    spec = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(frames.shape[1], 1.0 / sr)
    out = np.full(_BAND_CENTERS.shape, np.nan)
    for i, fc in enumerate(_BAND_CENTERS):
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        if hi >= sr / 2:
            continue
        sel = (freqs >= lo) & (freqs < hi)
        if np.any(sel):
            out[i] = 10 * np.log10(np.mean(spec[sel]) + 1e-18)
    valid = ~np.isnan(out)
    if valid.any():
        out[valid] -= np.mean(out[valid])
    return out


def spectral_centroid_hz(x: np.ndarray, sr: int) -> float:
    if x.size < int(0.1 * sr):
        return float("nan")
    frames, mask = speech_mask(x, sr, frame_ms=64.0, hop_ms=32.0)
    if not np.any(mask):
        return float("nan")
    frames = frames[mask] * np.hanning(frames.shape[1])
    mag = np.mean(np.abs(np.fft.rfft(frames, axis=1)), axis=0)
    freqs = np.fft.rfftfreq(frames.shape[1], 1.0 / sr)
    return float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))


def median_f0_hz(x: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 400.0) -> float:
    """F0 mediano por autocorrelação nos quadros vozeados (medida de tom, não de timbre)."""
    if x.size < int(0.2 * sr):
        return float("nan")
    work_sr = 16000
    y = resample(x, sr, work_sr)
    frame, hop = int(0.04 * work_sr), int(0.02 * work_sr)
    frames, mask = speech_mask(y, work_sr, frame_ms=40.0, hop_ms=20.0, rel_db=-30.0)
    frames = frames[mask]
    if frames.shape[0] == 0:
        return float("nan")
    frames = frames - frames.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(frames.shape[1] * 2)))
    spec = np.fft.rfft(frames, nfft, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, nfft, axis=1)[:, : frames.shape[1]]
    lo, hi = int(work_sr / fmax), int(work_sr / fmin)
    if hi <= lo or hi >= ac.shape[1]:
        return float("nan")
    norm = ac[:, 0] + 1e-12
    seg = ac[:, lo:hi]
    peak = np.argmax(seg, axis=1)
    strength = seg[np.arange(seg.shape[0]), peak] / norm
    voiced = strength > 0.45
    if not np.any(voiced):
        return float("nan")
    return float(np.median(work_sr / (peak[voiced] + lo)))


# --------------------------------------------------------------------------- ffmpeg
def find_ffmpeg() -> str:
    """Localiza o ffmpeg: variável MATRACA_FFMPEG, PATH ou imageio-ffmpeg."""
    env = os.environ.get("MATRACA_FFMPEG")
    if env and os.path.exists(env):
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:  # pragma: no cover - só em ambientes sem ffmpeg no PATH
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg não encontrado. Instale o ffmpeg ou defina MATRACA_FFMPEG.") from exc


def _atempo_chain(speed: float) -> str:
    filters = []
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5
    filters.append(f"atempo={speed:.6f}")
    return ",".join(filters)


def time_stretch(x: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Altera a duração (speed>1 acelera) via ffmpeg/atempo preservando o tom."""
    if abs(speed - 1.0) < 1e-4 or x.size == 0:
        return x
    cmd = [find_ffmpeg(), "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1",
           "-i", "pipe:0", "-af", _atempo_chain(speed), "-f", "f32le", "-ar", str(sr), "-ac", "1",
           "pipe:1"]
    res = subprocess.run(cmd, input=np.ascontiguousarray(x, dtype="<f4").tobytes(),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0:
        raise RuntimeError("ffmpeg falhou ao ajustar a duração: " + res.stderr.decode(errors="ignore")[-300:])
    return np.frombuffer(res.stdout, dtype="<f4").astype(np.float32)


def master_for_streaming(
    in_wav: str,
    out_wav: str,
    target_lufs: float = -14.0,
    true_peak_dbtp: float = -1.5,
    lra: float = 11.0,
    out_rate: Optional[int] = 48000,
    bit_depth: int = 24,
) -> dict:
    """Masterização final em duas passagens (loudnorm linear) para YouTube.

    Só altera o *volume* (ganho linear). Se o ffmpeg não conseguir manter o modo linear
    sem passar do pico, o ganho é reduzido, nunca comprimido. Retorna as medidas.
    """
    import json
    import re

    ff = find_ffmpeg()
    af1 = f"loudnorm=I={target_lufs}:TP={true_peak_dbtp}:LRA={lra}:print_format=json"
    p1 = subprocess.run([ff, "-nostdin", "-hide_banner", "-i", in_wav, "-af", af1, "-f", "null", "-"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    text = p1.stderr.decode(errors="ignore")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", text, flags=re.S)
    if not m:
        raise RuntimeError("Não foi possível medir o loudness para a masterização.")
    st = json.loads(m.group(0))
    measured_i = float(st["input_i"])
    if measured_i <= -69.0:
        raise RuntimeError("O áudio está praticamente silencioso; masterização cancelada.")
    af2 = (f"loudnorm=I={target_lufs}:TP={true_peak_dbtp}:LRA={lra}:"
           f"measured_I={st['input_i']}:measured_TP={st['input_tp']}:measured_LRA={st['input_lra']}:"
           f"measured_thresh={st['input_thresh']}:offset={st['target_offset']}:linear=true")
    codec = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_f32le"}[bit_depth]
    cmd = [ff, "-nostdin", "-hide_banner", "-v", "error", "-y", "-i", in_wav, "-af", af2,
           "-c:a", codec, "-ac", "1"]
    if out_rate:
        cmd += ["-ar", str(int(out_rate))]
    cmd.append(out_wav)
    subprocess.run(cmd, check=True)
    return {"input_lufs": measured_i, "input_true_peak": float(st["input_tp"]),
            "target_lufs": target_lufs, "true_peak_limit": true_peak_dbtp}
