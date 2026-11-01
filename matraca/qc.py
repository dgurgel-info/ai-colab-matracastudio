"""Controle de qualidade por parte: mede e compara — não altera o timbre da voz."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import dsp


def _clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, list):
        return [_clean(i) for i in v]
    return v


@dataclass
class Metrics:
    duration: float = 0.0
    lufs: float = dsp.SILENCE_LUFS
    peak_db: float = -120.0
    f0: float = float("nan")
    centroid: float = float("nan")
    shape: List[float] = field(default_factory=list)
    active_ratio: float = 0.0
    rate: float = float("nan")          # caracteres por segundo de fala ativa
    clipped: float = 0.0                # fração de amostras no limite digital
    speaker: float = float("nan")       # cosseno com a voz de referência (se disponível)

    def to_json(self) -> dict:
        return {k: _clean(v) for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, d: dict) -> "Metrics":
        def f(v):
            return float("nan") if v is None else float(v)

        m = cls()
        for k, v in (d or {}).items():
            if k == "shape":
                m.shape = [f(i) for i in (v or [])]
            elif hasattr(m, k):
                setattr(m, k, f(v))
        return m


@dataclass
class Tolerances:
    """Quanto uma parte pode se afastar das demais antes de ser sinalizada.

    São heurísticas conservadoras; ``sensitivity`` > 1 deixa o QC mais rígido.
    """
    f0_semitones: float = 2.5
    shape_db: float = 3.0
    centroid_ratio: float = 0.25
    rate_ratio: float = 0.40
    speaker_drop: float = 0.08
    loudness_db: float = 12.0
    min_active: float = 0.25
    sensitivity: float = 1.0
    min_parts_for_relative: int = 3


def measure(x, sr: int, text: Optional[str] = None,
            speaker_cos: Optional[float] = None) -> Metrics:
    x = dsp.as_mono(x)
    m = Metrics(duration=len(x) / sr if sr else 0.0)
    if x.size == 0:
        return m
    m.lufs = dsp.loudness_lufs(x, sr)
    m.peak_db = dsp.peak_db(x)
    m.clipped = float(np.mean(np.abs(x) >= 0.999))
    m.active_ratio = dsp.active_ratio(x, sr)
    m.f0 = dsp.median_f0_hz(x, sr)
    m.centroid = dsp.spectral_centroid_hz(x, sr)
    m.shape = [float(v) for v in dsp.spectral_shape_db(x, sr)]
    if text and m.active_ratio > 0 and m.duration > 0:
        m.rate = len(text.strip()) / (m.duration * m.active_ratio)
    if speaker_cos is not None:
        m.speaker = float(speaker_cos)
    return m


def _nanmedian(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def anchor_of(metrics: Sequence[Metrics]) -> Metrics:
    """Perfil “típico” do conjunto: mediana de cada medida (robusta a uma parte ruim)."""
    a = Metrics()
    a.lufs = _nanmedian([m.lufs for m in metrics])
    a.f0 = _nanmedian([m.f0 for m in metrics])
    a.centroid = _nanmedian([m.centroid for m in metrics])
    a.rate = _nanmedian([m.rate for m in metrics])
    a.speaker = _nanmedian([m.speaker for m in metrics])
    a.active_ratio = _nanmedian([m.active_ratio for m in metrics])
    shapes = [m.shape for m in metrics if m.shape]
    if shapes:
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                a.shape = [float(v) for v in np.nanmedian(np.asarray(shapes, dtype=float), axis=0)]
    return a


def absolute_problems(m: Metrics, tol: Tolerances) -> List[str]:
    problems = []
    if m.duration <= 0.05 or m.peak_db < -60:
        problems.append("áudio vazio ou silencioso")
    if m.clipped > 0.001:
        problems.append("saturação digital (clipping)")
    if m.duration > 0.05 and m.peak_db >= -60 and m.active_ratio < tol.min_active:
        problems.append("fala quase ausente (muito silêncio)")
    return problems


def deviation(m: Metrics, anchor: Metrics, tol: Tolerances) -> Tuple[float, List[str]]:
    """Retorna (pontuação, motivos). Pontuação > 1 significa fora da tolerância."""
    s = max(0.1, tol.sensitivity)
    terms: List[Tuple[float, str]] = []

    if not math.isnan(m.f0) and not math.isnan(anchor.f0) and m.f0 > 0 and anchor.f0 > 0:
        st = abs(12 * math.log2(m.f0 / anchor.f0))
        terms.append((st * s / tol.f0_semitones, f"tom (F0) difere {st:.1f} semitons"))
    if m.shape and anchor.shape:
        a, b = np.asarray(m.shape, float), np.asarray(anchor.shape, float)
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() >= 6:
            d = float(np.mean(np.abs(a[ok] - b[ok])))
            terms.append((d * s / tol.shape_db, f"cor tonal/microfone difere {d:.1f} dB"))
    if not math.isnan(m.centroid) and not math.isnan(anchor.centroid) and anchor.centroid > 0:
        r = abs(m.centroid / anchor.centroid - 1)
        terms.append((r * s / tol.centroid_ratio, f"brilho difere {r * 100:.0f}%"))
    if not math.isnan(m.rate) and not math.isnan(anchor.rate) and anchor.rate > 0:
        r = abs(m.rate / anchor.rate - 1)
        terms.append((r * s / tol.rate_ratio, f"ritmo de fala difere {r * 100:.0f}%"))
    if not math.isnan(m.speaker) and not math.isnan(anchor.speaker):
        drop = anchor.speaker - m.speaker
        if drop > 0:
            terms.append((drop * s / tol.speaker_drop, f"similaridade de voz {drop:.2f} abaixo das demais"))
    if m.lufs > dsp.SILENCE_LUFS and anchor.lufs > dsp.SILENCE_LUFS:
        d = abs(m.lufs - anchor.lufs)
        terms.append((d / tol.loudness_db, f"volume difere {d:.0f} dB"))

    if not terms:
        return 0.0, []
    score = max(t[0] for t in terms)
    reasons = [text for value, text in sorted(terms, reverse=True) if value > 1.0]
    return float(score), reasons
