"""Linha do tempo final: introdução + fala dublada + encerramento, e masterização.

O contexto necessário para remontar (trechos sem fala do original e parâmetros) é salvo
dentro da pasta da sessão, então corrigir uma parte e remontar não exige o áudio original
nem a GPU.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import soundfile as sf

from . import dsp
from .session import PartSession, Render

CONTEXT_FILE = "timeline.json"


@dataclass
class TimelineContext:
    speech_start: float = 0.0
    speech_end: float = 0.0
    total_duration: float = 0.0
    sync: bool = True
    max_stretch: float = 0.18
    master: bool = True
    target_lufs: float = -14.0
    true_peak: float = -1.5
    out_rate: int = 48000
    bit_depth: int = 24
    match_loudness: bool = True
    gap_ms: float = 180.0


def save_context(session: PartSession, ctx: TimelineContext, original_wav: Optional[str]) -> None:
    """Grava os parâmetros e os trechos sem fala (intro/outro) do áudio original."""
    sr = session.sample_rate
    if original_wav and os.path.exists(original_wav):
        x, osr = sf.read(original_wav, dtype="float32")
        x = dsp.as_mono(x)
        if osr != sr:
            x = dsp.resample(x, osr, sr)
        a, b = int(ctx.speech_start * sr), int(ctx.speech_end * sr)
        sf.write(os.path.join(session.root, "intro.wav"), x[:a], sr, subtype="FLOAT")
        sf.write(os.path.join(session.root, "outro.wav"), x[b:], sr, subtype="FLOAT")
    with open(os.path.join(session.root, CONTEXT_FILE), "w", encoding="utf-8") as fh:
        json.dump(asdict(ctx), fh, ensure_ascii=False, indent=1)


def load_context(session: PartSession) -> TimelineContext:
    path = os.path.join(session.root, CONTEXT_FILE)
    if not os.path.exists(path):
        return TimelineContext()
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return TimelineContext(**{k: v for k, v in data.items() if k in TimelineContext.__dataclass_fields__})


def _read_optional(path: str, sr: int) -> np.ndarray:
    if not os.path.exists(path):
        return np.zeros(0, np.float32)
    x, rate = sf.read(path, dtype="float32")
    x = dsp.as_mono(x)
    return dsp.resample(x, rate, sr) if rate != sr else x


def render_final(session: PartSession, out_wav: str, ctx: Optional[TimelineContext] = None) -> dict:
    """Monta intro + fala + encerramento, sincroniza a duração e masteriza.

    Retorna ``{"speed", "duration", "path", "master": {...} | None}``.
    """
    ctx = ctx or load_context(session)
    sr = session.sample_rate
    has_intro = ctx.speech_start >= 0.25
    has_outro = (ctx.total_duration - ctx.speech_end) >= 0.25 and ctx.sync
    target_speech = max(0.5, ctx.speech_end - ctx.speech_start) if ctx.sync else None

    rendered: Render = session.render(target_duration=target_speech, max_stretch=ctx.max_stretch,
                                      match_loudness=ctx.match_loudness, gap_ms=ctx.gap_ms)
    pieces = []
    if has_intro:
        pieces.append(dsp.fade_edges(_read_optional(os.path.join(session.root, "intro.wav"), sr), sr, 10.0))
    pieces.append(dsp.fade_edges(rendered.audio, sr, 10.0))
    if has_outro:
        pieces.append(dsp.fade_edges(_read_optional(os.path.join(session.root, "outro.wav"), sr), sr, 10.0))
    audio = np.concatenate([p for p in pieces if p.size]) if pieces else rendered.audio

    if ctx.sync and ctx.total_duration > 0:
        want = int(round(ctx.total_duration * sr))
        audio = np.concatenate([audio, np.zeros(want - len(audio), np.float32)]) if len(audio) < want else audio[:want]

    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    master_info = None
    if ctx.master:
        fd, tmp = tempfile.mkstemp(suffix="_premaster.wav")
        os.close(fd)
        try:
            sf.write(tmp, audio, sr, subtype="FLOAT")
            master_info = dsp.master_for_streaming(tmp, out_wav, target_lufs=ctx.target_lufs,
                                                   true_peak_dbtp=ctx.true_peak, out_rate=ctx.out_rate,
                                                   bit_depth=ctx.bit_depth)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    else:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.99:  # evita clipping ao converter para PCM
            audio = audio * (0.99 / peak)
        sf.write(out_wav, audio, sr, subtype="PCM_24")

    info = sf.info(out_wav)
    return {"speed": rendered.speed, "duration": info.frames / info.samplerate, "path": out_wav,
            "master": master_info, "sample_rate": info.samplerate, "ranges": rendered.ranges,
            "gains_db": rendered.gains_db}
