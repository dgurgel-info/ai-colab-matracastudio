"""Orquestração da dublagem por partes, independente do motor de TTS.

O notebook fornece um *adaptador de motor* (``adapter_factory``): uma função que recebe os
metadados da sessão (motor, idioma, referência, parâmetros) e devolve
``(synth_fn, speaker_fn, sample_rate)``. Tudo o mais (partes, QC, montagem, masterização,
persistência) vive aqui e é testado sem GPU.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from . import dsp, qc
from .session import PartSession, ProgressFn, StopFn, SynthFn, SpeakerFn, SyncError
from .timeline import TimelineContext, load_context, render_final, save_context

AdapterFactory = Callable[[dict], Tuple[SynthFn, Optional[SpeakerFn], int]]

SESSIONS_DIRNAME = "sessoes"


@dataclass
class DubOptions:
    part_chars: int = 600            # tamanho-alvo de cada parte (caracteres)
    max_retries: int = 1             # sementes alternativas para partes que destoam
    gap_ms: float = 180.0            # pausa entre partes
    base_seed: int = 42
    qc_sensitivity: float = 1.0
    match_loudness: bool = True      # iguala o volume entre partes (somente ganho)
    sync: bool = True                # ajusta ao tempo do original (limite max_stretch)
    max_stretch: float = 0.18
    master: bool = True              # masterização para streaming (loudnorm linear)
    target_lufs: float = -14.0
    true_peak: float = -1.5
    out_rate: int = 48000
    bit_depth: int = 24

    def timeline(self, speech_start: float, speech_end: float, total: float) -> TimelineContext:
        return TimelineContext(
            speech_start=speech_start, speech_end=speech_end, total_duration=total, sync=self.sync,
            max_stretch=self.max_stretch, master=self.master, target_lufs=self.target_lufs,
            true_peak=self.true_peak, out_rate=self.out_rate, bit_depth=self.bit_depth,
            match_loudness=self.match_loudness, gap_ms=self.gap_ms)


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", text).strip("_") or "audio"


def session_root(base_dir: str, name: str, lang_code: str) -> str:
    return os.path.join(base_dir, SESSIONS_DIRNAME, slug(name), slug(lang_code))


def _load_reference(path: str, rate: int) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32")
    x = dsp.as_mono(x)
    return (dsp.resample(x, sr, rate) if sr != rate else x), rate


def build_session(
    base_dir: str, name: str, lang_code: str, reference_wav: str, meta: dict,
    adapter_factory: AdapterFactory, opts: DubOptions,
) -> PartSession:
    """Cria a pasta da sessão, guarda a referência (para regenerar depois) e liga o motor."""
    root = session_root(base_dir, name, lang_code)
    if os.path.isdir(root):
        shutil.rmtree(root)            # nova dublagem do mesmo áudio/idioma substitui a anterior
    os.makedirs(root, exist_ok=True)
    ref_copy = os.path.join(root, "reference.wav")
    shutil.copyfile(reference_wav, ref_copy)
    meta = dict(meta, ref_file="reference.wav", lang_code=lang_code, name=name, options=asdict(opts))
    meta_for_engine = dict(meta, ref_path=ref_copy)
    synth, speaker, rate = adapter_factory(meta_for_engine)
    session = PartSession(
        root, lang_code, rate, synth_fn=synth, speaker_fn=speaker,
        reference=_load_reference(ref_copy, rate) if speaker else None,
        base_seed=opts.base_seed, gap_ms=opts.gap_ms,
        tolerances=qc.Tolerances(sensitivity=opts.qc_sensitivity),
        max_auto_retries=opts.max_retries, meta=meta)
    return session


def dub_language(
    session: PartSession, text: str, opts: DubOptions, speech_start: float, speech_end: float,
    total_duration: float, original_wav: Optional[str], out_wav: str,
    progress: Optional[ProgressFn] = None, should_stop: Optional[StopFn] = None,
) -> dict:
    """Divide o roteiro, sintetiza, avalia, monta a linha do tempo e masteriza."""
    session.plan(text, target_chars=opts.part_chars)
    summary = session.synthesize_all(progress=progress, should_stop=should_stop)
    save_context(session, opts.timeline(speech_start, speech_end, total_duration), original_wav)
    if summary["counts"].get("error"):
        return {"ok": False, "summary": summary, "path": None}
    try:
        info = render_final(session, out_wav)
    except SyncError as exc:
        # As partes já geradas ficam salvas: dá para editar o texto ou remontar sem sincronização.
        return {"ok": False, "sync_error": str(exc), "summary": summary, "path": None}
    info.update(ok=True, summary=summary)
    return info


def list_sessions(base_dir: str) -> List[Tuple[str, str]]:
    """[(rótulo, pasta)] das sessões salvas, mais recentes primeiro."""
    found = []
    for manifest in glob.glob(os.path.join(base_dir, SESSIONS_DIRNAME, "*", "*", "manifest.json")):
        root = os.path.dirname(manifest)
        lang = os.path.basename(root)
        name = os.path.basename(os.path.dirname(root))
        found.append((os.path.getmtime(manifest), f"{name} · {lang}", root))
    return [(label, root) for _, label, root in sorted(found, reverse=True)]


def restore_session(root: str) -> PartSession:
    """Carrega uma sessão salva. Sem motor: só ouvir, editar texto e remontar."""
    return PartSession.load(root)


def attach_engine(session: PartSession, adapter_factory: AdapterFactory) -> None:
    """Liga o motor a uma sessão restaurada (necessário para regenerar partes)."""
    meta = dict(session.meta, ref_path=os.path.join(session.root, session.meta.get("ref_file", "reference.wav")))
    synth, speaker, rate = adapter_factory(meta)
    if rate != session.sample_rate:
        raise RuntimeError(
            f"O motor atual gera {rate} Hz, mas a sessão foi criada com {session.sample_rate} Hz. "
            "Carregue o mesmo motor usado na dublagem para regenerar partes.")
    ref = _load_reference(meta["ref_path"], rate) if speaker and os.path.exists(meta["ref_path"]) else None
    session.attach_engine(synth, speaker, ref)


def output_name(name: str, lang_code: str) -> str:
    """Mesmo padrão dos arquivos anteriores: ``<nome>_<idioma>_dublado.wav``."""
    clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    clean_code = lang_code.replace("-", "_")
    return f"{clean_name}_{clean_code}_dublado.wav"
