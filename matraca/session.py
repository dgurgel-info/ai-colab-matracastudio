"""Sessão de dublagem por partes.

Cada parte do roteiro é gerada, medida e salva como um WAV próprio, com semente e texto
registrados em ``manifest.json``. Assim é possível ouvir, editar o texto, gerar outra
variação (take) ou remontar o áudio final sem refazer o restante.

Coerência de voz: todas as partes usam a mesma referência, o mesmo prompt de voz e a
mesma semente base (a escolha é do motor, via ``synth_fn``). O QC compara cada parte com
a mediana das demais e só *repete com outra semente* as que destoam; o áudio nunca é
filtrado ou equalizado. A única correção posterior ao áudio gerado é o **ganho** que
iguala o volume entre as partes.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from . import dsp, qc
from .parts import split_into_parts

SynthFn = Callable[[str, int], Tuple[np.ndarray, int]]
SpeakerFn = Callable[[np.ndarray, int], Optional[np.ndarray]]
ProgressFn = Callable[[dict], None]
StopFn = Callable[[], bool]

MANIFEST_VERSION = 1


class SyncError(RuntimeError):
    """A fala gerada é longa demais para caber na duração alvo sem cortar palavras."""


class PartError(RuntimeError):
    pass


@dataclass
class Take:
    seed: int
    file: str
    metrics: dict
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    text: str = ""               # texto exato usado neste take


@dataclass
class Part:
    id: int
    text: str
    takes: List[Take] = field(default_factory=list)
    active: int = -1
    status: str = "pending"      # pending | ok | flagged | error | stale
    error: str = ""
    failed_seeds: List[int] = field(default_factory=list)   # sementes que o motor não conseguiu gerar

    @property
    def take(self) -> Optional[Take]:
        return self.takes[self.active] if 0 <= self.active < len(self.takes) else None


@dataclass
class Render:
    audio: np.ndarray
    sample_rate: int
    speed: float                 # duração gerada / duração alvo (1.0 = sem ajuste)
    ranges: List[Tuple[int, float, float]]
    gains_db: Dict[int, float]


def _atomic_write_json(path: str, data: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


class PartSession:
    def __init__(
        self,
        root: str,
        label: str,
        sample_rate: int,
        synth_fn: Optional[SynthFn] = None,
        speaker_fn: Optional[SpeakerFn] = None,
        reference: Optional[Tuple[np.ndarray, int]] = None,
        base_seed: int = 42,
        gap_ms: float = 180.0,
        tolerances: Optional[qc.Tolerances] = None,
        max_auto_retries: int = 1,
        meta: Optional[dict] = None,
    ):
        self.root = root
        self.label = label
        self.sample_rate = int(sample_rate)
        self.synth_fn = synth_fn
        self.speaker_fn = speaker_fn
        self.base_seed = int(base_seed)
        self.gap_ms = float(gap_ms)
        self.tol = tolerances or qc.Tolerances()
        self.max_auto_retries = int(max_auto_retries)
        self.meta = dict(meta or {})
        self.parts: List[Part] = []
        self._lock = threading.RLock()
        self._ref_embedding: Optional[np.ndarray] = None
        os.makedirs(os.path.join(root, "parts"), exist_ok=True)
        if reference is not None and speaker_fn is not None:
            self._ref_embedding = self._safe_embed(reference[0], reference[1])

    def attach_engine(self, synth_fn: SynthFn, speaker_fn: Optional[SpeakerFn] = None,
                      reference: Optional[Tuple[np.ndarray, int]] = None) -> None:
        """Liga (ou religa) um motor a uma sessão carregada do disco, p. ex. para regenerar uma parte."""
        self.synth_fn = synth_fn
        self.speaker_fn = speaker_fn
        if reference is not None and speaker_fn is not None:
            self._ref_embedding = self._safe_embed(reference[0], reference[1])

    # ------------------------------------------------------------------ persistência
    @property
    def manifest_path(self) -> str:
        return os.path.join(self.root, "manifest.json")

    def save(self) -> None:
        with self._lock:
            data = {
                "version": MANIFEST_VERSION, "label": self.label, "sample_rate": self.sample_rate,
                "base_seed": self.base_seed, "gap_ms": self.gap_ms, "meta": self.meta,
                "parts": [{
                    "id": p.id, "text": p.text, "active": p.active, "status": p.status,
                    "error": p.error, "failed_seeds": p.failed_seeds,
                    "takes": [{"seed": t.seed, "file": t.file, "metrics": t.metrics, "score": t.score,
                               "reasons": t.reasons, "created": t.created, "text": t.text} for t in p.takes],
                } for p in self.parts],
            }
            _atomic_write_json(self.manifest_path, data)

    @classmethod
    def load(cls, root: str, **kwargs) -> "PartSession":
        with open(os.path.join(root, "manifest.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        s = cls(root, data["label"], data["sample_rate"], base_seed=data.get("base_seed", 42),
                gap_ms=data.get("gap_ms", 180.0), meta=data.get("meta"), **kwargs)
        for pd in data["parts"]:
            s.parts.append(Part(
                id=pd["id"], text=pd["text"], active=pd.get("active", -1), status=pd.get("status", "pending"),
                error=pd.get("error", ""), failed_seeds=list(pd.get("failed_seeds", [])),
                takes=[Take(seed=t["seed"], file=t["file"], metrics=t["metrics"], score=t.get("score", 0.0),
                            reasons=t.get("reasons", []), created=t.get("created", 0.0),
                            text=t.get("text", "")) for t in pd["takes"]]))
        return s

    # ------------------------------------------------------------------ planejamento
    def plan(self, text: str, target_chars: int = 600, max_chars: int = 1500) -> List[Part]:
        pieces = split_into_parts(text, target_chars=target_chars, max_chars=max_chars)
        if not pieces:
            raise PartError("O roteiro está vazio.")
        with self._lock:
            self.parts = [Part(id=i + 1, text=t) for i, t in enumerate(pieces)]
            self.save()
        return self.parts

    def get(self, part_id: int) -> Part:
        for p in self.parts:
            if p.id == int(part_id):
                return p
        raise PartError(f"Parte {part_id} não existe.")

    # ------------------------------------------------------------------ geração
    def _safe_embed(self, audio, sr) -> Optional[np.ndarray]:
        try:
            emb = self.speaker_fn(dsp.as_mono(audio), sr) if self.speaker_fn else None
            if emb is None:
                return None
            emb = np.asarray(emb, dtype=np.float64).reshape(-1)
            n = np.linalg.norm(emb)
            return emb / n if np.isfinite(n) and n > 1e-8 else None
        except Exception:
            return None

    def _speaker_cos(self, audio, sr) -> Optional[float]:
        if self._ref_embedding is None:
            return None
        emb = self._safe_embed(audio, sr)
        return None if emb is None else float(np.dot(emb, self._ref_embedding))

    def _synthesize(self, part: Part, seed: int) -> Take:
        if self.synth_fn is None:
            raise PartError("Esta sessão não tem um motor de síntese carregado (somente remontagem).")
        audio, sr = self.synth_fn(part.text, int(seed))
        x = dsp.as_mono(audio)
        if x.size == 0 or not np.all(np.isfinite(x)) or float(np.max(np.abs(x))) < 1e-5:
            raise PartError("o motor devolveu áudio vazio, inválido ou silencioso")
        if int(sr) != self.sample_rate:
            x = dsp.resample(x, int(sr), self.sample_rate)
        x = dsp.fade_edges(dsp.trim_dc(x), self.sample_rate, fade_ms=12.0)
        n = len(part.takes)
        fname = f"part_{part.id:03d}_t{n + 1:02d}_s{seed}.wav"
        sf.write(os.path.join(self.root, "parts", fname), x, self.sample_rate, subtype="FLOAT")
        m = qc.measure(x, self.sample_rate, part.text, self._speaker_cos(x, self.sample_rate))
        return Take(seed=int(seed), file=fname, metrics=m.to_json(), text=part.text)

    def _anchor(self, exclude_id: Optional[int] = None) -> Tuple[Optional[qc.Metrics], int]:
        usable = [p for p in self.parts if p.take is not None and p.status != "error"]
        # Com poucas partes a mediana é só a média delas: não dá para saber qual destoa.
        if len(usable) < self.tol.min_parts_for_relative:
            return None, len(usable)
        ms = [qc.Metrics.from_json(p.take.metrics) for p in usable if p.id != exclude_id]
        if len(ms) < 2:
            return None, len(ms)
        return qc.anchor_of(ms), len(ms)

    def _score(self, take: Take, anchor: Optional[qc.Metrics]) -> Tuple[float, List[str]]:
        m = qc.Metrics.from_json(take.metrics)
        problems = qc.absolute_problems(m, self.tol)
        score, reasons = (0.0, [])
        if anchor is not None:
            score, reasons = qc.deviation(m, anchor, self.tol)
        if problems:
            score = max(score, 2.0)
        return score, problems + reasons

    def _accept(self, part: Part, take: Take, score: float, reasons: List[str]) -> None:
        take.score, take.reasons = score, reasons
        part.takes.append(take)
        part.active = len(part.takes) - 1
        part.error = ""
        part.status = "flagged" if score > 1.0 else "ok"

    def _try_seeds(self, part: Part, seeds: List[int], anchor, progress, should_stop) -> bool:
        """Gera um take por semente até um passar no QC; mantém o melhor. True se gerou algo."""
        best: Optional[Tuple[Take, float, List[str]]] = None
        last_error = ""
        for attempt, seed in enumerate(seeds):
            if should_stop and should_stop():
                raise InterruptedError("Operação cancelada pelo usuário.")
            try:
                take = self._synthesize(part, seed)
            except InterruptedError:
                raise
            except Exception as exc:  # falha de um take não derruba o job inteiro
                last_error = str(exc) or exc.__class__.__name__
                if int(seed) not in part.failed_seeds:
                    part.failed_seeds.append(int(seed))
                if progress:
                    progress({"type": "part_retry", "part": part.id, "seed": seed, "reason": last_error})
                continue
            score, reasons = self._score(take, anchor)
            if best is None or score < best[1]:
                best = (take, score, reasons)
            if score <= 1.0:
                break
            if progress and attempt + 1 < len(seeds):
                progress({"type": "part_retry", "part": part.id, "seed": seed,
                          "reason": "; ".join(reasons) or "fora da tolerância"})
        if best is None:
            part.status, part.error = "error", last_error or "falha na síntese"
            return False
        self._accept(part, *best)
        return True

    def _next_seed(self, part: Part) -> int:
        used = {t.seed for t in part.takes} | set(part.failed_seeds)
        seed = self.base_seed if not used else max(used) + 1
        while seed in used:
            seed += 1
        return seed

    def _require_engine(self) -> None:
        if self.synth_fn is None:
            raise PartError("Esta sessão não tem um motor de síntese carregado (somente remontagem).")

    def synthesize_all(self, progress: Optional[ProgressFn] = None, should_stop: Optional[StopFn] = None,
                       only_pending: bool = True) -> dict:
        self._require_engine()
        total = len(self.parts)
        for i, part in enumerate(self.parts):
            if only_pending and part.take is not None and part.status in ("ok", "flagged"):
                continue
            if should_stop and should_stop():
                raise InterruptedError("Operação cancelada pelo usuário.")
            if progress:
                progress({"type": "part_start", "part": part.id, "index": i, "total": total, "chars": len(part.text)})
            t0 = time.time()
            with self._lock:
                # 1ª passada: semente base em todas as partes (a validada para a voz).
                self._try_seeds(part, [self.base_seed], None, progress, should_stop)
                self.save()
            if progress:
                progress({"type": "part_done", "part": part.id, "index": i, "total": total,
                          "status": part.status, "elapsed": time.time() - t0})

        summary = self.evaluate()
        # 2ª passada: só as que destoaram, com sementes alternativas, comparadas ao conjunto.
        if self.max_auto_retries > 0:
            for part in list(self.parts):
                if part.status not in ("flagged", "error"):
                    continue
                if should_stop and should_stop():
                    raise InterruptedError("Operação cancelada pelo usuário.")
                anchor, _ = self._anchor(exclude_id=part.id)
                current_seed = part.take.seed if part.take else self.base_seed
                seeds = [self.base_seed + k for k in range(1, self.max_auto_retries + 1)
                         if self.base_seed + k != current_seed]
                if progress:
                    why = "; ".join(part.take.reasons) if part.take and part.take.reasons else (part.error or "falha")
                    progress({"type": "part_requeue", "part": part.id, "reason": why, "seeds": seeds})
                with self._lock:
                    previous = (part.active, part.status, part.error)
                    before = part.take.score if part.take else float("inf")
                    ok = self._try_seeds(part, seeds, anchor, progress, should_stop)
                    # Só troca de take se o novo for melhor; senão volta ao anterior.
                    if previous[0] >= 0 and (not ok or part.take.score >= before):
                        part.active, part.status, part.error = previous
                    self.save()
            summary = self.evaluate()
        return summary

    def regenerate(self, part_id: int, text: Optional[str] = None, seed: Optional[int] = None,
                   auto: bool = False, progress: Optional[ProgressFn] = None,
                   should_stop: Optional[StopFn] = None) -> Part:
        """Gera um novo take de uma parte (opcionalmente com texto editado e/ou semente escolhida)."""
        self._require_engine()
        with self._lock:
            part = self.get(part_id)
            if text is not None and text.strip() and text.strip() != part.text:
                part.text = text.strip()
            anchor, _ = self._anchor(exclude_id=part.id)
            if seed is not None:
                seeds = [int(seed)]
            elif auto:
                first = self._next_seed(part)
                seeds = [first + k for k in range(0, max(1, self.max_auto_retries) + 1)]
            else:
                seeds = [self._next_seed(part)]
            previous = (part.active, part.status, part.error)
            ok = self._try_seeds(part, seeds, anchor, progress, should_stop)
            if not ok and previous[0] >= 0:  # mantém o take anterior utilizável
                part.active, part.status, part.error = previous
            self.evaluate()
            self.save()
            return part

    def set_text(self, part_id: int, text: str) -> Part:
        """Edita o texto sem gerar: a parte fica “stale” até ser regerada."""
        with self._lock:
            part = self.get(part_id)
            text = (text or "").strip()
            if not text:
                raise PartError("O texto da parte não pode ficar vazio.")
            part.text = text
            self.evaluate()
            self.save()
            return part

    def choose_take(self, part_id: int, index: int) -> Part:
        with self._lock:
            part = self.get(part_id)
            if not 0 <= int(index) < len(part.takes):
                raise PartError("Take inexistente.")
            part.active = int(index)
            self.evaluate()
            self.save()
            return part

    # ------------------------------------------------------------------ avaliação
    def evaluate(self) -> dict:
        """Recalcula pontuações com o conjunto atual e atualiza o status de cada parte."""
        with self._lock:
            anchor, n = self._anchor()
            counts = {"ok": 0, "flagged": 0, "error": 0, "stale": 0, "pending": 0}
            for p in self.parts:
                t = p.take
                if p.status == "error" and t is None:
                    counts["error"] += 1
                    continue
                if t is None:
                    p.status = "pending"
                elif t.text and p.text != t.text:
                    p.status = "stale"
                    t.score, t.reasons = self._score(t, anchor)
                else:
                    t.score, t.reasons = self._score(t, anchor)
                    p.status = "flagged" if t.score > 1.0 else "ok"
                counts[p.status] += 1
            return {"counts": counts, "relative_qc": anchor is not None, "parts": len(self.parts)}

    def table(self) -> List[list]:
        rows = []
        for p in self.parts:
            t = p.take
            m = qc.Metrics.from_json(t.metrics) if t else None
            rows.append([
                p.id, p.status, len(p.text), round(m.duration, 1) if m else None,
                t.seed if t else None, round(t.score, 2) if t else None,
                "; ".join(t.reasons) if (t and t.reasons) else (p.error or ""),
                p.text if len(p.text) <= 80 else p.text[:77] + "…",
            ])
        return rows

    def part_path(self, part_id: int) -> Optional[str]:
        t = self.get(part_id).take
        return os.path.join(self.root, "parts", t.file) if t else None

    def summary_stats(self) -> dict:
        ms = [qc.Metrics.from_json(p.take.metrics) for p in self.parts if p.take]
        return {"lufs_spread": (max(m.lufs for m in ms) - min(m.lufs for m in ms)) if ms else 0.0,
                "duration": sum(m.duration for m in ms)}

    # ------------------------------------------------------------------ montagem
    def _load_audio(self, part: Part) -> np.ndarray:
        path = self.part_path(part.id)
        if path is None or not os.path.exists(path):
            raise PartError(f"A parte {part.id} não tem áudio.")
        x, sr = sf.read(path, dtype="float32")
        x = dsp.as_mono(x)
        return dsp.resample(x, sr, self.sample_rate) if sr != self.sample_rate else x

    def render(self, target_duration: Optional[float] = None, max_stretch: float = 0.18,
               match_loudness: bool = True, gap_ms: Optional[float] = None,
               max_gain_db: float = 6.0) -> Render:
        """Monta as partes na ordem. Com ``target_duration``, ajusta o ritmo de forma uniforme.

        O mesmo fator de velocidade vale para todas as partes (nada de partes mais rápidas
        que outras). Se a fala for longa demais para caber, levanta ``SyncError`` em vez
        de cortar palavras.
        """
        with self._lock:
            bad = [p.id for p in self.parts if p.take is None or p.status in ("error", "pending")]
            if bad:
                raise PartError(f"Há partes sem áudio: {bad}. Gere ou regenere essas partes antes de montar.")
            stale = [p.id for p in self.parts if p.status == "stale"]
            if stale:
                raise PartError(f"As partes {stale} tiveram o texto editado. Regenere-as antes de montar.")
            sr = self.sample_rate
            gap = int(sr * (self.gap_ms if gap_ms is None else gap_ms) / 1000.0)
            audios = [self._load_audio(p) for p in self.parts]

            gains: Dict[int, float] = {}
            if match_loudness:
                lufs = [dsp.loudness_lufs(a, sr) for a in audios]
                usable = [v for v in lufs if v > -60]
                target = float(np.median(usable)) if usable else 0.0
                for p, a, l in zip(self.parts, audios, lufs):
                    wanted = 0.0 if l <= -60 else float(np.clip(target - l, -max_gain_db, max_gain_db))
                    gains[p.id] = dsp.safe_gain_db(a, wanted, ceiling_dbfs=-1.0)
                audios = [dsp.apply_gain_db(a, gains[p.id]) if abs(gains[p.id]) > 0.05 else a
                          for p, a in zip(self.parts, audios)]

            speed = 1.0
            n_gaps = max(0, len(audios) - 1)
            if target_duration:
                voice_target = target_duration - n_gaps * gap / sr
                if voice_target <= 0:
                    raise SyncError("A duração alvo é menor que as pausas entre as partes.")
                raw = sum(len(a) for a in audios) / (voice_target * sr)
                if raw > 1.0 + max_stretch:
                    raise SyncError(
                        f"A fala gerada ({sum(len(a) for a in audios)/sr + n_gaps*gap/sr:.1f}s) é longa demais para "
                        f"os {target_duration:.1f}s do original sem cortar palavras (precisaria acelerar "
                        f"{(raw-1)*100:.0f}%, limite {max_stretch*100:.0f}%). Desative a sincronização ou reduza o texto.")
                speed = raw
                applied = float(np.clip(raw, 1.0 - max_stretch, 1.0 + max_stretch)) * 1.001
                audios = [dsp.fade_edges(dsp.time_stretch(a, sr, applied), sr, 12.0) for a in audios]

            out: List[np.ndarray] = []
            ranges: List[Tuple[int, float, float]] = []
            cursor = 0
            for i, (p, a) in enumerate(zip(self.parts, audios)):
                if i:
                    out.append(np.zeros(gap, dtype=np.float32))
                    cursor += gap
                ranges.append((p.id, cursor / sr, (cursor + len(a)) / sr))
                out.append(a)
                cursor += len(a)
            audio = np.concatenate(out) if out else np.zeros(0, np.float32)
            if target_duration:
                want = int(round(target_duration * sr))
                if len(audio) < want:
                    audio = np.concatenate([audio, np.zeros(want - len(audio), np.float32)])
                elif len(audio) > want:
                    over = len(audio) - want
                    if over > int(0.15 * sr):  # sobra relevante: não esconder com corte
                        raise SyncError("A sincronização excedeu a duração alvo; nada foi cortado.")
                    audio = audio[:want]
            return Render(audio=audio, sample_rate=sr, speed=speed, ranges=ranges, gains_db=gains)
