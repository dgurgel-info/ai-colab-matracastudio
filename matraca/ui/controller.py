"""Lógica da aba “Revisão por partes”, sem dependência do Gradio (testável sem navegador)."""

from __future__ import annotations

import html
import os
import threading
from typing import Callable, Dict, List, Optional, Tuple

from .. import dubbing, qc
from ..session import Part, PartError, PartSession, SyncError
from ..timeline import TimelineContext, load_context, render_final

STATUS_LABEL = {
    "ok": "✅ OK", "flagged": "⚠️ Revisar", "error": "❌ Erro", "stale": "✏️ Texto editado", "pending": "⏳ Pendente",
}
TABLE_HEADERS = ["#", "Estado", "Chars", "Dur. (s)", "Semente", "Desvio", "Observações", "Texto"]


class BusyError(RuntimeError):
    pass


class ReviewController:
    """Mantém as sessões abertas e executa as ações da aba de revisão.

    ``adapter_factory`` liga o motor (GPU) quando for preciso regenerar uma parte;
    ``is_busy`` impede regenerar enquanto uma dublagem está em andamento.
    """

    def __init__(self, base_dir: str, outputs_dir: str,
                 adapter_factory: Optional[dubbing.AdapterFactory] = None,
                 is_busy: Optional[Callable[[], bool]] = None):
        self.base_dir = base_dir
        self.outputs_dir = outputs_dir
        self.adapter_factory = adapter_factory
        self.is_busy = is_busy or (lambda: False)
        self._open: Dict[str, PartSession] = {}
        self._engine_attached: set = set()
        self._gpu_lock = threading.Lock()

    # ----------------------------------------------------------------- sessões
    def choices(self) -> List[Tuple[str, str]]:
        return dubbing.list_sessions(self.base_dir)

    def session(self, root: str) -> PartSession:
        if not root:
            raise PartError("Escolha uma sessão.")
        if root not in self._open:
            self._open[root] = dubbing.restore_session(root)
        return self._open[root]

    def forget(self, root: str) -> None:
        self._open.pop(root, None)
        self._engine_attached.discard(root)

    def register(self, session: PartSession) -> None:
        """Usa a sessão viva (com motor) criada durante a dublagem."""
        self._open[session.root] = session
        if session.synth_fn is not None:
            self._engine_attached.add(session.root)

    # ----------------------------------------------------------------- leitura
    def summary_html(self, root: str) -> str:
        if not root:
            return "<div class='mt-empty'>Nenhuma sessão selecionada. Conclua uma dublagem ou escolha uma sessão salva.</div>"
        s = self.session(root)
        counts = s.evaluate()["counts"]
        stats = s.summary_stats()
        ctx = load_context(s)

        def chip(kind: str, label: str, n: int) -> str:
            return f"<span class='mt-chip mt-{kind}{' mt-zero' if not n else ''}'><b>{n}</b> {label}</span>"

        out_path = os.path.join(self.outputs_dir, dubbing.output_name(s.meta.get("name", "audio"), s.label))
        built = "sim" if os.path.exists(out_path) else "ainda não"
        return (
            "<div class='mt-summary'>"
            f"<div class='mt-summary-title'>{html.escape(str(s.meta.get('name', 'Áudio')))} · {html.escape(s.label)}</div>"
            "<div class='mt-chips'>"
            + chip("ok", "OK", counts["ok"]) + chip("flagged", "para revisar", counts["flagged"])
            + chip("stale", "texto editado", counts["stale"]) + chip("error", "com erro", counts["error"])
            + "</div>"
            f"<div class='mt-meta'>{len(s.parts)} partes · {stats['duration']:.0f}s de fala · "
            f"motor <b>{html.escape(str(s.meta.get('engine', '?')))}</b> · {s.sample_rate // 1000} kHz · "
            f"masterização {'ligada' if ctx.master else 'desligada'} · arquivo final montado: {built}</div>"
            "</div>"
        )

    def table(self, root: str) -> List[list]:
        if not root:
            return []
        s = self.session(root)
        s.evaluate()
        rows = s.table()
        for r in rows:
            r[1] = STATUS_LABEL.get(r[1], r[1])
        return rows

    def part_ids(self, root: str) -> List[int]:
        return [p.id for p in self.session(root).parts] if root else []

    def detail(self, root: str, part_id) -> dict:
        s = self.session(root)
        p = s.get(int(part_id))
        takes = []
        for i, t in enumerate(p.takes):
            mark = " ◀ em uso" if i == p.active else ""
            takes.append((f"Take {i + 1} · semente {t.seed} · desvio {t.score:.2f}{mark}", i))
        path = s.part_path(p.id)
        problem = ""
        if p.take and p.take.reasons:
            problem = "⚠️ " + "; ".join(p.take.reasons)
        elif p.error:
            problem = "❌ " + p.error
        elif p.status == "stale":
            problem = "✏️ O texto foi editado; regenere a parte para ouvir a mudança."
        return {"text": p.text, "audio": path if path and os.path.exists(path) else None,
                "takes": takes, "active": p.active if p.active >= 0 else None, "problem": problem,
                "status": STATUS_LABEL.get(p.status, p.status)}

    # ----------------------------------------------------------------- ações
    def _ensure_engine(self, root: str, s: PartSession) -> None:
        if root in self._engine_attached and s.synth_fn is not None:
            return
        if self.adapter_factory is None:
            raise PartError("Nenhum motor disponível para regenerar. Execute os Passos 1–3 do notebook.")
        dubbing.attach_engine(s, self.adapter_factory)
        self._engine_attached.add(root)

    def save_text(self, root: str, part_id, text: str) -> Part:
        return self.session(root).set_text(int(part_id), text)

    def regenerate(self, root: str, part_id, text: Optional[str] = None, seed=None) -> Part:
        if self.is_busy():
            raise BusyError("Há uma dublagem em andamento. Aguarde terminar para regenerar partes.")
        s = self.session(root)
        with self._gpu_lock:
            self._ensure_engine(root, s)
            seed_val = int(seed) if seed not in (None, "") else None
            return s.regenerate(int(part_id), text=text, seed=seed_val)

    def choose_take(self, root: str, part_id, index) -> Part:
        return self.session(root).choose_take(int(part_id), int(index))

    def sync_enabled(self, root: str) -> bool:
        return bool(load_context(self.session(root)).sync) if root else True

    def rebuild(self, root: str, master: Optional[bool] = None, target_lufs: Optional[float] = None,
                sync: Optional[bool] = None) -> dict:
        s = self.session(root)
        ctx = load_context(s)
        if sync is not None:
            ctx.sync = bool(sync)
        if master is not None:
            ctx.master = bool(master)
        if target_lufs is not None:
            ctx.target_lufs = float(target_lufs)
        out = os.path.join(self.outputs_dir, dubbing.output_name(s.meta.get("name", "audio"), s.label))
        info = render_final(s, out, ctx)
        info["path"] = out
        return info

    def friendly_error(self, exc: Exception) -> str:
        if isinstance(exc, SyncError):
            return f"⏱️ {exc}"
        if isinstance(exc, (PartError, BusyError)):
            return f"⚠️ {exc}"
        return f"❌ {exc.__class__.__name__}: {exc}"
