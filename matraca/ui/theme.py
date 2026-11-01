"""Tema e folha de estilo compartilhados da interface."""

from __future__ import annotations

import os
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_css() -> str:
    with open(os.path.join(_HERE, "style.css"), encoding="utf-8") as fh:
        return fh.read()


def build_theme(gr):
    """Tema base do Gradio alinhado aos tokens do CSS (emerald/blue, cantos suaves)."""
    try:
        return gr.themes.Base(
            primary_hue="emerald", secondary_hue="blue", neutral_hue="slate",
            radius_size=gr.themes.sizes.radius_md,
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        )
    except Exception:  # versões antigas sem todos os parâmetros
        return gr.themes.Soft(primary_hue="emerald", secondary_hue="blue")


def _gradio_major(gr) -> int:
    try:
        return int(str(gr.__version__).split(".")[0])
    except Exception:
        return 5


def style_kwargs(gr, theme, css: str) -> tuple:
    """(kwargs do ``gr.Blocks``, kwargs do ``launch``) para a versão instalada do Gradio.

    No Gradio 6 ``theme`` e ``css`` passaram do construtor do Blocks para o ``launch()``;
    nas versões anteriores continuam no construtor.
    """
    if _gradio_major(gr) >= 6:
        return {}, {"theme": theme, "css": css}
    return {"theme": theme, "css": css}, {}


def render_header(engine_label: str = "", gpu_label: Optional[str] = None, version: str = "") -> str:
    pills = []
    if gpu_label:
        pills.append(f"<span class='mt-pill'><span class='mt-dot'></span><b>{gpu_label}</b></span>")
    else:
        pills.append("<span class='mt-pill warn'><span class='mt-dot'></span>GPU não detectada</span>")
    if engine_label:
        pills.append(f"<span class='mt-pill'>Motor <b>{engine_label}</b></span>")
    if version:
        pills.append(f"<span class='mt-pill'>v<b>{version}</b></span>")
    return (
        "<div id='matraca-header' class='matraca-header'>"
        "<div class='matraca-brand'><div class='matraca-logo'>🎙️</div>"
        "<div><h1 class='matraca-title'>Matraca Studio</h1>"
        "<p class='matraca-subtitle'>Dublagem e clonagem de voz com IA · voz coerente, parte por parte</p></div></div>"
        f"<div class='mt-pills'>{''.join(pills)}</div></div>"
    )


def render_engine_progress(pct: Optional[float] = None, label: str = "", detail: str = "",
                           state: str = "running") -> str:
    """Barra fixa no topo durante a troca de modelo. ``pct=None`` esconde a barra (estado ocioso).

    ``state``: running | done | error. O texto vem do gerenciador de motores, nunca de entrada do usuário,
    mas é escapado de qualquer forma.
    """
    if pct is None:
        return ""
    import html
    value = max(0, min(100, int(round(pct))))
    detail_html = f"<pre class='mt-progress-detail'>{html.escape(detail)}</pre>" if detail else ""
    return (
        f"<div class='mt-progress {html.escape(state)}' role='progressbar' aria-valuemin='0' aria-valuemax='100' "
        f"aria-valuenow='{value}' aria-label='{html.escape(label)}'>"
        f"<div class='mt-progress-head'><span class='mt-progress-label'>{html.escape(label)}</span>"
        f"<span class='mt-progress-pct'>{value}%</span></div>"
        f"<div class='mt-progress-track'><div class='mt-progress-fill' style='width:{value}%'></div></div>"
        f"{detail_html}</div>"
    )
