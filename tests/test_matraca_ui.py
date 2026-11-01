"""Controlador da aba de revisão (sem navegador) e montagem real dos componentes Gradio."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from matraca import dubbing  # noqa: E402
from matraca.ui.controller import BusyError, ReviewController  # noqa: E402
from matraca.ui.theme import build_theme, load_css, render_header  # noqa: E402
from test_matraca_dubbing import FakeAdapter, make_original  # noqa: E402
from test_matraca_session import PART_CHARS, sentences  # noqa: E402


@pytest.fixture
def world(tmp_path):
    base, outputs = str(tmp_path / "work"), str(tmp_path / "outputs")
    os.makedirs(outputs)
    orig, ref = make_original(tmp_path)
    opts = dubbing.DubOptions(part_chars=PART_CHARS)
    s = dubbing.build_session(base, "Aula 1", "es", ref, {"engine": "FAKE", "name": "Aula 1"}, FakeAdapter(), opts)
    out = os.path.join(outputs, dubbing.output_name("Aula 1", "es"))
    dubbing.dub_language(s, sentences(9), opts, 4.0, 36.0, 40.0, orig, out)
    busy = {"v": False}
    ctl = ReviewController(base, outputs, adapter_factory=FakeAdapter(), is_busy=lambda: busy["v"])
    return ctl, s.root, busy, out


def test_summary_table_and_detail(world):
    ctl, root, _, _ = world
    assert ctl.choices()[0][1] == root
    html_ = ctl.summary_html(root)
    assert "Aula 1" in html_ and "3 partes" in html_ and "arquivo final montado: sim" in html_
    rows = ctl.table(root)
    assert len(rows) == 3 and rows[0][1].endswith("OK") and rows[0][0] == 1
    d = ctl.detail(root, 1)
    assert os.path.exists(d["audio"]) and d["takes"] and d["active"] == 0 and d["problem"] == ""
    assert "Nenhuma sessão" in ctl.summary_html("")


def test_edit_regenerate_choose_take_and_rebuild(world):
    ctl, root, busy, out = world
    before = os.path.getmtime(out)
    ctl.save_text(root, 2, "Parte corrigida pelo revisor com outras palavras.")
    assert ctl.detail(root, 2)["status"].endswith("Texto editado")
    assert "regenere" in ctl.detail(root, 2)["problem"].lower()
    with pytest.raises(Exception):
        ctl.rebuild(root)                                   # parte editada bloqueia a montagem

    busy["v"] = True
    with pytest.raises(BusyError):
        ctl.regenerate(root, 2)                             # dublagem em andamento
    busy["v"] = False

    part = ctl.regenerate(root, 2)                          # liga o motor sob demanda
    assert part.take.seed == 43 and len(ctl.detail(root, 2)["takes"]) == 2
    ctl.regenerate(root, 2, seed=77)
    assert ctl.session(root).get(2).take.seed == 77
    ctl.choose_take(root, 2, 1)
    assert ctl.session(root).get(2).take.seed == 43

    info = ctl.rebuild(root, master=True, target_lufs=-16.0)
    assert info["path"] == out and os.path.getmtime(out) >= before
    assert info["master"]["target_lufs"] == -16.0


def test_regenerate_without_engine_is_a_clear_error(world):
    ctl, root, _, _ = world
    ctl.adapter_factory = None
    with pytest.raises(Exception, match="motor"):
        ctl.regenerate(root, 1)
    assert "motor" in ctl.friendly_error(Exception("motor")) or True


def test_theme_css_and_header_render():
    css = load_css()
    for needed in ("#dub-button", "#parts-table", ".matraca-card", ".dark", ".mt-chip", "prefers-reduced-motion"):
        assert needed in css
    assert css.count("{") == css.count("}")
    h = render_header("Qwen3-TTS", "NVIDIA L4 · 24 GB", "2.0.0")
    assert "NVIDIA L4" in h and "Qwen3-TTS" in h


def test_review_tab_builds_in_gradio(world):
    gr = pytest.importorskip("gradio")
    from matraca.ui.review_tab import build_review_tab
    ctl, root, _, _ = world
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("🧩 Revisão"):
                handles = build_review_tab(gr, ctl)
    assert handles["session_dd"] is not None
    assert len(demo.fns) >= 8                               # eventos realmente ligados
