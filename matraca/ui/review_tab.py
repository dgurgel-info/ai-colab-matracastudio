"""Aba “Revisão por partes” (Gradio). Toda a lógica está em ``ReviewController``."""

# Sem "from __future__ import annotations": o Gradio precisa ler o tipo gr.SelectData em tempo de execução.
import html
from typing import Optional

from .controller import TABLE_HEADERS, ReviewController


def build_review_tab(gr, controller: ReviewController) -> dict:
    """Cria os componentes da aba. Retorna ``{"refresh": fn, "refresh_outputs": [...], ...}``.

    Deve ser chamado dentro de ``with gr.Blocks()`` e de um ``gr.TabItem``.
    """

    def session_choices():
        return [(label, root) for label, root in controller.choices()]

    def pick_default_part(root: str) -> Optional[int]:
        if not root:
            return None
        s = controller.session(root)
        for wanted in ("error", "flagged", "stale"):
            for p in s.parts:
                if p.status == wanted:
                    return p.id
        return s.parts[0].id if s.parts else None

    with gr.Column(elem_id="review-workspace"):
        with gr.Column(elem_classes=["matraca-card"]):
            gr.HTML(
                "<div class='matraca-section-title'><span class='section-number violet'>🧩</span><div>"
                "<h2>Revisão por partes</h2><p>Ouça cada parte, corrija o texto, gere outra variação ou escolha "
                "outro take e remonte o áudio final — sem refazer o restante.</p></div></div>"
            )
            with gr.Row():
                session_dd = gr.Dropdown(label="Sessão de dublagem", choices=session_choices(), value=None,
                                         interactive=True, scale=5, allow_custom_value=False)
                btn_refresh = gr.Button("↻ Atualizar", scale=1, elem_classes=["mini-actions"])
            summary = gr.HTML(controller.summary_html(""))
            table = gr.Dataframe(headers=TABLE_HEADERS, value=[], interactive=False, wrap=True,
                                 elem_id="parts-table", label="Partes (clique numa linha para editar)",
                                 column_widths=["5%", "13%", "8%", "8%", "8%", "8%", "22%", "28%"],
                                 max_height=340)

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_classes=["matraca-card"]):
                gr.HTML("<div class='matraca-section-title'><span class='section-number'>✎</span><div>"
                        "<h2>Editar parte</h2><p>O texto editado só vale depois de regenerar.</p></div></div>")
                part_dd = gr.Dropdown(label="Parte", choices=[], value=None, interactive=True)
                problem = gr.HTML("")
                part_audio = gr.Audio(label="Take em uso", type="filepath", interactive=False)
                part_text = gr.Textbox(label="Texto da parte", lines=6, max_lines=14, elem_id="part-text")
                with gr.Row():
                    btn_save = gr.Button("💾 Salvar texto")
                    btn_regen = gr.Button("🔁 Regenerar (nova variação)", variant="primary")
                with gr.Row():
                    seed_in = gr.Textbox(label="Semente (opcional)", placeholder="ex.: 123", max_lines=1, scale=1)
                    btn_regen_seed = gr.Button("Regenerar com esta semente", scale=1)
            with gr.Column(scale=2, elem_classes=["matraca-card"]):
                gr.HTML("<div class='matraca-section-title'><span class='section-number blue'>↺</span><div>"
                        "<h2>Takes</h2><p>Cada regeneração vira um take. Volte a um anterior quando preferir.</p></div></div>")
                take_dd = gr.Dropdown(label="Takes desta parte", choices=[], value=None, interactive=True)
                btn_take = gr.Button("Usar este take")
                msg = gr.HTML("", elem_classes=["mt-msg"])

        with gr.Column(elem_classes=["matraca-card"]):
            gr.HTML("<div class='matraca-section-title'><span class='section-number violet'>▶</span><div>"
                    "<h2>Remontar o áudio final</h2><p>Junta as partes, iguala o volume entre elas, sincroniza com o "
                    "original e masteriza para o YouTube.</p></div></div>")
            with gr.Row():
                sync_cb = gr.Checkbox(label="Ajustar ao tempo do original (limite de 18%)", value=True)
                master_cb = gr.Checkbox(label="Masterizar para YouTube (loudness + pico seguro)", value=True)
                lufs_dd = gr.Dropdown(label="Loudness alvo", value=-14.0, scale=1,
                                      choices=[("-14 LUFS · YouTube", -14.0), ("-16 LUFS · podcast", -16.0),
                                               ("-18 LUFS · mais dinâmico", -18.0)])
            btn_rebuild = gr.Button("🎚️ Remontar áudio final", elem_id="rebuild-button")
            final_msg = gr.HTML("", elem_classes=["mt-msg"])
            final_audio = gr.Audio(label="Resultado", type="filepath", interactive=False)
            final_file = gr.File(label="Arquivo WAV", interactive=False)

    # ------------------------------------------------------------------ callbacks
    def _player(path):
        """Entrega o áudio ao player como (taxa, array): não depende de ffprobe/ffmpeg."""
        if not path:
            return None
        import soundfile as sf
        data, rate = sf.read(path, dtype="int16", always_2d=False)
        return (int(rate), data)

    def _problem_html(text):
        return f"<div class='mt-problem'>{html.escape(text)}</div>" if text else ""

    def _part_updates(root, pid):
        if not root or pid in (None, ""):
            return (gr.update(value=None), "", gr.update(choices=[], value=None), "")
        d = controller.detail(root, pid)
        return (_player(d["audio"]), d["text"], gr.update(choices=d["takes"], value=d["active"]),
                _problem_html(d["problem"]))

    def _tables(root):
        return controller.summary_html(root), controller.table(root)

    def refresh_sessions(current=None):
        choices = session_choices()
        values = [v for _, v in choices]
        value = current if current in values else (values[0] if values else None)
        return gr.update(choices=choices, value=value)

    def on_session(root):
        if not root:
            return controller.summary_html(""), [], gr.update(choices=[], value=None), gr.update()
        try:
            ids = controller.part_ids(root)
            first = pick_default_part(root)
            s, t = _tables(root)
            return (s, t, gr.update(choices=[(f"Parte {i}", i) for i in ids], value=first),
                    gr.update(value=controller.sync_enabled(root)))
        except Exception as exc:
            return (f"<div class='mt-empty'>{controller.friendly_error(exc)}</div>", [],
                    gr.update(choices=[], value=None), gr.update())

    def on_part(root, pid):
        try:
            audio, text, takes, prob = _part_updates(root, pid)
            return audio, text, takes, prob
        except Exception as exc:
            return None, "", gr.update(choices=[], value=None), _problem_html(controller.friendly_error(exc))

    def on_row(root, evt: gr.SelectData):
        ids = controller.part_ids(root)
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        return gr.update(value=ids[row]) if 0 <= row < len(ids) else gr.update()

    def _after_change(root, pid, message):
        s, t = _tables(root)
        audio, text, takes, prob = on_part(root, pid)
        return s, t, audio, text, takes, prob, message

    def do_save(root, pid, text):
        try:
            controller.save_text(root, pid, text)
            return _after_change(root, pid, "✏️ Texto salvo. Clique em <b>Regenerar</b> para ouvir a mudança.")
        except Exception as exc:
            return _after_change(root, pid, controller.friendly_error(exc))

    def do_regen(root, pid, text, seed=None, use_seed=False):
        try:
            seed_value = None
            if use_seed:
                if str(seed or "").strip() == "":
                    return _after_change(root, pid, "⚠️ Informe um número de semente.")
                if not str(seed).strip().lstrip("-").isdigit():
                    return _after_change(root, pid, "⚠️ A semente precisa ser um número inteiro.")
                seed_value = int(str(seed).strip())
            part = controller.regenerate(root, pid, text=text, seed=seed_value)
            note = "✅ Nova variação gerada." if part.status in ("ok",) else (
                "⚠️ Gerada, mas ainda fora do padrão das demais — tente outra variação." if part.status == "flagged"
                else f"Estado: {part.status}")
            return _after_change(root, pid, note)
        except Exception as exc:
            return _after_change(root, pid, controller.friendly_error(exc))

    def do_take(root, pid, index):
        try:
            if index in (None, ""):
                return _after_change(root, pid, "Escolha um take.")
            controller.choose_take(root, pid, index)
            return _after_change(root, pid, "↺ Take selecionado.")
        except Exception as exc:
            return _after_change(root, pid, controller.friendly_error(exc))

    def do_rebuild(root, sync, master, lufs):
        try:
            info = controller.rebuild(root, master=master, target_lufs=lufs, sync=sync)
            m = info.get("master")
            extra = (f" · masterizado de {m['input_lufs']:.1f} para {m['target_lufs']:.0f} LUFS" if m else "")
            note = (f"✅ Áudio remontado: <b>{info['duration']:.1f}s</b> · {info['sample_rate'] // 1000} kHz · "
                    f"ritmo ×{info['speed']:.2f}{extra}")
            return note, _player(info["path"]), info["path"], *_tables(root)
        except Exception as exc:
            s, t = _tables(root) if root else (controller.summary_html(""), [])
            return controller.friendly_error(exc), None, None, s, t

    # ------------------------------------------------------------------ ligações
    session_dd.change(on_session, [session_dd], [summary, table, part_dd, sync_cb], show_progress="hidden").then(
        on_part, [session_dd, part_dd], [part_audio, part_text, take_dd, problem], show_progress="hidden")
    btn_refresh.click(refresh_sessions, [session_dd], [session_dd], show_progress="hidden")
    part_dd.change(on_part, [session_dd, part_dd], [part_audio, part_text, take_dd, problem], show_progress="hidden")
    table.select(on_row, [session_dd], [part_dd], show_progress="hidden")

    edit_outputs = [summary, table, part_audio, part_text, take_dd, problem, msg]
    btn_save.click(do_save, [session_dd, part_dd, part_text], edit_outputs, show_progress="hidden")
    btn_regen.click(lambda r, p, t: do_regen(r, p, t), [session_dd, part_dd, part_text], edit_outputs)
    btn_regen_seed.click(lambda r, p, t, sd: do_regen(r, p, t, sd, True), [session_dd, part_dd, part_text, seed_in],
                         edit_outputs)
    btn_take.click(do_take, [session_dd, part_dd, take_dd], edit_outputs, show_progress="hidden")
    btn_rebuild.click(do_rebuild, [session_dd, sync_cb, master_cb, lufs_dd],
                      [final_msg, final_audio, final_file, summary, table])

    return {
        "session_dd": session_dd,
        "refresh": refresh_sessions,
        "refresh_inputs": [session_dd],
        "refresh_outputs": [session_dd],
    }
