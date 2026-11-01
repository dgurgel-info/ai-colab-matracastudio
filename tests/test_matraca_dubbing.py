"""Fluxo completo de dublagem por partes com adaptador de motor falso (sem GPU)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from matraca import dsp, dubbing  # noqa: E402
from matraca.session import PartError  # noqa: E402
from test_matraca_session import FakeEngine, PART_CHARS, SR, sentences  # noqa: E402


class FakeAdapter:
    """Imita o adaptador de notebook: devolve (synth, speaker, taxa) a partir dos metadados."""

    def __init__(self, rate=SR):
        self.rate = rate
        self.engine = FakeEngine()
        self.metas = []

    def __call__(self, meta):
        self.metas.append(meta)
        assert os.path.exists(meta["ref_path"])           # a referência foi preservada na sessão
        return self.engine, None, self.rate


def make_original(tmp_path, total=40.0):
    rng = np.random.default_rng(3)
    path = tmp_path / "orig.wav"
    sf.write(path, rng.normal(0, 0.05, int(total * SR)).astype(np.float32), SR, subtype="FLOAT")
    ref = tmp_path / "ref.wav"
    sf.write(ref, rng.normal(0, 0.1, SR * 6).astype(np.float32), SR, subtype="FLOAT")
    return str(path), str(ref)


def test_end_to_end_then_fix_a_part_later_without_the_original(tmp_path):
    base = str(tmp_path / "work")
    orig, ref = make_original(tmp_path)
    adapter = FakeAdapter()
    opts = dubbing.DubOptions(part_chars=PART_CHARS, max_retries=1)
    meta = {"engine": "FAKE", "ref_text": "texto de referência", "num_steps": 30}

    session = dubbing.build_session(base, "Meu Vídeo 1", "pt-BR", ref, meta, adapter, opts)
    out = str(tmp_path / "out" / dubbing.output_name("Meu_Video_1", "pt-BR"))
    events = []
    result = dubbing.dub_language(session, sentences(9), opts, 4.0, 36.0, 40.0, orig, out, progress=events.append)

    assert result["ok"] and os.path.basename(out) == "Meu_Video_1_pt_BR_dublado.wav"
    assert result["duration"] == pytest.approx(40.0, abs=0.01) and result["sample_rate"] == 48000
    assert [e["type"] for e in events].count("part_done") == len(session.parts) == 3
    assert adapter.metas[0]["lang_code"] == "pt-BR" and adapter.metas[0]["engine"] == "FAKE"

    # —— depois: lista, restaura, edita, regenera e remonta; sem original e sem ter o motor antes
    os.remove(orig)
    found = dubbing.list_sessions(base)
    assert len(found) == 1 and "pt-BR" in found[0][0]
    restored = dubbing.restore_session(found[0][1])
    assert restored.synth_fn is None
    restored.set_text(2, "Segunda parte reescrita pelo revisor com outras palavras.")
    with pytest.raises(PartError):
        restored.regenerate(2)                              # sem motor
    dubbing.attach_engine(restored, FakeAdapter())
    restored.regenerate(2)
    fixed = str(tmp_path / "out" / "fixed.wav")
    from matraca.timeline import render_final
    info = render_final(restored, fixed)
    assert info["duration"] == pytest.approx(40.0, abs=0.01)
    assert restored.get(2).take.seed == 43 and len(restored.get(2).takes) == 2


def test_failed_part_is_reported_not_published(tmp_path):
    class Broken(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.engine.__class__ = type("E", (FakeEngine,), {
                "__call__": lambda self, text, seed: (_ for _ in ()).throw(RuntimeError("sem VRAM"))
                if "falha" in text else FakeEngine.__call__(self, text, seed)})

    base = str(tmp_path / "work")
    orig, ref = make_original(tmp_path)
    opts = dubbing.DubOptions(part_chars=PART_CHARS, max_retries=0)
    s = dubbing.build_session(base, "v", "en", ref, {"engine": "FAKE"}, Broken(), opts)
    out = str(tmp_path / "x.wav")
    r = dubbing.dub_language(s, sentences(3) + " " + sentences(3, "falha") + " " + sentences(3, "ok"),
                             opts, 4.0, 36.0, 40.0, orig, out)
    assert r["ok"] is False and not os.path.exists(out)
    assert r["summary"]["counts"]["error"] == 1


def test_attach_engine_rejects_a_different_sample_rate(tmp_path):
    base = str(tmp_path / "work")
    orig, ref = make_original(tmp_path)
    opts = dubbing.DubOptions(part_chars=PART_CHARS)
    s = dubbing.build_session(base, "v", "en", ref, {"engine": "FAKE"}, FakeAdapter(SR), opts)
    dubbing.dub_language(s, sentences(6), opts, 4.0, 36.0, 40.0, orig, str(tmp_path / "o.wav"))
    restored = dubbing.restore_session(s.root)
    with pytest.raises(RuntimeError, match="Hz"):
        dubbing.attach_engine(restored, FakeAdapter(48000))


def test_new_dubbing_of_same_audio_replaces_old_session(tmp_path):
    base = str(tmp_path / "work")
    orig, ref = make_original(tmp_path)
    opts = dubbing.DubOptions(part_chars=PART_CHARS)
    for text in (sentences(6), sentences(3)):
        s = dubbing.build_session(base, "v", "en", ref, {"engine": "FAKE"}, FakeAdapter(), opts)
        dubbing.dub_language(s, text, opts, 4.0, 36.0, 40.0, orig, str(tmp_path / "o.wav"))
    assert len(dubbing.list_sessions(base)) == 1
    assert len(dubbing.restore_session(dubbing.list_sessions(base)[0][1]).parts) == 1


def test_sync_failure_keeps_parts_and_can_be_rebuilt_without_sync(tmp_path):
    base = str(tmp_path / "work")
    orig, ref = make_original(tmp_path)
    opts = dubbing.DubOptions(part_chars=PART_CHARS)
    s = dubbing.build_session(base, "v", "en", ref, {"engine": "FAKE"}, FakeAdapter(), opts)
    out = str(tmp_path / "o.wav")
    # fala (~9 partes) não cabe em 5 s de original: antes derrubava o job inteiro
    r = dubbing.dub_language(s, sentences(9), opts, 1.0, 6.0, 8.0, orig, out)
    assert r["ok"] is False and "longa demais" in r["sync_error"] and not os.path.exists(out)
    assert all(p.take is not None for p in s.parts)          # nada foi perdido

    from matraca.ui.controller import ReviewController
    ctl = ReviewController(base, str(tmp_path), adapter_factory=None)
    assert ctl.sync_enabled(s.root) is True
    info = ctl.rebuild(s.root, sync=False)
    assert info["speed"] == 1.0 and os.path.exists(info["path"])
