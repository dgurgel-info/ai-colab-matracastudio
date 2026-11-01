"""Sessão por partes: divisão, QC, repetição de semente, edição, takes e montagem."""
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from matraca import dsp  # noqa: E402
from matraca.parts import split_into_parts, split_sentences  # noqa: E402
from matraca.session import PartError, PartSession, SyncError  # noqa: E402
from test_matraca_dsp import voiced_like  # noqa: E402

SR = 24000
SEC_PER_CHAR = 0.065


class FakeEngine:
    """Motor sintético: mesma voz sempre, exceto textos com BAD na semente 42 (defeito real)."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, seed):
        self.calls.append((text, seed))
        seconds = max(1.0, len(text) * SEC_PER_CHAR)
        bad = "BAD" in text and seed == 42
        x = voiced_like(seconds=seconds, f0=190.0 if bad else 120.0, tilt=1.3 if bad else 0.2,
                        seed=seed, level=0.6 if bad else 0.2)
        return x, SR


def sentences(n, prefix="Frase", extra=""):
    return " ".join(f"{prefix} número {i} fala sobre o tema com calma e clareza total{extra}." for i in range(n))


def make_session(tmp_path, engine=None, **kw):
    return PartSession(str(tmp_path / "sess"), "pt-br", SR, synth_fn=engine or FakeEngine(), **kw)


PART_CHARS = len(sentences(3)) + 20


# ----------------------------------------------------------------------------- divisão
def test_split_keeps_sentences_whole_and_respects_size():
    text = sentences(30)
    parts = split_into_parts(text, target_chars=300, max_chars=500)
    assert len(parts) > 3
    assert all(len(p) <= 500 for p in parts)
    assert all(p.rstrip().endswith(".") for p in parts)
    assert " ".join(parts) == text


def test_split_does_not_break_on_abbreviations_or_numbers():
    s = split_sentences("O Dr. Silva chegou. Eram 3. Depois saiu. Sr. João ficou.")
    assert s == ["O Dr. Silva chegou.", "Eram 3. Depois saiu.", "Sr. João ficou."]


def test_split_long_sentence_and_cjk():
    long = "palavra " * 400
    parts = split_into_parts(long, target_chars=300, max_chars=400)
    assert all(len(p) <= 400 for p in parts) and len(parts) >= 8
    cjk = split_sentences("你好。今天天气很好！我们去公园吗？")
    assert len(cjk) == 3


def test_tiny_last_part_is_merged():
    parts = split_into_parts(sentences(4) + " Fim.", target_chars=len(sentences(4)) // 2 + 10, max_chars=900)
    assert all(len(p) > 8 for p in parts)


# ----------------------------------------------------------------------------- QC
def test_qc_flags_a_drifted_part_and_retry_fixes_it(tmp_path):
    engine = FakeEngine()
    s = make_session(tmp_path, engine, max_auto_retries=2)
    text = " ".join([sentences(3), sentences(3, "Outra"), sentences(3, "BAD"),
                     sentences(3, "Mais"), sentences(3, "Final")])
    s.plan(text, target_chars=PART_CHARS)
    assert len(s.parts) == 5
    bad_id = next(p.id for p in s.parts if "BAD" in p.text)

    events = []
    summary = s.synthesize_all(progress=events.append)

    assert all(c == 0 for k, c in summary["counts"].items() if k in ("flagged", "error", "pending", "stale"))
    bad = s.get(bad_id)
    assert len(bad.takes) == 2                     # a 1ª semente destoou; a alternativa foi usada
    assert bad.take.seed != 42 and bad.take.score <= 1.0
    requeue = [e for e in events if e["type"] == "part_requeue" and e["part"] == bad_id]
    assert requeue and ("tom" in requeue[0]["reason"] or "cor tonal" in requeue[0]["reason"])
    # as partes boas ficaram com a semente base, sem repetições
    assert all(len(p.takes) == 1 and p.take.seed == 42 for p in s.parts if p.id != bad_id)


def test_qc_without_retries_leaves_the_part_flagged_with_reasons(tmp_path):
    s = make_session(tmp_path, max_auto_retries=0)
    s.plan(" ".join([sentences(3), sentences(3, "Outra"), sentences(3, "BAD"), sentences(3, "Mais")]),
           target_chars=PART_CHARS)
    summary = s.synthesize_all()
    assert summary["counts"]["flagged"] == 1
    flagged = next(p for p in s.parts if p.status == "flagged")
    assert "BAD" in flagged.text
    assert any("tom" in r or "cor tonal" in r or "brilho" in r for r in flagged.take.reasons)


def test_clean_run_does_not_flag_or_retry(tmp_path):
    engine = FakeEngine()
    s = make_session(tmp_path, engine)
    s.plan(sentences(12), target_chars=200)
    summary = s.synthesize_all()
    assert summary["counts"]["flagged"] == 0 and summary["counts"]["error"] == 0
    assert all(seed == 42 for _, seed in engine.calls)   # semente única em todas as partes


def test_engine_failure_is_isolated_and_recoverable(tmp_path):
    class Flaky(FakeEngine):
        def __call__(self, text, seed):
            if "QUEBRA" in text and seed == 42:
                raise RuntimeError("limite de tokens atingido")
            return super().__call__(text, seed)

    s = make_session(tmp_path, Flaky(), max_auto_retries=0)
    s.plan(" ".join([sentences(3), sentences(3, "QUEBRA"), sentences(3, "Mais")]), target_chars=PART_CHARS)
    summary = s.synthesize_all()
    assert summary["counts"]["error"] == 1 and summary["counts"]["ok"] == 2
    bad = next(p for p in s.parts if p.status == "error")
    assert "limite de tokens" in bad.error
    with pytest.raises(PartError, match="sem áudio"):
        s.render()
    s.regenerate(bad.id)                       # nova semente resolve
    assert s.get(bad.id).status == "ok"
    s.render()


# ----------------------------------------------------------------------------- edição / takes
def test_edit_marks_stale_then_regenerate_and_choose_take(tmp_path):
    s = make_session(tmp_path)
    s.plan(sentences(9), target_chars=PART_CHARS)
    s.synthesize_all()
    pid = s.parts[1].id
    first_file = s.get(pid).take.file

    s.set_text(pid, "Texto corrigido manualmente pelo revisor.")
    assert s.get(pid).status == "stale"
    with pytest.raises(PartError, match="Regenere"):
        s.render()

    s.regenerate(pid)
    p = s.get(pid)
    assert p.status in ("ok", "flagged") and p.take.text == "Texto corrigido manualmente pelo revisor."
    assert len(p.takes) == 2 and p.take.seed == 43      # próxima semente, a base não é repetida

    s.regenerate(pid, seed=99)
    assert s.get(pid).take.seed == 99 and len(s.get(pid).takes) == 3
    s.choose_take(pid, 0)                               # volta ao take do texto antigo => desatualizado
    assert s.get(pid).status == "stale" and s.get(pid).take.file == first_file


def test_manifest_roundtrip_allows_render_without_engine(tmp_path):
    s = make_session(tmp_path)
    s.plan(sentences(9), target_chars=PART_CHARS)
    s.synthesize_all()
    expected = s.render(match_loudness=False).audio

    loaded = PartSession.load(s.root)           # sem motor / sem GPU
    assert [p.text for p in loaded.parts] == [p.text for p in s.parts]
    assert loaded.synth_fn is None
    np.testing.assert_allclose(loaded.render(match_loudness=False).audio, expected, atol=1e-6)
    with pytest.raises(PartError, match="motor"):
        loaded.regenerate(1)


# ----------------------------------------------------------------------------- montagem
def test_render_matches_loudness_across_parts_without_changing_timbre(tmp_path):
    class Uneven(FakeEngine):
        def __call__(self, text, seed):
            x, sr = super().__call__(text, seed)
            gain = 0.6 if "Baixa" in text else (1.6 if "Alta" in text else 1.0)
            return np.clip(x * gain, -0.95, 0.95), sr

    s = make_session(tmp_path, Uneven())
    s.plan(" ".join([sentences(3), sentences(3, "Baixa"), sentences(3, "Alta"), sentences(3, "Normal")]),
           target_chars=PART_CHARS)
    s.synthesize_all()
    raw = [dsp.loudness_lufs(sf.read(s.part_path(p.id), dtype="float32")[0], SR) for p in s.parts]
    r = s.render()
    per = [dsp.loudness_lufs(r.audio[int(a * SR):int(b * SR)], SR) for _, a, b in r.ranges]
    assert max(raw) - min(raw) > 6
    assert max(per) - min(per) < 2.0               # volumes aproximados após o ganho
    assert max(abs(g) for g in r.gains_db.values()) <= 6.0 + 1e-6
    assert dsp.peak_db(r.audio) <= -0.99           # sem clipping
    # só ganho: o tom (F0) de cada trecho permanece o do take original
    f0s = [dsp.median_f0_hz(r.audio[int(a * SR):int(b * SR)], SR) for _, a, b in r.ranges]
    assert all(abs(f - 120.0) < 4 for f in f0s)


def test_render_syncs_to_target_duration_with_uniform_speed(tmp_path):
    s = make_session(tmp_path, gap_ms=200)
    s.plan(sentences(9), target_chars=PART_CHARS)
    s.synthesize_all()
    natural = s.render().audio.shape[0] / SR
    target = natural * 0.9                          # precisa acelerar ~10%
    r = s.render(target_duration=target)
    assert r.audio.shape[0] == round(target * SR)
    assert r.speed == pytest.approx(natural / target, rel=0.02)
    assert dsp.median_f0_hz(r.audio, SR) == pytest.approx(120.0, rel=0.05)   # atempo não muda o tom
    assert r.ranges[-1][2] <= target + 1e-6

    longer = s.render(target_duration=natural * 1.1)   # mais lento, completa com silêncio
    assert longer.audio.shape[0] == round(natural * 1.1 * SR)

    with pytest.raises(SyncError, match="longa demais"):
        s.render(target_duration=natural * 0.6)


def test_gap_is_applied_between_parts_only(tmp_path):
    s = make_session(tmp_path, gap_ms=250)
    s.plan(sentences(9), target_chars=PART_CHARS)
    s.synthesize_all()
    r = s.render(match_loudness=False)
    n = len(s.parts)
    voice = sum(sf.info(s.part_path(p.id)).frames for p in s.parts)
    assert r.audio.shape[0] == voice + (n - 1) * int(SR * 0.25)


def test_two_parts_are_never_compared_against_each_other(tmp_path):
    engine = FakeEngine()
    s = make_session(tmp_path, engine, max_auto_retries=2)
    s.plan(" ".join([sentences(3), sentences(3, "BAD")]), target_chars=PART_CHARS)
    assert len(s.parts) == 2
    summary = s.synthesize_all()
    assert summary["relative_qc"] is False
    assert summary["counts"]["flagged"] == 0
    assert all(len(p.takes) == 1 for p in s.parts)        # ninguém foi refeito à toa
    assert len(engine.calls) == 2
