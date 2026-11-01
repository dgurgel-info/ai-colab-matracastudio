"""Um motor por vez: descarregamento completo antes de iniciar o próximo, falhas sem sobras e progresso."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matraca.engines import manager as mgr  # noqa: E402
from matraca.engines.manager import EngineManager, engine_name  # noqa: E402


class Model:
    def __init__(self, name):
        self.name, self.closed = name, False

    def close(self):
        self.closed = True


def make_manager(events, fail=None):
    models = {}

    def loader(name):
        def _load(log, progress):
            events.append(f"load:{name}")
            progress(70, f"carregando {name}", 90)
            if fail == name:
                raise RuntimeError("sem memória")
            models[name] = Model(name)
            return models[name]
        return _load

    names = [mgr.QWEN3, mgr.OMNIVOICE]
    manager = EngineManager({n: loader(n) for n in names}, free_memory=lambda: events.append("free"),
                            log=lambda m: None)
    return manager, models


def test_engine_name_accepts_ui_labels_and_canonical_names():
    assert engine_name("⚡ Qwen3-TTS 1.7B (Alibaba)") == mgr.QWEN3
    assert engine_name("🎙️ OmniVoice (k2-fsa)") == mgr.OMNIVOICE
    assert engine_name("Qwen3-TTS") == mgr.QWEN3


def test_switch_unloads_previous_before_loading_next():
    events = []
    manager, models = make_manager(events)
    manager.activate("OmniVoice")
    manager.activate("Qwen3-TTS")
    assert events == ["load:OmniVoice", "free", "load:Qwen3-TTS"]      # o anterior sai ANTES de o novo entrar
    assert models["OmniVoice"].closed is True
    assert manager.name == mgr.QWEN3 and manager.model is models["Qwen3-TTS"]


def test_same_engine_is_not_reloaded():
    events = []
    manager, _ = make_manager(events)
    first = manager.activate("Qwen3-TTS")
    again = manager.activate("⚡ Qwen3-TTS 1.7B (Alibaba)")
    assert first[1] is again[1]
    assert events.count("load:Qwen3-TTS") == 1


def test_only_the_selected_engine_is_ever_started():
    events = []
    manager, _ = make_manager(events)
    manager.activate("OmniVoice")
    assert events == ["load:OmniVoice"]


def test_failed_load_leaves_nothing_resident():
    events = []
    manager, models = make_manager(events, fail=mgr.QWEN3)
    manager.activate("OmniVoice")
    with pytest.raises(RuntimeError):
        manager.activate("Qwen3-TTS")
    assert manager.name is None and manager.model is None
    assert models["OmniVoice"].closed is True


def test_unload_closes_worker_and_is_safe_to_repeat():
    events = []
    manager, models = make_manager(events)
    manager.activate("Qwen3-TTS")
    manager.unload()
    manager.unload()
    assert models["Qwen3-TTS"].closed is True
    assert events.count("free") == 1


def test_close_error_does_not_block_unload():
    events = []
    manager, models = make_manager(events)
    manager.activate("Qwen3-TTS")

    def boom():
        raise OSError("servidor já morreu")
    models["Qwen3-TTS"].close = boom
    manager.unload()
    assert manager.model is None and "free" in events


def test_unknown_engine_is_rejected_without_touching_the_current_one():
    events = []
    manager, models = make_manager(events)
    manager.activate("OmniVoice")
    manager.loaders.pop(mgr.QWEN3)
    with pytest.raises(KeyError):
        manager.activate("Qwen3-TTS")
    assert manager.name == mgr.OMNIVOICE and models["OmniVoice"].closed is False


def test_progress_runs_forward_to_100_and_reports_the_unload():
    events, steps = [], []
    manager, _ = make_manager(events)
    manager.activate("OmniVoice", progress=lambda pct, label, until=None: steps.append((pct, label)))
    manager.activate("Qwen3-TTS", progress=lambda pct, label, until=None: steps.append((pct, label)))
    pcts = [p for p, _ in steps]
    assert steps[0][0] < 10 and pcts[-1] == 100
    assert any("Descarregando OmniVoice" in l for _, l in steps)       # a troca mostra o que sai
    second = pcts[pcts.index(100) + 1:]
    assert second == sorted(second)                                  # nunca recua dentro de uma troca


def test_already_loaded_engine_reports_ready_immediately():
    events, steps = [], []
    manager, _ = make_manager(events)
    manager.activate("OmniVoice")
    manager.activate("OmniVoice", progress=lambda pct, label, until=None: steps.append(pct))
    assert steps == [100] and events.count("load:OmniVoice") == 1


def test_build_manager_registers_the_engines(tmp_path):
    m = mgr.build_manager(str(tmp_path))
    assert set(m.loaders) == {mgr.QWEN3, mgr.OMNIVOICE}
