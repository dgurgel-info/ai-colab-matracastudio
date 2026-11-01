"""Ciclo de vida dos motores de voz: um por vez, cada um no seu venv.

Regras:
* Só um motor fica ativo. Ao trocar, o anterior é fechado (o processo do worker termina e o sistema
  devolve a VRAM) *antes* de o novo ser instalado e iniciado.
* Cada motor tem venv e dependências próprios (``venv_engine`` / ``higgs``); nada é instalado no
  ambiente do notebook por causa de um motor.
* Falha ao instalar ou iniciar não deixa nada pela metade: nenhum motor fica ativo.
* ``progress(percentual, texto, até)`` alimenta a barra de progresso da interface durante a troca.
"""

from __future__ import annotations

import gc
import os
import threading
from typing import Callable, Dict, Optional, Tuple

Log = Callable[[str], None]
Progress = Callable[..., None]

QWEN3, OMNIVOICE, HIGGS = "Qwen3-TTS", "OmniVoice", "Higgs TTS 3"


def engine_name(choice) -> str:
    """Nome canônico a partir do rótulo da interface (ou do nome já canônico)."""
    text = str(choice)
    if "Higgs" in text:
        return HIGGS
    if "Qwen" in text:
        return QWEN3
    return OMNIVOICE


def _noop_progress(*args, **kwargs) -> None:
    pass


class EngineManager:
    """Mantém exatamente um motor ativo. ``loaders`` mapeia nome -> ``loader(log, progress) -> modelo``."""

    def __init__(self, loaders: Dict[str, Callable[..., object]],
                 free_memory: Optional[Callable[[], None]] = None, log: Log = print):
        self.loaders = dict(loaders)
        self.free_memory = free_memory or gc.collect
        self.log = log
        self.name: Optional[str] = None
        self.model = None
        self._lock = threading.RLock()

    def activate(self, choice, log: Optional[Log] = None,
                 progress: Optional[Progress] = None) -> Tuple[str, object]:
        """Garante o motor escolhido: descarrega o anterior, instala (se preciso) e inicia o novo."""
        log, progress = log or self.log, progress or _noop_progress
        target = engine_name(choice)
        if target not in self.loaders:
            raise KeyError(f"Motor desconhecido: {choice}")
        with self._lock:
            if self.name == target and self.model is not None:
                progress(100, f"{target} já está pronto.")
                return target, self.model
            progress(2, "Preparando a troca de modelo...")
            self.unload(log, progress)
            try:
                model = self.loaders[target](log, progress)
            except BaseException:
                self.unload(log)        # nada de motor pela metade
                raise
            self.name, self.model = target, model
            progress(100, f"{target} pronto.")
            return target, model

    def unload(self, log: Optional[Log] = None, progress: Optional[Progress] = None) -> None:
        """Encerra o motor atual e libera a memória. Seguro se não houver motor."""
        log, progress = log or self.log, progress or _noop_progress
        with self._lock:
            name, model = self.name, self.model
            self.name = self.model = None
            if model is None:
                return
            progress(4, f"Descarregando {name} da memória...", 9)
            log(f"🧹 Descarregando [{name}] da memória...")
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()             # encerra o worker/servidor do motor
                except Exception as exc:
                    log(f"⚠️ Falha ao fechar [{name}]: {type(exc).__name__}: {exc}")
            del model
            self.free_memory()
            log("✅ Memória liberada!")
            progress(9, "Memória liberada.")


def make_higgs_loader(root: str) -> Callable[..., object]:
    def load(log: Log = print, progress: Optional[Progress] = None):
        from matraca.engines.higgs import HiggsServer
        progress = progress or _noop_progress
        server = HiggsServer(workdir=os.path.join(root, "higgs"))
        try:
            if not server.installed():
                progress(10, "Instalando o servidor do Higgs TTS 3 (só na primeira vez; vários minutos)...", 62)
            server.ensure_installed(log)
            progress(64, "Iniciando o Higgs TTS 3 (baixa o modelo na primeira vez)...", 97)
            server.start(log)
        except BaseException:
            server.stop()
            raise
        return server
    return load


def build_manager(root: str, free_memory: Optional[Callable[[], None]] = None, log: Log = print) -> EngineManager:
    """Gerenciador com os quatro motores, cada um no seu venv sob ``root``."""
    from matraca.engines.venv_engine import make_remote_loader
    loaders = {name: make_remote_loader(name, root) for name in (QWEN3, OMNIVOICE)}
    loaders[HIGGS] = make_higgs_loader(root)
    return EngineManager(loaders, free_memory=free_memory, log=log)
