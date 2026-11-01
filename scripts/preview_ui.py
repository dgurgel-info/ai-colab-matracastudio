#!/usr/bin/env python3
"""Abre a interface do Matraca Studio no seu computador, sem GPU e sem internet.

Executa o código REAL dos Passos 3 e 4 do notebook (interface, fila de jobs, tradução em
partes, montagem, masterização, aba de revisão), trocando só o que exige GPU/internet:
  * Whisper         -> transcrição de exemplo
  * Tradutor        -> tradução de exemplo (prefixa o idioma)
  * VoxCPM2/Qwen3/OmniVoice -> voz sintética determinística por semente

Para demonstrar a revisão por partes, algumas partes saem "destoantes" na semente base
(tom e volume diferentes), como pode acontecer com um TTS de verdade. Desligue com --no-drift.

Uso:
    pip install gradio torch torchaudio numpy scipy soundfile requests imageio-ffmpeg
    python scripts/preview_ui.py                 # L4 (VoxCPM2, Qwen3, OmniVoice)
    python scripts/preview_ui.py --notebook T4 --port 7861
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import types
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", choices=["L4", "T4"], default="L4")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--workdir", default=None, help="pasta de trabalho (padrão: temporária)")
    ap.add_argument("--no-drift", action="store_true", help="não simular partes destoantes")
    ap.add_argument("--no-block", action="store_true", help="não bloqueia (uso em testes automatizados)")
    return ap.parse_args()


def ensure_ffmpeg():
    """Garante `ffmpeg` no PATH (usa o binário do imageio-ffmpeg se necessário)."""
    if shutil.which("ffmpeg"):
        return
    import imageio_ffmpeg

    src = imageio_ffmpeg.get_ffmpeg_exe()
    bindir = Path(tempfile.mkdtemp(prefix="matraca_ffbin_"))
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    shutil.copyfile(src, bindir / exe)
    os.chmod(bindir / exe, 0o755)
    os.environ["PATH"] = str(bindir) + os.pathsep + os.environ["PATH"]


def install_torchaudio_shims():
    """torchaudio recente exige torchcodec para load/save; aqui basta o soundfile."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    def load(path, *a, **k):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(data.T)), sr

    def save(path, tensor, sample_rate, *a, **k):
        arr = tensor.detach().cpu().numpy()
        sf.write(str(path), arr.T if arr.ndim == 2 else arr, int(sample_rate), subtype="PCM_16")

    torchaudio.load, torchaudio.save = load, save


# --------------------------------------------------------------------------- voz sintética
def synth_voice(text: str, sr: int, seed: int, drift: bool):
    """Fala sintética: harmônicas de um F0 com envelope silábico; duração proporcional ao texto."""
    import numpy as np

    rng = np.random.default_rng(int(seed))
    seconds = max(1.2, len(text) * 0.062)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    bad = drift and int(seed) == 42 and "proteja o seu sono" in text   # só uma parte destoa, na semente base
    f0 = (178.0 if bad else 118.0) * (1 + 0.04 * np.sin(2 * np.pi * 0.6 * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    tilt = 0.35 if bad else 1.15
    x = sum(np.sin(k * phase) / (k ** tilt) for k in range(1, 28))
    syll = np.clip(np.sin(2 * np.pi * 3.2 * t + rng.uniform(0, 6.28)), 0, None) ** 0.65
    x = x * (0.15 + 0.85 * syll) + rng.normal(0, 5e-4, n)
    x = x / np.max(np.abs(x)) * (0.62 if bad else 0.22)
    return x.astype("float32")


class FakeWhisper:
    TEXT = (
        "Olá, seja bem-vindo ao nosso canal. Hoje vamos falar sobre como organizar o seu dia para render mais "
        "sem abrir mão do descanso. Primeiro, defina três prioridades e escreva cada uma delas. "
        "Em seguida, reserve blocos de tempo curtos, sem notificações, para trabalhar de verdade. "
        "Muita gente acha que produtividade é fazer mais coisas, mas na prática é fazer as coisas certas. "
        "Pergunte-se toda manhã: o que realmente move o meu projeto para frente? "
        "Depois, proteja o seu sono, porque uma mente cansada decide mal e demora o dobro do tempo. "
        "No fim do dia, revise o que funcionou e ajuste o plano de amanhã. "
        "Se você gostou deste vídeo, deixe o seu comentário contando qual hábito vai testar primeiro, "
        "e não esqueça de se inscrever para acompanhar os próximos conteúdos. Até a próxima!"
    )

    def transcribe(self, path, language=None, task="transcribe", **kw):
        import soundfile as sf

        dur = sf.info(str(path)).duration
        start, end = min(2.5, dur * 0.1), max(dur - 2.0, dur * 0.8)
        # Texto proporcional à fala (a voz sintética fala ~16 caracteres por segundo)
        budget, sentences, used = (end - start) / 0.062, [], 0
        pool = [s.strip() for s in self.TEXT.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        while pool and used < budget:                       # repete as frases até cobrir a duração
            s = pool[len(sentences) % len(pool)]
            if len(sentences) >= len(pool):                 # 2ª volta: varia a frase gatilho do defeito
                s = s.replace("proteja o seu sono", "cuide do seu descanso")
            if sentences and used + len(s) >= budget:
                break
            sentences.append(s + ".")
            used += len(s) + 1
        text = " ".join(sentences)
        step = (end - start) / len(sentences)
        segs = [{"start": start + i * step, "end": start + (i + 1) * step - 0.1, "text": s}
                for i, s in enumerate(sentences)]
        return {"text": text, "segments": segs, "language": language or "pt"}


def make_fake_models(drift: bool):
    import numpy as np
    import torch

    from matraca import dsp

    class Vox:
        tts_model = types.SimpleNamespace(sample_rate=48000)

        def generate(self, text, **kw):
            text = re.sub(r"^\([^)]*\)\s*", "", text)      # o prefixo de sotaque não é falado
            return synth_voice(text, 48000, torch.initial_seed(), drift)

    class Qwen:
        def __init__(self):
            self.model = types.SimpleNamespace(extract_speaker_embedding=self._embed)

        def create_voice_clone_prompt(self, **kw):
            return [object()]

        def generate_voice_clone(self, text, **kw):
            return [torch.from_numpy(synth_voice(text, 24000, torch.initial_seed(), drift))], 24000

        @staticmethod
        def _embed(audio, sr):
            shape = np.nan_to_num(dsp.spectral_shape_db(np.asarray(audio, np.float32), int(sr)))
            return torch.from_numpy((shape + 20.0).astype("float32"))

    class Omni:
        def generate(self, text, **kw):
            return [torch.from_numpy(synth_voice(text, 24000, torch.initial_seed(), drift))]

    class Higgs:
        def synthesize(self, text, ref_audio, ref_text, seed=42):
            return synth_voice(text, 24000, seed, False), 24000

        def close(self):
            pass

    return {"VoxCPM2": Vox(), "Qwen3-TTS": Qwen(), "OmniVoice": Omni(), "Higgs TTS 3": Higgs()}


def main():
    args = parse_args()
    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="matraca_preview_"))
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    ensure_ffmpeg()

    import numpy as np
    import soundfile as sf
    import torch

    install_torchaudio_shims()

    # módulos que só existem no Colab
    sys.modules["whisper"] = types.SimpleNamespace(load_model=lambda *a, **k: FakeWhisper())
    dt = types.ModuleType("deep_translator")
    dt.GoogleTranslator = dt.MyMemoryTranslator = type("T", (), {"__init__": lambda self, *a, **k: None})
    sys.modules["deep_translator"] = dt

    nb = json.loads((ROOT / f"Matraca_Studio_{args.notebook}.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"]]
    models = make_fake_models(not args.no_drift)

    ns = {"__name__": "__main__", "np": np, "torch": torch}
    ns["whisper_model"] = FakeWhisper()
    ns["device"], ns["dtype"] = "cpu", torch.float32
    ns["current_tts_model_name"], ns["active_tts_model"] = None, None

    ns["engine_manager"] = types.SimpleNamespace(name=None, model=None)     # o Passo 2 real exige GPU

    def load_tts_model(choice, log=print, progress=None):
        import time
        c = str(choice)
        name = ("VoxCPM2" if "VoxCPM" in c else "Higgs TTS 3" if "Higgs" in c
                else "Qwen3-TTS" if "Qwen" in c else "OmniVoice")
        progress = progress or (lambda *a, **k: None)
        previous = ns["engine_manager"].name
        if previous and previous != name:                          # simula a troca para ver a barra de progresso
            progress(4, f"Descarregando {previous} da memória...", 9)
            time.sleep(1.2)
        for pct, label in ((12, "Criando o ambiente isolado..."), (30, "Instalando as dependências..."),
                           (66, f"Iniciando {name}..."), (92, "Carregando pesos...")):
            progress(pct, label)
            log(f"[prévia] {label}")
            time.sleep(1.0)
        ns["engine_manager"].name, ns["engine_manager"].model = name, models[name]
        progress(100, f"{name} pronto.")
        return name, models[name]

    ns["load_tts_model"] = load_tts_model

    exec(compile(cells[3], f"Passo3_{args.notebook}", "exec"), ns)       # motor (código real)
    ns["trim_qwen_nonspeech_edges"] = lambda t, sample_rate=24000: t       # sem baixar o VAD

    def get_audio_duration(path):
        try:
            return float(sf.info(str(path)).duration)
        except Exception:
            return 0.0

    def translate_text_robust(text, target_code, source_code="auto", progress_callback=None, **kw):
        if progress_callback:
            progress_callback(f"  [prévia] tradução de exemplo para {target_code}")
        sentences = [p.strip() + "." for p in text.replace("\n", " ").split(".") if p.strip()]
        return f"[{target_code}] " + " ".join(sentences)      # prévia: só marca o idioma de destino

    ns["get_audio_duration"], ns["translate_text_robust"] = get_audio_duration, translate_text_robust

    ui = cells[4]
    old = ("demo.queue(max_size=32, default_concurrency_limit=1).launch(\n    share=True,\n    debug=True,\n"
           "    allowed_paths=[OUTPUTS_DIR],\n    **LAUNCH_STYLE\n)")
    assert old in ui, "trecho de launch mudou; atualize scripts/preview_ui.py"
    ui = ui.replace(old, (
        "demo.queue(max_size=32, default_concurrency_limit=1).launch(\n"
        f"    server_name={args.host!r}, server_port={args.port}, share=False, inbrowser=False,\n"
        f"    prevent_thread_lock={bool(args.no_block)}, allowed_paths=[OUTPUTS_DIR, os.path.abspath('.')],\n"
        "    **LAUNCH_STYLE\n)"))
    print(f"\n🎛️  Prévia do Matraca Studio ({args.notebook}) — pasta de trabalho: {workdir}")
    print(f"👉 Abra http://{args.host}:{args.port}\n")
    exec(compile(ui, f"Passo4_{args.notebook}", "exec"), ns)
    return ns


if __name__ == "__main__":
    ns = main()
    if "--no-block" in sys.argv:
        import time
        while True:
            time.sleep(3600)
