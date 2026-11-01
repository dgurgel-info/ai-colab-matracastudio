"""Worker de mentira: usa o servidor real (matraca.engines.worker) com um modelo falso, sem GPU nem download.

    python fake_engine_worker.py <porta>
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matraca.engines import worker  # noqa: E402


class Prompt:
    def __init__(self, text):
        self.text = text


class FakeModel:
    tts_model = SimpleNamespace(sample_rate=48000)

    def __init__(self):
        self.model = SimpleNamespace(extract_speaker_embedding=self._embedding)

    def _embedding(self, audio, sr):
        import torch
        return torch.tensor([float(np.asarray(audio).mean()), float(sr)])

    def generate(self, **kw):
        import torch
        if kw.get("text") == "explode":
            raise ValueError("texto inválido")
        return (0.1 * torch.rand(8)).numpy()                       # depende da semente do pedido

    def create_voice_clone_prompt(self, **kw):
        return [Prompt(kw["ref_text"])]                           # objeto que NÃO cabe em JSON

    def generate_voice_clone(self, text, language, voice_clone_prompt, **kw):
        import torch
        return [torch.full((4,), float(len(voice_clone_prompt[0].text)))], 24000


if __name__ == "__main__":
    worker.serve("fake", int(sys.argv[1]), model=FakeModel())
