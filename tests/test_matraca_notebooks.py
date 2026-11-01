"""Cola dos notebooks com o núcleo: semente fixa em todos os motores, prompt reutilizável e adaptador.

Executa a célula real do Passo 3 (motor) de cada notebook, com módulos pesados substituídos.
"""
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def engine_namespace(label):
    stub = types.ModuleType('deep_translator')
    stub.GoogleTranslator = stub.MyMemoryTranslator = type('T', (), {'__init__': lambda self, *a, **k: None})
    sys.modules.setdefault('deep_translator', stub)
    nb = json.loads((ROOT / f'Matraca_Studio_{label}.ipynb').read_text(encoding='utf-8'))
    ns = {'__name__': f'nb_{label}'}
    exec(compile(''.join(nb['cells'][3]['source']), label, 'exec'), ns)
    ns['trim_qwen_nonspeech_edges'] = lambda t, sample_rate=24000: t     # sem baixar o VAD
    ns['print'] = lambda *a, **k: None
    return ns


class FakeQwen:
    def __init__(self):
        self.prompts, self.calls, self.seeds = [], [], []
        self.model = types.SimpleNamespace(extract_speaker_embedding=lambda audio, sr: torch.ones(8))

    def create_voice_clone_prompt(self, **kw):
        self.prompts.append(kw)
        return [object()]

    def generate_voice_clone(self, **kw):
        self.calls.append(kw)
        self.seeds.append(torch.initial_seed())
        wave = 0.1 * torch.sin(torch.arange(24000) * 0.05) + 0.01 * torch.rand(24000)
        return [wave], 24000


class FakeOmni:
    def generate(self, **kw):
        return [0.1 * torch.randn(24000)]


class NotebookGlueTests(unittest.TestCase):
    def each(self):
        for label in ('T4', 'L4'):
            yield label, engine_namespace(label)

    def test_seeded_generation_is_reproducible_and_preserves_external_rng(self):
        for label, ns in self.each():
            before = torch.random.get_rng_state().clone()
            with ns['seeded_generation'](5):
                a = torch.rand(4)
            with ns['seeded_generation'](5):
                b = torch.rand(4)
            with ns['seeded_generation'](6):
                c = torch.rand(4)
            self.assertTrue(torch.equal(a, b), label)
            self.assertFalse(torch.equal(a, c), label)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()), label)

    def test_qwen_reuses_prompt_and_accepts_seed(self):
        for label, ns in self.each():
            model, prompt = FakeQwen(), [object()]
            list(ns['generate_qwen3_tts_chunked_stream'](
                model, 'Uma frase completa.', 'ref.wav', 'Fala.', 'es', voice_prompt=prompt, seed=7))
            self.assertEqual(model.prompts, [], label)                 # nenhum prompt novo
            self.assertIs(model.calls[0]['voice_clone_prompt'], prompt)
            self.assertEqual(model.seeds, [7])
            # padrão inalterado: 1 prompt por chamada e semente 42
            model = FakeQwen()
            list(ns['generate_qwen3_tts_chunked_stream'](model, 'Uma frase.', 'ref.wav', 'Fala.', 'es'))
            self.assertEqual(len(model.prompts), 1)
            self.assertEqual(model.seeds, [42])

    def test_omnivoice_is_deterministic_per_seed_and_defaults_to_42(self):
        for label, ns in self.each():
            gen = ns['generate_omnivoice_chunked']
            a = gen(FakeOmni(), 'Texto de teste.', 'r.wav', 'ref', num_steps=4, seed=3)
            b = gen(FakeOmni(), 'Texto de teste.', 'r.wav', 'ref', num_steps=4, seed=3)
            c = gen(FakeOmni(), 'Texto de teste.', 'r.wav', 'ref', num_steps=4, seed=4)
            d1 = gen(FakeOmni(), 'Texto de teste.', 'r.wav', 'ref', num_steps=4)
            d2 = gen(FakeOmni(), 'Texto de teste.', 'r.wav', 'ref', num_steps=4)
            torch.testing.assert_close(a, b)
            self.assertFalse(torch.allclose(a, c))
            torch.testing.assert_close(d1, d2)

    def test_adapter_returns_synth_speaker_and_native_rate(self):
        for label, ns in self.each():
            qwen, omni = FakeQwen(), FakeOmni()
            models = {'Qwen3-TTS': qwen, 'OmniVoice': omni}

            def load(choice):
                name = 'Qwen3-TTS' if 'Qwen' in choice else 'OmniVoice'
                return name, models[name]
            ns['load_tts_model'] = load
            base = {'lang_code': 'es', 'ref_path': 'ref.wav', 'ref_text': 'Fala.', 'num_steps': 4}

            synth, speaker, rate = ns['matraca_adapter_factory'](dict(base, engine='Qwen3-TTS 1.7B'))
            self.assertEqual(rate, 24000)
            self.assertIsNotNone(speaker)
            x1, sr1 = synth('Primeira parte do roteiro.', 42)
            x2, _ = synth('Segunda parte do roteiro.', 42)
            self.assertEqual(sr1, 24000)
            self.assertEqual(x1.ndim, 1)
            self.assertEqual(len(qwen.prompts), 1, 'o prompt de voz é criado uma vez por referência')
            self.assertIs(qwen.calls[0]['voice_clone_prompt'], qwen.calls[1]['voice_clone_prompt'])
            self.assertEqual(qwen.seeds, [42, 42])
            self.assertEqual(speaker(np.zeros(24000, np.float32), 24000).shape, (8,))
            synth(' mais uma', 43)
            self.assertEqual(qwen.seeds[-1], 43)
            # Se o modelo foi recarregado (novo objeto), o prompt é recriado
            models['Qwen3-TTS'] = FakeQwen()
            synth('Depois de trocar de motor.', 42)
            self.assertEqual(len(models['Qwen3-TTS'].prompts), 1)

            synth, speaker, rate = ns['matraca_adapter_factory'](dict(base, engine='OmniVoice'))
            self.assertEqual(rate, 24000)
            self.assertIsNone(speaker)
            a, _ = synth('Texto.', 5)
            b, _ = synth('Texto.', 5)
            np.testing.assert_allclose(a, b)

    def test_qwen_prompt_helper_matches_stream_rules(self):
        for label, ns in self.each():
            for code, english in [('en', True), ('en-US', True), ('pt-BR', False), ('es', False), ('zh-CN', False)]:
                m = FakeQwen()
                ns['qwen_clone_prompt'](m, 'ref.wav', 'Fala.', code)
                self.assertEqual(m.prompts[0]['x_vector_only_mode'], english, (label, code))
                s = FakeQwen()
                list(ns['generate_qwen3_tts_chunked_stream'](s, 'Frase.', 'ref.wav', 'Fala.', code))
                self.assertEqual(s.prompts[0]['x_vector_only_mode'], m.prompts[0]['x_vector_only_mode'])
            with self.assertRaises(ValueError):
                ns['qwen_clone_prompt'](FakeQwen(), 'ref.wav', '  ', 'es')
            with self.assertRaises(ValueError):
                ns['qwen_clone_prompt'](FakeQwen(), 'ref.wav', 'Fala.', 'ar')


if __name__ == '__main__':
    unittest.main()
