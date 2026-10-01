"""Regressões de clonagem e áudio sem baixar pesos nem iniciar Gradio.

Execute com Python + numpy + torch: python -m unittest discover -s tests -v
"""
import ast
import gc
import json
from pathlib import Path
import re
import time
import unittest
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def notebook_namespace(label):
    nb = json.loads((ROOT / f'Matraca_Studio_{label}.ipynb').read_text(encoding='utf-8'))
    engine = ''.join(nb['cells'][3]['source'])
    parsed = ast.parse(engine)
    names = {'limit_audio_peak', 'fade_audio_edges', 'split_text_into_chunks',
             'generate_qwen3_tts_chunked_stream', 'generate_qwen3_tts_chunked'}
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    nodes += [n for n in parsed.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'QWEN_LANGUAGES' for t in n.targets)]
    namespace = dict(torch=torch, np=np, re=re, gc=gc, time=time,
                     trim_qwen_nonspeech_edges=lambda t: t, is_stop_requested=lambda: False, print=lambda *a, **k: None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), label, 'exec'), namespace)
    return nb, namespace


class FakeModel:
    def __init__(self, raw=None, sr=24000):
        self.prompts = []
        self.calls = []
        self.seeds = []
        self.prompt = [object()]
        self.raw = raw
        self.sr = sr

    def create_voice_clone_prompt(self, **kwargs):
        self.prompts.append(kwargs)
        return self.prompt

    def generate_voice_clone(self, **kwargs):
        self.calls.append(kwargs)
        self.seeds.append(torch.initial_seed())
        if self.raw is not None:
            return [self.raw], self.sr
        amplitude = 0.12 if len(self.calls) % 2 else 0.03
        wave = amplitude * torch.sin(torch.arange(2400) * 0.1) + 0.02
        wave += torch.rand(2400) * 0.001
        return [wave], self.sr


class QwenNotebookTests(unittest.TestCase):
    def each(self):
        for label in ('T4', 'L4'):
            yield label, notebook_namespace(label)[1]

    def test_all_cells_parse_and_shared_qwen_implementation(self):
        sources = []
        for label in ('T4', 'L4'):
            nb, ns = notebook_namespace(label)
            for cell in nb['cells']:
                if cell['cell_type'] == 'code':
                    code = '\n'.join(line for line in ''.join(cell['source']).splitlines()
                                     if not line.lstrip().startswith(('!', '%')))
                    ast.parse(code)
            tree = ast.parse(''.join(nb['cells'][3]['source']))
            sources.append(ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                        and n.name == 'generate_qwen3_tts_chunked_stream')))
        self.assertEqual(*sources)

    def test_language_and_fixed_full_prompt_across_blocks(self):
        for label, ns in self.each():
            for code, language in [('en', 'English'), ('fr-FR', 'French'), ('es-ES', 'Spanish'),
                                   ('de', 'German'), ('pt-BR', 'Portuguese'), ('zh-CN', 'Chinese')]:
                with self.subTest(notebook=label, language=code):
                    model = FakeModel()
                    text = ('Uma frase completa para testar a voz e as transições. ' * 15).strip()
                    events = list(ns['generate_qwen3_tts_chunked_stream'](
                        model, text, 'reference.wav', 'Transcrição da voz original.', code))
                    self.assertEqual(len(model.prompts), 1)
                    self.assertFalse(model.prompts[0]['x_vector_only_mode'])
                    self.assertGreater(len(model.calls), 1)
                    self.assertEqual(model.seeds, [42] * len(model.calls))
                    for call in model.calls:
                        self.assertIs(call['voice_clone_prompt'], model.prompt)
                        self.assertEqual(call['language'], language)
                        self.assertNotIn('instruct_ids', call)
                        self.assertFalse(call['non_streaming_mode'])
                        self.assertTrue(call['subtalker_dosample'])
                        self.assertEqual(call['subtalker_temperature'], 0.9)
                    blocks = [t for event, _, t in events if event == 'chunk_done']
                    final = events[-1][2]
                    self.assertEqual(final.shape[-1], sum(t.shape[-1] for t in blocks) + 1920 * (len(blocks)-1))
                    cursor = 0
                    for i, block in enumerate(blocks):
                        self.assertEqual(float(block[0, 0]), 0)
                        self.assertEqual(float(block[0, -1]), 0)
                        torch.testing.assert_close(final[:, cursor:cursor+block.shape[-1]], block)
                        cursor += block.shape[-1]
                        if i < len(blocks)-1:
                            self.assertEqual(float(final[:, cursor:cursor+1920].abs().max()), 0)
                            cursor += 1920
                    # Não nivelar o bloco mais silencioso ao mais alto.
                    self.assertLess(float(blocks[1].abs().max()), float(blocks[0].abs().max()) * 0.3)

    def test_peak_limiter_does_not_amplify_noise_and_fades_preserve_duration(self):
        for label, ns in self.each():
            with self.subTest(notebook=label):
                quiet = torch.randn(1, 2400) * 0.001
                torch.testing.assert_close(ns['limit_audio_peak'](quiet), quiet)
                self.assertLessEqual(float(ns['limit_audio_peak'](torch.tensor([[2.0, -2.0]])).abs().max()), 0.950001)
                for sr in (24000, 48000):
                    raw = torch.ones(1, sr)
                    result = ns['fade_audio_edges'](raw, sample_rate=sr)
                    self.assertEqual(result.shape, raw.shape)
                    self.assertEqual(result[0, 0], 0)
                    self.assertEqual(result[0, -1], 0)
                    self.assertEqual(result[0, sr // 2], 1)
                    self.assertEqual(raw[0, 0], 1)

    def test_qwen_groups_short_sentences_without_restarting_clone(self):
        # This passage used to require several separate model calls.
        sentence = 'Uma frase completa descreve a configuração e explica o próximo passo. '
        text = (sentence * 9).strip()
        self.assertGreater(len(text), 600)
        self.assertLessEqual(len(text), 700)
        for label, ns in self.each():
            for code in ('es', 'fr', 'en'):
                with self.subTest(notebook=label, language=code):
                    model = FakeModel()
                    events = list(ns['generate_qwen3_tts_chunked_stream'](
                        model, text, 'ref.wav', 'Fala original.', code))
                    self.assertEqual(len(model.calls), 1)
                    self.assertEqual(model.calls[0]['text'], text)
                    self.assertEqual(events[-1][2].qwen_chunk_lengths, [2400])
            model = FakeModel()
            long_text = text + ' ' + text
            list(ns['generate_qwen3_tts_chunked_stream'](
                model, long_text, 'ref.wav', 'Fala original.', 'es'))
            self.assertGreater(len(model.calls), 1)
            self.assertEqual(' '.join(c['text'] for c in model.calls), long_text)
            self.assertTrue(all(len(c['text']) <= 700 for c in model.calls))
            model = FakeModel()
            list(ns['generate_qwen3_tts_chunked_stream'](
                model, text, 'ref.wav', 'Fala original.', 'zh-CN'))
            self.assertGreater(len(model.calls), 1)
            self.assertTrue(all(len(c['text']) <= 320 for c in model.calls))

    def test_bad_reference_and_output_fail_explicitly(self):
        for label, ns in self.each():
            for text, reference, language in [('', 'Fala.', 'fr'), ('Texte.', '', 'fr'), ('Text.', 'Fala.', 'ar')]:
                model = FakeModel()
                with self.subTest(notebook=label, text=text, reference=reference, language=language), self.assertRaises(ValueError):
                    list(ns['generate_qwen3_tts_chunked_stream'](model, text, 'ref.wav', reference, language))
                self.assertFalse(model.calls)
            for raw in [torch.empty(0), torch.zeros(2400), torch.full((2400,), float('nan')),
                        torch.full((2400,), float('inf')), torch.ones(2400), torch.ones(2, 2400)]:
                with self.subTest(notebook=label, shape=raw.shape), self.assertRaises(ValueError):
                    list(ns['generate_qwen3_tts_chunked_stream'](FakeModel(raw), 'Bonjour.', 'ref.wav', 'Fala.', 'fr'))

    def test_cancel_and_prompt_failure_never_fall_back(self):
        for label, ns in self.each():
            model = FakeModel()
            ns['is_stop_requested'] = lambda: True
            with self.assertRaises(InterruptedError):
                list(ns['generate_qwen3_tts_chunked_stream'](model, 'Hola.', 'ref.wav', 'Fala.', 'es'))
            self.assertFalse(model.prompts)
            ns['is_stop_requested'] = lambda: False
            def fail(**kwargs):
                raise RuntimeError('Falha na referência')
            model.create_voice_clone_prompt = fail
            with self.assertRaises(RuntimeError):
                list(ns['generate_qwen3_tts_chunked_stream'](model, 'Hola.', 'ref.wav', 'Fala.', 'es'))
            self.assertFalse(model.calls)

    def test_reproducible_output_and_external_rng_preserved(self):
        for label, ns in self.each():
            before = torch.random.get_rng_state().clone()
            results = [ns['generate_qwen3_tts_chunked'](FakeModel(), 'Bonjour. ' * 50, 'ref.wav', 'Fala.', 'fr') for _ in range(2)]
            torch.testing.assert_close(*results)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_resampler_receives_native_rate_before_join(self):
        for label, ns in self.each():
            calls = []
            def resample(orig_freq, new_freq):
                calls.append((orig_freq, new_freq))
                return lambda t: torch.nn.functional.interpolate(
                    t.unsqueeze(0), size=int(t.shape[-1] * new_freq / orig_freq),
                    mode='linear', align_corners=False).squeeze(0)
            ns['torchaudio'] = SimpleNamespace(transforms=SimpleNamespace(Resample=resample))
            result = ns['generate_qwen3_tts_chunked'](FakeModel(sr=48000), 'Hola.', 'ref.wav', 'Fala.', 'es')
            self.assertEqual(calls, [(48000, 24000)])
            self.assertEqual(result.shape, (1, 1200))
            with self.assertRaises(ValueError):
                ns['generate_qwen3_tts_chunked'](FakeModel(sr=0), 'Hola.', 'ref.wav', 'Fala.', 'es')


if __name__ == '__main__':
    unittest.main()
