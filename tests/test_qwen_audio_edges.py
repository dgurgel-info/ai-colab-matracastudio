"""Verify that edge trimming leaves the speech samples untouched."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def edge_function(label, spans):
    nb = json.loads((ROOT / f'Matraca_Studio_{label}.ipynb').read_text(encoding='utf-8'))
    tree = ast.parse(''.join(nb['cells'][3]['source']))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'trim_qwen_nonspeech_edges')
    model = SimpleNamespace(cpu=lambda: model, eval=lambda: model)
    namespace = dict(torch=torch, qwen_vad_model=None,
                     torchaudio=SimpleNamespace(functional=SimpleNamespace(
                         resample=lambda t, a, b: torch.zeros(round(t.numel()*b/a)))))
    def timestamps(probe, vad, **kwargs):
        assert kwargs['sampling_rate'] == 16000
        assert kwargs['speech_pad_ms'] == 100
        return spans
    module = SimpleNamespace(load_silero_vad=lambda: model, get_speech_timestamps=timestamps)
    exec(compile(ast.Module(body=[node], type_ignores=[]), label, 'exec'), namespace)
    return namespace, module


class AudioEdgesTests(unittest.TestCase):
    def test_preserves_words_and_internal_pauses_exactly(self):
        for label in ('T4', 'L4'):
            with self.subTest(notebook=label):
                ns, module = edge_function(label, [{'start': 8000, 'end': 16000},
                                                  {'start': 20000, 'end': 24000}])
                raw = torch.randn(1, 48000)
                original = raw.clone()
                with patch.dict(sys.modules, silero_vad=module):
                    result = ns['trim_qwen_nonspeech_edges'](raw)
                torch.testing.assert_close(result, raw[:, 12000:36000], rtol=0, atol=0)
                torch.testing.assert_close(raw, original, rtol=0, atol=0)
                self.assertEqual(result.shape[-1], 24000)

    def test_rejects_noise_only_and_invalid_boundaries(self):
        for label in ('T4', 'L4'):
            for spans in ([], [{'start': 16000, 'end': 8000}]):
                ns, module = edge_function(label, spans)
                with patch.dict(sys.modules, silero_vad=module), self.assertRaises(ValueError):
                    ns['trim_qwen_nonspeech_edges'](torch.randn(1, 48000))

    def test_precision_requires_native_bf16_hardware(self):
        for label in ('T4', 'L4'):
            nb = json.loads((ROOT / f'Matraca_Studio_{label}.ipynb').read_text(encoding='utf-8'))
            tree = ast.parse(''.join(nb['cells'][2]['source']))
            assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'qwen_dtype'
                                      for t in n.targets))
            expression = compile(ast.Expression(body=assignment.value), label, 'eval')
            for available, capability, expected in [(True, (7, 5), torch.float16),
                                                    (True, (8, 9), torch.bfloat16),
                                                    (False, None, torch.float32)]:
                cuda = SimpleNamespace(is_available=lambda: available,
                                       get_device_capability=lambda: capability)
                fake = SimpleNamespace(cuda=cuda, bfloat16=torch.bfloat16)
                actual = eval(expression, dict(torch=fake,
                              dtype=torch.float16 if available else torch.float32))
                self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
