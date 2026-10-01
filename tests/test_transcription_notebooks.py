"""Contratos de conversão e transcrição sem GPU, pesos ou serviços externos."""
import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


def load_transcription(label):
    notebook = json.loads((ROOT / f'Matraca_Studio_{label}.ipynb').read_text(encoding='utf-8'))
    functions = []
    for cell in notebook['cells'][3:5]:
        functions.extend(n for n in ast.parse(''.join(cell['source'])).body
                         if isinstance(n, ast.FunctionDef)
                         and n.name in {'extract_audio_to_wav', 'transcribe_only'})
    converter = Mock()
    whisper = Mock()
    whisper.transcribe.return_value = {'text': 'Texto transcrito.', 'language': 'pt',
                                       'segments': [{'start': 0.5}]}
    azure = Mock(return_value=('Texto do Azure.', 'pt-BR'))
    namespace = dict(subprocess=Mock(run=converter, DEVNULL=subprocess.DEVNULL),
                     tempfile=tempfile, os=os, get_audio_duration=lambda _: 12.0,
                     ORIG_LANG_CHOICES={'Português': 'pt', 'Automático': 'auto'},
                     whisper_model=whisper, transcribe_with_azure_speech=azure)
    exec(compile(ast.Module(body=functions, type_ignores=[]), label, 'exec'), namespace)
    return namespace, converter, whisper, azure


class TranscriptionTests(unittest.TestCase):
    def test_converter_accepts_asr_rate_and_preserves_cloning_default(self):
        for label in ('T4', 'L4'):
            with self.subTest(notebook=label):
                ns, ffmpeg, _, _ = load_transcription(label)
                for kwargs, expected in [({}, '24000'), ({'sample_rate': 16000}, '16000')]:
                    ns['extract_audio_to_wav']('input.mp3', 'output.wav', **kwargs)
                    cmd = ffmpeg.call_args.args[0]
                    self.assertEqual(cmd[cmd.index('-ar') + 1], expected)
                    self.assertEqual(cmd[cmd.index('-ac') + 1], '1')
                    self.assertEqual(cmd[cmd.index('-acodec') + 1], 'pcm_s16le')
                    self.assertTrue(ffmpeg.call_args.kwargs['check'])

    def test_transcription_providers_and_fallback_receive_16khz(self):
        for label in ('T4', 'L4'):
            for provider in ('whisper', 'azure', 'fallback'):
                with self.subTest(notebook=label, provider=provider):
                    ns, ffmpeg, whisper, azure = load_transcription(label)
                    if provider == 'fallback':
                        azure.side_effect = RuntimeError('Serviço indisponível')
                    text, status = ns['transcribe_only']('input.mp3', 'Português',
                                                        azure_speech_enabled=provider != 'whisper')
                    self.assertEqual(text, 'Texto do Azure.' if provider == 'azure' else 'Texto transcrito.')
                    self.assertIn('Transcrição concluída com sucesso', status)
                    cmd = ffmpeg.call_args.args[0]
                    self.assertEqual(cmd[cmd.index('-ar') + 1], '16000')
                    self.assertFalse(os.path.exists(cmd[-1]))
                    if provider == 'azure':
                        whisper.transcribe.assert_not_called()
                    else:
                        whisper.transcribe.assert_called_once_with(cmd[-1], language='pt', task='transcribe')
                    if provider == 'fallback':
                        self.assertIn('Azure AI Speech falhou', status)

    def test_auto_detection_and_failed_conversion_cleanup(self):
        for label in ('T4', 'L4'):
            with self.subTest(notebook=label):
                ns, ffmpeg, whisper, _ = load_transcription(label)
                ns['transcribe_only']('input.mp3', 'Automático')
                self.assertIsNone(whisper.transcribe.call_args.kwargs['language'])
                whisper.reset_mock()
                ffmpeg.side_effect = subprocess.CalledProcessError(1, 'ffmpeg')
                text, status = ns['transcribe_only']('input.mp3', 'Português')
                self.assertEqual(text, '')
                self.assertIn('CalledProcessError', status)
                self.assertFalse(os.path.exists(ffmpeg.call_args.args[0][-1]))
                whisper.transcribe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
