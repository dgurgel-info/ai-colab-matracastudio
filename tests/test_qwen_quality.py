"""Validação por janelas, rejeição/regeneração e DSP real em FFmpeg opcional."""
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from test_qwen_notebooks import FakeModel, notebook_namespace


def signal(seconds, frequency=220, amplitude=0.1):
    return amplitude * torch.sin(torch.arange(int(24000 * seconds)) * (2 * torch.pi * frequency / 24000))


class FrequencyEncoderModel(FakeModel):
    """Encoder simulado distingue dois tons; teste de fluxo, não de identidade humana."""
    def __init__(self, always_bad=False):
        super().__init__()
        self.always_bad = always_bad
        self.model = SimpleNamespace(extract_speaker_embedding=self.embedding)

    def embedding(self, audio, sr):
        self.assert_rate = sr
        power = abs(np.fft.rfft(audio))
        dominant = np.fft.rfftfreq(len(audio), 1 / sr)[power.argmax()]
        return torch.tensor([0.0, 1.0]) if dominant > 300 else torch.tensor([1.0, 0.0])

    def generate_voice_clone(self, **kwargs):
        self.calls.append(kwargs)
        # A média de todo o bloco mantém o primeiro tom; só a janela final revela a troca.
        if self.always_bad or len(self.calls) == 1:
            wave = torch.cat([signal(5), signal(3, frequency=440)])
        else:
            wave = signal(8)
        return [wave], 24000


class QualityTests(unittest.TestCase):
    def namespace(self, label):
        ns = notebook_namespace(label)[1]
        ns['prepare_qwen_reference'] = lambda audio, text: ((signal(4).numpy(), 24000), text)
        ns['clean_qwen_audio'] = lambda audio, **kwargs: audio
        return ns

    def test_window_detects_voice_change_hidden_by_whole_score(self):
        for label in ('T4', 'L4'):
            ns = self.namespace(label)
            model = FrequencyEncoderModel()
            whole, window = ns['qwen_voice_scores'](model, torch.cat([signal(5), signal(3, 440)]).unsqueeze(0), torch.tensor([1.0, 0.0]))
            self.assertGreater(whole, 0.9)
            self.assertLess(window, 0.1)

    def test_inconsistent_reference_is_rejected_before_synthesis(self):
        for label in ('T4', 'L4'):
            ns = self.namespace(label)
            ns['prepare_qwen_reference'] = lambda audio, text: ((torch.cat([signal(5), signal(3, 440)]).numpy(), 24000), text)
            model = FrequencyEncoderModel()
            with self.assertRaisesRegex(ValueError, 'Referência vocal inconsistente'):
                list(ns['generate_qwen3_tts_chunked_stream'](model, 'Uma frase.', 'ref.wav', 'Referência.', 'es'))
            self.assertFalse(model.calls)

    def test_regenerates_rejected_window_with_same_reference(self):
        for label in ('T4', 'L4'):
            for language in ('en', 'fr', 'es'):
                ns = self.namespace(label)
                model = FrequencyEncoderModel()
                events = list(ns['generate_qwen3_tts_chunked_stream'](model, 'Uma frase para sintetizar.', 'ref.wav', 'Referência.', language))
                self.assertEqual(len(model.calls), 2)
                self.assertIs(model.calls[0]['voice_clone_prompt'], model.calls[1]['voice_clone_prompt'])
                self.assertFalse(model.calls[0]['subtalker_dosample'])
                self.assertLess(model.calls[1]['temperature'], model.calls[0]['temperature'])
                self.assertEqual(sum(event == 'chunk_done' for event, _, _ in events), 1)
                self.assertEqual(events[-1][0], 'complete')
                self.assertIn('SIM=', next(message for event, message, _ in events if event == 'chunk_done'))

    def test_three_failed_attempts_never_deliver_final_audio(self):
        for label in ('T4', 'L4'):
            ns = self.namespace(label)
            model = FrequencyEncoderModel(always_bad=True)
            delivered = []
            with self.assertRaisesRegex(ValueError, '3 tentativas'):
                for event in ns['generate_qwen3_tts_chunked_stream'](model, 'Uma frase.', 'ref.wav', 'Referência.', 'fr'):
                    delivered.append(event)
            self.assertEqual(len(model.calls), 3)
            self.assertFalse(any(e[0] in ('chunk_done', 'complete') for e in delivered))

    def test_cancel_during_retry_stops_before_second_generation(self):
        ns = self.namespace('T4')
        model = FrequencyEncoderModel(always_bad=True)
        stream = ns['generate_qwen3_tts_chunked_stream'](model, 'Uma frase.', 'ref.wav', 'Referência.', 'es')
        for event, message, _ in stream:
            if event == 'header' and 'rejeitada' in message:
                ns['is_stop_requested'] = lambda: True
                break
        with self.assertRaises(InterruptedError):
            next(stream)
        self.assertEqual(len(model.calls), 1)

    def test_silence_trim_retains_margin_and_speech(self):
        for label in ('T4', 'L4'):
            ns = self.namespace(label)
            voice = signal(1)
            raw = torch.cat([torch.zeros(24000), voice, torch.zeros(24000)]).unsqueeze(0)
            result = ns['trim_qwen_padding'](raw)
            self.assertEqual(result.shape[-1], 24000 + 2 * 960)
            torch.testing.assert_close(result[:, 960:-960], voice.unsqueeze(0))

    def test_long_reference_is_retranscribed_at_16khz(self):
        for label in ('T4', 'L4'):
            ns = notebook_namespace(label)[1]
            ns['clean_qwen_audio'] = lambda t: t
            def resample(orig_freq, new_freq):
                return lambda t: torch.nn.functional.interpolate(t.unsqueeze(0), size=int(t.shape[-1] * new_freq / orig_freq), mode='linear', align_corners=False).squeeze(0)
            ns['torchaudio'] = SimpleNamespace(transforms=SimpleNamespace(Resample=resample))
            whisper = Mock()
            whisper.transcribe.return_value = {'text': 'Texto do recorte.'}
            ns['whisper_model'] = whisper
            (audio, rate), text = ns['prepare_qwen_reference']((signal(20).numpy(), 24000), 'Texto do arquivo inteiro.')
            self.assertEqual(len(audio), 24000 * 12)
            self.assertEqual(rate, 24000)
            self.assertEqual(text, 'Texto do recorte.')
            self.assertEqual(len(whisper.transcribe.call_args.args[0]), 16000 * 12)
            with self.assertRaisesRegex(ValueError, '3 segundos'):
                ns['prepare_qwen_reference']((signal(1).numpy(), 24000), 'Curto.')

    def test_rubberband_selection_and_explicit_fallback(self):
        for label in ('T4', 'L4'):
            for available in (True, False):
                ns = self.namespace(label)
                run = Mock(return_value=SimpleNamespace(stdout='rubberband' if available else 'atempo'))
                ns['subprocess'] = SimpleNamespace(run=run, PIPE=subprocess.PIPE)
                result = ns['build_quality_tempo_filter'](1.1)
                self.assertTrue(result.startswith('rubberband=' if available else 'atempo='))
                if available:
                    self.assertIn('pitch=1', result)
                    self.assertIn('formant=preserved', result)
                ns['build_quality_tempo_filter'](1.05)
                self.assertEqual(run.call_count, 1)


class RealFFmpegTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import imageio_ffmpeg
            cls.executable = imageio_ffmpeg.get_ffmpeg_exe()
            from scipy.io import wavfile
            cls.wavfile = wavfile
        except ImportError:
            raise unittest.SkipTest('Instale imageio-ffmpeg e scipy para DSP real.')

    def namespace(self, label):
        ns = notebook_namespace(label)[1]
        def run(cmd, **kwargs):
            cmd = [self.executable, *cmd[1:]]
            return subprocess.run(cmd, **kwargs)
        ns.update(subprocess=SimpleNamespace(run=run, DEVNULL=subprocess.DEVNULL, PIPE=subprocess.PIPE),
                  tempfile=tempfile, os=os)
        return ns

    def test_real_noise_filter_preserves_samples_and_reduces_pause_noise(self):
        generator = torch.Generator().manual_seed(7)
        noise = torch.randn(1, 24000 * 5, generator=generator) * 0.002
        audio = noise.clone()
        audio[:, 24000:24000 * 4] += signal(3).unsqueeze(0)
        for label in ('T4', 'L4'):
            ns = self.namespace(label)
            cleaned = ns['clean_qwen_audio'](audio)
            self.assertEqual(cleaned.shape, audio.shape)
            self.assertTrue(torch.isfinite(cleaned).all())
            # Ignore startup and speech onset; compare stationary pause energy.
            self.assertLess(float(cleaned[:, 6000:18000].square().mean()), float(audio[:, 6000:18000].square().mean()))
            self.assertGreater(float(cleaned[:, 48000:72000].square().mean()), float(audio[:, 48000:72000].square().mean()) * 0.8)

    def test_real_rubberband_preserves_tone_pitch(self):
        ns = self.namespace('L4')
        tempo = ns['build_quality_tempo_filter'](1.1)
        self.assertTrue(tempo.startswith('rubberband='))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.wav'
            output = Path(directory) / 'output.wav'
            self.wavfile.write(source, 24000, signal(3).numpy())
            subprocess.run([self.executable, '-y', '-i', str(source), '-af', tempo, '-c:a', 'pcm_f32le', str(output)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            rate, samples = self.wavfile.read(output)
        self.assertEqual(rate, 24000)
        self.assertAlmostEqual(len(samples) / rate, 3 / 1.1, delta=0.03)
        frequencies = np.fft.rfftfreq(len(samples), 1 / rate)
        dominant = frequencies[abs(np.fft.rfft(samples)).argmax()]
        self.assertAlmostEqual(dominant, 220, delta=1)


if __name__ == '__main__':
    unittest.main()
