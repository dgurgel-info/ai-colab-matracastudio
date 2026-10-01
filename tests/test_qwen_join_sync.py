"""Real WAV/FFmpeg regression: synchronization must leave joins silent."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    return torch.from_numpy(audio.T.copy()), rate


def save(path, audio, rate):
    sf.write(path, audio.detach().numpy().T, rate, subtype='PCM_16')


def namespace(label):
    nb = json.loads((ROOT / f'Matraca_Studio_{label}.ipynb').read_text(encoding='utf8'))
    source = ''.join(nb['cells'][3]['source'])
    names = {'save_synth_wav', 'sync_qwen_blocks', 'time_sync_audio',
             'assemble_timeline_audio',
             'fade_audio_edges', 'limit_audio_peak', 'build_atempo_filter'}
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    executable = shutil.which('ffmpeg')
    if not executable:
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise unittest.SkipTest('FFmpeg unavailable')
    def run(cmd, **kwargs):
        return subprocess.run([executable, *cmd[1:]], **kwargs)
    ns = dict(torch=torch, json=json, os=os, tempfile=tempfile, shutil=shutil,
              torchaudio=SimpleNamespace(load=load, save=save),
              subprocess=SimpleNamespace(run=run, DEVNULL=subprocess.DEVNULL),
              is_stop_requested=lambda: False,
              get_audio_duration=lambda p: sf.info(p).duration)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), label, 'exec'), ns)
    return ns, nb


class JoinSyncTests(unittest.TestCase):
    def test_video_timeline_matches_original_and_preserves_intro_outro(self):
        for label in ('T4', 'L4'):
            with self.subTest(notebook=label), tempfile.TemporaryDirectory() as work:
                ns, _ = namespace(label)
                rate = 24000
                original = torch.full((1, 5*rate), .1)
                original[:, -rate//2:] = .2
                speech = .15*torch.sin(torch.arange(3*rate).float()*.1).unsqueeze(0)
                speech.qwen_chunk_lengths = [3*rate]
                source, raw, final = [os.path.join(work, n) for n in ('original.wav','raw.wav','final.wav')]
                save(source, original, rate)
                ns['save_synth_wav'](raw, speech)
                ns['assemble_timeline_audio'](source, raw, .5, 4.5, 5.0, final)
                result, actual_rate = load(final)
                self.assertEqual(actual_rate, rate)
                self.assertEqual(result.shape[-1], 5*rate)
                self.assertAlmostEqual(float(result[0,rate//4]), .1, places=4)
                self.assertAlmostEqual(float(result[0,-rate//4]), .2, places=4)
                self.assertGreater(float(result[:,rate:3*rate].square().mean().sqrt()), .05)

    def test_real_stretch_leaves_exact_silent_joins(self):
        for label in ('T4', 'L4'):
            with self.subTest(notebook=label), tempfile.TemporaryDirectory() as work:
                ns, _ = namespace(label)
                # Tone plus noise at the edges exposes WSOLA bleed into a gap.
                rng = np.random.default_rng(7)
                chunks = []
                for frequency in (220, 310, 440):
                    t = np.arange(24000) / 24000
                    wave = .2 * np.sin(2*np.pi*frequency*t) + rng.normal(0, .01, len(t))
                    chunks.append(torch.from_numpy(wave.astype('float32')).unsqueeze(0))
                gap = torch.zeros(1, 1920)
                audio = torch.cat([chunks[0], gap, chunks[1], gap, chunks[2]], dim=-1)
                audio.qwen_chunk_lengths = [24000] * 3
                source, output = os.path.join(work, 'source.wav'), os.path.join(work, 'out.wav')
                ns['save_synth_wav'](source, audio)
                target = 3 / .85 + .16
                ns['time_sync_audio'](source, target, output)
                result, sr = load(output)
                self.assertEqual(result.shape[-1], round(target*sr))
                zeros = result[0].numpy() == 0
                boundaries = np.diff(np.r_[False, zeros, False].astype(int))
                starts, ends = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
                interior = [(a,b) for a,b in zip(starts, ends)
                            if b-a >= 1920 and a > 0 and b < len(zeros)-100]
                self.assertEqual(len(interior), 2)
                for a,b in interior:
                    self.assertEqual(float(result[:, a:b].abs().max()), 0)
                    # The first nonzero PCM sample can differ by one quantization step.
                    self.assertLessEqual(abs(float(result[0,a-1])), 1/32768)
                    self.assertLessEqual(abs(float(result[0,b])), 1/32768)

    def test_metadata_dispatch_in_both_workflows_and_rejects_truncation(self):
        for label in ('T4', 'L4'):
            with self.subTest(notebook=label), tempfile.TemporaryDirectory() as work:
                ns, nb = namespace(label)
                source = '\n'.join(''.join(c['source']) for c in nb['cells'])
                self.assertIn('save_synth_wav(temp_synth_wav, audio_tensor,', source)
                self.assertIn('save_synth_wav(temp_sub_wav, sub_audio,', source)
                audio = torch.ones(1, 24000) * .1
                audio.qwen_chunk_lengths = [24000]
                path, out = os.path.join(work,'raw.wav'), os.path.join(work,'out.wav')
                ns['save_synth_wav'](path, audio)
                with self.assertRaises(ValueError):
                    ns['time_sync_audio'](path, .5, out)
                self.assertFalse(os.path.exists(out))
                ns['save_synth_wav'](path, torch.ones(1,24000)*.1)
                self.assertFalse(os.path.exists(path+'.qwen-blocks.json'))


if __name__ == '__main__':
    unittest.main()
