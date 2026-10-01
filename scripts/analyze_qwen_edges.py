"""Exercise the notebooks' real edge detector against local WAVs.

No media is uploaded or modified. Requires torch, torchaudio, soundfile,
numpy and silero-vad==6.2.3. Example:
python scripts/analyze_qwen_edges.py original.wav dubbed.wav
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--seconds', type=float, default=45)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    helpers = []
    for label in ('T4', 'L4'):
        nb = json.loads((root / f'Matraca_Studio_{label}.ipynb').read_text(encoding='utf-8'))
        tree = ast.parse(''.join(nb['cells'][3]['source']))
        helpers.append(next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == 'trim_qwen_nonspeech_edges'))
    if ast.dump(helpers[0]) != ast.dump(helpers[1]):
        raise RuntimeError('The notebooks have different edge detectors')
    namespace = dict(torch=torch, torchaudio=torchaudio, qwen_vad_model=None)
    exec(compile(ast.Module(body=[helpers[0]], type_ignores=[]), 'notebook', 'exec'), namespace)
    for path in args.files:
        info = sf.info(path)
        audio, rate = sf.read(path, frames=round(args.seconds*info.samplerate),
                              dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        original = torch.from_numpy(audio).unsqueeze(0)
        result = namespace['trim_qwen_nonspeech_edges'](original, rate)
        # Find the retained start from the unchanged first sample sequence.
        candidates = np.flatnonzero(audio == float(result[0, 0]))
        offset = next(int(i) for i in candidates if
                      i + result.numel() <= len(audio) and
                      torch.equal(original[:, i:i+result.numel()], result))
        assert torch.equal(original, torch.from_numpy(audio).unsqueeze(0))
        spectrum = np.abs(np.fft.rfft(audio[:2*rate]*np.hanning(min(len(audio), 2*rate))))**2
        freq = np.fft.rfftfreq(min(len(audio), 2*rate), 1/rate)
        removed = audio[:offset]
        print(json.dumps(dict(file=path.name, sample_rate=rate,
                              examined_seconds=len(audio)/rate,
                              trimmed_start_seconds=offset/rate,
                              trimmed_end_seconds=(len(audio)-offset-result.numel())/rate,
                              removed_start_rms=float(np.sqrt(np.mean(removed**2))) if offset else 0,
                              first2s_above6khz_fraction=float(spectrum[freq > 6000].sum()/max(spectrum.sum(), 1e-20)),
                              retained_samples_identical=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
