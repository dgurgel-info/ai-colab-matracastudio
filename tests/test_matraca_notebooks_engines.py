"""Notebooks L4/T4: cada motor no seu venv, nada de motor instalado/carregado no notebook, seleção troca o modelo."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cells(label):
    nb = json.loads((ROOT / f'Matraca_Studio_{label}.ipynb').read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in nb['cells']]


class OneEngineAtATimeNotebookTests(unittest.TestCase):
    def each(self):
        for label in ('L4', 'T4'):
            yield label, cells(label)

    def test_step1_installs_only_the_notebook_base(self):
        for label, nb in self.each():
            step1 = nb[1]
            body = '\n'.join(line for line in step1.splitlines() if line.startswith('!'))
            for engine_package in ('voxcpm', 'qwen-tts', 'omnivoice', 'transformers'):
                self.assertNotIn(engine_package, body, (label, engine_package))
            self.assertIn('openai-whisper', body)
            self.assertIn('uv', body.split())                                   # gerencia os venvs
            self.assertNotIn('higgs_audio_v2_tokenizer', step1, label)

    def test_step2_loads_only_whisper_and_builds_the_manager(self):
        for label, nb in self.each():
            step2 = nb[2]
            self.assertIn('engine_manager = build_manager(', step2, label)
            self.assertNotIn("load_tts_model('", step2, label)                  # nenhum motor no Passo 2
            for forbidden in ('from voxcpm', 'from qwen_tts', 'from omnivoice', 'transformers.', 'HiggsAudioV2'):
                self.assertNotIn(forbidden, step2, (label, forbidden))
            self.assertIn('def load_tts_model(model_choice, log=print, progress=None):', step2)

    def test_seed_reaches_the_worker_in_every_engine_path(self):
        for label, nb in self.each():
            step3 = nb[3]
            self.assertIn('ACTIVE_SEED.set(int(seed))', step3, label)
            self.assertIn('with seeded_generation(seed), torch.inference_mode():', step3, label)    # Qwen
            self.assertNotIn('rng_devices', step3, label)                       # sem RNG local que o worker não veria

    def test_both_model_selectors_switch_the_engine_and_show_progress(self):
        for label, nb in self.each():
            step4 = nb[4]
            self.assertIn('def switch_engine(model_choice):', step4, label)
            self.assertEqual(step4.count('fn=switch_engine'), 2, label)          # dublagem e clonagem livre
            self.assertEqual(step4.count("concurrency_id='engine'"), 2, label)
            bar = "engine_progress = gr.HTML(render_engine_progress(None), elem_id='engine-progress-slot')"
            self.assertIn(bar, step4, label)
            self.assertEqual(step4.count('engine_progress]'), 2, label)
            self.assertIn('job_manager.is_running()', step4.split('def switch_engine')[1][:400], label)
            self.assertLess(step4.index(bar), step4.index('gr.HTML(render_header('), label)   # barra acima do cabeçalho


if __name__ == '__main__':
    unittest.main()


class VoxCPM2RemovedTests(unittest.TestCase):
    """O VoxCPM2 saiu do projeto: nenhum vestígio nos notebooks e Qwen3-TTS é o motor inicial."""

    def test_no_trace_of_voxcpm2_in_the_notebooks(self):
        for label in ('L4', 'T4'):
            text = (ROOT / f'Matraca_Studio_{label}.ipynb').read_text(encoding='utf-8').lower()
            self.assertNotIn('voxcpm', text, label)

    def test_l4_offers_three_engines_with_qwen_as_default(self):
        step4 = cells('L4')[4]
        self.assertEqual(step4.count('🧪 Higgs TTS 3 4B'), 2)                  # dublagem e clonagem livre
        self.assertEqual(step4.count('⚡ Qwen3-TTS 1.7B'), 4)                  # opção + valor inicial, nos dois seletores
        self.assertEqual(step4.count("value='⚡ Qwen3-TTS 1.7B"), 2)
        self.assertIn("MATRACA_ENGINE_LABEL = 'Qwen3-TTS · OmniVoice · Higgs TTS 3'", step4)
