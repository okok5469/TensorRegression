from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

from empirical_analysis.lemon.config import load_config


class LemonPipelineTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('empirical_analysis.lemon.pipeline')
        except ModuleNotFoundError as exc:
            self.fail('Pipeline is not implemented: ' + str(exc))

    def fixture(self, root):
        (root / 'data/sub-1/eeg').mkdir(parents=True)
        (root / 'TMT.csv').write_text('ID,TMT_5\nsub-1,30\n', encoding='utf-8')
        (root / 'config.yaml').write_text('seed: 7\noutput_dir: out\ndata:\n  eeg_root: data\n  tmt_csv: TMT.csv\n  manifest_file: null\n  participants_file: null\n  participants_json: null\nanalysis:\n  mode: pilot\nevents:\n  min_diagnostic_subjects: 5\n', encoding='utf-8')
        return load_config(root / 'config.yaml')

    def test_cannot_skip_required_stage(self):
        pipeline = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'inventory'):
                pipeline.run_stage(cfg, 'tmt')

    def test_inventory_and_tmt_write_auditable_outputs(self):
        pipeline = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            pipeline.run_stage(cfg, 'inventory')
            pipeline.run_stage(cfg, 'tmt')
            state = json.loads((root / 'out/run_state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['completed'], ['inventory', 'tmt'])
            self.assertTrue((root / 'out/diagnostics/eeg_inventory.csv').is_file())
            self.assertTrue((root / 'out/config_used.json').is_file())

    def test_changed_input_cannot_reuse_previous_stage(self):
        pipeline = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            pipeline.run_stage(cfg, 'inventory')
            (root / 'TMT.csv').write_text('ID,TMT_5\nsub-1,31\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                pipeline.run_stage(cfg, 'tmt')

    def test_deleted_completed_artifact_cannot_be_reported_as_success(self):
        pipeline = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            pipeline.run_stage(cfg, 'inventory')
            (root / 'out/diagnostics/eeg_inventory.csv').unlink()
            with self.assertRaisesRegex(ValueError, 'artifact'):
                pipeline.run_stage(cfg, 'inventory')

    def test_no_complete_eeg_stops_and_saves_diagnostic(self):
        pipeline = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            pipeline.run_stage(cfg, 'inventory')
            pipeline.run_stage(cfg, 'tmt')
            with self.assertRaisesRegex(ValueError, 'complete'):
                pipeline.run_stage(cfg, 'events')
            self.assertTrue((root / 'out/logs/events_error.json').is_file())
            state = json.loads((root / 'out/run_state.json').read_text(encoding='utf-8'))
            self.assertNotIn('events', state['completed'])

    def test_inventory_obeys_explicit_download_selection(self):
        pipeline = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            (root / 'data/sub-2/eeg').mkdir(parents=True)
            selection = root / 'selection.json'
            selection.write_text(json.dumps({'selected_subjects': ['sub-2']}), encoding='utf-8')
            cfg['paths']['subject_selection_file'] = selection
            result = pipeline.run_stage(cfg, 'inventory')
            self.assertEqual(result['subjects_in_inventory'], 1)

    def test_programmatic_configuration_changes_invalidate_run(self):
        pipeline = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            pipeline.run_stage(cfg, 'inventory')
            cfg['seed'] = 99
            with self.assertRaisesRegex(ValueError, 'changed'):
                pipeline.run_stage(cfg, 'tmt')


if __name__ == '__main__':
    unittest.main()
