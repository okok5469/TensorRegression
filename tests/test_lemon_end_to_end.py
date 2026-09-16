"""Synthetic-only integration of all CLI stages; no scientific result fixture."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from empirical_analysis.lemon.config import load_config
from empirical_analysis.lemon.pipeline import run_stage, stage_names


class LemonEndToEndTests(unittest.TestCase):
    def test_all_stages_on_five_synthetic_brainvision_subjects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = Path(__file__).resolve().parents[1] / 'empirical_analysis/lemon/config.yaml'
            config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            config.update(output_dir='out')
            config['data'] = {'eeg_root': 'data', 'tmt_csv': 'TMT.csv', 'manifest_file': None,
                              'participants_file': None, 'participants_json': None}
            config['analysis'].update(mode='pilot', min_samples=5, formal_cv_folds=2)
            config['channels'].update(min_common_eeg_channels=2)
            config['model'].update(n_lambdas=2, cv_folds=2, max_iter=20, tol=1e-3, max_cycles=1, m_max=2)
            ids = [f'sub-{number:06d}' for number in range(10002, 10007)]
            for number, subject in enumerate(ids):
                self.write_subject(root / 'data' / subject / 'eeg', subject, number)
            pd.DataFrame({'ID': ids, 'TMT_5': [41., 55., 49., 62., 58.]}).to_csv(root / 'TMT.csv', index=False)
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            loaded = load_config(path)
            for stage in stage_names():
                run_stage(loaded, stage)
            state = json.loads((root / 'out/run_state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['completed'], list(stage_names()))
            self.assertEqual(state['stages']['dataset']['n'], 5)
            self.assertTrue((root / 'out/RUN_SUMMARY.md').is_file())
            oof = pd.read_csv(root / 'out/cv/oof_predictions.csv')
            self.assertEqual(len(oof), 15)
            self.assertTrue(np.isfinite(oof.predicted).all())
            self.assertEqual(oof.groupby(['model', 'subject_id']).size().unique().tolist(), [1])
            run_stage(loaded, 'report')  # Completed artifacts remain verifiable.

    def write_subject(self, eeg, subject, seed):
        eeg.mkdir(parents=True)
        stem = subject + '_task-resting'
        sfreq = 100
        sample_count = 960 * sfreq
        samples = np.random.default_rng(seed).normal(size=(sample_count, 3)) * (2 + seed)
        samples.astype('<f4').tofile(eeg / (stem + '_eeg.eeg'))
        header = (
            'Brain Vision Data Exchange Header File Version 1.0\n'
            '[Common Infos]\nCodepage=UTF-8\n'
            f'DataFile={stem}_eeg.eeg\nMarkerFile={stem}_eeg.vmrk\n'
            'DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=3\nSamplingInterval=10000\n'
            '[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n'
            '[Channel Infos]\nCh1=C3,,1,µV\nCh2=C4,,1,µV\nCh3=VEOG,,1,µV\n'
        )
        (eeg / (stem + '_eeg.vhdr')).write_text(header, encoding='utf-8')
        events = []
        for block in range(16):
            value, label = (4, 'S210') if block % 2 == 0 else (3, 'S200')
            for delta in (0, 2):
                onset = block * 60 + delta
                events.append({'onset': onset, 'duration': .01, 'trial_type': 'Stimulus/' + label,
                               'value': value, 'sample': onset * sfreq})
        pd.DataFrame(events).to_csv(eeg / (stem + '_events.tsv'), index=False, sep='\t')
        markers = '\n'.join(f'Mk{i}=Stimulus,S  {event["value"]},{event["sample"] + 1},1,0'
                            for i, event in enumerate(events, start=1))
        (eeg / (stem + '_eeg.vmrk')).write_text(
            'Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\n'
            f'DataFile={stem}_eeg.eeg\n[Marker Infos]\n' + markers, encoding='utf-8')
        sidecar = {'SamplingFrequency': sfreq, 'RecordingDuration': (sample_count - 1) / sfreq,
                   'TaskDescription': 'Synthetic test only. S200=eyes open, S210=eyes closed.'}
        (eeg / (stem + '_eeg.json')).write_text(json.dumps(sidecar), encoding='utf-8')
        (eeg / (stem + '_events.json')).write_text(json.dumps({
            'sample': {'Description': 'First sample is 0.'}, 'value': {}, 'trial_type': {}}), encoding='utf-8')
        (eeg / (stem + '_channels.tsv')).write_text('name\ttype\nC3\tEEG\nC4\tEEG\nVEOG\tEOG\n', encoding='utf-8')


if __name__ == '__main__':
    unittest.main()
