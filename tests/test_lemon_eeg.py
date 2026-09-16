"""Synthetic regression checks for the isolated LEMON EEG pipeline."""

import importlib
import json
from pathlib import Path
import tempfile
import unittest

import mne
import numpy as np
import pandas as pd


class LemonEEGTests(unittest.TestCase):
    def module(self, name):
        try:
            return importlib.import_module("empirical_analysis.lemon." + name)
        except ModuleNotFoundError as error:
            self.fail(f"LEMON implementation is missing: {error}")

    def fixture(self, directory, changes=None):
        eeg_dir = Path(directory) / "sub-010002" / "eeg"
        eeg_dir.mkdir(parents=True)
        stem = "sub-010002_task-resting"
        metadata = {
            "SamplingFrequency": 100.0,
            "RecordingDuration": 139.99,
            "TaskDescription": "S200=eyes open, S210=eyes closed.",
        }
        metadata.update(changes or {})
        (eeg_dir / f"{stem}_eeg.json").write_text(json.dumps(metadata), encoding="utf-8")
        (eeg_dir / f"{stem}_events.json").write_text(json.dumps({
            "sample": {"Description": "First sample is 0."},
            "trial_type": {"Description": "Original event."},
            "value": {"Description": "Event code."},
        }), encoding="utf-8")
        events = pd.DataFrame({
            "onset": [0., 2., 4., 20., 62., 64., 72., 74., 132.],
            "duration": [.01] * 9,
            "trial_type": ["Stimulus/S200", "Stimulus/S210", "Stimulus/S210",
                           "unrelated", "Stimulus/S200", "Stimulus/S200",
                           "Stimulus/S210", "Stimulus/S210", "Stimulus/S200"],
            "value": [3, 4, 4, 2, 3, 3, 4, 4, 3],
            "sample": [0, 200, 400, 2000, 6200, 6400, 7200, 7400, 13200],
        })
        events.to_csv(eeg_dir / f"{stem}_events.tsv", sep="\t", index=False)
        markers = [f"Mk{i}=Stimulus,S  {value},{sample + 1},1,0"
                   for i, (value, sample) in enumerate(zip(events.value, events['sample']), 1)]
        (eeg_dir / f"{stem}_eeg.vmrk").write_text(
            "Brain Vision Data Exchange Marker File, Version 1.0\n"
            f"[Common Infos]\nDataFile={stem}_eeg.eeg\n[Marker Infos]\n" +
            "\n".join(markers), encoding="utf-8")
        (eeg_dir / f"{stem}_eeg.vhdr").write_text(
            "Brain Vision Data Exchange Header File Version 1.0\n"
            f"[Common Infos]\nDataFile={stem}_eeg.eeg\nMarkerFile={stem}_eeg.vmrk\n"
            "SamplingInterval=10000\n", encoding="utf-8")
        return eeg_dir, events

    def test_cross_file_mapping_and_duplicate_states(self):
        module = self.module("events")
        with tempfile.TemporaryDirectory() as temporary:
            eeg_dir, expected = self.fixture(temporary)
            events, evidence = module.diagnose_subject(eeg_dir)
            pd.testing.assert_frame_equal(events, expected)
            self.assertEqual(evidence["ec_code"], "Stimulus/S  4")
            self.assertEqual(evidence["eo_code"], "Stimulus/S  3")
            self.assertEqual(evidence["ec_intervals"], [(2., 62.), (72., 132.)])

    def test_ambiguous_semantics_rejected(self):
        module = self.module("events")
        with tempfile.TemporaryDirectory() as temporary:
            eeg_dir, _ = self.fixture(temporary, {"TaskDescription": "alternating S200/S210"})
            with self.assertRaisesRegex(ValueError, "semantic|TaskDescription"):
                module.diagnose_subject(eeg_dir)

    def test_standard_events_table_excludes_nonbids_derived_table(self):
        module = self.module("events")
        with tempfile.TemporaryDirectory() as temporary:
            eeg_dir, expected = self.fixture(temporary)
            derived = eeg_dir / "sub-010002_task-resting_eeg_events.tsv"
            derived.write_text("onset\ttrial_type\n0\tderived\n", encoding="utf-8")
            events, evidence = module.diagnose_subject(eeg_dir)
            pd.testing.assert_frame_equal(events, expected)
            self.assertEqual(evidence["ec_code"], "Stimulus/S  4")
            self.assertTrue(derived.is_file())

    def test_marker_sample_mismatch_rejected(self):
        module = self.module("events")
        with tempfile.TemporaryDirectory() as temporary:
            eeg_dir, _ = self.fixture(temporary)
            marker = next(eeg_dir.glob("*.vmrk"))
            marker.write_text(marker.read_text().replace("S  4,201", "S  4,202"))
            with self.assertRaisesRegex(ValueError, "sample|VMRK|marker"):
                module.diagnose_subject(eeg_dir)

    def test_trial_value_mismatch_rejected(self):
        module = self.module("events")
        with tempfile.TemporaryDirectory() as temporary:
            eeg_dir, events = self.fixture(temporary)
            events.loc[2, "value"] = 3
            events.to_csv(next(eeg_dir.glob("*_events.tsv")), sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "value|mapping|marker"):
                module.diagnose_subject(eeg_dir)

    def test_sample_onset_mismatch_rejected(self):
        module = self.module("events")
        with tempfile.TemporaryDirectory() as temporary:
            eeg_dir, events = self.fixture(temporary)
            events.loc[2, "onset"] = 5.
            events.to_csv(next(eeg_dir.glob("*_events.tsv")), sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "onset|sample"):
                module.diagnose_subject(eeg_dir)

    def test_final_ec_uses_recording_end(self):
        module = self.module("events")
        events = pd.DataFrame({"onset": [0., 1., 3., 5.],
                               "trial_type": ["EO", "EC", "other", "EC"]})
        self.assertEqual(module.build_ec_intervals(events, "EC", "EO", 30.), [(1., 30.)])

    def test_conflicting_simultaneous_states_rejected(self):
        module = self.module("events")
        events = pd.DataFrame({"onset": [0., 0.], "trial_type": ["EO", "EC"]})
        with self.assertRaisesRegex(ValueError, "simultaneous|conflict"):
            module.build_ec_intervals(events, "EC", "EO", 30.)

    def test_windows_respect_segments_and_trim(self):
        module = self.module("events")
        self.assertEqual(module.make_candidate_windows([(2., 62.), (72., 132.)], n_windows=4),
                         [(7., 32., 0), (32., 57., 0), (77., 102., 1), (102., 127., 1)])

    def test_short_terminal_segment_cannot_supply_window(self):
        module = self.module("events")
        with self.assertRaisesRegex(ValueError, "Insufficient|insufficient"):
            module.make_candidate_windows([(0., 60.), (70., 75.)], n_windows=3)

    def test_overlapping_segments_rejected(self):
        module = self.module("events")
        with self.assertRaisesRegex(ValueError, "overlap|order"):
            module.make_candidate_windows([(0., 60.), (50., 110.)], n_windows=2)

    def raw(self, sfreq=250., duration=30.):
        time = np.arange(int(sfreq * duration)) / sfreq
        data = np.array([np.sin(2 * np.pi * 10 * time),
                         np.sin(2 * np.pi * 6 * time),
                         50 * np.sin(2 * np.pi * 10 * time)]) * 1e-6
        return mne.io.RawArray(data, mne.create_info(["C3", "C4", "VEOG"], sfreq,
                                                   ["eeg", "eeg", "eog"]), verbose=False)

    def bands(self):
        return {"delta": [1., 4.], "theta": [4., 8.], "alpha": [8., 13.],
                "beta": [13., 30.], "gamma": [30., 45.]}

    def test_eog_excluded_and_order_preserved(self):
        module = self.module("eeg_preprocess")
        raw = self.raw()
        self.assertEqual(module.get_eeg_channels(raw), ["C3", "C4"])
        result = module.compute_window_bandpower(raw, 0., 25., ["C4", "C3"], self.bands(),
                                                average_reference=False)
        self.assertEqual(result.shape, (2, 5))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(np.argmax(result, axis=1).tolist(), [1, 2])
        self.assertEqual(raw.n_times, 7500)

    def test_half_open_window_excludes_endpoint_impulse(self):
        module = self.module("eeg_preprocess")
        clean = self.raw()
        contaminated = clean.copy()
        contaminated._data[:, 6250:] = 1e3
        kwargs = dict(tmin=0., tmax=25., channels=["C3", "C4"], bands=self.bands(),
                      average_reference=False)
        np.testing.assert_array_equal(module.compute_window_bandpower(clean, **kwargs),
                                      module.compute_window_bandpower(contaminated, **kwargs))

    def test_average_reference_uses_only_selected_eeg(self):
        module = self.module("eeg_preprocess")
        result = module.compute_window_bandpower(self.raw(), 0., 25., ["C3", "C4"], self.bands())
        np.testing.assert_allclose(result[0], result[1], rtol=1e-12, atol=1e-12)

    def test_invalid_preprocessing_rejected(self):
        module = self.module("eeg_preprocess")
        for changes in ({"overlap": 1.}, {"target_sfreq": 80.}, {"log_epsilon": 0.},
                        {"segment_sec": 30.}, {"channels": ["VEOG"]},
                        {"tmax": 31.}):
            options = dict(tmin=0., tmax=25., channels=["C3", "C4"], bands=self.bands())
            options.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                module.compute_window_bandpower(self.raw(), **options)

    def test_nonfinite_data_rejected(self):
        module = self.module("eeg_preprocess")
        raw = self.raw()
        raw._data[0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            module.compute_window_bandpower(raw, 0., 25., ["C3", "C4"], self.bands())

    def test_tensor_shape_and_actual_window_metadata(self):
        module = self.module("tensorize")
        with tempfile.TemporaryDirectory() as temporary:
            eeg_dir, _ = self.fixture(temporary)
            raw = self.raw(sfreq=100., duration=140.)
            header = next(eeg_dir.glob("*.vhdr"))
            header.write_text(header.read_text() +
                              "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=3\n"
                              "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n"
                              "[Channel Infos]\nCh1=C3,,1,V\nCh2=C4,,1,V\nCh3=VEOG,,1,V\n")
            raw.get_data().T.astype("<f4").tofile(header.with_suffix(".eeg"))
            config = {"channels": {"eog_names": ["VEOG"]},
                      "events": {"n_windows": 3, "trim_edge_sec": 5., "window_sec": 25.,
                                 "eyes_closed_code": None},
                      "preprocess": {"target_sfreq": 100., "l_freq": 1., "h_freq": 45.,
                                     "average_reference": True},
                      "bands": self.bands(), "psd": {"method": "welch", "average": "median",
                      "segment_sec": 4., "overlap": .5, "log_epsilon": 1e-20}}
            loaded = self.module("eeg_preprocess").load_raw_brainvision(header)
            self.assertFalse(loaded.preload)
            self.assertEqual(loaded.get_channel_types(), ["eeg", "eeg", "eog"])
            loaded.close()
            tensor, meta = module.tensorize_subject(eeg_dir, ["C4", "C3"], config)
            self.assertEqual(meta.get("n_ec_segments"), 2)
            self.assertEqual(meta.get("n_candidate_windows"), 4)
            self.assertEqual(tensor.shape, (2, 3, 5))
            self.assertEqual(meta["channels"], ["C4", "C3"])
            self.assertEqual(meta["bands"], list(self.bands()))
            self.assertEqual(meta["subject_id"], "sub-010002")
            self.assertEqual([entry["start_sec"] for entry in meta["windows"]], [7., 32., 77.])
            self.assertEqual(meta["windows"][2]["source_ec_segment"], 1)


if __name__ == "__main__":
    unittest.main()
