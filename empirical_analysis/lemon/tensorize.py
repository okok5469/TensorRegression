"""Build a traceable subject EEG tensor from independently verified EC blocks."""

from pathlib import Path

import numpy as np

from .eeg_preprocess import compute_window_bandpower, get_eeg_channels, load_raw_brainvision
from .events import count_candidate_windows, diagnose_subject, make_candidate_windows


def tensorize_subject(eeg_dir: Path, channels: list[str], config: dict) -> tuple[np.ndarray, dict]:
    """Return channel × window × band features and subject-specific provenance."""
    eeg_dir = Path(eeg_dir)
    _, evidence = diagnose_subject(eeg_dir)
    event_config = config["events"]
    configured_code = event_config.get("eyes_closed_code")
    if configured_code is not None and configured_code != evidence["ec_code"]:
        raise ValueError("Configured eyes_closed_code conflicts with subject event evidence")
    windows = make_candidate_windows(evidence["ec_intervals"],
                                     trim_edge_sec=event_config.get("trim_edge_sec", 5.),
                                     window_sec=event_config.get("window_sec", 25.),
                                     n_windows=event_config.get("n_windows", 16))
    bands = config["bands"]
    if list(bands) != ["delta", "theta", "alpha", "beta", "gamma"]:
        raise ValueError("Bands must have the fixed delta/theta/alpha/beta/gamma order")
    psd = config["psd"]
    if psd.get("method", "welch") != "welch" or psd.get("average", "median") != "median":
        raise ValueError("The LEMON workflow requires median Welch PSD")
    raw = load_raw_brainvision(Path(evidence["files"]["vhdr"]),
                               eog_names=config.get("channels", {}).get("eog_names", ["VEOG"]))
    try:
        sfreq = float(raw.info["sfreq"])
        actual_duration = (raw.n_times - 1) / sfreq
        if (not np.isclose(sfreq, evidence["sampling_frequency"], rtol=1e-10, atol=0)
                or abs(actual_duration - evidence["recording_duration"]) > 1.01 / sfreq):
            raise ValueError("Raw sampling frequency/sample count disagrees with EEG sidecar duration")
        if not set(channels).issubset(get_eeg_channels(raw)):
            raise ValueError("Subject is missing requested common EEG channels")
        preprocess = config["preprocess"]
        options = {name: preprocess[name] for name in
                   ("target_sfreq", "l_freq", "h_freq", "average_reference") if name in preprocess}
        options.update({name: psd[name] for name in ("segment_sec", "overlap", "log_epsilon")
                        if name in psd})
        tensor = np.stack([compute_window_bandpower(raw, start, end, channels, bands, **options)
                           for start, end, _ in windows], axis=1)
        subject_matches = [part for part in eeg_dir.parts if part.startswith("sub-")]
        if len(subject_matches) != 1:
            raise ValueError("EEG directory must contain one explicit sub-* subject identifier")
        metadata = {
            "subject_id": subject_matches[0], "shape": list(tensor.shape),
            "channels": list(channels), "bands": list(bands),
            "n_ec_segments": len(evidence["ec_intervals"]),
            "n_candidate_windows": count_candidate_windows(
                evidence["ec_intervals"], trim_edge_sec=event_config.get("trim_edge_sec", 5.),
                window_sec=event_config.get("window_sec", 25.)),
            "windows": [{"window_index": index, "start_sec": start, "end_sec": end,
                         "source_ec_segment": segment}
                        for index, (start, end, segment) in enumerate(windows)],
            "event_evidence": evidence, "raw_sfreq": sfreq, "raw_n_times": int(raw.n_times),
            "raw_duration_sec": actual_duration, "preprocess": dict(preprocess), "psd": dict(psd),
            "band_edges_hz": {name: list(edges) for name, edges in bands.items()},
            "feature": "log_absolute_band_power", "window_convention": "[start_sec, end_sec)",
        }
        return tensor, metadata
    finally:
        raw.close()
