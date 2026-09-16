"""Evidence-based LEMON event decoding and nonoverlapping EC windows."""

from collections import Counter
import configparser
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def _one_file(directory: Path, pattern: str) -> Path:
    paths = sorted(Path(directory).glob(pattern))
    # Match inventory: a derived *_eeg_events.tsv is not the BIDS task events table.
    if pattern == "*_events.tsv":
        paths = [path for path in paths if not path.name.endswith("_eeg_events.tsv")]
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}; found {len(paths)}")
    return paths[0]


def _brainvision_sections(path: Path) -> configparser.ConfigParser:
    content = path.read_text(encoding="utf-8-sig")
    parser = configparser.ConfigParser(interpolation=None, delimiters=("=",))
    start = content.find("[Common Infos]")
    if start < 0:
        raise ValueError(f"Missing BrainVision Common Infos: {path}")
    parser.read_string(content[start:])
    return parser


def build_ec_intervals(
    events: pd.DataFrame, ec_trial_type: str, eo_trial_type: str, recording_duration: float,
) -> list[tuple[float, float]]:
    """Collapse repeated states; unrelated markers do not terminate an EC block."""
    if ec_trial_type == eo_trial_type or not np.isfinite(recording_duration) or recording_duration <= 0:
        raise ValueError("Distinct EO/EC labels and positive recording duration are required")
    if not {"onset", "trial_type"}.issubset(events.columns):
        raise ValueError("Events require onset and trial_type columns")
    onsets = pd.to_numeric(events["onset"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(onsets).all() or (onsets < 0).any()
            or (onsets > recording_duration).any() or (np.diff(onsets) < 0).any()):
        raise ValueError("Event onsets must be finite, ordered and within the recording")
    states = events.loc[events["trial_type"].isin([ec_trial_type, eo_trial_type]),
                        ["onset", "trial_type"]].copy()
    states["onset"] = pd.to_numeric(states["onset"], errors="raise")
    if states.empty or ec_trial_type not in set(states["trial_type"]):
        raise ValueError("No eyes-closed events")
    if (states.groupby("onset")["trial_type"].nunique() > 1).any():
        raise ValueError("Conflicting simultaneous EO/EC states")
    changes = states.loc[states["trial_type"].ne(states["trial_type"].shift())]
    starts = changes["onset"].to_numpy(dtype=float)
    ends = np.r_[starts[1:], recording_duration]
    return [(float(start), float(end))
            for start, end, label in zip(starts, ends, changes["trial_type"])
            if label == ec_trial_type and end > start]


def _candidate_windows(
    intervals, trim_edge_sec: float, window_sec: float,
) -> list[tuple[float, float, int]]:
    """Enumerate complete windows with the same boundaries for counting and selection."""
    if (not np.isfinite([trim_edge_sec, window_sec]).all() or trim_edge_sec < 0
            or window_sec <= 0):
        raise ValueError("Invalid window duration or trimming parameters")
    segments = np.asarray(intervals, dtype=float)
    if (segments.ndim != 2 or segments.shape[1] != 2 or not np.isfinite(segments).all()
            or (segments[:, 0] < 0).any() or (segments[:, 1] <= segments[:, 0]).any()):
        raise ValueError("Intervals must contain finite positive start/end segments")
    if (segments[1:, 0] < segments[:-1, 1]).any():
        raise ValueError("EC intervals must be in order without overlap")
    windows = []
    for index, (start, end) in enumerate(segments):
        usable_start, usable_end = start + trim_edge_sec, end - trim_edge_sec
        count = max(0, int(np.floor((usable_end - usable_start) / window_sec)))
        windows.extend((float(usable_start + offset * window_sec),
                        float(usable_start + (offset + 1) * window_sec), index)
                       for offset in range(count))
    return windows


def count_candidate_windows(intervals, trim_edge_sec: float = 5., window_sec: float = 25.) -> int:
    """Count all complete windows before the fixed analysis-window selection."""
    return len(_candidate_windows(intervals, trim_edge_sec, window_sec))


def make_candidate_windows(
    intervals, trim_edge_sec: float = 5., window_sec: float = 25., n_windows: int = 16,
) -> list[tuple[float, float, int]]:
    """Choose the first complete half-open windows without crossing an EC block."""
    if (isinstance(n_windows, bool) or not isinstance(n_windows, (int, np.integer))
            or n_windows < 1):
        raise ValueError("Invalid window count")
    windows = _candidate_windows(intervals, trim_edge_sec, window_sec)
    if len(windows) < n_windows:
        raise ValueError(f"Insufficient EC windows: {len(windows)} available, {n_windows} required")
    return windows[:n_windows]


def _semantic_labels(description: str) -> tuple[str, str]:
    mappings = re.findall(r"\b(S\s*\d+)\s*=\s*eyes[\s-]+(open|closed)\b", description, re.I)
    by_state = {state: {re.sub(r"\s+", "", code).upper()
                        for code, meaning in mappings if meaning.lower() == state}
                for state in ("open", "closed")}
    if any(len(codes) != 1 for codes in by_state.values()):
        raise ValueError("TaskDescription must explicitly provide unique EO/EC semantics")
    eo_label, ec_label = next(iter(by_state["open"])), next(iter(by_state["closed"]))
    if eo_label == ec_label:
        raise ValueError("TaskDescription gives conflicting EO/EC semantics")
    return ec_label, eo_label


def diagnose_subject(eeg_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Verify TSV values/samples against VMRK and explicit EEG sidecar semantics."""
    eeg_dir = Path(eeg_dir)
    paths = {name: _one_file(eeg_dir, pattern) for name, pattern in {
        "events_tsv": "*_events.tsv", "events_json": "*_events.json",
        "eeg_json": "*_eeg.json", "vhdr": "*.vhdr", "vmrk": "*.vmrk",
    }.items()}
    sidecar = json.loads(paths["eeg_json"].read_text(encoding="utf-8-sig"))
    events_sidecar = json.loads(paths["events_json"].read_text(encoding="utf-8-sig"))
    required = {"onset", "sample", "trial_type", "value"}
    if not {"sample", "trial_type", "value"}.issubset(events_sidecar):
        raise ValueError("events.json must document sample, trial_type and value")
    sample_description = str(events_sidecar["sample"].get("Description", ""))
    if not re.search(r"first\s+sample\s+is\s+0|zero[- ]based|0[- ]based", sample_description, re.I):
        raise ValueError("events.json must explicitly document zero-based sample indices")
    ec_label, eo_label = _semantic_labels(str(sidecar.get("TaskDescription", "")))
    sfreq = float(sidecar["SamplingFrequency"])
    duration = float(sidecar["RecordingDuration"])
    if not np.isfinite([sfreq, duration]).all() or min(sfreq, duration) <= 0:
        raise ValueError("Invalid sampling frequency or recording duration")
    header = _brainvision_sections(paths["vhdr"])
    marker_file = _brainvision_sections(paths["vmrk"])
    header_sfreq = 1e6 / float(header["Common Infos"]["SamplingInterval"])
    if not np.isclose(header_sfreq, sfreq, rtol=1e-10, atol=0.):
        raise ValueError("SamplingFrequency disagrees with VHDR SamplingInterval")
    header_common, marker_common = header["Common Infos"], marker_file["Common Infos"]
    if ((eeg_dir / header_common["MarkerFile"]).resolve() != paths["vmrk"].resolve()
            or header_common["DataFile"] != marker_common["DataFile"]):
        raise ValueError("VHDR/VMRK file references are inconsistent")
    events = pd.read_csv(paths["events_tsv"], sep="\t")
    if not required.issubset(events.columns) or events.empty:
        raise ValueError(f"Events TSV requires nonempty columns: {sorted(required)}")
    numeric = events[["onset", "sample", "value"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Event onset, sample and value must be finite")
    for column in ("sample", "value"):
        values = numeric[column].to_numpy(dtype=float)
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Events {column} must be nonnegative integers")
    if not np.allclose(numeric["onset"], numeric["sample"] / sfreq, rtol=0, atol=.51 / sfreq):
        raise ValueError("Event onset disagrees with the zero-based sample index")
    markers = []
    for key, text in marker_file["Marker Infos"].items():
        if not re.fullmatch(r"mk\d+", key):
            continue
        fields = text.split(",")
        if len(fields) < 3:
            raise ValueError(f"Invalid VMRK marker {key}")
        marker_type, description = (part.replace("\\1", ",") for part in fields[:2])
        sample = int(fields[2]) - 1  # BrainVision marker positions are one-based.
        code = re.fullmatch(r"S\s*(\d+)", description)
        if marker_type == "New Segment":
            continue
        if sample < 0 or marker_type != "Stimulus" or code is None:
            raise ValueError(f"Unsupported VMRK marker requiring review: {key}={text}")
        markers.append((sample, int(code.group(1)), f"{marker_type}/{description}"))
    expected = Counter(zip(numeric["sample"].astype(int), numeric["value"].astype(int)))
    observed = Counter((sample, value) for sample, value, _ in markers)
    if expected != observed:
        raise ValueError("TSV/VMRK marker sample/value correspondence is inconsistent")
    normalized = events["trial_type"].astype(str).map(
        lambda label: re.sub(r"\s+", "", label.rsplit("/", 1)[-1]).upper())
    state_evidence = {}
    for state, label in (("ec", ec_label), ("eo", eo_label)):
        selected = events.loc[normalized.eq(label)]
        if selected.empty or selected["trial_type"].nunique() != 1 or selected["value"].nunique() != 1:
            raise ValueError(f"Missing or inconsistent {state.upper()} trial_type/value mapping")
        value = int(selected["value"].iloc[0])
        descriptions = {description for _, code, description in markers if code == value}
        labels_for_value = set(normalized.loc[numeric["value"].eq(value)])
        if len(descriptions) != 1 or labels_for_value != {label}:
            raise ValueError(f"Ambiguous {state.upper()} value/marker mapping")
        state_evidence[state] = {"trial_type": str(selected["trial_type"].iloc[0]),
                                 "value": value, "brainvision_code": next(iter(descriptions)),
                                 "event_count": len(selected)}
    ec, eo = state_evidence["ec"], state_evidence["eo"]
    if ec["value"] == eo["value"]:
        raise ValueError("EO and EC cannot share a value")
    intervals = build_ec_intervals(events, ec["trial_type"], eo["trial_type"], duration)
    evidence = {
        "ec_code": ec["brainvision_code"], "eo_code": eo["brainvision_code"],
        "ec_intervals": intervals, "mapping": state_evidence,
        "sampling_frequency": sfreq, "recording_duration": duration,
        "task_description": sidecar["TaskDescription"],
        "sample_convention": "TSV zero-based; VMRK one-based; verified row correspondence",
        "verified_event_count": len(events),
        "files": {name: str(path.resolve()) for name, path in paths.items()},
    }
    return events, evidence
