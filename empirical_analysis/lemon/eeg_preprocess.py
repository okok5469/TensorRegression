"""Window-local EEG preprocessing without loading an entire BrainVision recording."""

from pathlib import Path

import mne
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


def load_raw_brainvision(vhdr: Path, eog_names=("VEOG",)):
    """Read header only and assign explicit EOG channel types."""
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose=False)
    mapping = {name: "eog" for name in eog_names if name in raw.ch_names}
    raw.set_channel_types(mapping, verbose=False)
    return raw


def get_eeg_channels(raw) -> list[str]:
    """Return usable EEG channel names in the recording's order."""
    picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude="bads")
    return [raw.ch_names[index] for index in picks if raw.ch_names[index] != "VEOG"]


def compute_window_bandpower(
    raw, tmin, tmax, channels, bands, target_sfreq=250., l_freq=1., h_freq=45.,
    average_reference=True, segment_sec=4., overlap=.5, log_epsilon=1e-20,
) -> np.ndarray:
    """Return channel × band log absolute powers from samples in [tmin, tmax)."""
    sfreq = float(raw.info["sfreq"])
    parameters = [tmin, tmax, target_sfreq, l_freq, h_freq, segment_sec, overlap, log_epsilon]
    if (not np.isfinite(parameters).all() or not 0 <= tmin < tmax <= raw.n_times / sfreq
            or not 0 < l_freq < h_freq < min(sfreq, target_sfreq) / 2
            or segment_sec <= 0 or not 0 <= overlap < 1 or log_epsilon <= 0):
        raise ValueError("Invalid preprocessing parameters or window outside recording")
    if (not channels or len(set(channels)) != len(channels)
            or not set(channels).issubset(get_eeg_channels(raw))):
        raise ValueError("Requested channels must be unique, present, usable EEG channels")
    if average_reference and len(channels) < 2:
        raise ValueError("Average reference requires at least two EEG channels")
    edges = np.asarray(list(bands.values()), dtype=float)
    if (edges.ndim != 2 or edges.shape[1] != 2 or not np.isfinite(edges).all()
            or (edges[:, 0] >= edges[:, 1]).any()
            or (edges[1:, 0] < edges[:-1, 1]).any()
            or edges[0, 0] < l_freq or edges[-1, 1] > h_freq):
        raise ValueError("Bands must be ordered, nonoverlapping and within the filter range")
    # nextafter prevents an exact integer represented one ULP high from losing a sample.
    start = int(np.ceil(np.nextafter(tmin * sfreq, -np.inf)))
    stop = int(np.ceil(np.nextafter(tmax * sfreq, -np.inf)))
    if stop <= start:
        raise ValueError("Window contains no samples")
    chunk = raw.copy().crop(tmin=start / sfreq, tmax=(stop - 1) / sfreq,
                            include_tmax=True).load_data(verbose=False)
    chunk.pick(channels)
    if not np.isfinite(chunk.get_data()).all():
        raise ValueError("EEG window contains nonfinite data")
    chunk.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose=False)
    chunk.resample(target_sfreq, verbose=False)
    if average_reference:
        chunk.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    n_per_seg = int(round(segment_sec * target_sfreq))
    n_overlap = int(round(overlap * n_per_seg))
    if n_per_seg < 2 or n_per_seg > chunk.n_times or n_overlap >= n_per_seg:
        raise ValueError("Welch segment length or overlap is invalid for the window")
    freqs, psd = welch(chunk.get_data(), fs=target_sfreq, window="hamming", nperseg=n_per_seg,
                       noverlap=n_overlap, nfft=n_per_seg, detrend="constant",
                       scaling="density", average="median", axis=-1)
    power = []
    for index, (low, high) in enumerate(edges):
        mask = (freqs >= low) & ((freqs <= high) if index == len(edges) - 1 else (freqs < high))
        if np.count_nonzero(mask) < 2:
            raise ValueError("Welch frequency resolution supplies fewer than two bins in a band")
        power.append(trapezoid(psd[:, mask], freqs[mask], axis=-1))
    result = np.log(np.stack(power, axis=-1) + log_epsilon)
    if not np.isfinite(result).all():
        raise ValueError("Bandpower produced nonfinite tensor cells")
    return result
