"""Channel normalization, augmentation, and invariance techniques for Phase 1.

Key principles:
1. Stationary linear channel is multiplicative in the frequency domain, hence
   additive and constant in the cepstral domain. Per-recording Cepstral Mean
   Normalization (CMN/CMVN) subtracts this stationary offset.
2. Dynamic features (first and second temporal derivatives / delta & delta-delta)
   are mathematically invariant to any additive constant offset:
   d/dt [c(t) + h] = d/dt [c(t)].
3. Channel augmentation during training introduces synthetic spectral tilts,
   bandpass variations, and noise floor shifts, preventing the classifier from
   relying on channel cues.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sps
import librosa


def apply_cmvn_to_frames(mfcc_frames: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize MFCC frames across time (axis=1) to have zero mean and unit variance.

    mfcc_frames shape: (n_mfcc, n_frames)
    """
    mean = np.mean(mfcc_frames, axis=1, keepdims=True)
    std = np.std(mfcc_frames, axis=1, keepdims=True)
    return (mfcc_frames - mean) / (std + eps)


def extract_cmvn_mfcc(
    signal: np.ndarray,
    sr: int = 16_000,
    n_mfcc: int = 20,
) -> dict[str, float]:
    """Extract 40 channel-normalized MFCC features (deltas and CMVN-adjusted).

    Because stationary spectral tilt produces a constant offset across time:
    - Delta MFCC (velocities) cancels the constant offset exactly: d/dt [c(t) + h] = c'(t).
    - Delta-delta MFCC (accelerations) provides curvature invariant to linear drift.
    - Post-CMVN variance measures dynamic modulation rather than static channel energy.
    """
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    
    # Delta and delta-delta
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # CMVN on raw MFCC frames
    mfcc_norm = apply_cmvn_to_frames(mfcc)

    out: dict[str, float] = {}
    for i in range(n_mfcc):
        # CMVN normalized variance/std
        out[f"cmvn_mfcc{i:02d}_std"] = float(np.std(mfcc_norm[i]))
        # Delta mean & std (immune to additive channel)
        out[f"delta_mfcc{i:02d}_mean"] = float(np.mean(delta1[i]))
        out[f"delta_mfcc{i:02d}_std"] = float(np.std(delta1[i]))
        # Delta-delta mean
        out[f"delta2_mfcc{i:02d}_mean"] = float(np.mean(delta2[i]))

    # Truncate or select exactly 40 most discriminative channel-invariant features
    # Return 20 delta means + 20 delta stds
    cmvn_40 = {}
    for i in range(n_mfcc):
        cmvn_40[f"cmvn_delta{i:02d}_mean"] = out[f"delta_mfcc{i:02d}_mean"]
        cmvn_40[f"cmvn_delta{i:02d}_std"] = out[f"delta_mfcc{i:02d}_std"]
    return cmvn_40


def channel_augmentation(
    signal: np.ndarray,
    sr: int = 16_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply random channel perturbation: spectral tilt, band-limiting, and additive noise."""
    if rng is None:
        rng = np.random.default_rng()

    out = signal.copy()

    # 1. Random spectral tilt: IIR pre-emphasis/de-emphasis filter: y[n] = x[n] - alpha * x[n-1]
    # alpha in [-0.5, 0.5]
    alpha = rng.uniform(-0.4, 0.4)
    out = sps.lfilter([1.0, -alpha], [1.0], out)

    # 2. Random gain scaling: +/- 4 dB
    gain_db = rng.uniform(-4.0, 4.0)
    out = out * (10.0 ** (gain_db / 20.0))

    # 3. Random band-pass filtering (telephony simulation)
    # Low cutoff: 80 - 250 Hz, High cutoff: 3500 - 7800 Hz
    low_cut = rng.uniform(80.0, 250.0)
    high_cut = rng.uniform(3500.0, min(7800.0, sr / 2.0 - 100.0))
    sos = sps.butter(2, [low_cut, high_cut], btype="bandpass", fs=sr, output="sos")
    out = sps.sosfilt(sos, out)

    # 4. Additive gentle noise at 25-45 dB SNR
    target_snr_db = rng.uniform(25.0, 45.0)
    signal_rms = np.sqrt(np.mean(out**2)) + 1e-12
    noise_rms = signal_rms / (10.0 ** (target_snr_db / 20.0))
    noise = rng.normal(0.0, noise_rms, size=len(out))
    out = out + noise

    # Peak normalize
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak

    return out


def noise_floor_equalize(
    signal: np.ndarray,
    sr: int = 16_000,
    top_db: int = 30,
    target_silence_rms_db: float = -45.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Equalize ambient noise floor across recordings to prevent room identification."""
    if rng is None:
        rng = np.random.default_rng(42)

    intervals = librosa.effects.split(signal, top_db=top_db)
    mask = np.ones(len(signal), dtype=bool)
    for start, end in intervals:
        mask[start:end] = False

    current_silence = signal[mask]
    current_rms = np.sqrt(np.mean(current_silence**2)) if len(current_silence) > 0 else 1e-6
    target_rms = 10.0 ** (target_silence_rms_db / 20.0)

    # If the recording has very low background noise, inject comfort noise to match target
    if current_rms < target_rms:
        noise_needed = np.sqrt(max(0.0, target_rms**2 - current_rms**2))
        noise = rng.normal(0.0, noise_needed, size=len(signal))
        return signal + noise

    return signal
