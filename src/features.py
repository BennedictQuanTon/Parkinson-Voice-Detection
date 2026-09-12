"""Acoustic feature extraction.

Two feature sets, deliberately:

``mfcc``
    Mean and standard deviation of 20 MFCCs. This is the closest available
    match to the ``MFCC-mean`` representation used by Klempir et al. (2024,
    Sensors 24:5520), so the leakage experiment compares like with like.
``egemaps``
    eGeMAPSv02, 88 parameters (Eyben et al. 2016). The standard interpretable
    baseline for paralinguistic tasks, included so the finding does not depend
    on one hand-rolled feature set.

Perturbation measures (jitter, shimmer, HNR) are intentionally *not* computed
on the read-text task: they are only well defined on sustained phonation.
"""

from __future__ import annotations

import os
from pathlib import Path

# librosa compiles parts of itself with numba on first use and needs somewhere to
# write the compilation cache. The default location is outside the project and is
# not always writable, which surfaces as a per-file
# ``RuntimeError: cannot cache function`` rather than as an import error. Set
# before importing librosa, or the setting is ignored.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(__file__).resolve().parents[1] / ".cache" / "numba"),
)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import librosa  # noqa: E402
import numpy as np
import opensmile
import pandas as pd

SAMPLE_RATE = 16_000
N_MFCC = 20


def load_audio(path: str | Path) -> np.ndarray:
    signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return signal


def mfcc_features(path: str | Path) -> dict[str, float]:
    signal = load_audio(path)
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    out: dict[str, float] = {}
    for i in range(N_MFCC):
        out[f"mfcc{i:02d}_mean"] = float(mfcc[i].mean())
        out[f"mfcc{i:02d}_std"] = float(mfcc[i].std())
    return out


_SMILE = None


def _smile() -> opensmile.Smile:
    global _SMILE
    if _SMILE is None:
        _SMILE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _SMILE


def egemaps_features(path: str | Path) -> dict[str, float]:
    frame = _smile().process_file(str(path))
    return {c: float(frame[c].iloc[0]) for c in frame.columns}


def silence_features(path: str | Path, top_db: int = 30) -> dict[str, float] | None:
    """MFCC statistics of the *discarded* non-speech regions.

    This is the negative control for channel and background noise: a model that
    separates the groups using only the silence between utterances is keying on
    the recording setup, not on the voice.
    """
    signal = load_audio(path)
    intervals = librosa.effects.split(signal, top_db=top_db)
    mask = np.ones(len(signal), dtype=bool)
    for start, end in intervals:
        mask[start:end] = False
    silence = signal[mask]
    if len(silence) < SAMPLE_RATE // 4:  # need at least 250 ms to be meaningful
        return None
    mfcc = librosa.feature.mfcc(y=silence, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    out = {"silence_seconds": float(len(silence) / SAMPLE_RATE)}
    for i in range(N_MFCC):
        out[f"sil_mfcc{i:02d}_mean"] = float(mfcc[i].mean())
        out[f"sil_mfcc{i:02d}_std"] = float(mfcc[i].std())
    return out


def channel_features(path: str | Path, top_db: int = 30) -> dict[str, float] | None:
    """Eight gross properties of the recording chain, carrying no phonetics.

    Level, noise floor, DC offset and broad spectral tilt describe the
    microphone, its distance and the room. None of them describes articulation,
    so their discriminative power is a direct measure of how far the label can be
    predicted from acquisition conditions alone.
    """
    signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)  # unnormalised
    intervals = librosa.effects.split(signal, top_db=top_db)
    mask = np.ones(len(signal), dtype=bool)
    for start, end in intervals:
        mask[start:end] = False
    silence, speech = signal[mask], signal[~mask]

    # A continuously phonated vowel can contain no detectable pause at all, and
    # there is then no noise floor to measure. Returning None drops the recording
    # from this feature set and reports it, rather than inventing a value.
    if len(silence) < SAMPLE_RATE // 10 or len(speech) == 0:
        return None

    def db(x: np.ndarray) -> float:
        return float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12))

    spectrum = np.abs(librosa.stft(signal, n_fft=512)).mean(axis=1)
    spectrum = spectrum / spectrum.sum()

    silence_db, speech_db = db(silence), db(speech)
    return {
        "ch_peak": float(np.max(np.abs(signal))),
        "ch_silence_db": silence_db,
        "ch_speech_db": speech_db,
        "ch_snr_db": speech_db - silence_db,
        "ch_dc_offset": float(np.mean(signal)),
        # Bins of a 512-point STFT at 16 kHz are 31.25 Hz wide.
        "ch_lf_ratio": float(spectrum[:20].sum()),    # below ~625 Hz
        "ch_hf_ratio": float(spectrum[200:].sum()),   # above ~6.25 kHz
        "ch_clipped_fraction": float(np.mean(np.abs(signal) > 0.99)),
    }


EXTRACTORS = {
    "mfcc": mfcc_features,
    "egemaps": egemaps_features,
    "silence": silence_features,
    "channel": channel_features,
}


def extract(meta: pd.DataFrame, kind: str, verbose: bool = True) -> pd.DataFrame:
    """Run an extractor over ``meta['path']`` and return a features frame.

    Rows that produce no features (e.g. a recording with no usable silence) are
    dropped and reported, never silently filled.
    """
    if kind not in EXTRACTORS:
        raise KeyError(f"unknown feature set {kind!r}")
    fn = EXTRACTORS[kind]

    rows, kept, skipped = [], [], []
    for pos, path in enumerate(meta["path"]):
        try:
            values = fn(path)
        except Exception as exc:  # noqa: BLE001 - surfaced below, not swallowed
            skipped.append((path, repr(exc)))
            continue
        if values is None:
            skipped.append((path, "extractor returned no features"))
            continue
        rows.append(values)
        kept.append(pos)
        if verbose and (pos + 1) % 25 == 0:
            print(f"  {kind}: {pos + 1}/{len(meta)}")

    if skipped:
        print(f"  {kind}: skipped {len(skipped)} recording(s)")
        for path, reason in skipped[:10]:
            print(f"    {Path(path).name}: {reason}")

    features = pd.DataFrame(rows, index=meta.index[kept])
    return features.dropna(axis=1, how="all")
