"""Spectral diagnostics on the raw beat signal.

Thin wrappers around scipy.signal with OFDR-sensible defaults.
Useful for verifying noise slopes (white, 1/f, 1/f^2) and spotting
sweep nonlinearity artefacts.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch, spectrogram as _spectrogram


def beat_psd(beat, dt, nperseg=None):
    """Power spectral density of the beat signal via Welch's method.

    Parameters
    ----------
    beat : 1-D real array
        Time-domain beat signal.
    dt : float
        Sample interval [s].
    nperseg : int or None
        Segment length for Welch.  Defaults to len(beat)//8 (gives
        reasonable frequency resolution with 8 averages).

    Returns
    -------
    f : 1-D float array -- frequency axis [Hz]
    psd : 1-D float array -- PSD [signal_units^2 / Hz]
    """
    beat = np.asarray(beat, dtype=np.float64)
    fs = 1.0 / dt
    if nperseg is None:
        nperseg = max(256, len(beat) // 8)
    f, psd = welch(beat, fs=fs, nperseg=nperseg)
    return f, psd


def beat_spectrogram(beat, dt, nperseg=None):
    """Time-frequency spectrogram of the beat signal.

    Handy for spotting sweep nonlinearity: a perfectly linear chirp
    hitting a single reflector gives a horizontal line; nonlinearity
    makes it curve.

    Parameters
    ----------
    beat : 1-D real array
        Time-domain beat signal.
    dt : float
        Sample interval [s].
    nperseg : int or None
        Window length.  Defaults to len(beat)//16.

    Returns
    -------
    t : 1-D float array -- time axis [s]
    f : 1-D float array -- frequency axis [Hz]
    Sxx : 2-D float array -- spectrogram power (f x t)
    """
    beat = np.asarray(beat, dtype=np.float64)
    fs = 1.0 / dt
    if nperseg is None:
        nperseg = max(256, len(beat) // 16)
    f, t, Sxx = _spectrogram(beat, fs=fs, nperseg=nperseg)
    return t, f, Sxx


def psd_slope(f, psd, f_low, f_high):
    """Fit a power-law slope to a PSD segment on a log-log scale.

    PSD ~ f^alpha  =>  log(PSD) = alpha * log(f) + const

    Parameters
    ----------
    f, psd : 1-D arrays from beat_psd.
    f_low, f_high : float
        Frequency range [Hz] over which to fit.

    Returns
    -------
    dict with keys:
        slope : float  -- fitted exponent alpha
        intercept : float  -- log-space intercept
    """
    mask = (f >= f_low) & (f <= f_high) & (f > 0) & (psd > 0)
    if np.sum(mask) < 2:
        raise ValueError("not enough points in the requested range")
    lf = np.log10(f[mask])
    lp = np.log10(psd[mask])
    coeffs = np.polyfit(lf, lp, 1)
    return {"slope": coeffs[0], "intercept": coeffs[1]}
