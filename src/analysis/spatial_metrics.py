"""Spatial quality metrics measured on a reflectogram.

Unlike budget.py (pure algebra from config), these operate on actual
simulation output -- the complex reflectogram H(z).

All functions take plain arrays so they work without the pipeline.
"""

from __future__ import annotations

import numpy as np


def measure_resolution(H, z, peak_z, level_dB=-6.0):
    """Measure spatial resolution from the width of a reflector peak.

    Parameters
    ----------
    H : 1-D complex array
        Complex reflectogram.
    z : 1-D float array
        Position axis [m].
    peak_z : float
        Approximate position of the reflector [m].  The actual peak
        is found within ±5 bins of the closest index.
    level_dB : float
        Level below peak at which width is measured (default -6 dB).

    Returns
    -------
    dict with keys:
        resolution : float  -- measured width [m]
        peak_z     : float  -- actual peak position [m]
        peak_dB    : float  -- peak amplitude [dB]
    """
    H = np.asarray(H)
    z = np.asarray(z)
    amp_dB = 20.0 * np.log10(np.abs(H) + 1e-30)

    # find peak near the requested position
    i_approx = int(np.argmin(np.abs(z - peak_z)))
    margin = 5
    lo = max(0, i_approx - margin)
    hi = min(len(H), i_approx + margin + 1)
    i_peak = lo + int(np.argmax(amp_dB[lo:hi]))

    peak_val = amp_dB[i_peak]
    threshold = peak_val + level_dB   # level_dB is negative

    # walk left and right to find the crossing
    i_left = i_peak
    while i_left > 0 and amp_dB[i_left] > threshold:
        i_left -= 1
    i_right = i_peak
    while i_right < len(amp_dB) - 1 and amp_dB[i_right] > threshold:
        i_right += 1

    resolution = z[i_right] - z[i_left]

    return {
        "resolution": resolution,
        "peak_z": z[i_peak],
        "peak_dB": float(peak_val),
    }


def measure_dynamic_range(H, signal_mask, noise_mask):
    """Measure dynamic range from signal and noise regions.

    Parameters
    ----------
    H : 1-D complex array
        Complex reflectogram.
    signal_mask : 1-D bool array
        Bins belonging to the signal region (e.g. inside the fiber).
    noise_mask : 1-D bool array
        Bins belonging to the noise floor (e.g. beyond fiber end).

    Returns
    -------
    dict with keys:
        dr_dB       : float  -- dynamic range [dB]
        signal_mean : float  -- mean signal amplitude [dB]
        noise_mean  : float  -- mean noise floor [dB]
    """
    amp = np.abs(np.asarray(H))
    sig_mean = np.mean(amp[signal_mask])
    noi_mean = np.mean(amp[noise_mask])

    dr_dB = 20.0 * np.log10(sig_mean / noi_mean) if noi_mean > 0 else float("inf")
    return {
        "dr_dB": dr_dB,
        "signal_mean": 20.0 * np.log10(sig_mean + 1e-30),
        "noise_mean": 20.0 * np.log10(noi_mean + 1e-30),
    }


def snr_profile(reflectograms, z):
    """Per-position SNR from multiple sweep reflectograms.

    Needs >= 2 sweeps.  Computes mean / std of |H(z)| across sweeps
    at each position.

    Parameters
    ----------
    reflectograms : sequence of 1-D complex arrays
        One reflectogram per sweep (same length).
    z : 1-D float array
        Position axis [m].

    Returns
    -------
    dict with keys:
        snr_dB : 1-D float array -- SNR at each position [dB]
        mean   : 1-D float array -- mean |H| across sweeps
        std    : 1-D float array -- std  |H| across sweeps
    """
    stack = np.abs(np.array(reflectograms))   # (n_sweeps, n_z)
    if stack.shape[0] < 2:
        raise ValueError("snr_profile needs at least 2 sweeps")

    mu = np.mean(stack, axis=0)
    sigma = np.std(stack, axis=0, ddof=1)

    # avoid log(0)
    snr_dB = 20.0 * np.log10(mu / (sigma + 1e-30) + 1e-30)

    return {"snr_dB": snr_dB, "mean": mu, "std": sigma}
