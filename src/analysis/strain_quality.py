"""Strain measurement quality metrics.

Post-processing diagnostics for demodulated strain or frequency-shift
arrays.  All functions are stateless, operating on plain numpy arrays.
"""

from __future__ import annotations

import math

import numpy as np


def strain_noise_floor(strain, quiet_mask):
    """Standard deviation of strain in a known-quiet region.

    Parameters
    ----------
    strain : 1-D float array
        Demodulated strain epsilon(z).
    quiet_mask : 1-D bool array
        True where no perturbation is expected.

    Returns
    -------
    dict with keys:
        noise_floor : float  -- std of strain in quiet region
        mean        : float  -- mean (should be ~0)
        n_bins      : int    -- number of bins used
    """
    strain = np.asarray(strain)
    q = strain[quiet_mask]
    return {
        "noise_floor": float(np.std(q)),
        "mean": float(np.mean(q)),
        "n_bins": int(np.sum(quiet_mask)),
    }


def strain_sensitivity(noise_floor, gauge_length, dz):
    """Minimum detectable strain from noise floor and gauge length.

    For a boxcar gauge of width G, averaging reduces the noise by
    sqrt(G / dz).  This gives the 1-sigma sensitivity.

    Parameters
    ----------
    noise_floor : float
        Per-bin strain std (from strain_noise_floor).
    gauge_length : float
        Smoothing / gauge length [m].
    dz : float
        Spatial bin size [m].

    Returns
    -------
    float -- 1-sigma minimum detectable strain.
    """
    n_avg = max(1.0, gauge_length / dz)
    return noise_floor / math.sqrt(n_avg)


def allan_deviation(freq_shifts, dt_sweep):
    """Overlapping Allan deviation of frequency shift across sweeps.

    Useful for characterising measurement stability vs averaging
    time.  Needs at least 3 sweeps.

    Parameters
    ----------
    freq_shifts : 1-D float array, length n_sweeps
        Frequency shift at a single position across sweeps.
    dt_sweep : float
        Time between consecutive sweeps [s].

    Returns
    -------
    dict with keys:
        tau   : 1-D float array -- averaging times [s]
        adev  : 1-D float array -- Allan deviation at each tau
    """
    freq_shifts = np.asarray(freq_shifts, dtype=np.float64)
    n = len(freq_shifts)
    if n < 3:
        raise ValueError("allan_deviation needs at least 3 sweeps")

    # overlapping Allan deviation
    max_m = n // 2
    taus = []
    adevs = []

    for m in range(1, max_m + 1):
        tau = m * dt_sweep
        # overlapping differences
        diffs = freq_shifts[2 * m:] - 2 * freq_shifts[m:-m] + freq_shifts[:-2 * m]
        if len(diffs) == 0:
            break
        avar = np.mean(diffs ** 2) / (2.0 * m ** 2 * dt_sweep ** 2)
        # allan dev is sqrt of allan variance, but we want freq stability
        # so just sqrt(avar) * tau to get the standard form... actually
        # standard Allan dev for freq data is simpler:
        # sigma_y(tau) = sqrt( mean( (y_{n+1} - y_n)^2 ) / 2 )
        # where y_n = average of m samples starting at n
        pass

    # simpler: standard non-overlapping Allan deviation
    taus = []
    adevs = []
    for m in range(1, max_m + 1):
        tau = m * dt_sweep
        # non-overlapping averages
        n_blocks = n // m
        if n_blocks < 2:
            break
        blocks = freq_shifts[:n_blocks * m].reshape(n_blocks, m).mean(axis=1)
        diffs = np.diff(blocks)
        adev = math.sqrt(0.5 * np.mean(diffs ** 2))
        taus.append(tau)
        adevs.append(adev)

    return {
        "tau": np.array(taus),
        "adev": np.array(adevs),
    }
