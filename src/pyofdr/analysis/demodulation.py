"""OFDR demodulation -- from raw beat signal to physical quantities.

Three levels of processing, all operating on numpy arrays:

  1. FFT reflectogram: beat(t) -> H(z) complex reflectogram
  2. Phase-difference strain: two spectra -> distributed strain via
     complex-plane smoothing, unwrap, gradient
  3. Cross-spectrum frequency shift: two spectra -> spectral shift
     in Hz at each position (the standard Rayleigh-based method)

Plus k-clock resampling (aux-MZI-based sweep linearisation) and
conversion helpers (freq shift <-> strain / temperature).

All core functions take plain arrays so they can be tested without
running the full simulation pipeline.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import hilbert as _hilbert

from pyofdr.utils.constants import C


# ── 1. FFT reflectogram ─────────────────────────────────────────────

def fft_reflectogram(beat, dz, window=None, n_pad=None):
    """Compute the complex spatial reflectogram from a beat signal.

    Parameters
    ----------
    beat : 1-D real array
        Time-domain beat signal (ADC samples, photocurrent, ...).
    dz : float
        Unpadded FFT spatial spacing [m]. For raw acquisition samples,
        use acq.fft_dz; acq.dz describes the fiber construction grid.
    window : str or None
        Tapering window applied before FFT.  'hanning', 'tukey'
        or None (rectangular).
    n_pad : int or None
        Zero-pad length.  If None, uses len(beat).

    Returns
    -------
    H : 1-D complex array, length (N+1)//2
        One-sided complex reflectogram, with N the FFT length. Excludes
        the Nyquist bin for even N; odd N has no Nyquist bin to discard.
    z : 1-D float array, same length
        Position axis [m].
    """
    beat = np.asarray(beat, dtype=np.float64)
    n = len(beat)

    if window == "hanning":
        beat = beat * np.hanning(n)
    elif window == "tukey":
        # tukey with alpha=0.1 -- gentle taper at the edges
        from scipy.signal.windows import tukey as _tukey
        beat = beat * _tukey(n, alpha=0.1)
    elif window is not None:
        raise ValueError(f"unknown window: {window!r}")

    N = n_pad if n_pad is not None else n
    H = np.fft.fft(beat, n=N)
    n_half = (N + 1) // 2
    # zero-padding refines the grid: bin spacing is dz * n/N, not dz
    z = np.arange(n_half) * dz * (n / N)
    return H[:n_half], z


def reflectogram_from_acq(acq, core=0, **kwargs):
    """FFT raw ADC samples using their nominal linear-sweep spatial scale."""
    beat = acq.digital_main[core].astype(np.float64)
    # Older/manual acquisitions may not carry the sampling calibration.
    dz = acq.fft_dz if acq.fft_dz > 0 else acq.dz
    return fft_reflectogram(beat, dz, **kwargs)


# ── 2. Phase-difference strain demodulation ──────────────────────────

def phase_difference_strain(spec_meas, spec_ref, dz,
                            gauge_length, wavelength, n=1.4682, p_e=0.22):
    """Recover distributed strain from two complex spectra.

    Uses the phase-difference method: smooth the cross-product in
    the complex plane (boxcar of width gauge_length), unwrap, then
    differentiate to get local strain.

    Parameters
    ----------
    spec_meas, spec_ref : 1-D complex arrays
        Complex reflectograms (same length).
    dz : float
        Spatial bin size [m].
    gauge_length : float
        Smoothing kernel width [m]. Controls the trade-off between
        spatial resolution and phase noise. The rounded bin count
        must not exceed the spectrum length.
    wavelength : float
        Center wavelength [m].
    n : float
        Core refractive index.
    p_e : float
        Photoelastic (strain-optic) coefficient.

    Returns
    -------
    strain : 1-D float array, same length as inputs
        Recovered strain epsilon(z).
    """
    spec_meas = np.asarray(spec_meas, dtype=np.complex128)
    spec_ref  = np.asarray(spec_ref,  dtype=np.complex128)
    if spec_meas.shape != spec_ref.shape:
        raise ValueError("spectra must have the same length")

    # cross-product
    cdiff = spec_meas * np.conj(spec_ref)

    # boxcar smoothing in the complex plane
    w = max(1, int(round(gauge_length / dz)))
    if w > len(spec_meas):
        raise ValueError("smoothing window is longer than the spectrum")
    kernel = np.ones(w, dtype=np.float64) / w
    smoothed = (np.convolve(cdiff.real, kernel, mode="same")
                + 1j * np.convolve(cdiff.imag, kernel, mode="same"))

    phi = np.unwrap(np.angle(smoothed))

    k0 = 2.0 * math.pi / wavelength
    prefactor = 2.0 * k0 * n * (1.0 - p_e)

    strain = np.gradient(phi, dz) / prefactor
    return strain


# ── 3. Cross-spectrum frequency shift ────────────────────────────────

def cross_spectrum_shift(H_meas, H_ref, dz, n=1.4682, smooth_bins=0):
    """Estimate Rayleigh spectral shift between two reflectograms.

    The perturbation phase accumulates along the fiber (see
    fiber/strain.py), so the local shift comes from the spatial slope
    of the cross-product phase, not from the phase itself. Sign
    follows Froggatt-Moore: stretch/heating -> negative shift.

    Parameters
    ----------
    H_meas, H_ref : 1-D complex arrays
        Complex reflectograms.
    dz : float
        Spatial bin [m].
    n : float
        Core refractive index.
    smooth_bins : int
        If > 0, apply a boxcar moving average (in the complex plane)
        over this many bins before extracting the phase.
        Must not exceed the spectrum length.

    Returns
    -------
    freq_shift : 1-D float array
        Spectral shift [Hz] at each position.
    """
    H_meas = np.asarray(H_meas, dtype=np.complex128)
    H_ref  = np.asarray(H_ref,  dtype=np.complex128)
    if H_meas.shape != H_ref.shape:
        raise ValueError("spectra must have the same length")

    cross = H_meas * np.conj(H_ref)

    if smooth_bins > len(H_meas):
        raise ValueError("smoothing window is longer than the spectrum")
    if smooth_bins > 1:
        kernel = np.ones(smooth_bins, dtype=np.float64) / smooth_bins
        cross = (np.convolve(cross.real, kernel, mode="same")
                 + 1j * np.convolve(cross.imag, kernel, mode="same"))

    phase = np.unwrap(np.angle(cross))

    # round-trip delay per spatial bin
    tau_cell = 2.0 * n * dz / C
    # cumulative phase -> differentiate to get the local shift
    freq_shift = -np.gradient(phase) / (2.0 * math.pi * tau_cell)
    return freq_shift


# ── 4. Windowed sub-spectrum cross-correlation ──────────────────────

def _continuous_correlation_lag(a, b):
    """Refine the strongest sampled peak of the periodic correlation."""
    W = len(a)
    # Scale before multiplying. Very faint windows still carry phase information.
    a_scale = max(np.max(np.abs(a.real)), np.max(np.abs(a.imag)))
    b_scale = max(np.max(np.abs(b.real)), np.max(np.abs(b.imag)))
    q = (b / b_scale) * np.conj(a / a_scale)
    if np.count_nonzero(q) < 2:
        return np.nan
    q /= np.sum(np.abs(q))

    oversampling = 8
    correlation = np.fft.fftshift(np.fft.ifft(q, n=W * oversampling))
    peak = int(np.argmax(np.abs(correlation)))
    coarse_lag = (peak - W * oversampling // 2) / oversampling

    positions = (np.arange(W) - (W - 1) / 2) / W
    weights = q * np.exp(2j * np.pi * coarse_lag * positions)

    def objective(offset):
        value = np.sum(weights * np.exp(2j * np.pi * offset * positions))
        return -float(np.abs(value) ** 2)

    # Optimize a small offset, so tolerance does not grow near the lag boundary.
    result = minimize_scalar(
        objective, bounds=(-1 / oversampling, 1 / oversampling),
        method="bounded", options={"xatol": 1e-10},
    )
    if not result.success or not np.isfinite(result.fun):
        return np.nan

    offset = result.x
    # Keep exact grid peaks when the apparent gain is just roundoff.
    if objective(0.0) <= result.fun + 4 * np.finfo(float).eps:
        offset = 0.0
    lag = coarse_lag + offset
    return (lag + W / 2) % W - W / 2


def windowed_xcorr_strain(H_meas, H_ref, dz, gauge_length, stride,
                          sweep_range_hz, center_freq, p_e=0.22, *,
                          peak_method="continuous"):
    """Distributed strain via local sub-spectrum cross-correlation.

    For each sliding window of `gauge_length`, takes the IFFT of the
    spatial reflectogram window to get the local optical spectrum and
    cross-correlates the strained vs reference version. The peak shift
    in optical frequency converts to strain via Froggatt-Moore:
    `eps = -dnu / (nu_0 (1 - p_e))`.

    The lag is periodic with W bins and is returned in [-W/2, W/2).
    Shifts outside this interval alias. Typical gauge 0.5-5 cm.

    Parameters
    ----------
    H_meas, H_ref : 1-D complex arrays
        Complex reflectograms (same length).
    dz : float
        Spatial bin [m].
    gauge_length : float
        Sliding window size in z [m]. Sets both spatial resolution and
        sub-spectrum bin width (dnu_bin = sweep_range_hz / W).
    stride : float
        Step between consecutive window centers [m]. Overlap when
        stride < gauge_length.
    sweep_range_hz : float
        DFT optical-frequency span [Hz]. For unpadded raw reflectograms,
        use acq.fft_sweep_range_hz (N * nominal frequency step). It equals
        the configured sweep span when sample_rate * duration is integral.
    center_freq : float
        Laser center optical frequency [Hz].
    p_e : float
        Photoelastic coefficient.
    peak_method : {"continuous", "parabolic"}, keyword-only
        Default: locate the peak with an 8x padded IFFT, then maximize the
        continuous correlation magnitude locally (offset tolerance 1e-10
        original lag bins). "parabolic" selects the previous three-sample
        estimate, which is faster but biased for fractional shifts.

    Returns
    -------
    z_centers : 1-D float array
        Window center positions [m].
    eps : 1-D float array
        Recovered local strain, same length as z_centers.
        NaN where either reflectogram window contains only zeros or
        has a non-finite sample (NaN or infinity). The continuous method
        also returns NaN if fewer than two spatial samples overlap with
        nonzero signal, or if the peak refinement fails to converge.
    """
    H_meas = np.asarray(H_meas, dtype=np.complex128)
    H_ref  = np.asarray(H_ref,  dtype=np.complex128)
    if H_meas.shape != H_ref.shape:
        raise ValueError("reflectograms must have the same length")
    if H_meas.ndim != 1:
        raise ValueError("reflectograms must be 1-D arrays")
    if peak_method not in ("continuous", "parabolic"):
        raise ValueError("peak_method must be 'continuous' or 'parabolic'")

    n_bins = len(H_ref)
    W = int(round(gauge_length / dz))
    if W < 8:
        raise ValueError(f"gauge too small: W = {W} bins (need >= 8)")
    if W > n_bins:
        raise ValueError(f"gauge ({W} bins) larger than reflectogram ({n_bins})")
    W -= W % 2           # force even so W//2 is clean
    S = max(1, int(round(stride / dz)))

    dnu_bin = sweep_range_hz / W

    z_centers = []
    eps_out = []

    for k0 in range(W // 2, n_bins - W // 2 + 1, S):
        a = H_ref[k0 - W // 2 : k0 + W // 2]
        b = H_meas[k0 - W // 2 : k0 + W // 2]

        # Both windows need a finite, nonzero signal to estimate a shift.
        if (not np.any(a) or not np.any(b)
                or not np.isfinite(a).all() or not np.isfinite(b).all()):
            z_centers.append(k0 * dz)
            eps_out.append(np.nan)
            continue

        if peak_method == "continuous":
            lag = _continuous_correlation_lag(a, b)
            dnu = lag * dnu_bin
            eps = -dnu / (center_freq * (1.0 - p_e))
            z_centers.append(k0 * dz)
            eps_out.append(eps)
            continue

        # local spectra in optical-frequency domain
        A = np.fft.ifft(a)
        B = np.fft.ifft(b)

        # circular cross-correlation via FFT. Peak lag = shift of B vs A.
        R = np.fft.fftshift(np.fft.ifft(
            np.fft.fft(B) * np.conj(np.fft.fft(A))
        ))
        mag = np.abs(R)
        k_peak = int(np.argmax(mag))

        # parabolic sub-bin refinement around the peak
        if 0 < k_peak < W - 1:
            y_m = mag[k_peak - 1]
            y_0 = mag[k_peak]
            y_p = mag[k_peak + 1]
            denom = (y_m - 2.0 * y_0 + y_p)
            frac = 0.5 * (y_m - y_p) / denom if denom != 0.0 else 0.0
        else:
            frac = 0.0

        lag = (k_peak - W // 2) + frac
        dnu = lag * dnu_bin
        eps = -dnu / (center_freq * (1.0 - p_e))

        z_centers.append(k0 * dz)
        eps_out.append(eps)

    return np.array(z_centers), np.array(eps_out)


# ── k-clock resampling (aux-MZI based sweep linearisation) ─────────

def kclock_resample(beat, aux, trim_start=0, n_out=None):
    """Resample the main beat onto a uniform optical-frequency grid.

    The auxiliary interferometer produces  cos(phi(t) - phi(t-tau))  whose
    unwrapped Hilbert phase is a monotonic function of the instantaneous
    optical frequency. Interpolating the main beat onto a uniform grid in
    that phase is equivalent to resampling at constant delta-nu, which
    cancels any sweep nonlinearity before the FFT.

    Parameters
    ----------
    beat : 1-D real array
        Main detector beat signal (uniform-time samples).
    aux : 1-D real array
        Auxiliary MZI signal on the same time grid.
    trim_start : int
        Drop the first `trim_start` samples from both arrays. The aux step
        leaves the initial n_tau samples invalid (see AuxMZI.aux_valid_start).
    n_out : int or None
        Length of the resampled output. Defaults to len(beat) - trim_start.

    Returns
    -------
    beat_resampled : 1-D real array
    """
    beat = np.asarray(beat, dtype=np.float64)
    aux  = np.asarray(aux,  dtype=np.float64)
    if beat.shape != aux.shape:
        raise ValueError("beat and aux must have the same length")
    if beat.ndim != 1:
        raise ValueError("kclock_resample expects 1-D arrays")

    if trim_start > 0:
        beat = beat[trim_start:]
        aux  = aux[trim_start:]
    if len(aux) < 4:
        raise ValueError("aux signal too short after trimming")

    # analytic signal -> unwrapped phase. Remove DC first so the Hilbert
    # transform doesn't produce a large low-freq component on the imaginary
    # part that corrupts the phase near the edges.
    analytic = _hilbert(aux - aux.mean())
    phi = np.unwrap(np.angle(analytic))

    # we want a monotonically increasing phase (sweep nu goes up). If the
    # Hilbert happened to pick the conjugate branch, flip.
    if phi[-1] < phi[0]:
        phi = -phi

    if n_out is None:
        n_out = len(beat)

    if not np.all(np.diff(phi) > 0):
        # allow a handful of tiny non-monotonic glitches near the edges (Hilbert
        # artefact) but bail out if the whole signal is broken.
        n_bad = int(np.sum(np.diff(phi) <= 0))
        if n_bad > len(phi) // 100:
            raise ValueError(
                f"aux phase is non-monotonic ({n_bad} samples); "
                "sweep may have reversed or aux is too noisy"
            )
        # drop the glitchy samples so np.interp sees a strictly increasing
        # xp. without this the tolerate-branch would silently distort the
        # resampling, since np.interp has no contract for non-monotonic xp.
        keep = _strict_increasing_mask(phi)
        phi  = phi[keep]
        beat = beat[keep]

    phi_uniform = np.linspace(phi[0], phi[-1], n_out)
    return np.interp(phi_uniform, phi, beat)


def _strict_increasing_mask(phi):
    """Mask selecting the left-to-right strictly increasing subsequence of phi.

    phi[0] is always kept. For i >= 1, phi[i] is kept iff it exceeds the running
    maximum over phi[:i]. This is vectorised via np.maximum.accumulate.
    """
    running_max = np.maximum.accumulate(phi)
    keep = np.empty(phi.shape, dtype=bool)
    keep[0]  = True
    keep[1:] = phi[1:] > running_max[:-1]
    return keep


# ── Conversions ──────────────────────────────────────────────────────

def freq_shift_to_strain(df, center_freq, p_e=0.22):
    """Convert Rayleigh frequency shift to strain.

    eps = -df / (nu_0 * (1 - p_e))
    """
    df = np.asarray(df, dtype=np.float64)
    return -df / (center_freq * (1.0 - p_e))


def freq_shift_to_temperature(df, center_freq,
                               alpha=5.5e-7, xi=6.5e-6):
    """Convert Rayleigh frequency shift to temperature change.

    dT = -df / (nu_0 * (alpha + xi))

    Default coefficients are for standard silica SMF at 1550 nm:
      alpha ~ 0.55e-6 /K  (thermal expansion)
      xi    ~ 6.5e-6  /K  (thermo-optic)
    """
    df = np.asarray(df, dtype=np.float64)
    return -df / (center_freq * (alpha + xi))
