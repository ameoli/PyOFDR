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
from scipy.signal import hilbert as _hilbert

from utils.constants import C


# ── 1. FFT reflectogram ─────────────────────────────────────────────

def fft_reflectogram(beat, dz, window=None, n_pad=None):
    """Compute the complex spatial reflectogram from a beat signal.

    Parameters
    ----------
    beat : 1-D real array
        Time-domain beat signal (ADC samples, photocurrent, ...).
    dz : float
        Spatial resolution [m].
    window : str or None
        Tapering window applied before FFT.  'hanning', 'tukey'
        or None (rectangular).
    n_pad : int or None
        Zero-pad length.  If None, uses len(beat).

    Returns
    -------
    H : 1-D complex array, length n_pad//2
        One-sided complex reflectogram.
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
    n_half = N // 2
    z = np.arange(n_half) * dz
    return H[:n_half], z


def reflectogram_from_acq(acq, core=0, **kwargs):
    """Convenience wrapper: extract beat from an Acquisition and FFT it."""
    beat = acq.digital_main[core].astype(np.float64)
    return fft_reflectogram(beat, acq.dz, **kwargs)


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
        spatial resolution and phase noise.
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

    For each spatial bin, computes the phase of the complex
    cross-product and converts to a frequency shift.

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

    if smooth_bins > 1:
        kernel = np.ones(smooth_bins, dtype=np.float64) / smooth_bins
        cross = (np.convolve(cross.real, kernel, mode="same")
                 + 1j * np.convolve(cross.imag, kernel, mode="same"))

    phase = np.angle(cross)

    # round-trip delay per spatial bin
    tau_cell = 2.0 * n * dz / C
    freq_shift = phase / (2.0 * math.pi * tau_cell)
    return freq_shift


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

    if not np.all(np.diff(phi) > 0):
        # allow a handful of tiny non-monotonic glitches near the edges (Hilbert
        # artefact) but bail out if the whole signal is broken.
        n_bad = int(np.sum(np.diff(phi) <= 0))
        if n_bad > len(phi) // 100:
            raise ValueError(
                f"aux phase is non-monotonic ({n_bad} samples); "
                "sweep may have reversed or aux is too noisy"
            )

    if n_out is None:
        n_out = len(beat)

    phi_uniform = np.linspace(phi[0], phi[-1], n_out)
    return np.interp(phi_uniform, phi, beat)


# ── Conversions ──────────────────────────────────────────────────────

def freq_shift_to_strain(df, center_freq, p_e=0.22):
    """Convert Rayleigh frequency shift to strain.

    eps = -df / (nu_0 * (1 - p_e))
    """
    df = np.asarray(df, dtype=np.float64)
    return -df / (center_freq * (1.0 - p_e))


def freq_shift_to_temperature(df, center_freq,
                               alpha=5.5e-7, xi=6.3e-6):
    """Convert Rayleigh frequency shift to temperature change.

    dT = -df / (nu_0 * (alpha + xi))

    Default coefficients are for standard silica SMF at 1550 nm:
      alpha ~ 0.55e-6 /K  (thermal expansion)
      xi    ~ 6.3e-6  /K  (thermo-optic)
    """
    df = np.asarray(df, dtype=np.float64)
    return -df / (center_freq * (alpha + xi))
