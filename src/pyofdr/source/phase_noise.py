"""Laser phase noise models.

Today three frequency-noise contributions are combined:

    S_nu(f) = h_0     +  h_{-1}/f  +  h_{-2}/f^2
              white       flicker     random walk

The white term is parametrized via linewidth (Lorentzian FWHM) as in
the original model: d_phi ~ N(0, sqrt(2*pi*lw*dt)). The coloured
contributions are generated with the Timmer & Koenig FFT method
(utils.colorednoise) and integrated to phase by cumulative sum.

The two coloured levels are expressed as *RMS frequency deviation
over the sampled bandwidth* [Hz], which is pragmatic to tune by
hand. A nicer h-coefficient parametrization can come later.
"""

from __future__ import annotations

import math

import numpy as np

from pyofdr.utils.colorednoise import powerlaw_psd_gaussian


def frequency_noise(
    n_samples: int,
    dt: float,
    linewidth: float,
    sigma_flicker: float,
    sigma_rw: float,
    rng,
    *,
    xp=np,
):
    """Return (nu_white, nu_colored) instantaneous frequency noise [Hz].

    - linewidth   : Lorentzian FWHM [Hz], drives the white FM component
    - sigma_flicker: RMS of 1/f frequency noise [Hz] over [1/T, fs/2]
    - sigma_rw    : RMS of 1/f^2 frequency noise [Hz] over [1/T, fs/2]
    - rng         : a numpy Generator (used for all three contributions)

    nu_white integrates to the usual Wiener phase. The coloured part is
    returned separately because downstream (MZI time-warp) can only absorb
    noise that is slow compared to the fiber round-trip delays.

    The contributions are statistically independent because
    powerlaw_psd_gaussian draws fresh samples each time from the same rng.
    """
    nu_white = xp.zeros(n_samples)
    if linewidth > 0:
        # integrates to d_phi ~ N(0, sqrt(2*pi*lw*dt)) per sample
        sigma_w = math.sqrt(linewidth / (2.0 * math.pi * dt))
        nu_white = sigma_w * rng.standard_normal(n_samples)

    nu_colored = xp.zeros(n_samples)
    if sigma_flicker > 0:
        nu_colored = nu_colored + sigma_flicker * powerlaw_psd_gaussian(
            1.0, n_samples, random_state=rng)
    if sigma_rw > 0:
        nu_colored = nu_colored + sigma_rw * powerlaw_psd_gaussian(
            2.0, n_samples, random_state=rng)

    return nu_white, nu_colored


def colored_frequency_noise(
    n_samples: int,
    dt: float,
    linewidth: float,
    sigma_flicker: float,
    sigma_rw: float,
    rng,
    *,
    xp=np,
):
    """Return the integrated phase-noise array phi_noise[t] in radians."""
    nu_w, nu_c = frequency_noise(
        n_samples, dt, linewidth, sigma_flicker, sigma_rw, rng, xp=xp)
    return 2.0 * math.pi * xp.cumsum(nu_w + nu_c) * dt
