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
    """Return a phase-noise array phi_noise[t] in radians.

    - linewidth   : Lorentzian FWHM [Hz], drives the white FM component
    - sigma_flicker: RMS of 1/f frequency noise [Hz] over [1/T, fs/2]
    - sigma_rw    : RMS of 1/f^2 frequency noise [Hz] over [1/T, fs/2]
    - rng         : a numpy Generator (used for all three contributions)

    The three contributions are statistically independent because
    powerlaw_psd_gaussian draws fresh samples each time from the same rng.
    """
    phi = xp.zeros(n_samples)

    # white FM -> Wiener phase
    if linewidth > 0:
        sigma_w = math.sqrt(2.0 * math.pi * linewidth * dt)
        phi = phi + xp.cumsum(sigma_w * rng.standard_normal(n_samples))

    # flicker FM (1/f)
    if sigma_flicker > 0:
        nu_f = sigma_flicker * powerlaw_psd_gaussian(
            1.0, n_samples, random_state=rng)
        phi = phi + 2.0 * math.pi * xp.cumsum(nu_f) * dt

    # random-walk FM (1/f^2)
    if sigma_rw > 0:
        nu_rw = sigma_rw * powerlaw_psd_gaussian(
            2.0, n_samples, random_state=rng)
        phi = phi + 2.0 * math.pi * xp.cumsum(nu_rw) * dt

    return phi
