"""Weak FBG arrays under Born approximation. See #40.

Each grating has a sinc-shaped amplitude response in optical frequency:
    r(t) = sqrt(R_max) * sinc(2 n L_g (nu(t) - nu_B) / C)
with sinc(x) = sin(pi x)/(pi x) (numpy convention). First null at
delta_nu = C / (2 n L_g), i.e. delta_lambda ~ lambda^2 / (2 n L_g) --
the classical FBG bandwidth.

The grating contributes a beat tone at its z-bin (just like a flat
reflector at z_0) modulated by the sinc envelope as the laser sweeps
across the Bragg resonance.

Caveats:
- Born approximation only -- peak_reflectivity above ~0.5 is wrong.
- Phase term is the ideal-chirp beat, not warped by sweep nonlinearity
  (small delta_nu/gamma error when both fbg and a2/a3/ripple are on).
- Broadcast across all cores; per-core inscription is a future refinement.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pyofdr.utils.constants import C


def weak_fbg_signal(
    fbgs: list[dict[str, Any]],
    dz: float,
    n_z: int,
    attenuation: np.ndarray | None,
    nu_inst: np.ndarray,
    n_core: float,
    *,
    xp=np,
) -> np.ndarray:
    """Real beat contribution summed over all FBGs in the list.

    Returns a (n_t,) real array to add into the per-core beat signal.
    Each fbg is a dict with keys 'z', 'bragg_wavelength', 'length',
    'peak_reflectivity'.
    """
    n_t = nu_inst.shape[0]
    out = xp.zeros(n_t, dtype=xp.float64)
    if not fbgs:
        return out

    n_idx = xp.arange(n_t)

    for fbg in fbgs:
        idx_z = int(round(fbg["z"] / dz))
        if idx_z == n_z and fbg["z"] <= n_z * dz:
            idx_z = n_z - 1   # end-face clamp, see #84
        if idx_z < 0 or idx_z >= n_z:
            continue

        nu_B  = C / fbg["bragg_wavelength"]
        L_g   = fbg["length"]
        R_max = fbg["peak_reflectivity"]
        att   = float(attenuation[idx_z]) if attenuation is not None else 1.0

        envelope = math.sqrt(R_max) * att * xp.sinc(
            2.0 * n_core * L_g * (nu_inst - nu_B) / C
        )
        # match the IFFT-based reflector convention: exp(2 pi i idx_z n / N)
        phase = 2.0 * math.pi * idx_z * n_idx / n_t
        out = out + envelope * xp.cos(phase)

    return out
