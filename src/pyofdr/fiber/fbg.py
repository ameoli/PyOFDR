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
- Propagation uses the same small-delay time-warp approximation as Rayleigh.
- Broadcast across all cores; per-core inscription is a future refinement.
- Strain and temperature use the grating's z-bin, assumed uniform over L_g.
  Only the resonance shifts; bandwidth and propagation phase stay unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pyrayleigh import WeakFBG
from pyrayleigh.backends import real_arrays

from pyofdr.utils.constants import C


def weak_fbg_signal(
    fbgs: list[dict[str, Any]],
    dz: float,
    n_z: int,
    attenuation: np.ndarray | None,
    nu_inst: np.ndarray,
    n_core: float,
    *,
    linewidth: float = 0.0,
    strain_field: np.ndarray | None = None,
    temperature_field: np.ndarray | None = None,
    photoelastic_coefficient: float = 0.22,
    thermal_expansion: float = 5.5e-7,
    thermo_optic: float = 6.5e-6,
    time_fraction: np.ndarray | None = None,
    xp=np,
) -> np.ndarray:
    """Real beat contribution summed over all FBGs in the list.

    Returns a (n_t,) real array to add into the per-core beat signal.
    Each fbg is a dict with keys 'z', 'bragg_wavelength', 'length',
    'peak_reflectivity'.  linewidth applies the same exp(-pi*lw*tau)
    coherence visibility the profile reflectors get in the MZI.
    Optional fields contain applied strain and temperature change [K] on
    the spatial grid. Missing fields mean zero perturbation.
    time_fraction is the propagation phase coordinate: t/T for a linear
    chirp, s(t)/T for a warped sweep. The MZI supplies the same coordinate
    used by the Rayleigh profile. It includes sweep errors, GVD and slow FM;
    nu_inst independently sets the spectral envelope. If omitted, assume
    a linear chirp with an integer number of samples per sweep.
    Native inputs must be finite, with a positive shifted Bragg wavelength.
    """
    parameters = [nu_inst, n_core, linewidth, photoelastic_coefficient, thermal_expansion, thermo_optic,
                  1.0 if attenuation is None else attenuation,
                  0.0 if strain_field is None else strain_field,
                  0.0 if temperature_field is None else temperature_field]
    parameters.extend(value for fbg in fbgs for value in
                      (fbg["bragg_wavelength"], fbg["length"], fbg["peak_reflectivity"]))
    if time_fraction is not None:
        parameters.append(time_fraction)
    xp, values = real_arrays(*parameters, xp=None if xp is np else xp)
    (nu_inst, n_core, linewidth, photoelastic_coefficient, thermal_expansion, thermo_optic,
     attenuation_values, strain_values, temperature_values) = values[:9]
    n_t = nu_inst.shape[0]
    out = xp.zeros(n_t, dtype=xp.float64)
    if not fbgs:
        return out

    if time_fraction is None:
        time_fraction = xp.arange(n_t, dtype=xp.float64) / n_t
    else:
        time_fraction = xp.asarray(time_fraction, dtype=xp.float64)
        if time_fraction.shape != (n_t,):
            raise ValueError("time_fraction must be finite and match nu_inst")
        if isinstance(time_fraction, np.ndarray) and not np.isfinite(time_fraction).all():
            raise ValueError("time_fraction must be finite and match nu_inst")

    for fbg in fbgs:
        idx_z = int(round(fbg["z"] / dz))
        if idx_z == n_z and fbg["z"] <= n_z * dz:
            idx_z = n_z - 1   # end-face clamp, see #84
        if idx_z < 0 or idx_z >= n_z:
            continue

        shift = 0.0
        if strain_field is not None:
            shift = shift + (1.0 - photoelastic_coefficient) * strain_values[idx_z]
        if temperature_field is not None:
            shift = shift + (thermal_expansion + thermo_optic) * temperature_values[idx_z]

        # Always start from the configured resonance. No drift between sweeps.
        grating = WeakFBG(fbg["z"], fbg["bragg_wavelength"], fbg["length"], fbg["peak_reflectivity"])
        wavelength = xp.asarray(fbg["bragg_wavelength"] * (1.0 + shift), dtype=xp.float64)
        if isinstance(wavelength, np.ndarray) and (not np.isfinite(wavelength) or wavelength <= 0):
            raise ValueError("effective FBG Bragg wavelength must be finite and positive")
        reflection = grating.reflection(nu_inst, n_core, shift, xp=xp)
        att = attenuation_values[idx_z] if attenuation is not None else attenuation_values
        tau = 2.0 * n_core * (idx_z * dz) / C
        envelope = xp.real(reflection) * att * xp.exp(-math.pi * linewidth * tau)
        phase = 2.0 * math.pi * idx_z * time_fraction
        out = out + envelope * xp.cos(phase)

    return out
