"""Inject discrete reflectors (connectors, splices, etc) into the fiber profile."""

from __future__ import annotations

from typing import Any

import numpy as np
from pyrayleigh import lossless_reflector
from pyrayleigh.backends import complex_arrays, real_arrays
from pyrayleigh.units import db_to_linear


def reflection_coefficients(reflectors, phases, *, xp=None):
    """Bare field coefficients for one core, keeping reflector and phase order."""
    xp, values = real_arrays(phases, *(ref["R"] for ref in reflectors),
                             xp=None if xp is np else xp)
    phases = values[0]
    if phases.shape != (len(reflectors),):
        raise ValueError("give one phase per reflector")
    if not reflectors:
        return xp.zeros((0,), dtype=xp.complex128)
    return xp.stack([lossless_reflector(ref["z"], ref["R"], phases[index], xp=xp).r_left
                     for index, ref in enumerate(reflectors)])


def reflector_bin(ref, dz, n_z):
    idx = int(round(ref["z"] / dz))
    if idx == n_z and ref["z"] <= n_z * dz:
        idx = n_z - 1
    return idx


def profile_with_reflectors(profile, dz, reflectors, *, reflection=None, xp=None):
    """Return the profile plus fixed-bin point returns, without propagation losses."""
    parameters = (reflection,) if reflection is not None else tuple(ref["R"] for ref in reflectors)
    xp, values = complex_arrays(profile, *parameters, xp=None if xp is np else xp)
    profile = values[0]
    if profile.ndim != 2:
        raise ValueError("profile must have shape (n_cores, n_z)")
    if reflection is not None:
        reflection = values[1]
        if reflection.shape != (profile.shape[0], len(reflectors)):
            raise ValueError("give one reflection coefficient per core and reflector")
    n_z = profile.shape[1]
    bins = xp.arange(n_z)
    for i, ref in enumerate(reflectors):
        idx = reflector_bin(ref, dz, n_z)
        if idx < 0 or idx >= n_z:
            continue
        if reflection is None:
            point = lossless_reflector(ref["z"], ref["R"], xp=xp)
            addition = xp.real(point.r_left)
        else:
            addition = reflection[:, i:i + 1]
        profile = xp.where(bins == idx, profile + addition, profile)
    return profile


def inject_reflectors(
    profile: np.ndarray,
    z: np.ndarray,
    dz: float,
    reflectors: list[dict[str, Any]],
    *,
    reflection=None,
    xp=np,
) -> np.ndarray:
    """Add discrete reflectors on top of the Rayleigh backscatter profile.

    Each reflector is a dict with keys 'z' (position in meters) and
    'R' (power reflectivity, 0-1). Optional complex reflection coefficients
    have shape (n_cores, n_reflectors); otherwise sqrt(R) is used.

    Returns the modified profile (in-place).
    """
    if not reflectors:
        return profile

    profile[...] = profile_with_reflectors(profile, dz, reflectors, reflection=reflection, xp=xp)
    return profile


def attenuation_with_connectors(attenuation, dz, reflectors, *, xp=None):
    """Return the direct-return envelope with connector losses on fixed spatial bins."""
    xp, values = real_arrays(attenuation,
                             *(value for ref in reflectors for value in
                               (ref["R"], ref.get("loss_dB", 0.))),
                             xp=None if xp is np else xp)
    attenuation = values[0]
    if attenuation.ndim != 1:
        raise ValueError("attenuation must be a vector")
    n_z = attenuation.shape[0]
    bins = xp.arange(n_z)
    for i, ref in enumerate(reflectors):
        idx = reflector_bin(ref, dz, n_z)
        if idx < 0 or idx >= n_z:
            continue
        reflectance, loss = values[1 + 2*i:3 + 2*i]
        point = lossless_reflector(ref["z"], reflectance, xp=xp)
        factor = db_to_linear(-loss, xp=xp)
        attenuation = xp.where(bins >= idx, attenuation*factor, attenuation)
        transmission = point.t_left_to_right * point.t_right_to_left
        attenuation = xp.where(bins > idx, attenuation*transmission, attenuation)
    return attenuation


def apply_connector_losses(
    attenuation: np.ndarray,
    dz: float,
    reflectors: list[dict[str, Any]],
    *,
    xp=np,
) -> np.ndarray:
    """Apply direct-return transmission and insertion-loss steps.

    For each reflector with loss_dB > 0 the envelope beyond that point
    is scaled down by the round-trip amplitude factor:
        factor = 10^(-loss_dB / 10)
    (one-way loss_dB applied twice, converted to amplitude).

    The extra loss sits just before the reflecting interface. A return
    from that interface sees the loss, but not its own (1-R) transmission.
    Returns from later bins cross the interface twice.
    """
    if not reflectors:
        return attenuation
    attenuation[...] = attenuation_with_connectors(attenuation, dz, reflectors, xp=xp)
    return attenuation
