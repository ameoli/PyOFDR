"""Inject discrete reflectors (connectors, splices, etc) into the fiber profile."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def reflector_bin(ref, dz, n_z):
    idx = int(round(ref["z"] / dz))
    if idx == n_z and ref["z"] <= n_z * dz:
        idx = n_z - 1
    return idx


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

    n_z = profile.shape[-1]

    for i, ref in enumerate(reflectors):
        R = ref["R"]
        idx = reflector_bin(ref, dz, n_z)
        if idx < 0 or idx >= n_z:
            continue
        # amplitude for a point reflector with power reflectivity R
        profile[:, idx] += math.sqrt(R) if reflection is None else reflection[:, i]

    return profile


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
    n_z = len(attenuation)
    for ref in reflectors:
        idx = reflector_bin(ref, dz, n_z)
        if idx < 0 or idx >= n_z:
            continue
        # round-trip amplitude factor for one-way loss_dB
        factor = 10.0 ** (-ref.get("loss_dB", 0.0) / 10.0)
        attenuation[idx:] *= factor
        attenuation[idx + 1:] *= 1.0 - ref["R"]

    return attenuation
