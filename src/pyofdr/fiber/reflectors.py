"""Inject discrete reflectors (connectors, splices, etc) into the fiber profile."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def inject_reflectors(
    profile: np.ndarray,
    z: np.ndarray,
    dz: float,
    reflectors: list[dict[str, Any]],
    *,
    xp=np,
) -> np.ndarray:
    """Add discrete reflectors on top of the Rayleigh backscatter profile.

    Each reflector is a dict with keys 'z' (position in meters) and
    'R' (power reflectivity, 0-1).  The reflector amplitude sqrt(R)
    is added to every core at the nearest spatial bin.

    Returns the modified profile (in-place).
    """
    if not reflectors:
        return profile

    n_z = profile.shape[-1]

    for ref in reflectors:
        pos = ref["z"]
        R = ref["R"]
        idx = int(round(pos / dz))
        # with n_z = ceil(length/dz) the end face can round to n_z --
        # clamp to the last bin instead of silently dropping it (#84)
        if idx == n_z and pos <= n_z * dz:
            idx = n_z - 1
        if idx < 0 or idx >= n_z:
            continue
        # amplitude for a point reflector with power reflectivity R
        profile[:, idx] += math.sqrt(R)

    return profile


def apply_connector_losses(
    attenuation: np.ndarray,
    dz: float,
    reflectors: list[dict[str, Any]],
    *,
    xp=np,
) -> np.ndarray:
    """Apply insertion-loss steps to the attenuation envelope.

    For each reflector with loss_dB > 0 the envelope beyond that point
    is scaled down by the round-trip amplitude factor:
        factor = 10^(-loss_dB / 10)
    (one-way loss_dB applied twice, converted to amplitude).

    Reflectors are processed in order of position so that cascaded
    connectors accumulate correctly.
    """
    lossy = [r for r in reflectors if r.get("loss_dB", 0) > 0]
    if not lossy:
        return attenuation

    n_z = len(attenuation)
    # sort by position so losses accumulate left-to-right
    for ref in sorted(lossy, key=lambda r: r["z"]):
        idx = int(round(ref["z"] / dz))
        if idx == n_z and ref["z"] <= n_z * dz:
            idx = n_z - 1   # same end-face clamp as inject_reflectors
        if idx < 0 or idx >= n_z:
            continue
        # round-trip amplitude factor for one-way loss_dB
        factor = 10.0 ** (-ref["loss_dB"] / 10.0)
        attenuation[idx:] *= factor

    return attenuation
