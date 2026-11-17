"""Multi-bounce returns from discrete reflectors."""

from __future__ import annotations

import math

import numpy as np
from pyrayleigh import reciprocal_transmission_factors, reflection_paths_from_positions

from pyofdr.fiber.reflectors import reflector_bin


def add_ghost_reflections(
    profile, dz, reflectors, *, max_order=2, reflection=None,
    loss_to_z=None, xp=np,
):
    """Add ghosts with all path losses already included, in-place.

    Order N has 2N-1 reflections. Positions are snapped before forming
    the apparent bin; ghosts outside the array are dropped.
    loss_to_z is the cumulative one-way distributed loss in dB, without
    connector losses. reflection contains the left-side coefficients,
    with shape (n_cores, n_reflectors).
    """
    if max_order < 2 or len(reflectors) < 2:
        return profile

    n_cores, n_z = profile.shape
    selected = sorted(
        (i for i, r in enumerate(reflectors)
         if 0 <= reflector_bin(r, dz, n_z) < n_z),
        key=lambda i: reflectors[i]["z"],
    )
    refs = [reflectors[i] for i in selected]
    bins = [reflector_bin(r, dz, n_z) for r in refs]
    if len(set(bins)) != len(bins):
        raise ValueError("discrete reflectors must occupy distinct spatial bins")
    if reflection is None:
        reflection = xp.tile(xp.asarray([math.sqrt(r["R"]) for r in refs]),
                             (n_cores, 1))
    else:
        reflection = reflection[:, selected]

    # Node 0 is the input, before a possible reflector at z=0.
    losses = [0.0] + [0.0 if loss_to_z is None else float(loss_to_z[b])
                      for b in bins]
    extra = [10.0 ** (-r.get("loss_dB", 0.0) / 20.0) for r in refs]
    transmission = [math.sqrt(1.0 - r["R"]) * q for r, q in zip(refs, extra)]
    travel = reciprocal_transmission_factors(
        losses, transmission, xp=None if xp is np else xp)

    active = [i for i, r in enumerate(refs) if r["R"] > 0]
    paths = reflection_paths_from_positions([bins[i] for i in active], max_order=max_order)
    for route in paths:
        if len(route) == 1:
            continue
        path = tuple(active[i] for i in route)
        b = sum(bins[idx] if m % 2 == 0 else -bins[idx]
                for m, idx in enumerate(path))
        if not 0 <= b < n_z:
            continue

        amp = xp.ones(n_cores, dtype=xp.complex128)
        previous = 0
        for m, idx in enumerate(path):
            amp *= travel[previous][idx + 1]
            if m % 2 == 0:
                # The extra loss sits on the left of the interface.
                amp *= reflection[:, idx] * extra[idx] ** 2
            else:
                # Real transmission fixes the reverse reflection phase.
                amp *= -xp.conj(reflection[:, idx])
            previous = idx + 1
        amp *= travel[previous][0]
        profile[:, b] += amp

    return profile
