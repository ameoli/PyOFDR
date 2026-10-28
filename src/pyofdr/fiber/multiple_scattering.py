"""Multi-bounce returns from discrete reflectors."""

from __future__ import annotations

import itertools
import math

import numpy as np

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
    n_r = len(refs)
    travel = [[1.0] * (n_r + 1) for _ in range(n_r + 1)]
    for i in range(n_r + 1):
        for j in range(i + 1, n_r + 1):
            factor = 10.0 ** (-abs(losses[j] - losses[i]) / 20.0)
            for k in range(i + 1, j):
                factor *= transmission[k - 1]
            travel[i][j] = travel[j][i] = factor

    active = [i for i, r in enumerate(refs) if r["R"] > 0]
    for order in range(2, max_order + 1):
        for path in itertools.product(active, repeat=2 * order - 1):
            if any(bins[path[m]] >= bins[path[m - 1]]
                   or bins[path[m]] >= bins[path[m + 1]]
                   for m in range(1, len(path), 2)):
                continue
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
