"""Cascading multi-bounce ghost reflections from discrete reflectors (#35).

Light reflected by one strong reflector can be re-scattered by another and
end up back at the detector via a longer optical path. Each such path
appears in the OFDR signal as a ghost peak at an apparent depth set by
the round-trip geometry.

For an order-N path (2N-1 reflection events) with reflector indices
(i_0, i_1, ..., i_{2N-2}) the apparent depth and amplitude are

    z_app = z_{i_0} - z_{i_1} + z_{i_2} - ... + z_{i_{2N-2}}
    a     = prod sqrt(R_{i_m})

with the trajectory constraint that every odd-indexed reflector (the
"valleys" where the light bounces back forward) sits strictly below both
of its neighbours: z_{i_{2m+1}} < z_{i_{2m}} and z_{i_{2m+1}} < z_{i_{2m+2}}.

The continuum Rayleigh-Rayleigh background is several orders of magnitude
weaker in standard SMF -- tracked separately in #79.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np


def add_ghost_reflections(
    profile: np.ndarray,
    dz: float,
    reflectors: list[dict[str, Any]],
    *,
    max_order: int = 2,
    xp=np,
) -> np.ndarray:
    """Inject multi-bounce ghost peaks on top of the existing profile.

    Ghosts whose apparent bin falls outside [0, n_z) are dropped --
    extending the array to 2L would change a lot of downstream code,
    so for now we only model ghosts that land within the fiber range.
    For the canonical input-connector + end-face case the user can pad
    the fiber length so 2L_eff stays inside the array.

    Bin convention: each reflector z is rounded to its bin first, then
    the apparent bin is the signed sum of those bins (NOT round(z_app/dz)
    of the real-valued sum -- the two differ by at most 1).

    Caveat: if an interior "valley" reflector carries a non-zero loss_dB,
    the per-bin attenuation envelope undercounts the actual loss along
    a ghost path (the ghost crosses the valley twice). For typical OFDR
    setups -- where the loss reflector is the input connector at z=0 --
    this is exact.
    """
    if max_order < 2 or len(reflectors) < 2:
        return profile

    n_z = profile.shape[-1]
    bins = [int(round(r["z"] / dz)) for r in reflectors]
    amps = [math.sqrt(r["R"]) for r in reflectors]
    n_r = len(reflectors)

    # enumerate orders 2..max_order. each order N has 2N-1 reflection
    # events; itertools.product is in C so the loop body is the only
    # thing that runs in Python.
    for N in range(2, max_order + 1):
        path_len = 2 * N - 1
        for path in itertools.product(range(n_r), repeat=path_len):
            # valley constraint -- every odd-position reflector must sit
            # strictly below both neighbours
            valid = True
            for m in range(1, path_len, 2):
                v = bins[path[m]]
                if v >= bins[path[m - 1]] or v >= bins[path[m + 1]]:
                    valid = False
                    break
            if not valid:
                continue
            # signed sum -> apparent depth bin
            b = 0
            for i, idx in enumerate(path):
                b += bins[idx] if i % 2 == 0 else -bins[idx]
            if b < 0 or b >= n_z:
                continue
            amp = 1.0
            for idx in path:
                amp *= amps[idx]
            profile[:, b] += amp

    return profile
