"""Core-to-core crosstalk for multicore fibers (#47).

First-order phase-scrambled model: each directed pair (j -> i) of
coupling cores bleeds a fraction of the source profile into the
target, with an independent U(0, 2pi) phase per bin.  The amplitude
coupling grows as sqrt(xt_linear * z/1000), mimicking distributed
weak coupling along the fiber. Power "lost" back into neighbors is
not subtracted -- for typical datasheet xt (~ -30 dB/km) that's <<1
and well below any other attenuation term.

Good enough for SNR / differential-strain trade studies. Full
coupled-mode equations (with mode beating + wavelength dependence +
bend-induced XT) are tracked in #69 as a V2 upgrade.
"""

from __future__ import annotations

import math

import numpy as np
from pyrayleigh import crosstalk_from_phases, crosstalk_pairs
from pyrayleigh.backends import complex_arrays


def neighbors_from_topology(n_cores: int, topology: str) -> list[list[int]]:
    """List where entry [i] is the cores that couple INTO core i."""
    neighbors = [[] for _ in range(n_cores)]
    for source, target in crosstalk_pairs(n_cores, topology):
        neighbors[target].append(source)
    return neighbors


def apply_crosstalk(profile, z, xt_dB_per_km, topology, rng, xp=np):
    """Mix neighbour cores through PyRayleigh, keeping the old RNG draw order.

    Reuse the same draws when differentiating the numerical parameters.
    """
    n_cores, n_z = profile.shape
    if n_cores < 2:
        return profile

    pairs = crosstalk_pairs(n_cores, topology)
    xp, _ = complex_arrays(profile, z, xt_dB_per_km, xp=None if xp is np else xp)
    phases = xp.stack([xp.asarray(rng.uniform(0.0, 2.0 * math.pi, n_z), dtype=xp.float64)
                       for _ in pairs])
    return crosstalk_from_phases(profile, z, xt_dB_per_km, phases, topology=topology, xp=xp)
