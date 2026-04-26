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

from utils.units import dB_to_linear


def _neighbors_hex7() -> list[list[int]]:
    # core 0 at centre, 1..6 around it (ring). Each outer core couples
    # to the centre and its two ring neighbours.
    return [
        [1, 2, 3, 4, 5, 6],    # centre <- all outer
        [0, 2, 6],             # outer 1 <- centre + 2, 6
        [0, 1, 3],             # outer 2 <- centre + 1, 3
        [0, 2, 4],             # outer 3 <- centre + 2, 4
        [0, 3, 5],             # outer 4 <- centre + 3, 5
        [0, 4, 6],             # outer 5 <- centre + 4, 6
        [0, 5, 1],             # outer 6 <- centre + 5, 1
    ]


def neighbors_from_topology(n_cores: int, topology: str) -> list[list[int]]:
    """List where entry [i] is the cores that couple INTO core i."""
    if topology == "hex7":
        if n_cores != 7:
            raise ValueError(f"topology 'hex7' needs n_cores=7, got {n_cores}")
        return _neighbors_hex7()
    if topology == "linear":
        out = []
        for i in range(n_cores):
            ns = []
            if i > 0:
                ns.append(i - 1)
            if i < n_cores - 1:
                ns.append(i + 1)
            out.append(ns)
        return out
    raise ValueError(f"unknown crosstalk topology: {topology!r}")


def apply_crosstalk(profile, z, xt_dB_per_km, topology, rng, xp=np):
    """Return a new (n_cores, n_z) profile with neighbour cores mixed in."""
    n_cores, n_z = profile.shape
    if n_cores < 2:
        return profile

    nbs = neighbors_from_topology(n_cores, topology)

    # amplitude coupling c(z) = sqrt(xt_power_per_km * z_in_km)
    xt_pwr = dB_to_linear(xt_dB_per_km)
    c_amp = xp.sqrt(xt_pwr * z / 1000.0)    # (n_z,)

    out = xp.array(profile, copy=True)
    for i in range(n_cores):
        for j in nbs[i]:
            phi = rng.uniform(0.0, 2.0 * math.pi, n_z)
            out[i] = out[i] + profile[j] * c_amp * xp.exp(1j * phi)
    return out
