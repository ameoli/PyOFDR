"""Central seed derivation for stochastic pipeline steps. See #14."""

from __future__ import annotations

from hashlib import sha256
from operator import index


SEED_SCHEME = "sha256-v1"

# Keep these IDs fixed when adding components.
_COMPONENT_IDS = {
    "fiber":               0,
    "laser":               1,
    "detector":            2,
    "adc":                 3,
    "crosstalk":           4,
    "index_fluctuations":  5,
}


def derive_seed(base: int, *, component: str, core: int = 0, sweep: int = 0, sub: int = 0) -> int:
    """Deterministic integer seed for one component/core/sweep/substream.

    Inputs must be non-negative integers. The old additive seeds are
    not preserved: the same config now gives a different realisation.
    """
    base = index(base)
    core = index(core)
    sweep = index(sweep)
    sub = index(sub)
    if min(base, core, sweep, sub) < 0:
        raise ValueError("seed and stream indices must be non-negative")

    c = _COMPONENT_IDS[component]
    # Separators keep the fields distinct even for very large counters.
    key = f"{SEED_SCHEME}:{base}:{c}:{core}:{sweep}:{sub}"
    data = sha256(key.encode("ascii")).digest()
    return int.from_bytes(data,  "big")
