"""Central seed derivation for stochastic pipeline steps. See #14."""

from __future__ import annotations

_OFFSETS = {
    "fiber":    0,
    "laser":    1000,
    "detector": 2000,
    "adc":      3000,
}

_CORE_STRIDE = 1_000_000


def derive_seed(base: int, *, component: str, core: int = 0, sweep: int = 0, sub: int = 0) -> int:
    return base + _OFFSETS[component] + core * _CORE_STRIDE + sub * 100_000 + sweep
