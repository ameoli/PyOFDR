"""Array backend abstraction.

Wraps numpy for now. The point of this module is that the rest of
the code never imports numpy directly for *array operations*: it
goes through `get_backend()` instead. That way when we want to add
cupy (GPU) or jax (autodiff/jit) we just write a new backend class
without touching the pipeline steps.

What lives here:
- xp:                  the array module (numpy / cupy / jax.numpy)
- fft:                 fft submodule (xp.fft)
- random_generator:    factory for a stateful PRNG with .standard_normal(n)
- sosfilt_lowpass:     butterworth low pass (scipy.signal for now)
- to_numpy / asarray:  host <-> device conversion helpers (no-op on numpy)

NOTE: cupy and jax come later. Calling get_backend("cupy") or
get_backend("jax") raises NotImplementedError on purpose, so we
don't accidentally hide the fact that they aren't wired up yet.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt


class NumpyBackend:
    """CPU backend, the default."""

    name = "numpy"

    def __init__(self) -> None:
        self.xp = np
        self.fft = np.fft

    # ---- conversion ----------------------------------------------------
    def asarray(self, x: Any) -> np.ndarray:
        return np.asarray(x)

    def to_numpy(self, x: Any) -> np.ndarray:
        # no-op for numpy; cupy will need .get(), jax will need np.asarray()
        return np.asarray(x)

    # ---- random --------------------------------------------------------
    def random_generator(self, seed: int):
        """Return a stateful PRNG with .standard_normal(n).

        numpy and cupy both expose this same API via default_rng,
        so the wrapper is trivial. jax will need a small adapter
        around its split-key model.
        """
        return np.random.default_rng(seed)

    # ---- signal --------------------------------------------------------
    def sosfilt_lowpass(self, x, order: int, cutoff_norm: float):
        """Butterworth low-pass via SOS filtering.

        cutoff_norm is normalized to Nyquist (in [0, 1]). Filters along
        the last axis so it works for 1D and (n_cores, n_t) inputs.
        """
        sos = butter(order, cutoff_norm, btype="low", output="sos")
        return sosfilt(sos, x, axis=-1)


_DEFAULT: NumpyBackend | None = None


def get_backend(name: str | None = None):
    """Return a backend instance. Only 'numpy' is implemented for now."""
    global _DEFAULT
    if name is None or name == "numpy":
        if _DEFAULT is None:
            _DEFAULT = NumpyBackend()
        return _DEFAULT
    if name == "cupy":
        raise NotImplementedError("cupy backend will be added in a later release")
    if name == "jax":
        raise NotImplementedError("jax backend will be added in a later release")
    raise ValueError(f"Unknown backend '{name}'")


def set_backend(name: str):
    """Reset and return the default backend (mostly for tests / CLI)."""
    global _DEFAULT
    _DEFAULT = None
    return get_backend(name)
