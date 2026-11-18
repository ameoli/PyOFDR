"""Ask a supplied PyRayleigh fiber for its sampled optical response."""

from numbers import Integral
from collections.abc import Sequence

from pyrayleigh import Fiber, FiberState, MultiCoreFiber, compute_response

from pyofdr.core.acquisition import Acquisition
from pyofdr.core.config_models import InstrumentConfig
from pyofdr.source.swept_laser import SweptLaser


def _state_input(state):
    if state is None or isinstance(state, FiberState):
        return state
    if isinstance(state, Sequence) and not isinstance(state, (str, bytes)):
        return tuple(state)
    raise TypeError("state must be a FiberState, a sequence of per-core states, or None")


def interrogate(config: dict | InstrumentConfig, fiber: Fiber | MultiCoreFiber,
                state: FiberState | Sequence[FiberState | None] | None = None, *, sweep_index=0, max_order=1,
                xp=None) -> Acquisition:
    """One zero-linewidth sweep, before interferometry and detection.

    The supplied fiber and state are retained without rebuilding them.
    Independent cores accept one state or None per core, in core order.
    Per-core inputs are stored as a tuple; shared state objects are retained.
    A time history is evaluated at each laboratory sample time by PyRayleigh.
    xp selects the native response backend; the laser still uses NumPy.
    """
    if not isinstance(fiber, (Fiber, MultiCoreFiber)):
        raise TypeError("fiber must be a PyRayleigh Fiber or MultiCoreFiber")
    state = _state_input(state)
    if isinstance(sweep_index, bool) or not isinstance(sweep_index, Integral) or sweep_index < 0:
        raise ValueError("sweep_index must be a nonnegative integer")
    if isinstance(config, InstrumentConfig):
        config = config.model_dump()
    settings = InstrumentConfig(**config)
    laser = SweptLaser(settings)
    if laser.linewidth > 0:
        raise NotImplementedError("native interrogation currently requires source.linewidth=0")
    acq = laser.process(Acquisition(sweep_index=sweep_index))
    if ((acq.nu_inst < fiber.frequency_min) | (acq.nu_inst > fiber.frequency_max)).any():
        raise ValueError("laser frequencies lie outside the declared fiber band")
    probe = laser.make_probe(acq, xp=xp)
    acq.fiber_response = compute_response(fiber, probe, state, max_order=max_order, xp=xp)
    acq.fiber_model = fiber
    acq.fiber_state = state
    acq.add_log("fiber_response", n_cores=fiber.n_cores if isinstance(fiber, MultiCoreFiber) else 1,
                max_order=max_order)
    return acq
