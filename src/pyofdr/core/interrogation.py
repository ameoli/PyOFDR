"""Ask a supplied PyRayleigh fiber for its sampled optical response."""

from numbers import Integral
from collections.abc import Sequence

from pyrayleigh import Fiber, FiberState, MultiCoreFiber, compute_response, compute_response_diagnostics

from pyofdr.core.acquisition import Acquisition
from pyofdr.core.config_models import InstrumentConfig
from pyofdr.source.swept_laser import SweptLaser


_UNSET = object()


def _state_input(state):
    if state is None or isinstance(state, FiberState):
        return state
    if isinstance(state, Sequence) and not isinstance(state, (str, bytes)):
        return tuple(state)
    raise TypeError("state must be a FiberState, a sequence of per-core states, or None")


def interrogate(config: dict | InstrumentConfig, fiber: Fiber | MultiCoreFiber,
                state: FiberState | Sequence[FiberState | None] | None = None, *, sweep_index=0, max_order=_UNSET,
                xp=None, method="truncated", loss_convention=None, denominator_tolerance=_UNSET,
                diagnostics=False) -> Acquisition:
    """One zero-linewidth sweep, before interferometry and detection.

    The supplied fiber and state are retained without rebuilding them.
    Independent cores accept one state or None per core, in core order.
    Per-core inputs are stored as a tuple; shared state objects are retained.
    A time history is evaluated at each laboratory sample time by PyRayleigh.
    xp selects the native response backend; the laser still uses NumPy.
    method="truncated" keeps max_order=1 by default. method="exact" requires
    max_order to be omitted and loss_convention="background". It supports
    one scalar core and a static or frozen state, including Bragg intervals.
    method="direct_consistent" uses the same local transmissions while keeping
    one reflection per physical block, with the same scalar static restrictions.
    denominator_tolerance is forwarded when supplied. PyRayleigh checks
    these solver options and physical constraints.
    diagnostics=True requires method="exact" and keeps direct-consistent,
    residual and interference data in acq.fiber_diagnostics.
    fiber_response is its total response, so later optics still use the exact
    field. The default call does not calculate or retain diagnostic fields.
    """
    if not isinstance(diagnostics, bool):
        raise TypeError("diagnostics must be a bool")
    if diagnostics and (method != "exact" or max_order is not _UNSET):
        raise ValueError("diagnostics requires method='exact' and max_order omitted")
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
    options = {"method": method, "loss_convention": loss_convention}
    if max_order is not _UNSET:
        options["max_order"] = max_order
    if denominator_tolerance is not _UNSET:
        options["denominator_tolerance"] = denominator_tolerance
    if diagnostics:
        diagnostic_options = dict(options)
        del diagnostic_options["method"]
        acq.fiber_diagnostics = compute_response_diagnostics(fiber, probe, state, xp=xp, **diagnostic_options)
        acq.fiber_response = acq.fiber_diagnostics.total
    else:
        acq.fiber_response = compute_response(fiber, probe, state, xp=xp, **options)
    acq.fiber_model = fiber
    acq.fiber_state = state
    if method == "truncated":
        details = {"max_order": 1 if max_order is _UNSET else max_order}
    else:
        details = dict(options)
    if diagnostics:
        details["diagnostics"] = True
    acq.add_log("fiber_response", n_cores=fiber.n_cores if isinstance(fiber, MultiCoreFiber) else 1, **details)
    return acq
