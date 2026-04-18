"""Motion kinds for dynamic strain perturbations (#41).

Each motion kind used to appear in three places that parallel-changed:
the pydantic model in config_models.py (validation), an evaluator for
eps(t), and a sweep-rate under-sampling check. The evaluators and checks
live here now behind a small registry; the pydantic classes stay in
config_models because they share the unit-aware Time/Frequency types
with the rest of the config (moving them would need a shared pydantic
base module and doesn't pay off yet).

Adding a new motion kind = add one Pydantic class in config_models,
plus one _eval_* + _check_* pair + one registry entry here.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable

# eval:  (motion_dict, t)                           -> float
# check: (motion_dict, sweep_duration, seg_index)   -> str | None
EvalFn  = Callable[[dict, float], float]
CheckFn = Callable[[dict, float, int], "str | None"]


def _eval_harmonic(motion: dict, t: float) -> float:
    return motion["amplitude"] * math.sin(
        2.0 * math.pi * motion["frequency"] * t + motion["phase"])


def _check_harmonic(motion: dict, sweep_duration: float, idx: int) -> str | None:
    f = motion.get("frequency", 0.0)
    f_nyq = 0.5 / sweep_duration
    # f == Nyquist with phase=0 lands on zero-crossings every sweep -- warn
    if f >= f_nyq:
        return (f"strain segment {idx}: motion frequency {f:g} Hz >= "
                f"sweep-rate Nyquist {f_nyq:g} Hz; expect aliasing")
    return None


def _eval_thermal(motion: dict, t: float) -> float:
    # first-order relaxation; 0 at t=0, asymptotes to amplitude
    return motion["amplitude"] * (1.0 - math.exp(-t / motion["tau"]))


def _check_thermal(motion: dict, sweep_duration: float, idx: int) -> str | None:
    tau = motion.get("tau", 0.0)
    if 0 < tau < 2.0 * sweep_duration:
        return (f"strain segment {idx}: thermal tau {tau:g} s < "
                f"2*sweep_duration ({2*sweep_duration:g} s); "
                f"transient under-sampled across sweeps")
    return None


def _eval_impulsive(motion: dict, t: float) -> float:
    # gaussian pulse; peak at center_time, std dev = width
    dt = t - motion["center_time"]
    w  = motion["width"]
    return motion["amplitude"] * math.exp(-0.5 * (dt / w) ** 2)


def _check_impulsive(motion: dict, sweep_duration: float, idx: int) -> str | None:
    width = motion.get("width", 0.0)
    if 0 < width < 2.0 * sweep_duration:
        return (f"strain segment {idx}: impulsive width {width:g} s < "
                f"2*sweep_duration ({2*sweep_duration:g} s); "
                f"pulse peak under-sampled across sweeps")
    return None


# kind -> (evaluator, sampling check). adding a row here registers a new kind.
_MOTION_HANDLERS: dict[str, tuple[EvalFn, CheckFn]] = {
    "harmonic":  (_eval_harmonic,  _check_harmonic),
    "thermal":   (_eval_thermal,   _check_thermal),
    "impulsive": (_eval_impulsive, _check_impulsive),
}


def evaluate_motion(motion: dict | None, t: float) -> float:
    """Return the extra epsilon contributed by a dynamic motion at lab time t."""
    if motion is None:
        return 0.0
    kind = motion["kind"]
    try:
        eval_fn = _MOTION_HANDLERS[kind][0]
    except KeyError:
        raise ValueError(f"unknown motion kind: {kind!r}") from None
    return eval_fn(motion, t)


def check_motion_sampling(motion: dict | None, sweep_duration: float,
                          segment_index: int) -> None:
    """Emit a UserWarning if the sweep rate under-samples the motion."""
    if motion is None:
        return
    handler = _MOTION_HANDLERS.get(motion["kind"])
    if handler is None:
        return   # evaluate_motion will raise for us when it's used
    msg = handler[1](motion, sweep_duration, segment_index)
    if msg:
        # user -> StrainPerturbation.__init__ -> here: stacklevel=3 points at user
        warnings.warn(msg, stacklevel=3)
