from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from collections.abc import Sequence
from numbers import Real

import numpy as np
from pyrayleigh import Fiber, FiberState, MultiCoreFiber

from pyofdr import __version__
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.config import compute_derived, compute_instrument_derived
from pyofdr.core.config_models import InstrumentConfig, RootConfig
from pyofdr.core.interrogation import _state_input, interrogate
logger = logging.getLogger(__name__)


def run_campaign(cfg: dict | InstrumentConfig, keep_acquisitions: bool = True, *,
                 fiber: Fiber | MultiCoreFiber | None = None,
                 state: FiberState | Sequence[FiberState | None] | None = None, max_order=1,
                 fft_dz: float | None = None) -> list[Acquisition]:
    """Run one or more sweeps and return the list of Acquisitions.

    The supplied fiber or generated legacy profile is reused across sweeps.
    Noise sources (laser phase noise, RIN, detector noise, ADC jitter)
    are independent for each sweep.

    Parameters
    ----------
    cfg : dict or InstrumentConfig
        A legacy recipe, or instrument-only settings when fiber is supplied.
    keep_acquisitions : bool
        Keep each sweep in memory. If False, output.path is required
        and acquisitions are released after writing them to HDF5.
    fiber, state
        External PyRayleigh objects, reused across all sweeps. Time histories
        are sampled in laboratory time. Native sweeps require linewidth zero.
        Independent cores can receive a sequence of states in core order.
        The sequence is copied once; PyRayleigh checks coupled-core constraints.
    max_order : int
        Point reflection order for the supplied fiber (1, 2 or 3).
    fft_dz : float or None
        Optional nominal FFT spatial spacing [m] for the external fiber.
        None leaves the range calibration unset. The full campaign uses NumPy.

    Returns
    -------
    list[Acquisition]
        One Acquisition per sweep, or an empty list if keep_acquisitions=False.
    """
    from pyofdr.fiber.profile import FiberGenerator
    from pyofdr.fiber.strain import StrainPerturbation
    from pyofdr.fiber.temperature import TemperaturePerturbation
    from pyofdr.fiber.pipeline import FiberModel
    from pyofdr.source.swept_laser import SweptLaser
    from pyofdr.optics.mach_zehnder import MachZehnder
    from pyofdr.optics.aux_mzi import AuxMZI
    from pyofdr.detection.detector import Detector
    from pyofdr.detection.filter import AntiAliasFilter
    from pyofdr.digitizer.adc import ADC
    from pyofdr.output.hdf5_writer import HDF5Writer

    native = fiber is not None
    if native:
        if not isinstance(fiber, (Fiber, MultiCoreFiber)):
            raise TypeError("fiber must be a PyRayleigh Fiber or MultiCoreFiber")
        state = _state_input(state)
        if fft_dz is not None and (isinstance(fft_dz, bool) or not isinstance(fft_dz, Real)
                                   or not math.isfinite(fft_dz) or fft_dz <= 0):
            raise ValueError("fft_dz must be a finite positive spatial spacing [m]")
    elif state is not None or max_order != 1 or fft_dz is not None:
        raise ValueError("state, max_order and fft_dz require an external fiber")
    if isinstance(cfg, InstrumentConfig):
        if not native and not isinstance(cfg, RootConfig):
            raise ValueError("InstrumentConfig requires an external fiber")
        cfg = cfg.model_dump()
    if native:
        settings = InstrumentConfig(**cfg)
        cfg = settings.model_dump()

    output_path = cfg.get("output", {}).get("path")
    if not keep_acquisitions and not output_path:
        raise ValueError("output.path is required when keep_acquisitions=False")

    if native:
        derived = compute_instrument_derived(settings)
        derived.update(length=fiber.length,
                       n_cores=fiber.n_cores if isinstance(fiber, MultiCoreFiber) else 1,
                       fft_dz=0. if fft_dz is None else fft_dz)
        step_config = settings
        steps = []
    else:
        derived = compute_derived(cfg)
        step_config = cfg
        steps = [FiberGenerator(cfg), StrainPerturbation(cfg),
                 TemperaturePerturbation(cfg), FiberModel(cfg), SweptLaser(cfg)]

    n_sweeps = cfg.get("simulation", {}).get("n_sweeps", 1)

    logger.info("PyOFDR v%s -- starting simulation (%d sweep%s)",
                __version__, n_sweeps, "s" if n_sweeps > 1 else "")
    if native:
        logger.info("  external fiber: L = %g m, cores = %d, N_t = %d",
                    derived["length"], derived["n_cores"], derived["n_t"])
    else:
        logger.info("  dz = %.4f mm, N_z = %d, N_t = %d",
                    derived["dz"] * 1e3, derived["n_z"], derived["n_t"])

    # build pipeline steps once -- stateful steps (FiberGenerator) cache
    # across sweeps automatically
    steps += [
        MachZehnder(step_config),
        AuxMZI(step_config),     # no-op unless optics.aux_mzi.enabled
        Detector(step_config),
        AntiAliasFilter(step_config),
        ADC(step_config),
    ]

    acquisitions = []
    n_samples = 0

    with (HDF5Writer(output_path) if output_path else nullcontext()) as writer:
        for i in range(n_sweeps):
            if native:
                acq = interrogate(settings, fiber, state, sweep_index=i, max_order=max_order)
                for name in ("reflection", "transmission"):
                    if not np.all(np.isfinite(getattr(acq.fiber_response, name))):
                        raise ValueError(f"PyRayleigh {name} contains non-finite values in sweep {i}")
                if fft_dz is not None:
                    acq.fft_dz = fft_dz
            else:
                acq = Acquisition(sweep_index=i)
            for step in steps:
                acq = step.process(acq)

            if writer is not None:
                if i == 0:
                    writer.write_config(cfg, derived)
                    writer.write_fiber(acq)
                writer.write_sweep(acq, sweep_index=i)

            n_samples = acq.n_samples
            if keep_acquisitions:
                acquisitions.append(acq)
            else:
                # The shared fiber stays cached. This sweep's arrays can go now.
                del acq
            logger.info("  sweep %d/%d done", i + 1, n_sweeps)

    if output_path:
        logger.info("Output written to %s", output_path)
    logger.info("Simulation complete, %d sweep%s, %d samples each",
                n_sweeps, "s" if n_sweeps > 1 else "", n_samples)
    return acquisitions
