from __future__ import annotations

import logging
from core.acquisition import Acquisition
from core.config import compute_derived
logger = logging.getLogger(__name__)


def run_campaign(cfg: dict) -> list[Acquisition]:
    """Run one or more sweeps and return the list of Acquisitions.

    The fiber profile is generated once and reused across sweeps.
    Noise sources (laser phase noise, RIN, detector noise, ADC jitter)
    are independent for each sweep.

    Paramters
    ---------
    cfg : dict
        Config dict from load_config.

    Returns
    -------
    list[Acquisition]
        One Acquisition per sweep.
    """
    from fiber.profile import FiberGenerator
    from fiber.strain import StrainPerturbation
    from source.swept_laser import SweptLaser
    from optics.mach_zehnder import MachZehnder
    from optics.aux_mzi import AuxMZI
    from detection.detector import Detector
    from detection.filter import AntiAliasFilter
    from digitizer.adc import ADC
    from output.hdf5_writer import HDF5Writer

    derived = compute_derived(cfg)
    n_sweeps = cfg.get("simulation", {}).get("n_sweeps", 1)

    logger.info("PyOFDR v0.1 -- starting simulation (%d sweep%s)",
                n_sweeps, "s" if n_sweeps > 1 else "")
    logger.info("  dz = %.4f mm, N_z = %d, N_t = %d",
                derived["dz"] * 1e3, derived["n_z"], derived["n_t"])

    # build pipeline steps once -- stateful steps (FiberGenerator) cache
    # across sweeps automatically
    steps = [
        FiberGenerator(cfg),
        StrainPerturbation(cfg),
        SweptLaser(cfg),
        MachZehnder(cfg),
        AuxMZI(cfg),     # no-op unless optics.aux_mzi.enabled
        Detector(cfg),
        AntiAliasFilter(cfg),
        ADC(cfg),
    ]

    output_path = cfg.get("output", {}).get("path")
    writer = None
    acquisitions = []

    try:
        if output_path:
            writer = HDF5Writer(output_path)
            writer.__enter__()

        for i in range(n_sweeps):
            acq = Acquisition(sweep_index=i)
            for step in steps:
                acq = step.process(acq)

            if writer is not None:
                if i == 0:
                    writer.write_config(cfg, derived)
                    writer.write_fiber(acq)
                writer.write_sweep(acq, sweep_index=i)

            acquisitions.append(acq)
            logger.info("  sweep %d/%d done", i + 1, n_sweeps)

    finally:
        if writer is not None:
            writer.__exit__(None, None, None)

    if output_path:
        logger.info("Output written to %s", output_path)
    logger.info("Simulation complete, %d sweep%s, %d samples each",
                n_sweeps, "s" if n_sweeps > 1 else "", acquisitions[0].n_samples)
    return acquisitions
