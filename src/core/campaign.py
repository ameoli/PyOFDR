from __future__ import annotations

import logging
from core.acquisition import Acquisition
from core.config import compute_derived
logger = logging.getLogger(__name__)


def run_campaign(cfg: dict) -> Acquisition:
    """Run a single-sweep simulation and return the Acquisition.

    Paramters
    ---------
    cfg : dict
        Config dict from load_config.

    Returns
    -------
    Acquisition
        The completed acquisition with digital signal.
    """
    # import here to avoid circular imports
    # (there's probably a better way to do this)
    from fiber.profile import FiberGenerator
    from source.swept_laser import SweptLaser
    from optics.mach_zehnder import MachZehnder
    from detection.detector import Detector
    from detection.filter import AntiAliasFilter
    from digitizer.adc import ADC

    derived = compute_derived(cfg)
    logger.info("PyOFDR v0.1 -- starting simulation")
    logger.info("  dz = %.4f mm, N_z = %d, N_t = %d",
                derived["dz"] * 1e3, derived["n_z"], derived["n_t"])

    # Build pipeline
    # TODO: make this configurable / use a registry pattern
    steps = [
        FiberGenerator(cfg),
        SweptLaser(cfg),
        MachZehnder(cfg),
        Detector(cfg),
        AntiAliasFilter(cfg),
        ADC(cfg),
    ]

    acq = Acquisition()
    for step in steps:
        acq = step.process(acq)

    logger.info("Simulation complete, %d samples generated", acq.n_samples)
    return acq
