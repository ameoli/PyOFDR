"""Photodetector with noise sources.

Signal chain:
    optical power [W]  -->  photocurrent (responsivity)
                       -->  + shot noise
                       -->  + thermal noise
                       -->  + dark current noise
                       -->  * transimpedance  -->  voltage [V]

Shot noise comes from the quantum nature of photons. Each photon
arriving at the detector generates an electron with some probability.
The resulting current has Poisson statistics, which for large
currents is well approximated by Gaussian with variance 2*e*I*B.

Thermal noise comes from the transimpedance amplifier and is
specified via the NEP (noise-equivalent power). The current noise
is sigma = R * NEP * sqrt(B) where R is responsivity and B is
the noise bandwith (= fs/2 for our sampled signal).

Dark current is a small current that flows even with no light.
It contributes additional shot noise with variance 2*e*I_dark*B.

Anti-alias filter is a separate step (detection/filter.py).
"""

from __future__ import annotations

import math
from typing import Any

from core.acquisition import Acquisition
from core.pipeline import PipelineStep
from utils.constants import E_CHARGE


class Detector(PipelineStep):

    name = "detection"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        det = config.get("detection", {})
        adc = config.get("adc", {})
        simulation = config.get("simulation", {})

        self.responsivity = det.get("responsivity", 1.0)       # A/W
        self.impedance = adc.get("input_impedance", 50.0)      # ohm
        self.shot_noise_enabled = det.get("shot_noise", True)
        self.thermal_nep = det.get("thermal_nep", 1.0e-11)     # W/sqrt(Hz)
        self.dark_current = det.get("dark_current", 1.0e-9)    # A
        self.seed = simulation.get("seed", 42)

    def process(self, acq: Acquisition) -> Acquisition:
        if acq.photocurrent_main is None:
            raise RuntimeError("Detector: photocurrent not set")

        xp = self.bk.xp
        rng = self.bk.random_generator(self.seed + 2000 + acq.sweep_index)
        n = acq.n_samples
        dt = acq.dt

        # noise bandwidth (one-sided Nyquist)
        bw = 1.0 / (2.0 * dt)

        # convert optical power to current
        I = acq.photocurrent_main * self.responsivity

        # shot noise: sigma^2 = 2 * e * |I| * B
        if self.shot_noise_enabled:
            shot_var = 2.0 * E_CHARGE * xp.abs(I) * bw
            I = I + xp.sqrt(shot_var) * rng.standard_normal(n)

        # thermal noise: sigma = R * NEP * sqrt(B)
        if self.thermal_nep > 0:
            sigma_thermal = self.responsivity * self.thermal_nep * math.sqrt(bw)
            I = I + sigma_thermal * rng.standard_normal(n)

        # dark current shot noise: sigma^2 = 2 * e * I_dark * B
        if self.dark_current > 0:
            sigma_dark = math.sqrt(2.0 * E_CHARGE * self.dark_current * bw)
            I = I + sigma_dark * rng.standard_normal(n)

        # transimpedance: I -> V
        acq.analog_main = I * self.impedance

        acq.add_log("detection",
                     shot_noise=self.shot_noise_enabled,
                     thermal_nep=self.thermal_nep,
                     dark_current=self.dark_current)
        return acq
