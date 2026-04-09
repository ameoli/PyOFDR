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
from utils.seeding import derive_seed


class Detector(PipelineStep):

    name = "detection"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        det = self.config["detection"]
        adc = self.config["adc"]

        self.responsivity = det["responsivity"]                # A/W
        self.impedance = adc["input_impedance"]                # ohm
        self.shot_noise_enabled = det["shot_noise"]
        self.thermal_nep = det["thermal_nep"]                  # W/sqrt(Hz)
        self.dark_current = det["dark_current"]                # A
        self.seed = self.config["simulation"]["seed"]

    def process(self, acq: Acquisition) -> Acquisition:
        if acq.photocurrent_main is None:
            raise RuntimeError("Detector: photocurrent not set")

        xp = self.bk.xp
        n_c, n = acq.photocurrent_main.shape
        dt = acq.dt
        bw = 1.0 / (2.0 * dt)

        # one rng per core per noise type, so toggling shot/thermal/dark
        # doesn't reshuffle the others. each detector is physically independent.
        def _rngs(sub: int):
            return [
                self.bk.random_generator(
                    derive_seed(self.seed, component="detector",
                                core=c, sweep=acq.sweep_index, sub=sub)
                )
                for c in range(n_c)
            ]
        rngs_shot = _rngs(0)
        rngs_therm = _rngs(1)
        rngs_dark = _rngs(2)

        # convert optical power to current
        I = acq.photocurrent_main * self.responsivity

        # shot noise: sigma^2 = 2 * e * |I| * B
        if self.shot_noise_enabled:
            shot_var = 2.0 * E_CHARGE * xp.abs(I) * bw
            shot = xp.stack([r.standard_normal(n) for r in rngs_shot])
            I = I + xp.sqrt(shot_var) * shot

        # thermal noise: sigma = R * NEP * sqrt(B)
        if self.thermal_nep > 0:
            sigma_thermal = self.responsivity * self.thermal_nep * math.sqrt(bw)
            therm = xp.stack([r.standard_normal(n) for r in rngs_therm])
            I = I + sigma_thermal * therm

        # dark current shot noise: sigma^2 = 2 * e * I_dark * B
        if self.dark_current > 0:
            sigma_dark = math.sqrt(2.0 * E_CHARGE * self.dark_current * bw)
            dark = xp.stack([r.standard_normal(n) for r in rngs_dark])
            I = I + sigma_dark * dark

        # transimpedance: I -> V
        acq.analog_main = I * self.impedance

        acq.add_log("detection",
                     shot_noise=self.shot_noise_enabled,
                     thermal_nep=self.thermal_nep,
                     dark_current=self.dark_current)
        return acq
