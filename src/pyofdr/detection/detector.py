"""Photodetector with noise sources.

Signal chain:
    optical power [W]  -->  photocurrent (responsivity)
                       -->  per-diode nonlinearity / saturation, then AC beat
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

from pyofdr.core.acquisition import Acquisition
from pyofdr.core.pipeline import PipelineStep
from pyofdr.utils.constants import E_CHARGE
from pyofdr.utils.seeding import derive_seed


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
        self.balanced = det["balanced"]
        self.saturation_current = det["saturation_current"]   # None or A
        self.nl_coeffs = det["nonlinearity_coefficients"]      # [a2, a3, ...] or []
        self.seed = self.config["simulation"]["seed"]

        # DC current per PD for shot noise. reference arm dominates
        # (signal is Rayleigh backscatter, tiny) and the 50/50 recombiner
        # splits it between the two outputs; a single-ended PD sits on one
        # of them, so the same value serves both modes (#91).
        eta  = self.config["optics"]["splitting_ratio"]
        P    = self.config["source"]["power"]
        self.dc_current = 0.5 * self.responsivity  * eta * P

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

        # convert optical power to current
        I = acq.photocurrent_main * self.responsivity
        if self.nl_coeffs or self.saturation_current is not None or self.shot_noise_enabled:
            I, current_a, current_b = self._diode_response(I)

        if self.balanced:
            # In the linear limit, the two PDs see complementary MZI arms.
            # Arm A -> I_dc + I_beat/2,  arm B -> I_dc - I_beat/2.
            # Subtraction recovers the full-difference beat the MZI already
            # stores, plus the noise of BOTH PDs (uncorrelated, so it adds).
            rngs_shot_a   = _rngs(0)
            rngs_shot_b   = _rngs(3)
            rngs_therm    = _rngs(1)
            rngs_dark_a   = _rngs(2)
            rngs_dark_b   = _rngs(5)

            if self.shot_noise_enabled:
                sa = xp.stack([r.standard_normal(n) for r in rngs_shot_a])
                sb = xp.stack([r.standard_normal(n) for r in rngs_shot_b])
                sigma_a = xp.sqrt(2.0 * E_CHARGE * xp.maximum(current_a, 0.0) * bw)
                sigma_b = xp.sqrt(2.0 * E_CHARGE * xp.maximum(current_b, 0.0) * bw)
                I = I + sigma_a*sa - sigma_b*sb

            if self.thermal_nep > 0:
                # single TIA sits after the subtraction node
                sigma_thermal = self.responsivity * self.thermal_nep * math.sqrt(bw)
                th = xp.stack([r.standard_normal(n) for r in rngs_therm])
                I  = I + sigma_thermal * th

            if self.dark_current > 0:
                # dark current is per-PD and does not split
                sigma_dark = math.sqrt(2.0 * E_CHARGE * self.dark_current  * bw)
                da = xp.stack([r.standard_normal(n) for r in rngs_dark_a])
                db = xp.stack([r.standard_normal(n) for r in rngs_dark_b])
                I  = I + sigma_dark * (da - db)

        else:
            # single-ended
            rngs_shot  = _rngs(0)
            rngs_therm = _rngs(1)
            rngs_dark  = _rngs(2)

            # Shot noise uses the total current, even though the static
            # baseline has been removed from the signal path.
            if self.shot_noise_enabled:
                collected = xp.maximum(current_a, 0.0)
                shot_var = 2.0 * E_CHARGE * collected * bw
                shot = xp.stack([r.standard_normal(n) for r in rngs_shot])
                I = I + xp.sqrt(shot_var) * shot

            # thermal noise: sigma = R * NEP * sqrt(B)
            if self.thermal_nep > 0:
                sigma_thermal = self.responsivity * self.thermal_nep * math.sqrt(bw)
                therm = xp.stack([r.standard_normal(n) for r in rngs_therm])
                I = I + sigma_thermal * therm

            # dark current shot noise: sigma^2 = 2*e*I_dark*B
            if self.dark_current > 0:
                sigma_dark = math.sqrt(2.0 * E_CHARGE * self.dark_current * bw)
                dark = xp.stack([r.standard_normal(n) for r in rngs_dark])
                I = I + sigma_dark * dark

        # transimpedance: I -> V
        acq.analog_main = I * self.impedance

        acq.add_log("detection",
                     shot_noise=self.shot_noise_enabled,
                     thermal_nep=self.thermal_nep,
                     dark_current=self.dark_current,
                     balanced=self.balanced,
                     saturation_current=self.saturation_current,
                     nonlinearity_coefficients=tuple(self.nl_coeffs))
        return acq

    def _diode_response(self, beat):
        """Return the AC signal and total currents before DC removal."""
        xp = self.bk.xp
        if self.balanced:
            input_a = self.dc_current + beat/2
            input_b = self.dc_current - beat/2
        else:
            input_a = self.dc_current + beat
            input_b = self.dc_current  # static baseline

        current_a, current_b = input_a, input_b
        for order, coeff in enumerate(self.nl_coeffs, start=2):
            if coeff != 0:
                a, b = input_a**order, input_b**order
                current_a = current_a + coeff*a
                current_b = current_b + coeff*b
                beat = beat + coeff*(a - b)

        if self.saturation_current is not None:
            limited_a = xp.clip(current_a, 0.0, self.saturation_current)
            limited_b = xp.clip(current_b, 0.0, self.saturation_current)
            clipped = (limited_a != current_a) | (limited_b != current_b)
            # Keep small beats clear of DC subtraction unless a diode clips.
            beat = xp.where(clipped, limited_a - limited_b, beat)
            return beat, limited_a, limited_b
        return beat, current_a, current_b
