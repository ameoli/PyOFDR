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

        # photodiode saturation: real PDs have a max linear photocurrent
        # above which the response flattens. We model this as a hard clip
        # applied BEFORE the noise sources (noise is added by the TIA /
        # on the clipped current). symmetric because balanced mode can
        # swing negative.
        if self.saturation_current is not None:
            I_sat = self.saturation_current
            I = xp.clip(I, -I_sat, I_sat)

        # small-signal photodiode nonlinearity. In balanced mode we
        # reconstruct per-PD currents as I_dc +- I_beat, apply the polynomial
        # to each (matched PDs), and subtract -- this keeps the DC*beat
        # mixing that would otherwise be absent from a polynomial applied
        # to the already-differenced signal. Single-ended: the one PD sits
        # at I_dc + I_beat, so expand the polynomial there and drop the
        # static part (signal path is AC-coupled, #91) -- the mixing terms
        # survive just like in the balanced case (#98).
        if self.nl_coeffs:
            if self.balanced:
                # per-PD beat is half the stored full difference
                I_A = self.dc_current + I / 2.0
                I_B = self.dc_current - I / 2.0
                I_A_nl = I_A
                I_B_nl = I_B
                for k, a in enumerate(self.nl_coeffs, start=2):
                    if a != 0:
                        I_A_nl = I_A_nl + a * I_A ** k
                        I_B_nl = I_B_nl + a * I_B ** k
                I = I_A_nl - I_B_nl
            else:
                I_pd = self.dc_current + I
                for k, a in enumerate(self.nl_coeffs, start=2):
                    if a != 0:
                        I = I + a * (I_pd ** k - self.dc_current ** k)

        if self.balanced:
            # Balanced detection: two photodiodes see complementary MZI arms.
            # Arm A -> I_dc + I_beat/2,  arm B -> I_dc - I_beat/2.
            # Subtraction recovers the full-difference beat the MZI already
            # stores, plus the noise of BOTH PDs (uncorrelated, so it adds).
            rngs_shot_a   = _rngs(0)
            rngs_shot_b   = _rngs(3)
            rngs_therm    = _rngs(1)
            rngs_dark_a   = _rngs(2)
            rngs_dark_b   = _rngs(5)

            if self.shot_noise_enabled:
                # shot noise from the per-PD DC current (signal arm negligible)
                sigma_shot = math.sqrt(2.0 * E_CHARGE * self.dc_current * bw)
                sa = xp.stack([r.standard_normal(n) for r in rngs_shot_a])
                sb = xp.stack([r.standard_normal(n) for r in rngs_shot_b])
                I  = I + sigma_shot * (sa - sb)

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

            # shot noise: sigma^2 = 2*e*(I_dc + I)*B. The reference-arm DC
            # dominates; the beat only modulates it. The signal path itself
            # stays AC-coupled -- I_dc is noise bookkeeping only (#91).
            if self.shot_noise_enabled:
                shot_var = 2.0 * E_CHARGE * xp.maximum(self.dc_current + I, 0.0) * bw
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
