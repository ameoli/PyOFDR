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
        self.balanced = det["balanced"]
        self.saturation_current = det["saturation_current"]   # None or A
        self.nl_coeffs = det["nonlinearity_coefficients"]      # [a2, a3, ...] or []
        self.seed = self.config["simulation"]["seed"]

        # balanced mode needs the DC current per arm for shot noise.
        # reference arm dominates (signal is Rayleigh backscatter, tiny)
        if self.balanced:
            eta  = self.config["optics"]["splitting_ratio"]
            P    = self.config["source"]["power"]
            self.dc_current = self.responsivity  * eta * P

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
        # to the already-differenced signal. In single-ended mode we apply
        # the polynomial to the combined current directly (note: the MZI
        # output currently carries only the AC beat; the DC*beat mixing
        # is therefore not modelled for single-ended -- see dev/notes).
        if self.nl_coeffs:
            if self.balanced:
                I_A = self.dc_current + I
                I_B = self.dc_current - I
                I_A_nl = I_A
                I_B_nl = I_B
                for k, a in enumerate(self.nl_coeffs, start=2):
                    if a != 0:
                        I_A_nl = I_A_nl + a * I_A ** k
                        I_B_nl = I_B_nl + a * I_B ** k
                I = (I_A_nl - I_B_nl) / 2.0
            else:
                I_lin = I
                for k, a in enumerate(self.nl_coeffs, start=2):
                    if a != 0:
                        I = I + a * I_lin ** k

        if self.balanced:
            # Balanced detection: two photodiodes see complementary MZI arms.
            # Arm A -> I_dc + I_beat,  arm B -> I_dc - I_beat
            # Subtraction gives 2*I_beat + noiseA - noiseB.
            # We normalize to 1x signal by halving the noise difference,
            # net effect is sqrt(2) less noise => 3dB SNR gain.
            rngs_shot_a   = _rngs(0)
            rngs_shot_b   = _rngs(3)
            rngs_therm_a  = _rngs(1)
            rngs_therm_b  = _rngs(4)
            rngs_dark_a   = _rngs(2)
            rngs_dark_b   = _rngs(5)

            if self.shot_noise_enabled:
                # shot noise from DC current (signal arm is negligible)
                sigma_shot = math.sqrt(2.0 * E_CHARGE * self.dc_current * bw)
                sa = xp.stack([r.standard_normal(n) for r in rngs_shot_a])
                sb = xp.stack([r.standard_normal(n) for r in rngs_shot_b])
                I  = I + sigma_shot * (sa - sb) / 2.0

            if self.thermal_nep > 0:
                sigma_thermal = self.responsivity * self.thermal_nep * math.sqrt(bw)
                ta = xp.stack([r.standard_normal(n) for r in rngs_therm_a])
                tb = xp.stack([r.standard_normal(n) for r in rngs_therm_b])
                I  = I + sigma_thermal * (ta - tb) / 2.0

            if self.dark_current > 0:
                sigma_dark = math.sqrt(2.0 * E_CHARGE * self.dark_current  * bw)
                da = xp.stack([r.standard_normal(n) for r in rngs_dark_a])
                db = xp.stack([r.standard_normal(n) for r in rngs_dark_b])
                I  = I + sigma_dark * (da - db) / 2.0

        else:
            # single-ended
            rngs_shot  = _rngs(0)
            rngs_therm = _rngs(1)
            rngs_dark  = _rngs(2)

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
