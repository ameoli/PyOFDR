"""ADC model.

Ideal uniform mid-tread quantizer + clipping at +/- V_range/2.

Optional ENOB (effective number of bits) injects extra Gaussian noise
*before* quantization, so that the total RMS noise matches what a
2^enob-level ideal ADC would have. This is the standard way to lump
together aperture jitter, INL, thermal noise of the front-end, etc.
into one number, see e.g. ADI MT-003.

  sigma_total = V_range / (2^enob * sqrt(12))
  sigma_quant = V_range / (2^bits * sqrt(12))   (already there from quant)
  sigma_extra = sqrt(sigma_total^2 - sigma_quant^2)

If enob is unset (or equals bits) we don't add anything and the ADC
stays purely quantization-noise limited.

DNL (differential non-linearity) and INL (integral non-linearity) are
the per-code deviations from the ideal transfer characteristic. We
build a single LSB-offset curve at __init__ time (once per chip) and
apply it after quantization. INL is a sinusoid across the code range,
DNL is a random walk with zero endpoint drift. See e.g. Kester, "The
Data Conversion Handbook", chapter 2.
"""

from __future__ import annotations

import math
from typing import Any

from core.acquisition import Acquisition
from core.pipeline import PipelineStep
from utils.seeding import derive_seed


class ADC(PipelineStep):

    name = "adc"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        adc = self.config["adc"]
        self.bits = adc["bits"]
        self.voltage_range = adc["voltage_range"]
        self.v_max = self.voltage_range / 2.0
        self.n_levels = 2 ** self.bits
        self.enob = adc["enob"]   # may be None
        self.jitter_rms = adc["jitter_rms"]
        self.dnl_rms_lsb  = adc["dnl_rms_lsb"]
        self.inl_peak_lsb = adc["inl_peak_lsb"]
        self.seed = self.config["simulation"]["seed"]

        # DNL / INL: these are physical chip-level imperfections, fixed
        # once the part is manufactured. So we compute one realisation up
        # front and reuse it across all sweeps. INL is modelled as a
        # single sinusoid spanning the code range, DNL as a random
        # cumulative walk (each step ~ N(0, dnl_rms)). Final curve[k] is
        # the offset in LSB to subtract from code k.
        self._nl_curve = None
        if self.dnl_rms_lsb > 0 or self.inl_peak_lsb > 0:
            self._nl_curve = self._build_nl_curve()

    def _build_nl_curve(self):
        xp = self.bk.xp
        k = xp.arange(self.n_levels) - self.n_levels // 2
        curve = xp.zeros(self.n_levels)
        if self.inl_peak_lsb > 0:
            curve += self.inl_peak_lsb * xp.sin(
                2.0 * math.pi * (k + self.n_levels // 2) / self.n_levels)
        if self.dnl_rms_lsb > 0:
            rng = self.bk.random_generator(
                derive_seed(self.seed, component="adc", sub=2))
            # cumulative DNL, zero-mean-detrended so it doesn't add a
            # monotonic ramp (that would just be a gain error).
            walk = xp.cumsum(rng.standard_normal(self.n_levels)) * self.dnl_rms_lsb
            walk -= xp.linspace(0, walk[-1], self.n_levels)
            curve += walk
        return curve

    def process(self, acq: Acquisition) -> Acquisition:
        if acq.analog_main is None:
            raise RuntimeError("ADC: analog_main is None")

        xp = self.bk.xp
        analog = acq.analog_main

        # ENOB-equivalent extra noise, injected before quantization
        if self.enob is not None and self.enob < self.bits:
            sigma_total = self.voltage_range / (2.0 ** self.enob * math.sqrt(12.0))
            sigma_quant = self.voltage_range / (2.0 ** self.bits * math.sqrt(12.0))
            sigma_extra = math.sqrt(sigma_total ** 2 - sigma_quant ** 2)
            n_c, n = analog.shape
            rngs = [
                self.bk.random_generator(
                    derive_seed(self.seed, component="adc",
                                core=c, sweep=acq.sweep_index)
                )
                for c in range(n_c)
            ]
            noise = xp.stack([r.standard_normal(n) for r in rngs])
            analog = analog + sigma_extra * noise

        # aperture jitter -- each sample is taken at t + dt_err, so the
        # voltage error is  dV/dt * dt_err  where dt_err ~ N(0, sigma_j)
        # (first order Taylor approx, good enough for sub-ns jitter)
        if self.jitter_rms > 0:
            n_c, n  = analog.shape

            # dV/dt via forward difference, pad last sample
            dvdt = xp.empty_like(analog)
            dvdt[:, :-1] = xp.diff(analog, axis=-1) / acq.dt
            dvdt[:,  -1] = dvdt[:, -2]

            jitter_rngs = [
                self.bk.random_generator(
                    derive_seed(self.seed, component="adc",
                                core=c, sweep=acq.sweep_index, sub=1))
                for c in range(n_c)
            ]
            dt_err = xp.stack([r.standard_normal(n) for r in jitter_rngs])
            analog = analog  +  dvdt * self.jitter_rms * dt_err

        half = self.n_levels // 2   # 32768 for 16-bit

        clipped = xp.clip(analog, -self.v_max, self.v_max)
        normalized = clipped / self.v_max           # [-1, 1]
        digital = xp.floor(normalized * half).astype(xp.int32)
        digital = xp.clip(digital, -half, half - 1)

        # DNL/INL: shift each output code by the precomputed nonlinearity
        # curve (in LSB).  index with (code + half) to land in [0, n_levels).
        if self._nl_curve is not None:
            curve = xp.asarray(self._nl_curve)
            offset = curve[digital + half]
            digital = xp.floor(digital + offset).astype(xp.int32)
            digital = xp.clip(digital, -half, half - 1)

        acq.digital_main = digital.astype(xp.int16)

        acq.add_log("adc", bits=self.bits, enob=self.enob,
                     jitter_rms_ps=self.jitter_rms * 1e12,
                     dnl_rms_lsb=self.dnl_rms_lsb,
                     inl_peak_lsb=self.inl_peak_lsb,
                     lsb_uV=self.voltage_range / self.n_levels * 1e6)
        return acq
