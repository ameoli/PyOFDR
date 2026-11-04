"""ADC model.

Ideal uniform floor quantizer + clipping at +/- V_range/2.

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
the code-width and transition-level errors. A fixed threshold table
is built once per chip up to 20 bits. Above that, INL thresholds are
evaluated on demand and random DNL is unsupported. INL displaces the
thresholds sinusoidally;
DNL uses positive random code widths, normalized to the full input
range. The sampled voltage is compared with these thresholds before
assigning a code. See Kester, "The Data Conversion Handbook", chapter 2.
"""

from __future__ import annotations

import math
from typing import Any

from pyofdr.core.acquisition import Acquisition
from pyofdr.core.config_models import ADC_TABLE_MAX_BITS
from pyofdr.core.pipeline import PipelineStep
from pyofdr.utils.seeding import derive_seed


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

        # Fixed chip imperfections, shared by all sweeps and channels.
        self._thresholds = None
        if self.bits <= ADC_TABLE_MAX_BITS and (self.dnl_rms_lsb > 0 or self.inl_peak_lsb > 0):
            self._thresholds = self._build_thresholds()

    def _build_thresholds(self):
        xp = self.bk.xp
        edges = xp.arange(self.n_levels + 1, dtype=float)
        if self.dnl_rms_lsb > 0:
            rng = self.bk.random_generator(
                derive_seed(self.seed, component="adc", sub=2))
            # A lognormal width has positive support and CV = dnl_rms_lsb.
            # Normalize the finite realization so there is no net gain error.
            sigma = math.sqrt(math.log1p(self.dnl_rms_lsb * self.dnl_rms_lsb))
            if not math.isfinite(sigma):
                raise ValueError("DNL magnitude is outside the numerical range")
            log_width = sigma * rng.standard_normal(self.n_levels)
            widths = xp.exp(log_width - xp.max(log_width))
            widths *= self.n_levels / xp.sum(widths)
            edges[1:] = xp.cumsum(widths)
            edges[-1] = self.n_levels
        if self.inl_peak_lsb > 0:
            k = xp.arange(self.n_levels + 1, dtype=float)
            inl = self.inl_peak_lsb * xp.sin(2.0 * math.pi * k / self.n_levels)
            inl[[0, self.n_levels // 2, -1]] = 0.0
            edges += inl
        edges = (edges - self.n_levels // 2) * (self.voltage_range / self.n_levels)
        if not xp.all(xp.isfinite(edges)) or xp.any(xp.diff(edges) <= 0):
            raise ValueError("ADC thresholds must be strictly increasing; reduce DNL/INL")
        return edges[1:-1]

    def _quantize_inl(self, voltage):
        xp = self.bk.xp
        half = self.n_levels // 2
        lsb = self.voltage_range / self.n_levels
        flat = voltage.reshape(-1)
        codes = xp.empty(flat.size, dtype=xp.int64)
        # Binary search on the same thresholds as the table. Small blocks keep
        # the temporary arrays bounded even for a long, multicore acquisition.
        for start in range(0, flat.size, 65536):
            v = flat[start:start+65536]
            lo = xp.zeros(v.size, dtype=xp.int64)
            hi = xp.full(v.size, self.n_levels-1, dtype=xp.int64)
            for _ in range(self.bits):
                mid = (lo+hi+1)//2
                offset = self.inl_peak_lsb * xp.sin(2*math.pi*mid/self.n_levels)
                offset = xp.where(mid == half, 0., offset)
                threshold = (mid+offset-half)*lsb
                above = v >= threshold
                lo = xp.where(above, mid, lo)
                hi = xp.where(above, hi, mid-1)
            codes[start:start+v.size] = lo-half
        return codes.reshape(voltage.shape)

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
        if self._thresholds is not None:
            # At a transition, the upper code wins, just as with floor quantization.
            digital = xp.searchsorted(self._thresholds, clipped, side="right") - half
        elif self.inl_peak_lsb > 0:
            digital = self._quantize_inl(clipped)
        else:
            normalized = clipped / self.v_max           # [-1, 1]
            # At 32 bits, +full scale is 2**31 before clipping. Too big for int32.
            digital = xp.floor(normalized * half).astype(xp.int64)
            digital = xp.clip(digital, -half, half - 1)

        # int16 would silently wrap the codes for bits > 16 (#89)
        out_dtype = xp.int16 if self.bits <= 16 else xp.int32
        acq.digital_main = digital.astype(out_dtype)

        acq.add_log("adc", bits=self.bits, enob=self.enob,
                     jitter_rms_ps=self.jitter_rms * 1e12,
                     dnl_rms_lsb=self.dnl_rms_lsb,
                     inl_peak_lsb=self.inl_peak_lsb,
                     lsb_uV=self.voltage_range / self.n_levels * 1e6)
        return acq
