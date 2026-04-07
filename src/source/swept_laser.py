"""
Physics:
    nu(t) = nu_start + gamma * t        (instantanous frequency)
    phi(t) = 2pi * integral(nu) dt      (accumulated phase)
    E(t) = sqrt(P) * exp(j*phi(t))      (optical field)

No phase noise, no RIN, no sweep non-linearity for now.
These will be important later but let's get the basic structure
working first.

TODO: add phase noise
add non-linear sweep correction
"""

from __future__ import annotations

import math
from typing import Any

from core.acquisition import Acquisition
from core.pipeline import PipelineStep
from utils.constants import C
from utils.units import wavelength_range_to_freq_range


class SweptLaser(PipelineStep):

    name = "source"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        source = config.get("source", {})
        adc = config.get("adc", {})

        self.center_wl = source.get("center_wavelength", 1550e-9)
        self.sweep_range_wl = source.get("sweep_range", 40e-9)
        self.sweep_duration = source.get("sweep_duration", 0.01)
        self.power = source.get("power", 10e-3)
        self.sample_rate = adc.get("sample_rate", 200e6)

        # derived quantities
        self.nu_center = C / self.center_wl
        self.sweep_range_hz = wavelength_range_to_freq_range(
            self.center_wl, self.sweep_range_wl
        )
        self.gamma = self.sweep_range_hz / self.sweep_duration   # Hz/s

    def process(self, acq: Acquisition) -> Acquisition:
        xp = self.bk.xp
        dt = 1.0 / self.sample_rate
        n_samples = int(math.ceil(self.sweep_duration * self.sample_rate))
        t = xp.arange(n_samples) * dt

        # instantanous frequency (linear ramp)
        nu_start = self.nu_center - self.sweep_range_hz / 2.0
        nu_inst = nu_start + self.gamma * t

        # phase = 2pi * cumulative sum (rectangle rule integration)
        phi = 2.0 * math.pi * xp.cumsum(nu_inst) * dt

        # optical field -- just sqrt(P) * exp(j*phi), no noise
        E = math.sqrt(self.power) * xp.exp(1j * phi)

        acq.t = t
        acq.dt = dt
        acq.n_samples = n_samples
        acq.nu_inst = nu_inst
        acq.E_source = E

        acq.add_log("source", n_samples=n_samples, gamma=self.gamma)
        return acq
