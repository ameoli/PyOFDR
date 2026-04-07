"""
Physics:
    nu(t) = nu_start + gamma * t        (instantanous frequency)
    phi(t) = 2pi * integral(nu) dt + phi_noise(t)
    E(t) = sqrt(P) * exp(j*phi(t))      (optical field)

phi_noise is a Wiener process: white frequency noise with PSD h0
gives phase increments d_phi ~ N(0, sqrt(2*pi*linewidth*dt)). The
linewidth here is the Lorentzian FWHM, in Hz.

TODO: full PSD phase noise (flicker, random walk)
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
        sim = config.get("simulation", {})

        self.center_wl = source.get("center_wavelength", 1550e-9)
        self.sweep_range_wl = source.get("sweep_range", 40e-9)
        self.sweep_duration = source.get("sweep_duration", 0.01)
        self.power = source.get("power", 10e-3)
        self.linewidth = source.get("linewidth", 0.0)
        self.sample_rate = adc.get("sample_rate", 200e6)
        self.seed = sim.get("seed", 42)

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

        # Wiener phase noise: dphi ~ N(0, sqrt(2*pi*lw*dt))
        if self.linewidth > 0:
            rng = self.bk.random_generator(self.seed + 1000 + acq.sweep_index)
            sigma = math.sqrt(2.0 * math.pi * self.linewidth * dt)
            phi = phi + xp.cumsum(sigma * rng.standard_normal(n_samples))

        E = math.sqrt(self.power) * xp.exp(1j * phi)

        acq.t = t
        acq.dt = dt
        acq.n_samples = n_samples
        acq.nu_inst = nu_inst
        acq.E_source = E

        acq.add_log("source", n_samples=n_samples, gamma=self.gamma,
                     linewidth=self.linewidth)
        return acq
