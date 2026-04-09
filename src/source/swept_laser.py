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
from utils.seeding import derive_seed
from utils.units import wavelength_range_to_freq_range


class SweptLaser(PipelineStep):

    name = "source"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        source = self.config["source"]

        self.center_wl = source["center_wavelength"]
        self.sweep_range_wl = source["sweep_range"]
        self.sweep_duration = source["sweep_duration"]
        self.power = source["power"]
        self.linewidth = source["linewidth"]
        self.sample_rate = self.config["adc"]["sample_rate"]
        self.seed = self.config["simulation"]["seed"]

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
            rng = self.bk.random_generator(
                derive_seed(self.seed, component="laser", sweep=acq.sweep_index)
            )
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
