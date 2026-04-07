

from __future__ import annotations

import math
from typing import Any

from core.acquisition import Acquisition
from core.pipeline import PipelineStep
from fiber.attenuation import round_trip_attenuation
from utils.constants import C
from utils.seeding import derive_seed
from utils.units import dB_to_linear, wavelength_range_to_freq_range


class FiberGenerator(PipelineStep):

    name = "fiber"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        fiber = config.get("fiber", {})
        source = config.get("source", {})

        self.length = fiber.get("length", 10.0)
        self.n_core = fiber.get("n_core", 1.4682)
        self.rayleigh_dB = fiber.get("rayleigh_coefficient_dB", -82.0)
        self.attenuation_dB_km = fiber.get("attenuation_dB_per_km", 0.0)
        self.seed = config.get("simulation", {}).get("seed", 42)

        # we need the sweep range to compute dz
        center_wl = source.get("center_wavelength", 1550e-9)
        sweep_wl = source.get("sweep_range", 40e-9)
        self.sweep_range_hz = wavelength_range_to_freq_range(center_wl, sweep_wl)

        self._done = False

    def process(self, acq: Acquisition) -> Acquisition:
        if self._done:
            return acq   # don't regenerate on subsequent sweeps

        xp = self.bk.xp
        rng = self.bk.random_generator(derive_seed(self.seed, component="fiber"))

        # Spatial resolution: dz = c / (2 * n * delta_nu)
        dz = C / (2.0 * self.n_core * self.sweep_range_hz)
        n_z = int(math.ceil(self.length / dz))
        z = xp.arange(n_z) * dz

        # Rayleigh backscatter coefficient (power per meter)
        R_per_m = dB_to_linear(self.rayleigh_dB)
        sigma = math.sqrt(R_per_m * dz)

        # Circular gaussian phasors: E[|r|^2] = sigma^2
        # Each r(z) = sigma/sqrt(2) * (X + jY), X,Y ~ N(0,1)
        re = rng.standard_normal(n_z)
        im = rng.standard_normal(n_z)
        profile = (sigma / math.sqrt(2.0)) * (re + 1j * im)

        # round-trip attenuation envelope
        attenuation = round_trip_attenuation(z, self.attenuation_dB_km, xp=xp)

        acq.z = z
        acq.dz = dz
        acq.fiber_profile = profile
        acq.attenuation_envelope = attenuation

        acq.add_log("fiber", n_z=n_z, dz_mm=dz * 1e3,
                     attenuation_dB_km=self.attenuation_dB_km)
        self._done = True
        return acq
