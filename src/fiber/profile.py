

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
        fiber = self.config["fiber"]
        source = self.config["source"]

        self.length = fiber["length"]
        self.n_core = fiber["n_core"]
        self.n_cores = fiber["n_cores"]
        self.rayleigh_dB = fiber["rayleigh_coefficient_dB"]
        self.attenuation_dB_km = fiber["attenuation_dB_per_km"]
        self.seed = self.config["simulation"]["seed"]

        # we need the sweep range to compute dz
        self.sweep_range_hz = wavelength_range_to_freq_range(
            source["center_wavelength"], source["sweep_range"]
        )

        self._done = False

    def process(self, acq: Acquisition) -> Acquisition:
        if self._done:
            return acq   # don't regenerate on subsequent sweeps

        xp = self.bk.xp

        # Spatial resolution: dz = c / (2 * n * delta_nu)
        dz = C / (2.0 * self.n_core * self.sweep_range_hz)
        n_z = int(math.ceil(self.length / dz))
        z = xp.arange(n_z) * dz

        # Rayleigh backscatter coefficient (power per meter)
        R_per_m = dB_to_linear(self.rayleigh_dB)
        sigma = math.sqrt(R_per_m * dz)

        # one independent profile per core (circular gaussian phasors)
        parts = []
        for c in range(self.n_cores):
            rng_c = self.bk.random_generator(
                derive_seed(self.seed, component="fiber", core=c)
            )
            re = rng_c.standard_normal(n_z)
            im = rng_c.standard_normal(n_z)
            parts.append((sigma / math.sqrt(2.0)) * (re + 1j * im))
        profile = xp.stack(parts)

        # round-trip attenuation envelope
        attenuation = round_trip_attenuation(z, self.attenuation_dB_km, xp=xp)

        acq.z = z
        acq.dz = dz
        acq.fiber_profile = profile
        acq.attenuation_envelope = attenuation

        acq.add_log("fiber", n_z=n_z, dz_mm=dz * 1e3,
                     attenuation_dB_km=self.attenuation_dB_km,
                     n_cores=self.n_cores)
        self._done = True
        return acq
