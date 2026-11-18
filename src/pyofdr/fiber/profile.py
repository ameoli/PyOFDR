

from __future__ import annotations

import math
from typing import Any

from pyrayleigh import PointReflector, crosstalk_from_phases
from pyrayleigh.backends import real_arrays

from pyofdr.core.acquisition import Acquisition
from pyofdr.core.pipeline import PipelineStep
from pyofdr.fiber.attenuation import (
    attenuation_samples_from_segments, cumulative_loss_dB,
    round_trip_attenuation, round_trip_attenuation_varying,
)
from pyofdr.fiber.bends import bend_loss_to_positions, bend_loss_total_dB
from pyofdr.fiber.multiple_scattering import add_ghost_reflections
from pyofdr.fiber.model import build_fiber
from pyofdr.fiber.phase import phase_at_bins
from pyofdr.fiber.reflectors import (attenuation_with_connectors, profile_with_reflectors,
                                    reflector_bin)
from pyofdr.utils.constants import C
from pyofdr.utils.units import wavelength_range_to_freq_range


class FiberGenerator(PipelineStep):

    name = "fiber"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        fiber = self.config["fiber"]
        source = self.config["source"]

        self.length = fiber["length"]
        self.n_core = fiber["n_core"]
        self.n_cores = fiber["n_cores"]
        self.attenuation_dB_km = fiber["attenuation_dB_per_km"]
        self.attenuation_segments = fiber.get("attenuation_segments", [])
        self.bends = fiber.get("bends", [])
        self.reflectors = sorted(fiber.get("reflectors", []), key=lambda r: r["z"])
        if len({ref["z"] for ref in self.reflectors}) != len(self.reflectors):
            raise ValueError("discrete reflectors must occupy distinct spatial bins")
        self.multiple_scattering = fiber.get("multiple_scattering", None)
        self.center_wl = source["center_wavelength"]

        # we need the sweep range to compute dz
        self.sweep_range_hz = wavelength_range_to_freq_range(
            source["center_wavelength"], source["sweep_range"]
        )

        self._cached_z = None
        self._cached_dz = None
        self._cached_profile = None
        self._cached_atten = None
        self._cached_ghosts = None
        self._cached_model = None

    def process(self, acq: Acquisition) -> Acquisition:
        if self._cached_profile is not None:
            # re-attach cached fiber to a fresh Acquisition
            acq.z = self._cached_z
            acq.dz = self._cached_dz
            acq.fiber_profile = self._cached_profile
            acq.attenuation_envelope = self._cached_atten
            acq.ghost_profile = self._cached_ghosts
            acq.fiber_model = self._cached_model
            return acq

        xp = self.bk.xp

        # Spatial resolution: dz = c / (2 * n * delta_nu)
        dz = C / (2.0 * self.n_core * self.sweep_range_hz)
        frequency = C / self.center_wl
        native_xp, _ = real_arrays(xp.asarray(dz))
        model = build_fiber(self.config, cell_length=dz, reference_temperature=293.15,
                            frequency_min=frequency - self.sweep_range_hz/2,
                            frequency_max=frequency + self.sweep_range_hz/2, xp=native_xp)
        z = xp.asarray(model.cores[0].rayleigh.positions).copy()
        n_z = len(z)
        profile = xp.stack([core.rayleigh.coefficients for core in model.cores])

        # discrete reflectors (connectors, splices, etc)
        reflection = None
        if self.reflectors:
            bins = [reflector_bin(r, dz, n_z) for r in self.reflectors]
            if len(set(bins)) != len(bins):
                raise ValueError("discrete reflectors must occupy distinct spatial bins")
            # r_right has no input-side loss in the configured point model.
            reflection = xp.stack([
                xp.stack([-xp.conj(point.r_right) for point in core.elements
                          if isinstance(point, PointReflector)])
                for core in model.cores
            ])
            profile = profile_with_reflectors(profile, dz, self.reflectors,
                                              reflection=reflection, xp=xp)

        index = model.cores[0].index_perturbation
        phi_n = None
        if index is not None:
            # Configured index profiles are steps, using their right-hand values.
            columns = index.positions.searchsorted(z, side="right") - 1
            inside = (z >= index.positions[0]) & (z <= index.positions[-1])
            delta_n = xp.where(inside, index.values[columns], 0.)
            k0 = 2.0 * math.pi / self.center_wl
            phi_n = phase_at_bins(delta_n, dz, 2.0 * k0, xp)
            profile = profile * xp.exp(1j * phi_n)

        # MCF core-to-core crosstalk. Applied after all per-core phase
        # modifications so whatever each core carries gets mixed.
        coupling = model.crosstalk
        if coupling is not None:
            phases = coupling.phases[:, :n_z]
            profile = crosstalk_from_phases(profile, z, model.xt_db_per_km, phases,
                                           topology=coupling.topology, xp=native_xp)

        # Base attenuation, including any per-segment overrides.
        if self.attenuation_segments:
            alpha_z = attenuation_samples_from_segments(
                self.attenuation_dB_km, self.attenuation_segments, z, xp=xp)
            attenuation = round_trip_attenuation_varying(z, alpha_z, dz, xp=xp)
            loss_to_z = cumulative_loss_dB(alpha_z, dz, xp=xp)
        else:
            attenuation = round_trip_attenuation(z, self.attenuation_dB_km, xp=xp)
            loss_to_z = z * (self.attenuation_dB_km / 1000.0)

        for bend in self.bends:
            total_dB = bend_loss_total_dB(bend, wavelength=self.center_wl)
            loss_to_z = loss_to_z + bend_loss_to_positions(
                z, bend["start"], bend["end"], total_dB, xp=xp)
        if self.bends:
            attenuation = xp.exp(-loss_to_z * math.log(10.0) / 10.0)

        # connector insertion losses (step-downs in the envelope)
        if self.reflectors:
            points = [point for point in model.cores[0].elements if isinstance(point, PointReflector)]
            forward_transmission = xp.stack([point.t_left_to_right for point in points])
            transmission = xp.stack([point.t_left_to_right * point.t_right_to_left
                                     for point in points])
            attenuation = attenuation_with_connectors(attenuation, dz, self.reflectors,
                                                       transmission=transmission, xp=xp)

        ghosts = None
        if self.multiple_scattering is not None and self.reflectors:
            ghosts = xp.zeros_like(profile)
            add_ghost_reflections(
                ghosts, dz, self.reflectors, reflection=reflection,
                loss_to_z=loss_to_z,
                forward_transmission=forward_transmission,
                max_order=self.multiple_scattering["max_order"], xp=xp,
            )
            # Index/strain/temperature still use the apparent-bin phase.
            if phi_n is not None:
                ghosts *= xp.exp(1j * phi_n)
            if coupling is not None:
                ghosts = crosstalk_from_phases(ghosts, z, model.xt_db_per_km, phases,
                                              topology=coupling.topology, xp=native_xp)

        acq.z = z
        acq.dz = dz
        acq.fiber_profile = profile
        acq.attenuation_envelope = attenuation
        acq.ghost_profile = ghosts
        acq.fiber_model = model

        # cache so subsequent sweeps reuse the same fiber
        self._cached_z = z
        self._cached_dz = dz
        self._cached_profile = profile
        self._cached_atten = attenuation
        self._cached_ghosts = ghosts
        self._cached_model = model

        acq.add_log("fiber", n_z=n_z, dz_mm=dz * 1e3,
                     attenuation_dB_km=self.attenuation_dB_km,
                     n_cores=self.n_cores)
        return acq
