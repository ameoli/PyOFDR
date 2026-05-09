

from __future__ import annotations

import math
from typing import Any

from pyofdr.core.acquisition import Acquisition
from pyofdr.core.pipeline import PipelineStep
from pyofdr.fiber.attenuation import round_trip_attenuation, round_trip_attenuation_varying
from pyofdr.fiber.bends import bend_loss_dB
from pyofdr.fiber.crosstalk import apply_crosstalk
from pyofdr.fiber.multiple_scattering import add_ghost_reflections
from pyofdr.fiber.reflectors import apply_connector_losses, inject_reflectors
from pyofdr.utils.constants import C
from pyofdr.utils.seeding import derive_seed
from pyofdr.utils.units import dB_to_linear, wavelength_range_to_freq_range


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
        self.rayleigh_segments = fiber.get("rayleigh_segments", [])
        self.attenuation_dB_km = fiber["attenuation_dB_per_km"]
        self.attenuation_segments = fiber.get("attenuation_segments", [])
        self.bends = fiber.get("bends", [])
        self.reflectors = fiber.get("reflectors", [])
        self.index_segments = fiber.get("index_segments", [])
        self.index_fluctuations = fiber.get("index_fluctuations", None)
        self.crosstalk = fiber.get("crosstalk", None)
        self.multiple_scattering = fiber.get("multiple_scattering", None)
        self.seed = self.config["simulation"]["seed"]
        self.center_wl = source["center_wavelength"]

        # we need the sweep range to compute dz
        self.sweep_range_hz = wavelength_range_to_freq_range(
            source["center_wavelength"], source["sweep_range"]
        )

        self._cached_z = None
        self._cached_dz = None
        self._cached_profile = None
        self._cached_atten = None

    def process(self, acq: Acquisition) -> Acquisition:
        if self._cached_profile is not None:
            # re-attach cached fiber to a fresh Acquisition
            acq.z = self._cached_z
            acq.dz = self._cached_dz
            acq.fiber_profile = self._cached_profile
            acq.attenuation_envelope = self._cached_atten
            return acq

        xp = self.bk.xp

        # Spatial resolution: dz = c / (2 * n * delta_nu)
        dz = C / (2.0 * self.n_core * self.sweep_range_hz)
        n_z = int(math.ceil(self.length / dz))
        z = xp.arange(n_z) * dz

        # Rayleigh backscatter coefficient (power per meter). When
        # segments are given we build a per-z R(z) array; the overall
        # sigma then also varies along the fiber. Backward compat:
        # no segments -> scalar, identical to the old path.
        if self.rayleigh_segments:
            R_z = xp.full(n_z, dB_to_linear(self.rayleigh_dB))
            for seg in self.rayleigh_segments:
                mask = (z >= seg["start"]) & (z < seg["end"])
                R_z = xp.where(mask,
                               dB_to_linear(seg["rayleigh_coefficient_dB"]),
                               R_z)
            sigma = xp.sqrt(R_z * dz)
        else:
            sigma = math.sqrt(dB_to_linear(self.rayleigh_dB) * dz)

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

        # discrete reflectors (connectors, splices, etc)
        if self.reflectors:
            inject_reflectors(profile, z, dz, self.reflectors, xp=xp)

        # cascading multi-bounce ghosts between those reflectors (#35).
        # added before the per-bin index phase so each ghost picks up
        # the same phi_n at its apparent bin -- a small-perturbation
        # approximation (ghost path is longer than 2*z_app, so strictly
        # the phase is not phi_n[bin_app]; ok while delta_n stays small).
        if self.multiple_scattering is not None and self.reflectors:
            add_ghost_reflections(
                profile, dz, self.reflectors,
                max_order=self.multiple_scattering["max_order"],
                xp=xp,
            )

        # small-signal n(z) perturbation -- each scatterer at z picks up
        # a round-trip phase from the integrated delta_n up to z. Two
        # contributions: deterministic segments (#68) plus stochastic OU
        # fluctuations (partial #33). Valid when |delta_n/n_core| is
        # small enough that the dz grid doesn't need to move (full OPL
        # treatment is the rest of #33).
        delta_n = None
        if self.index_segments:
            delta_n = xp.zeros(n_z)
            for seg in self.index_segments:
                mask = (z >= seg["start"]) & (z < seg["end"])
                delta_n = xp.where(mask, delta_n + seg["delta_n"], delta_n)

        if (self.index_fluctuations is not None
                and self.index_fluctuations["sigma"] > 0):
            sigma  = self.index_fluctuations["sigma"]
            L_corr = self.index_fluctuations["correlation_length"]
            # AR(1) form of OU: dn[k] = a*dn[k-1] + sigma*sqrt(1-a^2)*eta[k],
            # init from the stationary distribution -> dn[0] = sigma*eta[0].
            a = math.exp(-dz / L_corr)
            rng_n = self.bk.random_generator(
                derive_seed(self.seed, component="index_fluctuations"))
            eta = rng_n.standard_normal(n_z)
            dn_rand = xp.empty(n_z)
            dn_rand[0] = sigma * eta[0]
            scale = sigma * math.sqrt(1.0 - a * a)
            # python loop -- n_z up to ~1e5 in typical OFDR runs, ~50 ms.
            # vectorized AR(1) via lfilter would need stationary-init zi math.
            for k in range(1, n_z):
                dn_rand[k] = a * dn_rand[k-1] + scale * eta[k]
            delta_n = dn_rand if delta_n is None else delta_n + dn_rand

        if delta_n is not None:
            k0 = 2.0 * math.pi / self.center_wl
            phi_n = 2.0 * k0 * xp.cumsum(delta_n) * dz
            profile = profile * xp.exp(1j * phi_n)

        # MCF core-to-core crosstalk. Applied after all per-core phase
        # modifications so whatever each core carries gets mixed.
        if self.crosstalk is not None and self.n_cores > 1:
            rng_xt = self.bk.random_generator(
                derive_seed(self.seed, component="crosstalk"))
            profile = apply_crosstalk(
                profile, z,
                self.crosstalk["xt_dB_per_km"],
                self.crosstalk["topology"],
                rng_xt, xp=xp,
            )

        # round-trip attenuation envelope. With per-segment overrides
        # or bends we build an alpha(z) vector and integrate cumulatively.
        if self.attenuation_segments or self.bends:
            alpha_z = xp.full(n_z, float(self.attenuation_dB_km))
            for seg in self.attenuation_segments:
                mask = (z >= seg["start"]) & (z < seg["end"])
                alpha_z = xp.where(mask,
                                   seg["attenuation_dB_per_km"],
                                   alpha_z)
            # bends add excess dB/km on top of the base alpha across
            # their section. Total loss for the bend is spread
            # uniformly over [start, end).
            for bend in self.bends:
                seg_len_m = bend["end"] - bend["start"]
                if seg_len_m <= 0:
                    continue
                total_dB = bend_loss_dB(bend["radius"], bend["turns"],
                                         bend["A_dB_per_turn"], bend["R_c"])
                extra_dB_km = total_dB / (seg_len_m / 1000.0)
                mask = (z >= bend["start"]) & (z < bend["end"])
                alpha_z = xp.where(mask, alpha_z + extra_dB_km, alpha_z)
            attenuation = round_trip_attenuation_varying(z, alpha_z, dz, xp=xp)
        else:
            attenuation = round_trip_attenuation(z, self.attenuation_dB_km, xp=xp)

        # connector insertion losses (step-downs in the envelope)
        if self.reflectors:
            apply_connector_losses(attenuation, dz, self.reflectors, xp=xp)

        acq.z = z
        acq.dz = dz
        acq.fiber_profile = profile
        acq.attenuation_envelope = attenuation

        # cache so subsequent sweeps reuse the same fiber
        self._cached_z = z
        self._cached_dz = dz
        self._cached_profile = profile
        self._cached_atten = attenuation

        acq.add_log("fiber", n_z=n_z, dz_mm=dz * 1e3,
                     attenuation_dB_km=self.attenuation_dB_km,
                     n_cores=self.n_cores)
        return acq
