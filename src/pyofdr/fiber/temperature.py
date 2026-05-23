"""Distributed temperature perturbation on the Rayleigh profile.

Each scatterer at position z picks up a round-trip phase from the
integrated temperature change up to z:

    d phi(z) = 2 k0 n (alpha_L + xi) integral_0^z dT(z') dz'

with alpha_L the thermal expansion of the fiber and xi = (1/n) dn/dT
the thermo-optic coefficient. This is the temperature half of the
Froggatt-Moore Rayleigh-shift relation -- combined with StrainPerturbation
it produces the strain-temperature cross-sensitivity (#75).

Segments may carry an optional `motion` field (harmonic / thermal /
impulsive). The same registry in strain_transfer.motions evaluates it,
just interpreted as dT(t) instead of eps(t).

No host->fiber transfer model here -- the user's dT(z) is applied directly
to the fiber. Realistic thermal diffusion / time-space coupling is the
job of #76 (V2).

Implementation note: we cache the *phase array*, not the output profile,
so this step composes correctly with any dynamic upstream step. The
strain step caches its output and gets away with it because nothing
upstream is currently dynamic; if that changes, the same trick should
move there too.
"""

from __future__ import annotations

import math
from typing import Any

from pyofdr.core.acquisition import Acquisition
from pyofdr.core.config import sweep_period
from pyofdr.core.pipeline import PipelineStep
from pyofdr.strain_transfer.motions import check_motion_sampling, evaluate_motion


class TemperaturePerturbation(PipelineStep):

    name = "temperature"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        temp = self.config["temperature"]

        self.segments = list(temp["segments"])
        self.alpha_L = temp["thermal_expansion"]
        self.xi = temp["thermo_optic"]
        self.center_wl = self.config["source"]["center_wavelength"]
        self.n_core = self.config["fiber"]["n_core"]
        self.sweep_period = sweep_period(self.config)

        self._has_motion = any(s.get("motion") is not None for s in self.segments)
        self._cached_phase = None
        self._cached_dT = None

        if self._has_motion:
            for i, seg in enumerate(self.segments):
                check_motion_sampling(seg.get("motion"), self.sweep_period, i)

    def process(self, acq: Acquisition) -> Acquisition:
        if not self.segments:
            return acq

        if acq.fiber_profile is None or acq.z is None:
            raise RuntimeError("TemperaturePerturbation: fiber_profile/z not set")

        xp = self.bk.xp

        # static fast path: reuse cached phase but multiply the *current*
        # incoming profile so a dynamic upstream step still composes
        if not self._has_motion and self._cached_phase is not None:
            acq.fiber_profile = acq.fiber_profile * xp.exp(1j * self._cached_phase)
            acq.temperature_field = self._cached_dT
            return acq

        z = acq.z
        dz = acq.dz
        t_sweep = acq.sweep_index * self.sweep_period

        # piecewise-constant dT(z) at this sweep's lab time
        dT = xp.zeros_like(z)
        for seg in self.segments:
            value = seg["delta_T"] + evaluate_motion(seg.get("motion"), t_sweep)
            mask = (z >= seg["start"]) & (z <= seg["end"])
            dT = xp.where(mask, dT + value, dT)

        k0 = 2.0 * math.pi / self.center_wl
        prefactor = 2.0 * k0 * self.n_core * (self.alpha_L + self.xi)
        # left Riemann sum, same as the strain step
        phase = prefactor * xp.cumsum(dT) * dz

        acq.temperature_field = dT
        acq.fiber_profile = acq.fiber_profile * xp.exp(1j * phase)

        if not self._has_motion:
            self._cached_phase = phase
            self._cached_dT = dT

        acq.add_log("temperature",
                     n_segments=len(self.segments),
                     max_dT=float(xp.max(xp.abs(dT))),
                     dynamic=self._has_motion,
                     t_sweep=t_sweep,
                     alpha_L=self.alpha_L, xi=self.xi)
        return acq
