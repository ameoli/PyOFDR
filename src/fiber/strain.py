"""Distributed strain perturbation on the Rayleigh profile.

Each scatterer at position z picks up an extra round-trip phase
proportional to the integrated strain up to z:

    d phi(z) = 2 k0 n (1 - p_e) integral_0^z eps(z') dz'

p_e is the strain-optic (photoelastic) coefficient, ~0.22 for silica
at 1550 nm. The (1 - p_e) factor folds in the index change due to the
strain itself, otherwise we'd be double-counting the geometric stretch.

For now: uniform (in time) strain on a list of segments. Harmonic /
propagating / thermal-transient perturbations come later, see the
roadmap. Host->fiber transfer is delegated to strain_transfer; default
is ideal, cox shear-lag is available via config.

TODO: when multi-sweep lands (#4), drop the _done cache and let the
user pass a time-varying eps(t) per segment.
"""

from __future__ import annotations

import math
from typing import Any

from core.acquisition import Acquisition
from core.pipeline import PipelineStep
from strain_transfer import CoxShearLag, IdealTransfer


class StrainPerturbation(PipelineStep):

    name = "strain"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        strain = config.get("strain", {}) or {}
        source = config.get("source", {})
        fiber  = config.get("fiber", {})

        self.segments = list(strain.get("segments", []))
        self.p_e = strain.get("photoelastic_coefficient", 0.22)
        self.center_wl = source.get("center_wavelength", 1550e-9)
        self.n_core = fiber.get("n_core", 1.4682)

        kind = strain.get("transfer", "ideal")
        if kind == "ideal":
            self.transfer = IdealTransfer()
        elif kind == "cox":
            cox = strain.get("cox") or {}
            beta = cox.get("beta")
            if beta is None:
                # pydantic catches this too, but we may be called from a
                # raw dict (tests do that), so guard here as well
                raise ValueError("strain.transfer=cox requires strain.cox.beta")
            self.transfer = CoxShearLag(beta=beta)
        else:
            raise ValueError(f"unknown strain.transfer: {kind!r}")

        # mirror FiberGenerator: don't re-apply on subsequent sweeps,
        # otherwise the phase shift would accumulate every call
        self._done = False

    def process(self, acq: Acquisition) -> Acquisition:
        if self._done or not self.segments:
            return acq
        if acq.fiber_profile is None or acq.z is None:
            raise RuntimeError("StrainPerturbation: fiber_profile/z not set")

        xp = self.bk.xp
        z = acq.z
        dz = acq.dz

        eps_fiber = self.transfer.apply(self.segments, z, xp=xp)

        k0 = 2.0 * math.pi / self.center_wl
        prefactor = 2.0 * k0 * self.n_core * (1.0 - self.p_e)
        # plain left Riemann sum -- dz is the spatial bin so it's
        # accurate to within ~half a bin past the segment edge
        phase = prefactor * xp.cumsum(eps_fiber) * dz

        # uniform axial strain affects every core the same way
        acq.fiber_profile = acq.fiber_profile * xp.exp(1j * phase)

        acq.add_log("strain",
                     n_segments=len(self.segments),
                     transfer=type(self.transfer).__name__,
                     max_eps=float(xp.max(xp.abs(eps_fiber))),
                     p_e=self.p_e)
        self._done = True
        return acq
