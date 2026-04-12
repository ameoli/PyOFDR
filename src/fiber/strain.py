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

TODO: let the user pass a time-varying eps(t) per segment for
multi-sweep scenarios.
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
        strain = self.config["strain"]

        self.segments = list(strain["segments"])
        self.p_e = strain["photoelastic_coefficient"]
        self.center_wl = self.config["source"]["center_wavelength"]
        self.n_core = self.config["fiber"]["n_core"]

        kind = strain["transfer"]
        if kind == "ideal":
            self.transfer = IdealTransfer()
        elif kind == "cox":
            self.transfer = CoxShearLag(beta=strain["cox"]["beta"])
        else:
            raise ValueError(f"unknown strain.transfer: {kind!r}")

        self._cached_profile = None
        self._cached_strain = None

    def process(self, acq: Acquisition) -> Acquisition:
        if not self.segments:
            return acq
        if self._cached_profile is not None:
            acq.fiber_profile = self._cached_profile
            acq.strain_field = self._cached_strain
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

        acq.strain_field = eps_fiber

        # uniform axial strain affects every core the same way
        acq.fiber_profile = acq.fiber_profile * xp.exp(1j * phase)
        self._cached_profile = acq.fiber_profile
        self._cached_strain = eps_fiber

        acq.add_log("strain",
                     n_segments=len(self.segments),
                     transfer=type(self.transfer).__name__,
                     max_eps=float(xp.max(xp.abs(eps_fiber))),
                     p_e=self.p_e)
        return acq
