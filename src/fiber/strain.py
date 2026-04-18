"""Distributed strain perturbation on the Rayleigh profile.

Each scatterer at position z picks up an extra round-trip phase
proportional to the integrated strain up to z:

    d phi(z) = 2 k0 n (1 - p_e) integral_0^z eps(z') dz'

p_e is the strain-optic (photoelastic) coefficient, ~0.22 for silica
at 1550 nm. The (1 - p_e) factor folds in the index change due to the
strain itself, otherwise we'd be double-counting the geometric stretch.

Segments may carry an optional `motion` field (harmonic for now) that
varies the strain across sweeps at the sweep rate (t_sweep =
sweep_index * sweep_duration). Static segments are cached as before;
segments with motion trigger recomputation every sweep.

Host->fiber transfer is delegated to strain_transfer; default is
ideal, cox shear-lag is available via config.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

from core.acquisition import Acquisition
from core.pipeline import PipelineStep
from strain_transfer import CoxShearLag, IdealTransfer, realize_segments


class StrainPerturbation(PipelineStep):

    name = "strain"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        strain = self.config["strain"]

        self.segments = list(strain["segments"])
        self.p_e = strain["photoelastic_coefficient"]
        self.center_wl = self.config["source"]["center_wavelength"]
        self.n_core = self.config["fiber"]["n_core"]
        self.sweep_duration = self.config["source"]["sweep_duration"]

        kind = strain["transfer"]
        if kind == "ideal":
            self.transfer = IdealTransfer()
        elif kind == "cox":
            self.transfer = CoxShearLag(beta=strain["cox"]["beta"])
        else:
            raise ValueError(f"unknown strain.transfer: {kind!r}")

        self._has_motion = any(s.get("motion") is not None for s in self.segments)
        self._cached_profile = None
        self._cached_strain = None
        self._cached_base_profile = None    # base (unstrained) profile reference

        # cheap sanity check: the inter-sweep sample rate is 1/sweep_duration,
        # so motions with f >= 1/(2*T_sweep) alias. at f == Nyquist with
        # phase=0 you can even land on zero-crossings every sweep, which is an
        # easy trap. warn; don't refuse (user may want to probe that regime).
        # For thermal, the analogous check is tau >> T_sweep -- a transient
        # with tau < 2*T_sweep is basically a step by the second sample.
        if self._has_motion:
            f_nyq = 0.5 / self.sweep_duration
            for i, seg in enumerate(self.segments):
                motion = seg.get("motion")
                if motion is None:
                    continue
                kind = motion["kind"]
                if kind == "harmonic":
                    f = motion.get("frequency", 0.0)
                    if f >= f_nyq:
                        warnings.warn(
                            f"strain segment {i}: motion frequency {f:g} Hz >= "
                            f"sweep-rate Nyquist {f_nyq:g} Hz; expect aliasing",
                            stacklevel=2,
                        )
                elif kind == "thermal":
                    tau = motion.get("tau", 0.0)
                    if 0 < tau < 2.0 * self.sweep_duration:
                        warnings.warn(
                            f"strain segment {i}: thermal tau {tau:g} s < "
                            f"2*sweep_duration ({2*self.sweep_duration:g} s); "
                            f"transient under-sampled across sweeps",
                            stacklevel=2,
                        )

    def process(self, acq: Acquisition) -> Acquisition:
        if not self.segments:
            return acq

        # static case: cache strained profile, reuse it across sweeps
        if not self._has_motion and self._cached_profile is not None:
            acq.fiber_profile = self._cached_profile
            acq.strain_field = self._cached_strain
            return acq

        if acq.fiber_profile is None or acq.z is None:
            raise RuntimeError("StrainPerturbation: fiber_profile/z not set")

        xp = self.bk.xp
        z = acq.z
        dz = acq.dz

        # remember the base (unstrained) profile so dynamic sweeps keep applying
        # the phase to the fresh profile, not to an already-strained one.
        if self._cached_base_profile is None:
            self._cached_base_profile = acq.fiber_profile
        base_profile = self._cached_base_profile

        # freeze motion at the sweep's lab time
        t_sweep = acq.sweep_index * self.sweep_duration
        current_segments = realize_segments(self.segments, t_sweep)

        eps_fiber = self.transfer.apply(current_segments, z, xp=xp)

        k0 = 2.0 * math.pi / self.center_wl
        prefactor = 2.0 * k0 * self.n_core * (1.0 - self.p_e)
        # plain left Riemann sum -- dz is the spatial bin so it's
        # accurate to within ~half a bin past the segment edge
        phase = prefactor * xp.cumsum(eps_fiber) * dz

        acq.strain_field = eps_fiber
        # uniform axial strain affects every core the same way
        acq.fiber_profile = base_profile * xp.exp(1j * phase)

        if not self._has_motion:
            self._cached_profile = acq.fiber_profile
            self._cached_strain = eps_fiber

        acq.add_log("strain",
                     n_segments=len(self.segments),
                     transfer=type(self.transfer).__name__,
                     max_eps=float(xp.max(xp.abs(eps_fiber))),
                     dynamic=self._has_motion,
                     t_sweep=t_sweep,
                     p_e=self.p_e)
        return acq
