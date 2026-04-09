"""Mach-Zehnder interferomter

For a perfectly linear sweep, the beat signal is just the IFFT of the fiber profile. This is the key insight that makes the simulation O(N*log(N)) instead of O(N^2).
Each spatial bin k corresponds to beat frequency f_k, so the IFFT directly gives us the time-domain beat signal.
For now no auxiliary interferometer (k-clock) -- that comes later. Also no circulator model, no phase noise on the beat.

TODO: add auxiliary MZI for k-linearization
add circulator insertion loss / isolation
add differential phase noise at each delay
"""

from __future__ import annotations

import math
from typing import Any

from core.acquisition import Acquisition
from core.pipeline import PipelineStep


class MachZehnder(PipelineStep):

    name = "optics"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.splitting_ratio = self.config["optics"]["splitting_ratio"]

    def process(self, acq: Acquisition) -> Acquisition:
        if acq.fiber_profile is None:
            raise RuntimeError("Optics: fiber_profile not set")
        if acq.E_source is None:
            raise RuntimeError("Optics: E_source not set")

        xp = self.bk.xp
        n_c, n_z = acq.fiber_profile.shape
        eta = self.splitting_ratio

        # weighted profile: Rayleigh phasors * attenuation envelope
        # broadcast (n_c, n_z) * (n_z,) -> (n_c, n_z)
        weighted = acq.fiber_profile
        if acq.attenuation_envelope is not None:
            weighted = weighted * acq.attenuation_envelope

        # zero-pad up to n_samples along the time axis
        n_pad = acq.n_samples - n_z
        h = xp.concatenate([
            weighted.astype(xp.complex128),
            xp.zeros((n_c, n_pad), dtype=xp.complex128),
        ], axis=-1)
        beat = self.bk.fft.ifft(h, axis=-1) * acq.n_samples

        P_avg = float(xp.mean(xp.abs(acq.E_source) ** 2))
        scale = 2.0 * math.sqrt(eta * (1.0 - eta)) * P_avg

        acq.photocurrent_main = scale * xp.real(beat)

        acq.add_log("optics", topology="mach_zehnder", scale=float(scale))
        return acq
