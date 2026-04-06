"""Mach-Zehnder interferomter

For a perfectly linear sweep, the beat signal is just the IFFT of the fiber profile. This is the key insight that makes the simulation O(N*log(N)) instead of O(N^2).
Each spatial bin k corresponds to beat frequency f_k, so the IFFT directly gives us the time-domain beat signal.
For now no auxiliary interferometer (k-clock) -- that comes later. Also no circulator model, no phase noise on the beat.

TODO: add auxiliary MZI for k-linearization
add circulator insertion loss / isolation
add differential phase noise at each delay
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.acquisition import Acquisition
from core.pipeline import PipelineStep


class MachZehnder(PipelineStep):

    name = "optics"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        optics = config.get("optics", {})
        self.splitting_ratio = optics.get("splitting_ratio", 0.5)

    def process(self, acq: Acquisition) -> Acquisition:
        if acq.fiber_profile is None:
            raise RuntimeError("Optics: fiber_profile not set")
        if acq.E_source is None:
            raise RuntimeError("Optics: E_source not set")

        n_z = len(acq.fiber_profile)
        eta = self.splitting_ratio


        h = np.zeros(acq.n_samples, dtype=np.complex128)
        h[:n_z] = acq.fiber_profile
        beat = np.fft.ifft(h) * acq.n_samples

        P_avg = float(np.mean(np.abs(acq.E_source) ** 2))
        scale = 2.0 * np.sqrt(eta * (1.0 - eta)) * P_avg

        acq.photocurrent_main = scale * np.real(beat)

        acq.add_log("optics", topology="mach_zehnder", scale=float(scale))
        return acq
