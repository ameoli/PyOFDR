"""Auxiliary Mach-Zehnder interferometer (k-clock).

A short MZI with a known delay tau_aux, fed by the same swept laser as the
main interferometer. Its beat is

    I_aux(t)  ~  cos( phi(t) - phi(t - tau) )
              =  cos( 2*pi * integral_{t-tau}^{t} nu(u) du )

For a perfectly linear sweep this is a clean sinusoid at gamma*tau. When the
sweep wobbles, the aux signal wobbles with it, so its unwrapped Hilbert phase
is a monotonic function of the instantaneous optical frequency. That phase is
used downstream (see analysis.demodulation.kclock_resample) as a clock to
resample the main beat onto a uniform-nu grid, canceling the sweep
nonlinearity before the FFT.

This step produces only the clean aux interferogram. Real aux detector noise /
ADC quantisation on the aux channel is deferred -- in practice the aux is
digitised at the same fs as the main and the noise on the k-clock clock
budgets separately from the main noise.
"""

from __future__ import annotations

import math
from typing import Any

from pyofdr.core.acquisition import Acquisition
from pyofdr.core.pipeline import PipelineStep


class AuxMZI(PipelineStep):

    name = "aux_mzi"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        aux = self.config["optics"]["aux_mzi"]
        self.enabled = aux["enabled"]
        self.delay = aux["delay"]

    def process(self, acq: Acquisition) -> Acquisition:
        if not self.enabled:
            return acq

        if acq.nu_inst is None:
            raise RuntimeError("AuxMZI: nu_inst not set (run SweptLaser first)")

        xp = self.bk.xp
        dt = acq.dt
        n = acq.n_samples

        n_tau = int(round(self.delay / dt))
        if n_tau < 2:
            raise ValueError(
                f"aux MZI delay ({self.delay*1e9:.2f} ns) is too short "
                f"at fs={1.0/dt*1e-6:.1f} MHz (n_tau={n_tau})"
            )
        if n_tau >= n:
            raise ValueError(
                f"aux MZI delay ({self.delay*1e9:.2f} ns) exceeds sweep duration"
            )

        # integrated frequency (= phi/2pi): rectangle rule, same as SweptLaser
        integ = xp.cumsum(acq.nu_inst) * dt

        # phi(t) - phi(t-tau) via integer sample lag
        phi_diff = xp.empty(n, dtype=xp.float64)
        phi_diff[n_tau:] = 2.0 * math.pi * (integ[n_tau:] - integ[:-n_tau])
        # first n_tau samples: hold the first valid value so the aux array is
        # the same shape as the main beat. Consumers must trim by aux_valid_start.
        phi_diff[:n_tau] = phi_diff[n_tau]

        acq.aux_signal = xp.cos(phi_diff)
        acq.aux_valid_start = n_tau

        # mean aux beat frequency = <dnu/dt> * tau  ~  gamma*tau for a linear sweep
        mean_beat = float((phi_diff[-1] - phi_diff[n_tau])
                          / (2.0 * math.pi * (n - 1 - n_tau) * dt))
        acq.add_log("aux_mzi", delay=self.delay, n_tau=n_tau,
                     mean_beat_Hz=mean_beat)
        return acq
