"""Mach-Zehnder interferomter

For a perfectly linear sweep, the beat signal is just the IFFT of the
fiber profile. This is the key insight that makes the simulation
O(N*log(N)) instead of O(N^2). Each spatial bin k corresponds to beat
frequency f_k, so the IFFT directly gives us the time-domain beat signal.

When the laser has sweep nonlinearity (a2, a3, ripple), each scatterer's
beat frequency wobbles in the same way. This is equivalent to a
time-warp of the ideal linear beat signal. The warp is computed from
the integrated frequency deviation, then applied via linear interpolation.

non-Lorentzian coherence decay (flicker / random-walk phase noise)
needs the full phase structure function; only the Lorentzian term
is modelled here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.acquisition import Acquisition
from core.pipeline import PipelineStep
from optics.components import Circulator
from utils.constants import C
from utils.units import wavelength_range_to_freq_range


class MachZehnder(PipelineStep):

    name = "optics"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.splitting_ratio = self.config["optics"]["splitting_ratio"]
        circ_cfg = self.config["optics"].get("circulator", {})
        self.circulator = Circulator(**circ_cfg)

        src = self.config["source"]
        self.nl_a2 = src["sweep_nonlinearity_a2"]
        self.nl_a3 = src["sweep_nonlinearity_a3"]
        self.ripple_amp = src["sweep_ripple_amplitude"]
        self.ripple_period = src["sweep_ripple_period"]
        self._has_nonlinearity = (
            self.nl_a2 != 0 or self.nl_a3 != 0
            or (self.ripple_amp > 0 and self.ripple_period > 0)
        )
        self.linewidth = src["linewidth"]
        self.n_core = self.config["fiber"]["n_core"]

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

        # laser coherence roll-off: for a Lorentzian-linewidth laser each
        # scatterer at round-trip delay tau = 2*n*z/c sees its beat amplitude
        # decay as exp(-pi * linewidth * tau). The IFFT trick ignores this
        # because it doesn't carry a delayed reference arm explicitly, so we
        # fold it in as a z-dependent multiplier on the profile.
        if self.linewidth > 0:
            z = xp.arange(n_z) * acq.dz
            tau = 2.0 * self.n_core * z / C
            visibility = xp.exp(-math.pi * self.linewidth * tau)
            weighted = weighted * visibility

        # zero-pad up to n_samples along the time axis
        n_pad = acq.n_samples - n_z
        h = xp.concatenate([
            weighted.astype(xp.complex128),
            xp.zeros((n_c, n_pad), dtype=xp.complex128),
        ], axis=-1)
        beat = self.bk.fft.ifft(h, axis=-1) * acq.n_samples

        # time-warp for sweep nonlinearity: if the chirp rate varies,
        # each scatterer's beat freq scales by dnu/dt / gamma. This is
        # equivalent to resampling the ideal beat at warped time indices.
        if self._has_nonlinearity:
            beat = self._apply_time_warp(beat, acq)

        # circulator: signal passes through twice (to fiber and back)
        IL2 = self.circulator.round_trip_transmission

        # source power may vary sample-to-sample (edge droop, RIN). the beat
        # amplitude is sqrt(P(t) * P(t-tau)) which for slow envelopes and short
        # round-trip delays collapses to P(t). Using the mean here would throw
        # away both the envelope and RIN before they reach the detector.
        P_t = xp.abs(acq.E_source) ** 2
        prefactor = 2.0 * math.sqrt(eta * (1.0 - eta)) * IL2

        acq.photocurrent_main = prefactor * P_t * xp.real(beat)

        acq.add_log("optics", topology="mach_zehnder",
                     scale=float(prefactor * xp.mean(P_t)),
                     circulator_IL_dB=self.circulator.insertion_loss_dB,
                     sweep_nonlinearity=self._has_nonlinearity,
                     coherence_rolloff=self.linewidth > 0)
        return acq

    def _apply_time_warp(self, beat, acq):
        """Resample the ideal-chirp beat to account for sweep nonlinearity.

        For a scatterer at round-trip delay tau the physical beat phase
        is 2*pi*tau*nu(t) (small tau), while the ideal-chirp beat at time
        s has phase 2*pi*gamma*tau*s. Setting them equal ->
            s(t) = t + delta_nu(t) / gamma
        with delta_nu the instantaneous frequency deviation from the linear
        ramp.  Same mapping for a general fiber since it's tau-independent.
        """
        xp = self.bk.xp
        src = self.config["source"]
        wl = src["center_wavelength"]
        dwl = src["sweep_range"]
        T = src["sweep_duration"]
        gamma = wavelength_range_to_freq_range(wl, dwl) / T

        dt = acq.dt
        n = acq.n_samples
        t = xp.arange(n) * dt

        # instantaneous frequency deviation from the linear ramp
        delta_nu = xp.zeros(n, dtype=xp.float64)
        if self.nl_a2 != 0 or self.nl_a3 != 0:
            delta_nu = delta_nu + self.nl_a2 * t**2 + self.nl_a3 * t**3
        if self.ripple_amp > 0 and self.ripple_period > 0:
            delta_nu = delta_nu + self.ripple_amp * xp.sin(
                2.0 * math.pi * t / self.ripple_period)

        # warped time: where in the "ideal" timeline each real sample falls
        s = t + delta_nu / gamma

        # resample each core via linear interpolation
        t_ideal = t   # the uniform grid the IFFT beat lives on
        n_c = beat.shape[0]
        warped = xp.empty_like(beat)
        for c in range(n_c):
            warped[c] = xp.interp(s, t_ideal, xp.real(beat[c]))

        return warped
