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
import warnings
from typing import Any

import numpy as np

from pyofdr.core.acquisition import Acquisition
from pyofdr.core.pipeline import PipelineStep
from pyofdr.fiber.fbg import weak_fbg_signal
from pyofdr.optics.components import Circulator
from pyofdr.utils.constants import C
from pyofdr.utils.units import wavelength_range_to_freq_range


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
        # group-velocity dispersion. beta2 = -lambda^2 D / (2 pi c).
        # both signs valid; only |beta2|>0 triggers the warp.
        self.center_wl = src["center_wavelength"]
        self.sweep_duration = src["sweep_duration"]
        self.D = self.config["fiber"].get("dispersion_D", 0.0)
        self.beta2 = (
            -self.center_wl**2 * self.D / (2.0 * math.pi * C)
            if self.D != 0 else 0.0
        )
        self._has_dispersion = (self.beta2 != 0.0)
        self._has_nonlinearity = (
            self.nl_a2 != 0 or self.nl_a3 != 0
            or (self.ripple_amp > 0 and self.ripple_period > 0)
            or self._has_dispersion   # GVD reuses the time-warp engine
        )
        self.linewidth = src["linewidth"]
        self.n_core = self.config["fiber"]["n_core"]
        self.fbg_arrays = self.config["fiber"].get("fbg_arrays", [])

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

        # circulator return-loss: reflection at the port-2 face, zero
        # round-trip delay -> a discrete reflector sitting in the z=0 bin.
        # picks up IL^2 naturally via the prefactor further down.
        rl = self.circulator.return_loss
        if rl > 0:
            if weighted is acq.fiber_profile:
                weighted = weighted.copy()
            weighted[:, 0] = weighted[:, 0] + rl

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

        # weak FBG arrays. added after the time-warp so the sinc envelope
        # uses the actual nu_inst; the phase term doesn't get warped, but
        # the residual delta_nu/gamma error is small in practice.
        if self.fbg_arrays:
            fbg_real = weak_fbg_signal(
                self.fbg_arrays, acq.dz, n_z,
                acq.attenuation_envelope, acq.nu_inst,
                self.n_core, xp=xp,
            )
            beat = beat + fbg_real[None, :]

        # circulator: signal passes through twice (to fiber and back)
        IL2 = self.circulator.round_trip_transmission

        # source power may vary sample-to-sample (edge droop, RIN). the beat
        # amplitude is sqrt(P(t) * P(t-tau)) which for slow envelopes and short
        # round-trip delays collapses to P(t). Using the mean here would throw
        # away both the envelope and RIN before they reach the detector.
        P_t = xp.abs(acq.E_source) ** 2
        prefactor = 2.0 * math.sqrt(eta * (1.0 - eta)) * IL2

        acq.photocurrent_main = prefactor * P_t * xp.real(beat)

        # circulator port-1 -> port-3 leakage: light bypasses the fiber
        # (and the round-trip IL) entirely and lands at zero delay -> a
        # DC offset on the beat, riding the source power envelope (so it
        # also carries RIN / edge droop).
        iso = self.circulator.isolation
        if iso > 0:
            iso_amp = 2.0 * math.sqrt(eta * (1.0 - eta)) * iso
            acq.photocurrent_main = acq.photocurrent_main + iso_amp * P_t

        acq.add_log("optics", topology="mach_zehnder",
                     scale=float(prefactor * xp.mean(P_t)),
                     circulator_IL_dB=self.circulator.insertion_loss_dB,
                     circulator_iso_dB=self.circulator.isolation_dB,
                     circulator_RL_dB=self.circulator.return_loss_dB,
                     sweep_nonlinearity=self._has_nonlinearity,
                     coherence_rolloff=self.linewidth > 0,
                     n_fbgs=len(self.fbg_arrays))
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

        # GVD adds 4*pi^2 * beta2 * gamma^2 * z * (t-T/2)^2 to the beat
        # phase. factoring out z gives phi(z,t) = z*K(t), and the warp
        # engine just needs an extra delta_nu = pi*beta2*gamma^2*c/n*(t-T/2)^2
        # (so that gamma * s = K when s = t + delta_nu/gamma). z-independent
        # because the broadening shows up in the FFT itself.
        if self._has_dispersion:
            tc = t - T / 2.0
            delta_nu = delta_nu + (
                math.pi * self.beta2 * gamma**2 * C / self.n_core * tc**2
            )

        # warped time: where in the "ideal" timeline each real sample falls
        s = t + delta_nu / gamma

        # np.interp just clamps past [t[0], t[-1]]; large nonlinearity pushes
        # s outside and the edges end up zero-order-held. warn if that
        # excursion is a meaningful slice of the sweep.
        t_ideal = t
        T = n * dt
        excess = max(float(xp.max(s)) - float(t_ideal[-1]), -float(xp.min(s)))
        if excess > 0.05 * T:
            warnings.warn(
                f"MZI time-warp excursion {100.0 * excess / T:.1f}% of "
                f"sweep, beat will extrapolate at the edges",
                stacklevel=2,
            )

        # resample each core via linear interpolation
        n_c = beat.shape[0]
        warped = xp.empty_like(beat)
        for c in range(n_c):
            warped[c] = xp.interp(s, t_ideal, xp.real(beat[c]))

        return warped
