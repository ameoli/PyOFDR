"""
Physics:
    nu(t) = nu_start + gamma * t + a2*t^2 + a3*t^3 + ripple(t)
    phi(t) = 2pi * integral(nu) dt + phi_noise(t)
    E(t) = sqrt(P) * exp(j*phi(t))

phi_noise combines three frequency-noise contributions:
  - white FM   (Lorentzian linewidth) -> d_phi ~ N(0, sqrt(2*pi*lw*dt))
  - flicker FM (1/f)                  -> powerlaw_psd_gaussian(beta=1)
  - random-walk FM (1/f^2)            -> powerlaw_psd_gaussian(beta=2)
all generated in source/phase_noise.py.

Sweep nonlinearity (a2, a3, ripple) is opt-in: zero coefficients give
the original linear chirp. The nonlinearity affects nu_inst and phi,
and the MZI time-warps the beat signal accordingly (see mach_zehnder.py).
"""

from __future__ import annotations

import math
from typing import Any

from pyofdr.core.acquisition import Acquisition
from pyofdr.core.pipeline import PipelineStep
from pyofdr.source.phase_noise import colored_frequency_noise
from pyofdr.utils.constants import C
from pyofdr.utils.seeding import derive_seed
from pyofdr.utils.units import wavelength_range_to_freq_range


class SweptLaser(PipelineStep):

    name = "source"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        source = self.config["source"]

        self.center_wl = source["center_wavelength"]
        self.sweep_range_wl = source["sweep_range"]
        self.sweep_duration = source["sweep_duration"]
        self.power = source["power"]
        self.linewidth = source["linewidth"]
        self.sigma_flicker = source["flicker_noise_Hz"]
        self.sigma_rw      = source["random_walk_noise_Hz"]
        self.rin_dB_per_Hz = source["rin_dB_per_Hz"]   # None or float
        self.envelope_edge_dB = source["power_envelope_edge_dB"]
        self.nl_a2 = source["sweep_nonlinearity_a2"]
        self.nl_a3 = source["sweep_nonlinearity_a3"]
        self.ripple_amp = source["sweep_ripple_amplitude"]
        self.ripple_period = source["sweep_ripple_period"]
        self.sample_rate = self.config["adc"]["sample_rate"]
        self.seed = self.config["simulation"]["seed"]

        # derived quantities
        self.nu_center = C / self.center_wl
        self.sweep_range_hz = wavelength_range_to_freq_range(
            self.center_wl, self.sweep_range_wl
        )
        self.gamma = self.sweep_range_hz / self.sweep_duration   # Hz/s

    def process(self, acq: Acquisition) -> Acquisition:
        xp = self.bk.xp
        dt = 1.0 / self.sample_rate
        n_samples = int(math.ceil(self.sweep_duration * self.sample_rate))
        t = xp.arange(n_samples) * dt

        # instantanous frequency: linear ramp + optional nonlinearity
        nu_start = self.nu_center - self.sweep_range_hz / 2.0
        nu_inst = nu_start + self.gamma * t

        if self.nl_a2 != 0 or self.nl_a3 != 0:
            nu_inst = nu_inst + self.nl_a2 * t**2 + self.nl_a3 * t**3
        if self.ripple_amp > 0 and self.ripple_period > 0:
            nu_inst = nu_inst + self.ripple_amp * xp.sin(
                2.0 * math.pi * t / self.ripple_period)

        # phase = 2pi * cumulative sum (rectangle rule integration)
        phi = 2.0 * math.pi * xp.cumsum(nu_inst) * dt

        # phase noise: white (Lorentzian) + flicker (1/f) + random walk (1/f^2)
        if self.linewidth > 0 or self.sigma_flicker > 0 or self.sigma_rw > 0:
            rng = self.bk.random_generator(
                derive_seed(self.seed, component="laser", sweep=acq.sweep_index)
            )
            phi = phi + colored_frequency_noise(
                n_samples, dt,
                linewidth=self.linewidth,
                sigma_flicker=self.sigma_flicker,
                sigma_rw=self.sigma_rw,
                rng=rng, xp=xp,
            )

        # power envelope: real tunable lasers droop at the edges of the
        # sweep. Parabolic model -- P(center)=P0, P(edges)=P0*10^(-edge/10).
        # edge=0 keeps the old flat behaviour.
        if self.envelope_edge_dB > 0:
            edge_lin = 1.0 - 10.0 ** (-self.envelope_edge_dB / 10.0)
            u = t / self.sweep_duration - 0.5        # -0.5 .. +0.5
            P_t = self.power * (1.0 - 4.0 * edge_lin * u * u)
        else:
            P_t = self.power  # scalar, broadcasts fine

        # RIN -- relative intensity noise.  Modelled as white multiplicative
        # noise on optical power: P(t) = P0*(1 + n(t)),  n ~ N(0, sigma_rin),
        # sigma_rin = sqrt(RIN_linear * BW).
        if self.rin_dB_per_Hz is not None:
            rin_linear = 10.0 ** (self.rin_dB_per_Hz / 10.0)
            bw  = self.sample_rate / 2.0
            sigma_rin = math.sqrt(rin_linear * bw)

            rng_rin = self.bk.random_generator(
                derive_seed(self.seed, component="laser",
                            sweep=acq.sweep_index, sub=1))
            n_rin = sigma_rin  * rng_rin.standard_normal(n_samples)
            P_noisy = P_t * (1.0 + n_rin)
            P_noisy = xp.maximum(P_noisy, 0.0)   # shouldn't happen for realistic RIN values
            E = xp.sqrt(P_noisy) * xp.exp(1j * phi)
        else:
            E = xp.sqrt(P_t) * xp.exp(1j * phi) if self.envelope_edge_dB > 0 \
                else math.sqrt(self.power) * xp.exp(1j * phi)

        acq.t = t
        acq.dt = dt
        acq.n_samples = n_samples
        acq.nu_inst = nu_inst
        acq.E_source = E

        acq.add_log("source", n_samples=n_samples, gamma=self.gamma,
                     linewidth=self.linewidth,
                     flicker_noise_Hz=self.sigma_flicker,
                     random_walk_noise_Hz=self.sigma_rw,
                     rin_dB_per_Hz=self.rin_dB_per_Hz,
                     envelope_edge_dB=self.envelope_edge_dB)
        return acq
