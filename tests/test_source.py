"""Tests for swept laser, phase noise, and RIN."""

import numpy as np
import pytest

from helpers import CFG
from core.acquisition import Acquisition
from fiber.profile import FiberGenerator
from source.swept_laser import SweptLaser


class TestSweptLaser:

    def _make_acq(self):
        acq = FiberGenerator(CFG).process(Acquisition())
        return SweptLaser(CFG).process(acq)

    def test_time_axis_exists(self):
        acq = self._make_acq()
        assert acq.t is not None
        assert acq.n_samples > 0

    def test_field_is_complex(self):
        acq = self._make_acq()
        assert acq.E_source.dtype == np.complex128

    def test_frequency_increases_monotonically(self):
        acq = self._make_acq()
        dnu = np.diff(acq.nu_inst)
        assert np.all(dnu > 0)

    def test_optical_power_matches_config(self):
        acq = self._make_acq()
        P = np.mean(np.abs(acq.E_source) ** 2)
        np.testing.assert_allclose(P, 10e-3, rtol=0.01)


class TestPhaseNoise:

    def _run(self, linewidth):
        cfg = {**CFG, "source": {**CFG["source"], "linewidth": linewidth}}
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        return SweptLaser(cfg).process(acq)

    def test_zero_linewidth_matches_noiseless(self):
        # backwards compat: linewidth=0 must give the old result
        a = self._run(0.0)
        b = SweptLaser(CFG).process(FiberGenerator(CFG).process(Acquisition()))
        np.testing.assert_array_equal(a.E_source, b.E_source)

    def test_nonzero_linewidth_changes_field(self):
        a = self._run(0.0)
        b = self._run(1e5)
        assert not np.array_equal(a.E_source, b.E_source)

    def test_power_unchanged_by_phase_noise(self):
        # |E|^2 must stay equal to P regardless of phi noise
        acq = self._run(1e6)
        P = np.mean(np.abs(acq.E_source) ** 2)
        np.testing.assert_allclose(P, 10e-3, rtol=0.01)

    def test_phase_increment_variance_matches_theory(self):
        # var(d phi_noise) ~ 2*pi*lw*dt. Cancel the optical carrier by
        # dividing the noisy field by the noiseless one, then unwrap.
        lw = 1e6
        a = self._run(lw)
        b = self._run(0.0)
        phi_noise = np.unwrap(np.angle(a.E_source / b.E_source))
        dphi = np.diff(phi_noise)
        expected = 2.0 * np.pi * lw * a.dt
        np.testing.assert_allclose(np.var(dphi), expected, rtol=0.05)

    def test_same_seed_same_noise(self):
        a = self._run(1e5)
        b = self._run(1e5)
        np.testing.assert_array_equal(a.E_source, b.E_source)


class TestRIN:

    def _run(self, rin_dB=None):
        cfg = {**CFG, "source": {**CFG["source"], "rin_dB_per_Hz": rin_dB}}
        acq = FiberGenerator(cfg).process(Acquisition())
        return SweptLaser(cfg).process(acq)

    def test_no_rin_constant_power(self):
        acq = self._run(None)
        P = np.abs(acq.E_source)**2
        np.testing.assert_allclose(P, 10e-3, rtol=1e-10)

    def test_rin_changes_amplitude(self):
        clean = self._run(None)
        noisy = self._run(-120.0)
        assert not np.allclose(np.abs(clean.E_source)**2,
                               np.abs(noisy.E_source)**2)

    def test_rin_mean_power_unchanged(self):
        # RIN is zero-mean, so average power ~ P0
        acq = self._run(-130.0)
        P = np.abs(acq.E_source)**2
        np.testing.assert_allclose(np.mean(P), 10e-3, rtol=0.01)

    def test_rin_variance_matches_theory(self):
        # var(P) / P0^2 should equal RIN_linear * BW
        rin_dB = -130.0
        acq = self._run(rin_dB)
        P  = np.abs(acq.E_source)**2
        P0 = 10e-3
        rin_lin = 10.0**(rin_dB / 10.0)
        bw = 200e6 / 2.0
        expected = P0**2 * rin_lin * bw
        np.testing.assert_allclose(np.var(P), expected, rtol=0.05)

    def test_rin_reproducible(self):
        a = self._run(-130.0)
        b = self._run(-130.0)
        np.testing.assert_array_equal(a.E_source, b.E_source)

    def test_rin_does_not_affect_phase_noise(self):
        # RIN and phase noise use different sub-seeds, so toggling RIN
        # must not change the phase realisation
        cfg_ph   = {**CFG, "source": {**CFG["source"],
                                       "linewidth": 1e5, "rin_dB_per_Hz": None}}
        cfg_both = {**CFG, "source": {**CFG["source"],
                                       "linewidth": 1e5, "rin_dB_per_Hz": -130.0}}
        a = SweptLaser(cfg_ph).process(FiberGenerator(cfg_ph).process(Acquisition()))
        b = SweptLaser(cfg_both).process(FiberGenerator(cfg_both).process(Acquisition()))
        phi_a = np.unwrap(np.angle(a.E_source))
        phi_b = np.unwrap(np.angle(b.E_source))
        np.testing.assert_allclose(phi_a, phi_b,  atol=1e-10)
