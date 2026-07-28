"""Tests for colored phase noise (see #28).

The slope tests fit a line on log(PSD) vs log(f) in the middle of
the band. Finite-sample estimation is noisy, so tolerances are
generous; we just want to catch gross regressions.
"""

import math

import numpy as np
import pytest

from scipy.signal import welch

from helpers import CFG
from pyofdr.core.acquisition import Acquisition
from pyofdr.fiber.profile import FiberGenerator
from pyofdr.optics.aux_mzi import AuxMZI
from pyofdr.optics.mach_zehnder import MachZehnder
from pyofdr.source.phase_noise import colored_frequency_noise, frequency_noise
from pyofdr.source.swept_laser import SweptLaser
from pyofdr.utils.colorednoise import powerlaw_psd_gaussian


def _slope(f, psd):
    """log-log slope on the middle 60% of the spectrum."""
    mask = (f > 0) & np.isfinite(psd) & (psd > 0)
    f, psd = f[mask], psd[mask]
    lo = int(0.2 * len(f))
    hi = int(0.8 * len(f))
    lf = np.log(f[lo:hi])
    lp = np.log(psd[lo:hi])
    a, _ = np.polyfit(lf, lp, 1)
    return a


class TestColoredNoisePrimitive:

    def test_white_is_flat(self):
        y = powerlaw_psd_gaussian(0.0, 2**14, random_state=np.random.default_rng(0))
        f, psd = welch(y, fs=1.0, nperseg=1024)
        assert abs(_slope(f, psd)) < 0.3

    def test_flicker_slope(self):
        y = powerlaw_psd_gaussian(1.0, 2**14, random_state=np.random.default_rng(1))
        f, psd = welch(y, fs=1.0, nperseg=1024)
        assert _slope(f, psd) == pytest.approx(-1.0, abs=0.3)

    def test_random_walk_slope(self):
        y = powerlaw_psd_gaussian(2.0, 2**14, random_state=np.random.default_rng(2))
        f, psd = welch(y, fs=1.0, nperseg=1024)
        assert _slope(f, psd) == pytest.approx(-2.0, abs=0.3)


class TestPhaseNoiseWrapper:

    def test_returns_zero_with_all_disabled(self):
        phi = colored_frequency_noise(
            1024, 1e-9,
            linewidth=0.0, sigma_flicker=0.0, sigma_rw=0.0,
            rng=np.random.default_rng(0),
        )
        np.testing.assert_array_equal(phi, np.zeros(1024))

    def test_wiener_matches_legacy_formula(self):
        """With only linewidth>0 we should get a cumsum of N(0, sqrt(2 pi lw dt))."""
        n, dt, lw = 4096, 1e-9, 1e5
        rng1 = np.random.default_rng(42)
        phi = colored_frequency_noise(n, dt, lw, 0.0, 0.0, rng=rng1)

        rng2 = np.random.default_rng(42)
        sigma = math.sqrt(2.0 * math.pi * lw * dt)
        phi_ref = np.cumsum(sigma * rng2.standard_normal(n))
        np.testing.assert_allclose(phi, phi_ref)

    def test_flicker_only_phase_psd_slope(self):
        # integrated 1/f FM -> phase PSD goes as 1/f^3
        n, dt = 2**15, 1e-9
        phi = colored_frequency_noise(
            n, dt, 0.0, sigma_flicker=1e5, sigma_rw=0.0,
            rng=np.random.default_rng(7))
        f, psd = welch(phi, fs=1.0/dt, nperseg=2048)
        assert _slope(f, psd) == pytest.approx(-3.0, abs=0.5)


class TestSweptLaserIntegration:

    def test_defaults_are_backward_compatible(self):
        # config without flicker/rw fields -> same E as before
        cfg_a = {**CFG, "source": {**CFG["source"], "linewidth": 1e4}}
        cfg_b = {**cfg_a, "source": {**cfg_a["source"],
                                      "flicker_noise_Hz": 0.0,
                                      "random_walk_noise_Hz": 0.0}}
        a = SweptLaser(cfg_a).process(Acquisition(sweep_index=0))
        b = SweptLaser(cfg_b).process(Acquisition(sweep_index=0))
        np.testing.assert_array_equal(a.E_source, b.E_source)

    def test_flicker_changes_the_field(self):
        cfg_clean = {**CFG, "source": {**CFG["source"], "linewidth": 0.0}}
        cfg_noisy = {**cfg_clean, "source": {**cfg_clean["source"],
                                               "flicker_noise_Hz": 1e5}}
        a = SweptLaser(cfg_clean).process(Acquisition(sweep_index=0))
        b = SweptLaser(cfg_noisy).process(Acquisition(sweep_index=0))
        # the clean field has zero phase noise, the flicker one doesn't
        assert not np.allclose(a.E_source, b.E_source)


class TestNoiseReachesTheBeat:
    """#82 -- stochastic phase noise used to die in |E_source|^2."""

    def _beat(self, **src_extra):
        cfg = {**CFG,
               "source": {**CFG["source"], **src_extra},
               "fiber": {**CFG["fiber"],
                          "rayleigh_coefficient_dB": -200.0,
                          "reflectors": [{"z": 0.5, "R": 0.01}]}}
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        acq = SweptLaser(cfg).process(acq)
        return MachZehnder(cfg).process(acq)

    def test_flicker_changes_the_photocurrent(self):
        clean = self._beat()
        noisy = self._beat(flicker_noise_Hz=1e6)
        assert not np.allclose(clean.photocurrent_main,
                               noisy.photocurrent_main)

    def test_flicker_raises_the_pedestal(self):
        # noise floor away from the reflector peak must come up
        clean = self._beat()
        noisy = self._beat(flicker_noise_Hz=3e7)
        spec_c = np.abs(np.fft.rfft(clean.photocurrent_main[0]))
        spec_n = np.abs(np.fft.rfft(noisy.photocurrent_main[0]))
        peak = np.argmax(spec_c)
        # look well away from the peak (and from DC)
        lo = peak + peak // 2
        hi = 3 * peak
        assert np.median(spec_n[lo:hi]) > 10.0 * np.median(spec_c[lo:hi])

    def test_noise_reaches_the_aux_clock(self):
        # the k-clock shares the laser so it must jitter too
        cfg_clean = {**CFG,
                     "optics": {**CFG["optics"],
                                "aux_mzi": {"enabled": True, "delay": 50e-9}}}
        cfg_noisy = {**cfg_clean,
                     "source": {**cfg_clean["source"], "linewidth": 1e5}}
        a = AuxMZI(cfg_clean).process(
            SweptLaser(cfg_clean).process(Acquisition()))
        b = AuxMZI(cfg_noisy).process(
            SweptLaser(cfg_noisy).process(Acquisition()))
        assert not np.allclose(a.aux_signal, b.aux_signal)

    def test_white_fm_alone_keeps_main_beat_deterministic(self):
        # the Lorentzian part is deliberately not warped: two sweeps with
        # different noise realizations give the same photocurrent (only the
        # ensemble-average visibility roll-off applies).
        cfg = {**CFG,
               "source": {**CFG["source"], "linewidth": 1e5},
               "fiber": {**CFG["fiber"], "reflectors": [{"z": 0.5, "R": 0.01}]}}
        acq0 = Acquisition(sweep_index=0)
        acq0 = FiberGenerator(cfg).process(acq0)
        prof = acq0.fiber_profile.copy()
        acq0 = MachZehnder(cfg).process(SweptLaser(cfg).process(acq0))

        acq1 = Acquisition(sweep_index=1)
        acq1 = FiberGenerator(cfg).process(acq1)
        acq1.fiber_profile = prof   # same fiber, different laser noise draw
        acq1 = MachZehnder(cfg).process(SweptLaser(cfg).process(acq1))

        np.testing.assert_allclose(acq0.photocurrent_main,
                                   acq1.photocurrent_main)


class TestFrequencyNoisePrimitive:

    def test_white_integrates_to_wiener(self):
        n, dt, lw = 4096, 1e-9, 1e5
        nu_w, _ = frequency_noise(n, dt, lw, 0.0, 0.0,
                                  rng=np.random.default_rng(3))
        phi = 2.0 * math.pi * np.cumsum(nu_w) * dt
        # per-sample increment must be N(0, sqrt(2 pi lw dt))
        sigma = math.sqrt(2.0 * math.pi * lw * dt)
        assert np.std(np.diff(phi)) == pytest.approx(sigma, rel=0.05)

    def test_colored_part_is_separate(self):
        nu_w, nu_c = frequency_noise(1024, 1e-9, 0.0, 1e5, 0.0,
                                     rng=np.random.default_rng(4))
        np.testing.assert_array_equal(nu_w, np.zeros(1024))
        assert np.any(nu_c != 0.0)
