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
from pyofdr.source.phase_noise import colored_frequency_noise
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
