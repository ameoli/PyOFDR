"""Tests for strain_quality analysis module."""

import math

import numpy as np
import pytest

from pyofdr.analysis.strain_quality import (
    strain_noise_floor,
    strain_sensitivity,
    allan_deviation,
)


class TestStrainNoiseFloor:

    def test_zero_strain_gives_zero_noise(self):
        strain = np.zeros(500)
        mask = np.ones(500, dtype=bool)
        r = strain_noise_floor(strain, mask)
        assert r["noise_floor"] == 0.0
        assert r["mean"] == 0.0

    def test_known_noise_level(self):
        rng = np.random.default_rng(0)
        sigma = 1e-6
        strain = rng.normal(0, sigma, 10000)
        mask = np.ones(10000, dtype=bool)
        r = strain_noise_floor(strain, mask)
        assert r["noise_floor"] == pytest.approx(sigma, rel=0.05)

    def test_mask_selects_subset(self):
        strain = np.zeros(100)
        strain[50:] = 1e-3   # strained region
        quiet = np.zeros(100, dtype=bool)
        quiet[:50] = True
        r = strain_noise_floor(strain, quiet)
        assert r["noise_floor"] == 0.0
        assert r["n_bins"] == 50


class TestStrainSensitivity:

    def test_longer_gauge_improves_sensitivity(self):
        nf = 1e-6
        dz = 20e-6
        s1 = strain_sensitivity(nf, gauge_length=1e-3, dz=dz)
        s2 = strain_sensitivity(nf, gauge_length=10e-3, dz=dz)
        assert s2 < s1

    def test_sqrt_scaling(self):
        nf = 1e-6
        dz = 20e-6
        G = 5e-3
        s = strain_sensitivity(nf, G, dz)
        expected = nf / math.sqrt(G / dz)
        assert s == pytest.approx(expected)

    def test_single_bin_gauge(self):
        nf = 1e-6
        dz = 20e-6
        s = strain_sensitivity(nf, gauge_length=dz, dz=dz)
        assert s == pytest.approx(nf)


class TestAllanDeviation:

    def test_white_noise_decreases_with_tau(self):
        rng = np.random.default_rng(42)
        # white frequency noise: adev should decrease as 1/sqrt(tau)
        n = 1000
        freq = rng.standard_normal(n) * 1e6
        r = allan_deviation(freq, dt_sweep=0.01)
        # first tau should have higher adev than last
        assert r["adev"][0] > r["adev"][-1]

    def test_constant_freq_gives_zero_adev(self):
        freq = np.ones(100) * 1e9
        r = allan_deviation(freq, dt_sweep=0.01)
        np.testing.assert_allclose(r["adev"], 0.0, atol=1e-6)

    def test_too_few_sweeps_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            allan_deviation(np.array([1.0, 2.0]), dt_sweep=0.01)

    def test_tau_spacing(self):
        freq = np.random.default_rng(0).standard_normal(50)
        r = allan_deviation(freq, dt_sweep=0.1)
        # first tau = dt_sweep, second = 2*dt_sweep, etc.
        assert r["tau"][0] == pytest.approx(0.1)
        assert r["tau"][1] == pytest.approx(0.2)

    def test_output_lengths_match(self):
        freq = np.random.default_rng(0).standard_normal(50)
        r = allan_deviation(freq, dt_sweep=0.01)
        assert len(r["tau"]) == len(r["adev"])
        assert len(r["tau"]) > 0
