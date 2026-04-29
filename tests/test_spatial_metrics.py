"""Tests for spatial_metrics analysis module."""

import numpy as np
import pytest

from pyofdr.analysis.spatial_metrics import (
    measure_resolution,
    measure_dynamic_range,
    snr_profile,
)


class TestMeasureResolution:

    def _make_reflector(self, N, dz, z_pos, width_bins=3):
        """Synthetic reflectogram: noise floor + gaussian peak."""
        z = np.arange(N) * dz
        amp = np.ones(N) * 1e-3   # noise floor
        i_peak = int(z_pos / dz)
        for di in range(-width_bins * 3, width_bins * 3 + 1):
            idx = i_peak + di
            if 0 <= idx < N:
                amp[idx] += np.exp(-0.5 * (di / width_bins) ** 2)
        return amp.astype(np.complex128), z

    def test_resolution_is_positive(self):
        H, z = self._make_reflector(1000, 20e-6, 0.01)
        r = measure_resolution(H, z, 0.01)
        assert r["resolution"] > 0

    def test_wider_peak_gives_larger_resolution(self):
        dz = 20e-6
        H_narrow, z = self._make_reflector(1000, dz, 0.01, width_bins=2)
        H_wide, _   = self._make_reflector(1000, dz, 0.01, width_bins=6)
        r_n = measure_resolution(H_narrow, z, 0.01)
        r_w = measure_resolution(H_wide, z, 0.01)
        assert r_w["resolution"] > r_n["resolution"]

    def test_peak_position_is_found(self):
        H, z = self._make_reflector(1000, 20e-6, 0.01)
        r = measure_resolution(H, z, 0.01)
        assert abs(r["peak_z"] - 0.01) < 0.001


class TestMeasureDynamicRange:

    def test_known_dr(self):
        N = 1000
        H = np.ones(N, dtype=np.complex128) * 0.001
        H[:500] = 1.0   # signal
        sig = np.zeros(N, dtype=bool); sig[:500] = True
        noi = np.zeros(N, dtype=bool); noi[500:] = True
        r = measure_dynamic_range(H, sig, noi)
        assert r["dr_dB"] == pytest.approx(60.0, abs=0.1)

    def test_equal_regions_gives_zero_dr(self):
        H = np.ones(100, dtype=np.complex128)
        mask = np.ones(100, dtype=bool)
        r = measure_dynamic_range(H, mask, mask)
        assert r["dr_dB"] == pytest.approx(0.0, abs=0.01)


class TestSNRProfile:

    def test_constant_signal_gives_high_snr(self):
        N = 256
        z = np.arange(N) * 1e-3
        # identical sweeps -> std ~ 0 -> snr very high
        sweeps = [np.ones(N, dtype=np.complex128)] * 3
        r = snr_profile(sweeps, z)
        assert np.all(r["snr_dB"] > 100)

    def test_noisy_signal_gives_finite_snr(self):
        rng = np.random.default_rng(0)
        N = 512
        z = np.arange(N) * 1e-3
        base = np.ones(N)
        sweeps = [base + 0.1 * rng.standard_normal(N) for _ in range(10)]
        r = snr_profile(sweeps, z)
        # SNR ~ 20*log10(1/0.1) ~ 20 dB, give some margin
        median_snr = np.median(r["snr_dB"])
        assert 10 < median_snr < 30

    def test_single_sweep_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            snr_profile([np.ones(10)], np.arange(10))

    def test_output_shapes(self):
        N = 64
        z = np.arange(N) * 1e-3
        sweeps = [np.ones(N, dtype=np.complex128)] * 3
        r = snr_profile(sweeps, z)
        assert r["snr_dB"].shape == (N,)
        assert r["mean"].shape == (N,)
        assert r["std"].shape == (N,)
