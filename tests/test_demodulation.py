"""Tests for the demodulation module (see #42)."""

import math

import numpy as np
import pytest

from helpers import CFG
from analysis.demodulation import (
    fft_reflectogram,
    reflectogram_from_acq,
    phase_difference_strain,
    cross_spectrum_shift,
    freq_shift_to_strain,
    freq_shift_to_temperature,
)


# ── FFT reflectogram ────────────────────────────────────────────────

class TestFFTReflectogram:

    def test_round_trip_identity(self):
        """IFFT -> fft_reflectogram recovers the original spectrum."""
        rng = np.random.default_rng(0)
        N = 2048
        dz = 0.02e-3
        # random complex spectrum (one-sided)
        H_orig = rng.standard_normal(N // 2) + 1j * rng.standard_normal(N // 2)
        # make a real beat from it: mirror to get conjugate-symmetric spectrum
        full = np.zeros(N, dtype=np.complex128)
        full[:N // 2] = H_orig
        full[N // 2 + 1:] = np.conj(H_orig[-1:0:-1])
        beat = np.fft.ifft(full).real

        H_rec, z = fft_reflectogram(beat, dz)
        assert len(z) == N // 2
        # skip DC bin (index 0) -- Nyquist folding makes it special
        np.testing.assert_allclose(np.abs(H_rec[1:]), np.abs(H_orig[1:]),
                                    rtol=1e-10)

    def test_zero_pad_increases_points(self):
        beat = np.random.default_rng(1).standard_normal(512)
        H1, z1 = fft_reflectogram(beat, 1e-3)
        H2, z2 = fft_reflectogram(beat, 1e-3, n_pad=2048)
        assert len(z2) == 1024
        assert len(z1) == 256

    def test_hanning_window_runs(self):
        beat = np.random.default_rng(2).standard_normal(256)
        H, z = fft_reflectogram(beat, 1e-3, window="hanning")
        assert H.shape == z.shape

    def test_unknown_window_raises(self):
        with pytest.raises(ValueError, match="unknown window"):
            fft_reflectogram(np.ones(64), 1e-3, window="blackman-harris")


# ── Phase-difference strain ──────────────────────────────────────────

class TestPhaseDifference:

    def test_recover_uniform_strain(self):
        """Apply a known linear phase ramp (=uniform strain) and recover it."""
        N = 4096
        wl = 1550e-9
        n = 1.4682
        p_e = 0.22
        dz = 20e-6   # 20 um bins

        # constant-amplitude reference -- isolates the phase-gradient
        # math from Rayleigh-like amplitude fluctuations
        spec_ref = np.ones(N, dtype=np.complex128)

        # apply uniform strain over [z0, z1]
        eps_true = 5e-5
        z = np.arange(N) * dz
        z0, z1 = 0.02, 0.06   # 2 cm to 6 cm

        k0 = 2.0 * math.pi / wl
        prefactor = 2.0 * k0 * n * (1.0 - p_e)
        eps_field = np.where((z >= z0) & (z <= z1), eps_true, 0.0)
        phase = prefactor * np.cumsum(eps_field) * dz
        spec_meas = spec_ref * np.exp(1j * phase)

        strain = phase_difference_strain(
            spec_meas, spec_ref, dz,
            gauge_length=2e-3,  # 2 mm gauge
            wavelength=wl, n=n, p_e=p_e,
        )

        # well inside the strained region (away from edges by gauge_length)
        inside = (z > z0 + 5e-3) & (z < z1 - 5e-3)
        np.testing.assert_allclose(strain[inside], eps_true,
                                    rtol=0.02, atol=1e-7)

        # unstrained region well before the segment
        before = z < z0 - 5e-3
        np.testing.assert_allclose(strain[before], 0.0, atol=1e-7)

    def test_zero_strain_gives_zero(self):
        rng = np.random.default_rng(20)
        spec = rng.standard_normal(512) + 1j * rng.standard_normal(512)
        strain = phase_difference_strain(spec, spec, 20e-6, 1e-3, 1550e-9)
        np.testing.assert_allclose(strain, 0.0, atol=1e-12)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            phase_difference_strain(np.ones(10), np.ones(20),
                                     1e-3, 1e-3, 1550e-9)


# ── Cross-spectrum frequency shift ───────────────────────────────────

class TestCrossSpectrumShift:

    def test_recover_known_shift(self):
        """Apply a constant frequency shift and recover it."""
        rng = np.random.default_rng(30)
        N = 2048
        dz = 20e-6
        n = 1.4682

        H_ref = rng.standard_normal(N) + 1j * rng.standard_normal(N)

        # apply a uniform spectral shift: phase = 2*pi*df * tau_cell
        df_true = 1e9   # 1 GHz shift
        tau_cell = 2.0 * n * dz / 299792458.0
        phase = 2.0 * math.pi * df_true * tau_cell
        H_meas = H_ref * np.exp(1j * phase)

        df_rec = cross_spectrum_shift(H_meas, H_ref, dz, n=n)
        # every bin should give df_true
        np.testing.assert_allclose(df_rec, df_true, rtol=1e-10)

    def test_zero_shift(self):
        rng = np.random.default_rng(31)
        H = rng.standard_normal(256) + 1j * rng.standard_normal(256)
        df = cross_spectrum_shift(H, H, 20e-6)
        # H*conj(H) = |H|^2 is real+positive, angle ~ 0 up to float rounding
        np.testing.assert_allclose(df, 0.0, atol=0.1)

    def test_smooth_bins(self):
        rng = np.random.default_rng(32)
        N = 512
        dz = 20e-6
        n = 1.4682
        H_ref = rng.standard_normal(N) + 1j * rng.standard_normal(N)

        df_true = 5e8
        tau_cell = 2.0 * n * dz / 299792458.0
        phase = 2.0 * math.pi * df_true * tau_cell
        H_meas = H_ref * np.exp(1j * phase)

        df_rec = cross_spectrum_shift(H_meas, H_ref, dz, n=n, smooth_bins=10)
        # smoothing shouldn't affect a uniform shift
        np.testing.assert_allclose(df_rec[20:-20], df_true, rtol=1e-4)


# ── Conversion helpers ───────────────────────────────────────────────

class TestConversions:

    def test_strain_roundtrip(self):
        """freq_shift -> strain -> back should be consistent."""
        nu0 = 299792458.0 / 1550e-9
        p_e = 0.22
        eps_in = 1e-4
        df = -eps_in * nu0 * (1.0 - p_e)
        eps_out = freq_shift_to_strain(df, nu0, p_e=p_e)
        assert eps_out == pytest.approx(eps_in)

    def test_temperature_sign(self):
        # heating -> positive df shift (blueshift in Rayleigh)
        # so negative df -> cooling -> positive dT... wait, convention:
        # standard: heating causes negative freq shift (redshift)
        # dT = -df / (nu0 * (alpha + xi)) => positive df -> negative dT
        nu0 = 299792458.0 / 1550e-9
        dT = freq_shift_to_temperature(np.array([-1e9]), nu0)
        assert dT[0] > 0   # negative freq shift -> positive temp change

    def test_array_input(self):
        nu0 = 299792458.0 / 1550e-9
        df = np.array([0.0, 1e9, -1e9])
        eps = freq_shift_to_strain(df, nu0)
        assert eps.shape == (3,)
        assert eps[0] == pytest.approx(0.0)


# ── Integration: full pipeline round-trip ────────────────────────────

class TestIntegration:

    def test_pipeline_strain_recovery(self):
        """Run 2 sweeps -- one unstrained, one strained -- and recover."""
        from core.campaign import run_campaign

        eps_true = 5e-5
        cfg_ref = {**CFG, "simulation": {**CFG["simulation"], "n_sweeps": 1},
                   "strain": {"segments": []}}
        cfg_str = {**CFG, "simulation": {**CFG["simulation"], "n_sweeps": 1},
                   "strain": {"segments": [{"start": 0.3, "end": 0.7,
                                              "epsilon": eps_true}]}}
        # turn off noise for a clean test
        cfg_base = {"detection": {**CFG["detection"],
                                    "shot_noise": False, "thermal_nep": 0.0,
                                    "dark_current": 0.0},
                    "source": {**CFG["source"], "linewidth": 0.0}}
        cfg_ref.update(cfg_base)
        cfg_str.update(cfg_base)

        acq_ref = run_campaign(cfg_ref)[-1]
        acq_str = run_campaign(cfg_str)[-1]

        H_ref, z = fft_reflectogram(
            acq_ref.digital_main[0].astype(np.float64), acq_ref.dz)
        H_str, _ = fft_reflectogram(
            acq_str.digital_main[0].astype(np.float64), acq_str.dz)

        wl = cfg_ref["source"]["center_wavelength"]

        # phase-difference method: recovers strain from cumulative phase
        eps_rec = phase_difference_strain(
            H_str, H_ref, acq_ref.dz,
            gauge_length=0.01,  # 1 cm gauge
            wavelength=wl,
        )

        # inside the strained region, median should be close to eps_true
        inside = (z > 0.35) & (z < 0.65)
        median_eps = np.median(eps_rec[inside])
        assert abs(median_eps - eps_true) / eps_true < 0.30
