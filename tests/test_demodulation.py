"""Tests for the demodulation module (see #42)."""

import math

import numpy as np
import pytest

from helpers import CFG
from pyofdr.analysis.demodulation import (
    fft_reflectogram,
    reflectogram_from_acq,
    phase_difference_strain,
    cross_spectrum_shift,
    windowed_xcorr_strain,
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

    def test_recover_uniform_shift(self):
        """Uniform strain -> constant shift, Froggatt-Moore sign."""
        rng = np.random.default_rng(30)
        N = 2048
        dz = 20e-6
        n = 1.4682
        p_e = 0.22
        wl = 1550e-9

        H_ref = rng.standard_normal(N) + 1j * rng.standard_normal(N)

        # forward model applies *cumulative* phase (see fiber/strain.py)
        eps = 1e-5
        k0 = 2.0 * math.pi / wl
        prefactor = 2.0 * k0 * n * (1.0 - p_e)
        phase = prefactor * np.cumsum(np.full(N, eps)) * dz
        H_meas = H_ref * np.exp(1j * phase)

        df_rec = cross_spectrum_shift(H_meas, H_ref, dz, n=n)
        # stretch -> redshift, same value at every bin
        nu0 = 299792458.0 / wl
        df_true = -eps * nu0 * (1.0 - p_e)
        np.testing.assert_allclose(df_rec, df_true, rtol=1e-9)

        # round trip back to strain
        eps_rec = freq_shift_to_strain(df_rec, nu0, p_e=p_e)
        np.testing.assert_allclose(eps_rec, eps, rtol=1e-9)

    def test_localised_segment(self):
        """Strain over a segment: shift inside, zero elsewhere."""
        rng = np.random.default_rng(31)
        N = 2048
        dz = 20e-6
        n = 1.4682
        p_e = 0.22
        wl = 1550e-9

        H_ref = rng.standard_normal(N) + 1j * rng.standard_normal(N)

        z = np.arange(N) * dz
        z0, z1 = 0.01, 0.03
        eps = 1e-4
        eps_field = np.where((z >= z0) & (z <= z1), eps, 0.0)
        k0 = 2.0 * math.pi / wl
        prefactor = 2.0 * k0 * n * (1.0 - p_e)
        phase = prefactor * np.cumsum(eps_field) * dz
        H_meas = H_ref * np.exp(1j * phase)

        df_rec = cross_spectrum_shift(H_meas, H_ref, dz, n=n)
        df_true = -eps * (299792458.0 / wl) * (1.0 - p_e)

        # np.gradient smears the step over +-1 bin, keep a small margin
        inside  = (z > z0 + 5 * dz) & (z < z1 - 5 * dz)
        outside = (z < z0 - 5 * dz) | (z > z1 + 5 * dz)
        np.testing.assert_allclose(df_rec[inside], df_true, rtol=1e-9)
        np.testing.assert_allclose(df_rec[outside], 0.0, atol=1e3)

    def test_zero_shift(self):
        rng = np.random.default_rng(32)
        H = rng.standard_normal(256) + 1j * rng.standard_normal(256)
        df = cross_spectrum_shift(H, H, 20e-6)
        # H*conj(H) = |H|^2 is real+positive, angle ~ 0 up to float rounding
        np.testing.assert_allclose(df, 0.0, atol=0.1)

    def test_smooth_bins(self):
        # constant-amplitude reflectogram: smoothing must leave the
        # phase ramp (and thus the recovered shift) untouched
        N = 512
        dz = 20e-6
        n = 1.4682
        p_e = 0.22
        wl = 1550e-9

        H_ref = np.ones(N, dtype=np.complex128)

        eps = 1e-4
        k0 = 2.0 * math.pi / wl
        prefactor = 2.0 * k0 * n * (1.0 - p_e)
        phase = prefactor * np.cumsum(np.full(N, eps)) * dz
        H_meas = H_ref * np.exp(1j * phase)

        df_rec = cross_spectrum_shift(H_meas, H_ref, dz, n=n, smooth_bins=10)
        df_true = -eps * (299792458.0 / wl) * (1.0 - p_e)
        np.testing.assert_allclose(df_rec[20:-20], df_true, rtol=1e-9)


# ── Windowed sub-spectrum cross-correlation ─────────────────────────

class TestWindowedXcorrStrain:

    def test_synthetic_integer_shift(self):
        """Rolling the local spectrum by an integer number of bins
        must show up as the matching freq shift, and thus as a strain
        derivable from Froggatt-Moore."""
        rng = np.random.default_rng(40)
        W = 256
        dz = 20e-6
        k_shift = 7        # roll(A, +k) == compressed fiber, so eps < 0

        A = rng.standard_normal(W) + 1j * rng.standard_normal(W)
        B = np.roll(A, k_shift)

        # make reflectograms so that the algo's IFFT of the window recovers A/B
        H_ref = np.fft.fft(A)
        H_str = np.fft.fft(B)

        sweep_hz = 5e12
        nu0      = 193.4e12
        p_e      = 0.22

        z, eps = windowed_xcorr_strain(
            H_str, H_ref, dz,
            gauge_length=W * dz, stride=W * dz,
            sweep_range_hz=sweep_hz, center_freq=nu0, p_e=p_e,
        )

        # expected: dnu = +k_shift * (sweep/W) (compressed -> higher freq)
        # eps      = -dnu / (nu0 * (1 - p_e))
        eps_true = -k_shift * (sweep_hz / W) / (nu0 * (1.0 - p_e))
        assert len(eps) == 1
        np.testing.assert_allclose(eps[0], eps_true, rtol=1e-10)

    def test_zero_shift(self):
        rng = np.random.default_rng(41)
        W = 256
        n = 4 * W
        H = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        dz = 20e-6

        _, eps = windowed_xcorr_strain(
            H, H, dz,
            gauge_length=W * dz, stride=W * dz,
            sweep_range_hz=5e12, center_freq=193.4e12,
        )
        # same reflectogram -> no spectral shift
        np.testing.assert_allclose(eps, 0.0, atol=1e-12)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            windowed_xcorr_strain(np.ones(100), np.ones(200),
                                   dz=20e-6, gauge_length=1e-3, stride=1e-3,
                                   sweep_range_hz=5e12, center_freq=193e12)

    def test_gauge_too_small_raises(self):
        with pytest.raises(ValueError, match="gauge too small"):
            windowed_xcorr_strain(np.ones(100), np.ones(100),
                                   dz=20e-6, gauge_length=50e-6, stride=50e-6,
                                   sweep_range_hz=5e12, center_freq=193e12)

    def test_pipeline_strain_recovery(self):
        """End-to-end: run the simulator with a known 1000 ustrain step
        and check we recover it within a few percent on the inside bins."""
        from pyofdr.core.campaign import run_campaign
        from pyofdr.utils.constants import C
        from pyofdr.utils.units import wavelength_range_to_freq_range

        eps_true = 1e-3    # 1000 ustrain
        cfg_ref = {**CFG, "simulation": {**CFG["simulation"], "n_sweeps": 1},
                   "strain": {"segments": []}}
        cfg_str = {**CFG, "simulation": {**CFG["simulation"], "n_sweeps": 1},
                   "strain": {"segments": [{"start": 0.3, "end": 0.7,
                                             "epsilon": eps_true}]}}
        # kill noise so the test is deterministic
        noise_off = {"detection": {**CFG["detection"],
                                    "shot_noise": False, "thermal_nep": 0.0,
                                    "dark_current": 0.0},
                     "source": {**CFG["source"], "linewidth": 0.0}}
        cfg_ref.update(noise_off)
        cfg_str.update(noise_off)

        acq_ref = run_campaign(cfg_ref)[-1]
        acq_str = run_campaign(cfg_str)[-1]

        H_ref, _ = fft_reflectogram(
            acq_ref.digital_main[0].astype(np.float64), acq_ref.dz)
        H_str, _ = fft_reflectogram(
            acq_str.digital_main[0].astype(np.float64), acq_str.dz)

        wl = cfg_ref["source"]["center_wavelength"]
        sweep_hz = wavelength_range_to_freq_range(wl, cfg_ref["source"]["sweep_range"])
        nu0      = C / wl

        zc, eps_rec = windowed_xcorr_strain(
            H_str, H_ref, acq_ref.dz,
            gauge_length=0.01, stride=2e-3,
            sweep_range_hz=sweep_hz, center_freq=nu0, p_e=0.22,
        )
        inside  = (zc > 0.38) & (zc < 0.62)
        outside = (zc > 0.05) & (zc < 0.25)
        assert inside.sum()  > 0
        assert outside.sum() > 0

        np.testing.assert_allclose(np.median(eps_rec[inside]),  eps_true, rtol=0.05)
        np.testing.assert_allclose(np.median(eps_rec[outside]), 0.0,       atol=1e-5)


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
        from pyofdr.core.campaign import run_campaign

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


class TestStrainRecoveryRobustness:
    """Strain recovery via windowed sub-spectrum xcorr must hold across
    realistic detector / gauge combinations. Sweeps shot noise on/off
    against three gauges (2 mm, 1 cm, 5 cm). Part of #5."""

    @staticmethod
    def _run_pair(shot_on):
        from pyofdr.core.campaign import run_campaign
        eps_true = 1e-3
        cfg_ref = {**CFG, "simulation": {**CFG["simulation"], "n_sweeps": 1},
                   "strain": {"segments": []},
                   "detection": {**CFG["detection"],
                                 "shot_noise":   shot_on,
                                 "thermal_nep":  0.0,
                                 "dark_current": 0.0},
                   "source": {**CFG["source"], "linewidth": 0.0}}
        cfg_str = {**cfg_ref, "strain": {"segments":
                   [{"start": 0.3, "end": 0.7, "epsilon": eps_true}]}}
        acq_ref = run_campaign(cfg_ref)[0]
        acq_str = run_campaign(cfg_str)[0]
        return acq_ref, acq_str, eps_true

    @pytest.mark.parametrize("shot_on",      [False, True])
    @pytest.mark.parametrize("gauge_length", [2e-3, 1e-2, 5e-2])
    def test_recovers_within_5pct(self, shot_on, gauge_length):
        from pyofdr.utils.constants import C
        from pyofdr.utils.units import wavelength_range_to_freq_range

        acq_ref, acq_str, eps_true = self._run_pair(shot_on)

        H_ref, _ = fft_reflectogram(
            acq_ref.digital_main[0].astype(np.float64), acq_ref.dz)
        H_str, _ = fft_reflectogram(
            acq_str.digital_main[0].astype(np.float64), acq_str.dz)

        wl       = CFG["source"]["center_wavelength"]
        sweep_hz = wavelength_range_to_freq_range(wl, CFG["source"]["sweep_range"])
        nu0      = C / wl

        # stride ~ gauge / 5 keeps a sensible overlap across sizes
        zc, eps_rec = windowed_xcorr_strain(
            H_str, H_ref, acq_ref.dz,
            gauge_length=gauge_length, stride=max(gauge_length / 5, 2e-3),
            sweep_range_hz=sweep_hz, center_freq=nu0, p_e=0.22,
        )
        # half-gauge margin keeps the window fully inside the strained range
        margin = gauge_length / 2.0
        inside = (zc > 0.3 + margin) & (zc < 0.7 - margin)
        assert inside.sum() > 0, "no fully-inside windows for this gauge"

        np.testing.assert_allclose(
            np.median(eps_rec[inside]), eps_true, rtol=0.05,
            err_msg=f"shot={shot_on}, gauge={gauge_length*1e3:.0f} mm")
