"""Tests for spectral analysis module."""

import numpy as np
import pytest

from pyofdr.analysis.spectral import beat_psd, beat_spectrogram, psd_slope


class TestBeatPSD:

    def test_white_noise_is_flat(self):
        rng = np.random.default_rng(0)
        N = 100_000
        dt = 1e-8   # 100 MHz
        beat = rng.standard_normal(N)
        f, psd = beat_psd(beat, dt)
        # for white noise, PSD should be roughly constant
        # check that max/min ratio is less than 10 dB
        psd_mid = psd[len(psd) // 4 : 3 * len(psd) // 4]
        ratio_dB = 10 * np.log10(np.max(psd_mid) / np.min(psd_mid))
        assert ratio_dB < 10

    def test_tone_shows_peak(self):
        dt = 1e-8
        N = 50_000
        t = np.arange(N) * dt
        f_tone = 1e7   # 10 MHz
        beat = np.sin(2 * np.pi * f_tone * t)
        f, psd = beat_psd(beat, dt)
        i_peak = np.argmax(psd)
        assert abs(f[i_peak] - f_tone) < f[1] - f[0]   # within one bin

    def test_output_shapes(self):
        beat = np.random.default_rng(1).standard_normal(1024)
        f, psd = beat_psd(beat, 1e-8, nperseg=256)
        assert f.shape == psd.shape
        assert len(f) > 0


class TestBeatSpectrogram:

    def test_output_shapes(self):
        beat = np.random.default_rng(0).standard_normal(4096)
        t, f, Sxx = beat_spectrogram(beat, 1e-8, nperseg=256)
        assert Sxx.shape == (len(f), len(t))

    def test_chirp_shows_time_varying_peak(self):
        dt = 1e-8
        N = 20_000
        t = np.arange(N) * dt
        # linear chirp from 1 MHz to 10 MHz
        phase = 2 * np.pi * (1e6 * t + 0.5 * (10e6 - 1e6) / (N * dt) * t ** 2)
        beat = np.cos(phase)
        t_s, f_s, Sxx = beat_spectrogram(beat, dt, nperseg=512)
        # peak frequency should increase with time
        peak_freq = f_s[np.argmax(Sxx, axis=0)]
        assert peak_freq[-1] > peak_freq[0]


class TestPSDSlope:

    def test_white_noise_slope_near_zero(self):
        rng = np.random.default_rng(10)
        N = 200_000
        dt = 1e-8
        beat = rng.standard_normal(N)
        f, psd = beat_psd(beat, dt, nperseg=8192)
        r = psd_slope(f, psd, 1e6, 40e6)
        # white noise -> slope ~ 0
        assert abs(r["slope"]) < 0.3

    def test_not_enough_points_raises(self):
        f = np.array([1.0, 2.0, 3.0])
        psd = np.array([1.0, 0.5, 0.25])
        with pytest.raises(ValueError, match="not enough"):
            psd_slope(f, psd, 100, 200)

    def test_known_slope(self):
        # construct a 1/f PSD (slope = -1)
        f = np.logspace(1, 5, 500)
        psd = 1.0 / f
        r = psd_slope(f, psd, 100, 50_000)
        assert r["slope"] == pytest.approx(-1.0, abs=0.05)
