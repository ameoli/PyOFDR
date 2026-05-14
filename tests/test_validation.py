"""End-to-end analytical validation (issue #5).

These are *pipeline-level* checks: run a minimal simulation through
``run_campaign`` and compare the output statistics to what the analytical
budget predicts. Each test isolates one physical axis by turning off the
other noise sources / scatter contributions.

For noise-floor tests we run the pipeline twice (noise on / noise off)
and subtract the deterministic baseline in quadrature -- this avoids
needing Rayleigh-free fiber and keeps the comparison honest.

Kept short on purpose -- the door stays open for more validation
scenarios (RIN floor, Rayleigh mean power, strain shift, ...).
"""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pytest

from helpers import CFG
from pyofdr.analysis.budget import compute_budget
from pyofdr.core.campaign import run_campaign
from pyofdr.core.config import compute_derived


def _noiseless_cfg():
    """Base config: every noise source off, Rayleigh heavily suppressed."""
    cfg = deepcopy(CFG)
    cfg["simulation"] = {"seed": 7}
    cfg["source"] = dict(cfg["source"])
    cfg["source"]["linewidth"] = 0.0
    cfg["source"]["rin_dB_per_Hz"] = None
    cfg["fiber"] = dict(cfg["fiber"])
    cfg["fiber"]["rayleigh_coefficient_dB"] = -120.0
    cfg["fiber"]["attenuation_dB_per_km"] = 0.0
    cfg["detection"] = {"responsivity": 1.0, "bandwidth": 1.0e8,
                        "shot_noise": False, "thermal_nep": 0.0,
                        "dark_current": 0.0}
    return cfg


def _std_inner(acq, Z):
    """RMS of the detector current over the middle 80 % of the sweep."""
    sig = np.asarray(acq.analog_main[0]) / Z
    n   = sig.size
    lo, hi = n // 10, n - n // 10
    chunk = sig[lo:hi]
    return float(np.std(chunk - chunk.mean()))


def _injected_noise_std(cfg_on, cfg_off):
    """std of noise that was added on top of the noiseless baseline."""
    acq_on  = run_campaign(cfg_on)[0]
    acq_off = run_campaign(cfg_off)[0]
    Z = cfg_on["adc"]["input_impedance"]
    s_on  = _std_inner(acq_on,  Z)
    s_off = _std_inner(acq_off, Z)
    return math.sqrt(max(s_on ** 2 - s_off ** 2, 0.0))


class TestBeatFrequencyFromReflector:
    """A point reflector at z0 must show up at f_beat = 2 n gamma z0 / c."""

    def test_reflector_peak_at_expected_frequency(self):
        cfg = _noiseless_cfg()
        cfg["fiber"]["length"] = 5.0
        z0 = 3.0
        cfg["fiber"]["reflectors"] = [{"z": z0, "R": 0.1}]

        d = compute_derived(cfg)
        f_expected = 2.0 * cfg["fiber"]["n_core"] * d["gamma"] * z0 / 2.998e8

        acq = run_campaign(cfg)[0]
        sig = np.asarray(acq.analog_main[0])
        sig = sig - sig.mean()                     # drop DC

        n     = sig.size
        lo    = n // 10
        hi    = n - lo
        sig_c = sig[lo:hi]
        spec  = np.abs(np.fft.rfft(sig_c))
        freqs = np.fft.rfftfreq(sig_c.size, d=acq.dt)

        peak_idx = int(np.argmax(spec))
        f_peak   = freqs[peak_idx]
        bin_hz   = freqs[1] - freqs[0]

        # reflector position quantizes on the dz grid, which shifts the
        # beat by 2*n*gamma*dz/c -- a few hundred Hz. allow ~5 bins slack.
        assert abs(f_peak - f_expected) < 5 * bin_hz, \
            f"peak {f_peak:.3e} Hz vs expected {f_expected:.3e} Hz (bin {bin_hz:.1f} Hz)"


class TestNoiseFloorMatchesBudget:
    """Signal-independent noise sources: the measured std on the
    post-filter output must equal compute_budget's sigma within ~15 %."""

    def test_thermal_noise_matches_budget(self):
        cfg_off = _noiseless_cfg()
        cfg_off["fiber"]["length"] = 0.5
        cfg_on  = deepcopy(cfg_off)
        cfg_on["detection"]["thermal_nep"] = 1.0e-11

        b        = compute_budget(cfg_on)
        measured = _injected_noise_std(cfg_on, cfg_off)
        assert measured == pytest.approx(b["sigma_thermal"], rel=0.15), \
            f"thermal {measured:.3e} A vs budget {b['sigma_thermal']:.3e} A"

    def test_dark_noise_matches_budget(self):
        cfg_off = _noiseless_cfg()
        cfg_off["fiber"]["length"] = 0.5
        cfg_on  = deepcopy(cfg_off)
        cfg_on["detection"]["dark_current"] = 1.0e-7

        b        = compute_budget(cfg_on)
        measured = _injected_noise_std(cfg_on, cfg_off)
        assert measured == pytest.approx(b["sigma_dark"], rel=0.15), \
            f"dark {measured:.3e} A vs budget {b['sigma_dark']:.3e} A"


class TestAttenuationSlope:
    """Reflectogram |H(z)|^2 must decay at -2*alpha dB/km (round-trip on power).

    Same baseline-subtraction trick as the noise tests: the pipeline has a
    small intrinsic |H|^2 slope (FFT edge / windowing) even at alpha=0;
    subtracting the alpha=0 baseline isolates the configured attenuation
    and the linear fit becomes tight (sub-percent).
    """

    @staticmethod
    def _reflectogram_dB(cfg):
        acq  = run_campaign(cfg)[0]
        beat = np.asarray(acq.analog_main[0], dtype=np.float64)
        H    = np.fft.fft(beat)
        nh   = len(H) // 2
        z    = np.arange(nh) * acq.dz
        return z, 10.0 * np.log10(np.abs(H[:nh]) ** 2 + 1e-30)

    def _fit_slope(self, alpha_dB_km, W_smooth=1000):
        cfg_off = _noiseless_cfg()
        cfg_off["fiber"]["length"] = 5.0
        cfg_off["fiber"]["attenuation_dB_per_km"] = 0.0
        cfg_off["fiber"]["rayleigh_coefficient_dB"] = -82.0     # normal Rayleigh
        cfg_on  = deepcopy(cfg_off)
        cfg_on["fiber"]["attenuation_dB_per_km"] = alpha_dB_km

        z, P_off = self._reflectogram_dB(cfg_off)
        _, P_on  = self._reflectogram_dB(cfg_on)

        # trim away the reflectogram edges (roll-off + boundary artefacts)
        mask = (z > 0.3) & (z < 4.7)
        dP   = (P_on - P_off)[mask]
        zm   = z[mask]

        # heavy boxcar smoothing to collapse speckle
        smooth = np.convolve(dP, np.ones(W_smooth) / W_smooth, mode="valid")
        z_fit  = zm[W_smooth // 2 : W_smooth // 2 + len(smooth)]

        slope_dB_per_m, _ = np.polyfit(z_fit, smooth, 1)
        return slope_dB_per_m

    def test_slope_matches_2alpha(self):
        # 100 dB/km one-way -> -200 dB/km on reflectogram power (round trip)
        alpha    = 100.0
        expected = -2.0 * alpha / 1000.0        # dB/m
        measured = self._fit_slope(alpha)
        assert measured == pytest.approx(expected, rel=0.01), \
            f"measured {measured:.4f} dB/m vs expected {expected:.4f} dB/m"

    def test_zero_attenuation_flat(self):
        # if alpha=0 on both sides the slope difference is zero to
        # numerical noise
        slope = self._fit_slope(0.0)
        assert abs(slope) < 1e-6


# ── Froggatt-Moore sensitivities (strain + temperature) ────────────

def _xcorr_df(cfg_ref, cfg_meas):
    """Run two pipelines and pull the local freq shift on the inside
    of the perturbed segment via windowed_xcorr_strain. Returns
    (df_measured_Hz, nu0_Hz, p_e)."""
    from pyofdr.analysis.demodulation import (fft_reflectogram,
                                              windowed_xcorr_strain)
    from pyofdr.utils.constants import C
    from pyofdr.utils.units import wavelength_range_to_freq_range

    acq_ref  = run_campaign(cfg_ref)[0]
    acq_meas = run_campaign(cfg_meas)[0]

    H_ref,  _ = fft_reflectogram(
        acq_ref.digital_main[0].astype(np.float64), acq_ref.dz)
    H_meas, _ = fft_reflectogram(
        acq_meas.digital_main[0].astype(np.float64), acq_meas.dz)

    wl       = cfg_ref["source"]["center_wavelength"]
    sweep_hz = wavelength_range_to_freq_range(wl, cfg_ref["source"]["sweep_range"])
    nu0      = C / wl
    p_e      = 0.22

    zc, eps_rec = windowed_xcorr_strain(
        H_meas, H_ref, acq_ref.dz,
        gauge_length=0.01, stride=2e-3,
        sweep_range_hz=sweep_hz, center_freq=nu0, p_e=p_e,
    )
    # 5 cm margin inside the [0.3, 0.7] segment avoids edge effects from
    # the gauge straddling the boundary
    inside = (zc > 0.38) & (zc < 0.62)
    df = -float(np.median(eps_rec[inside])) * nu0 * (1.0 - p_e)
    return df, nu0, p_e


def _strain_cfg(eps_true):
    cfg = _noiseless_cfg()
    cfg["fiber"]["length"] = 1.0
    cfg["fiber"]["rayleigh_coefficient_dB"] = -82.0   # normal Rayleigh
    cfg_ref = deepcopy(cfg)
    cfg_str = deepcopy(cfg)
    cfg_str["strain"] = {"segments":
        [{"start": 0.3, "end": 0.7, "epsilon": eps_true}]}
    return cfg_ref, cfg_str


def _temp_cfg(dT_true):
    cfg = _noiseless_cfg()
    cfg["fiber"]["length"] = 1.0
    cfg["fiber"]["rayleigh_coefficient_dB"] = -82.0
    cfg_ref = deepcopy(cfg)
    cfg_dT  = deepcopy(cfg)
    cfg_dT["temperature"] = {"segments":
        [{"start": 0.3, "end": 0.7, "delta_T": dT_true}]}
    return cfg_ref, cfg_dT


class TestStrainShiftSensitivity:
    """End-to-end check of df = d_nu/d_eps * eps with
    d_nu/d_eps = -(1-p_e)*nu_0. compute_budget exposes this; the
    simulator must reproduce it after FFT + windowed xcorr (issue #5)."""

    def test_uniform_strain_matches_budget(self):
        eps_true = 1.0e-3
        cfg_ref, cfg_str = _strain_cfg(eps_true)
        df_meas, _, _ = _xcorr_df(cfg_ref, cfg_str)
        df_expected   = compute_budget(cfg_ref)["d_nu_d_eps"] * eps_true
        assert df_meas == pytest.approx(df_expected, rel=0.05), \
            f"strain shift {df_meas:.3e} Hz vs budget {df_expected:.3e} Hz"

    def test_sign_flip_on_compression(self):
        # negative eps (compression) must flip the sign of df
        df_t,  _, _ = _xcorr_df(*_strain_cfg(+1.0e-3))
        df_c,  _, _ = _xcorr_df(*_strain_cfg(-1.0e-3))
        assert df_t * df_c < 0
        assert abs(df_t + df_c) < 0.1 * abs(df_t)   # symmetric to within 10 %


class TestTemperatureShiftSensitivity:
    """df = d_nu/d_T * dT with d_nu/d_T = -(alpha_L + xi)*nu_0.
    Same machinery as strain via the Froggatt-Moore cross-sensitivity
    (a piece of #75 closed here)."""

    def test_uniform_dT_matches_budget(self):
        # 50 K so the shift is comfortably above the gauge sub-bin
        # resolution (~10 GHz at gauge=1 cm, sweep=40 nm)
        dT_true = 50.0
        cfg_ref, cfg_dT = _temp_cfg(dT_true)
        df_meas, _, _ = _xcorr_df(cfg_ref, cfg_dT)
        df_expected   = compute_budget(cfg_ref)["d_nu_d_T"] * dT_true
        assert df_meas == pytest.approx(df_expected, rel=0.05), \
            f"temperature shift {df_meas:.3e} Hz vs budget {df_expected:.3e} Hz"
