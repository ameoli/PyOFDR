"""End-to-end analytical validation (issue #5).

These are *pipeline-level* checks: run a minimal simulation through
``run_campaign`` and compare the output statistics to what the analytical
budget predicts. Each test isolates one physical axis by turning off the
other noise sources / scatter contributions.

For noise-floor tests we run the pipeline twice (noise on / noise off)
and subtract the paired traces before measuring the standard deviation.
The deterministic fiber and source are identical, including for beat RIN.
"""

from __future__ import annotations

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


def _inner(sig):
    """Discard filter settling and sweep-edge transients."""
    return sig[len(sig) // 10:len(sig) - len(sig) // 10]


def _injected_noise_std(cfg_on, cfg_off):
    """std of noise that was added on top of the noiseless baseline."""
    acq_on  = run_campaign(cfg_on)[0]
    acq_off = run_campaign(cfg_off)[0]
    Z = cfg_on["adc"]["input_impedance"]
    difference = (acq_on.analog_main[0] - acq_off.analog_main[0]) / Z
    return float(np.std(_inner(difference)))


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
    """Settled pipeline RMS agrees within 3%, including receiver topology
    and digital-filter ENBW. Short runs still retain > 6000 independent
    noise samples in the narrowest test band."""

    @staticmethod
    def _cfg(balanced, bandwidth, order):
        cfg_off = _noiseless_cfg()
        cfg_off["fiber"]["length"] = 0.5
        cfg_off["fiber"]["rayleigh_coefficient_dB"] = -300.0
        cfg_off["source"]["sweep_duration"] = 0.002
        cfg_off["adc"]["sample_rate"] = 1e8
        cfg_off["detection"].update(balanced=balanced, bandwidth=bandwidth,
                                    filter_order=order)
        return cfg_off

    @pytest.mark.parametrize("balanced", [False, True])
    @pytest.mark.parametrize("bandwidth,order", [(2e6, 1), (20e6, 4), (80e6, 8)])
    @pytest.mark.parametrize("noise", ["shot", "thermal", "dark", "receiver"])
    def test_receiver_noise_matches_budget(self, balanced, bandwidth, order, noise):
        cfg_off = self._cfg(balanced, bandwidth, order)
        cfg_on = deepcopy(cfg_off)
        if noise in ("shot", "receiver"):
            cfg_on["detection"]["shot_noise"] = True
        if noise in ("thermal", "receiver"):
            cfg_on["detection"]["thermal_nep"] = 1e-11
        if noise in ("dark", "receiver"):
            cfg_on["detection"]["dark_current"] = 1e-7
        b = compute_budget(cfg_on)
        measured = _injected_noise_std(cfg_on, cfg_off)
        assert measured == pytest.approx(b[f"sigma_{noise}"], rel=0.03)

    @pytest.mark.parametrize("balanced", [False, True])
    @pytest.mark.parametrize("reflectance", [1e-4, 4e-4])
    @pytest.mark.parametrize("bandwidth", [2e6, 20e6])
    def test_beat_rin_matches_budget(self, balanced, reflectance, bandwidth):
        # A deterministic reflector makes the multiplicative RIN measurable;
        # its beat can lie outside the filter passband. Use PRE-filter RMS.
        cfg_off = self._cfg(balanced, bandwidth, 4)
        cfg_off["fiber"]["reflectors"] = [{"z": 0.3, "R": reflectance}]
        cfg_on = deepcopy(cfg_off)
        cfg_on["source"]["rin_dB_per_Hz"] = -110.0
        acq_off = run_campaign(cfg_off)[0]
        acq_on = run_campaign(cfg_on)[0]
        beat_current = acq_off.photocurrent_main[0] * cfg_off["detection"]["responsivity"]
        beat_rms = float(np.sqrt(np.mean(_inner(beat_current) ** 2)))
        b = compute_budget(cfg_on, beat_rms_current=beat_rms)
        Z = cfg_on["adc"]["input_impedance"]
        difference = (acq_on.analog_main[0] - acq_off.analog_main[0]) / Z
        assert np.std(_inner(difference)) == pytest.approx(b["sigma_rin"], rel=0.03)

    @pytest.mark.parametrize("balanced", [False, True])
    @pytest.mark.parametrize("enob", [None, 12.0])
    def test_sampled_total_with_rin_and_adc(self, balanced, enob):
        cfg_off = self._cfg(balanced, 10e6, 2)
        cfg_off["fiber"]["reflectors"] = [{"z": 0.3, "R": 1e-4}]
        cfg_on = deepcopy(cfg_off)
        cfg_on["source"]["rin_dB_per_Hz"] = -110.0
        cfg_on["detection"].update(shot_noise=True, thermal_nep=1e-9, dark_current=1e-7)
        cfg_on["adc"]["enob"] = enob
        acq_off = run_campaign(cfg_off)[0]
        acq_on = run_campaign(cfg_on)[0]
        beat_current = acq_off.photocurrent_main[0] * cfg_off["detection"]["responsivity"]
        beat_rms = float(np.sqrt(np.mean(_inner(beat_current) ** 2)))
        b = compute_budget(cfg_on, beat_rms_current=beat_rms)
        Z = cfg_on["adc"]["input_impedance"]
        analog_error = (acq_on.analog_main[0] - acq_off.analog_main[0]) / Z
        assert np.std(_inner(analog_error)) == pytest.approx(b["sigma_analog"], rel=0.03)
        # Subtract the analog baseline: quantizing BOTH traces would add a
        # second quantization error not present in the budget. Std removes
        # the half-LSB bias from the ADC's floor quantizer.
        lsb = cfg_on["adc"]["voltage_range"] / 2 ** cfg_on["adc"]["bits"]
        reconstructed = acq_on.digital_main[0].astype(float) * lsb
        digital_error = (reconstructed - acq_off.analog_main[0]) / Z
        assert np.std(_inner(digital_error)) == pytest.approx(b["sigma_total"], rel=0.03)


class TestAttenuationSlope:
    """Reflectogram |H(z)|^2 must decay at -2*alpha dB/km (round-trip on power).

    The pipeline has a
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
    p_e      = cfg_ref.get("strain", {}).get("photoelastic_coefficient", 0.22)

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

    @pytest.mark.parametrize("p_e", [0.22, 0.4])
    def test_uniform_strain_matches_budget(self, p_e):
        eps_true = 1.0e-3
        cfg_ref, cfg_str = _strain_cfg(eps_true)
        cfg_ref["strain"] = {"photoelastic_coefficient": p_e}
        cfg_str["strain"]["photoelastic_coefficient"] = p_e
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

    @pytest.mark.parametrize("alpha_L,xi", [(5.5e-7, 6.5e-6), (1e-6, 8e-6)])
    def test_uniform_dT_matches_budget(self, alpha_L, xi):
        # 50 K so the shift is comfortably above the gauge sub-bin
        # resolution (~10 GHz at gauge=1 cm, sweep=40 nm)
        dT_true = 50.0
        cfg_ref, cfg_dT = _temp_cfg(dT_true)
        cfg_ref["temperature"] = {"thermal_expansion": alpha_L, "thermo_optic": xi}
        cfg_dT["temperature"].update(thermal_expansion=alpha_L, thermo_optic=xi)
        df_meas, _, _ = _xcorr_df(cfg_ref, cfg_dT)
        df_expected   = compute_budget(cfg_ref)["d_nu_d_T"] * dT_true
        assert df_meas == pytest.approx(df_expected, rel=0.05), \
            f"temperature shift {df_meas:.3e} Hz vs budget {df_expected:.3e} Hz"
