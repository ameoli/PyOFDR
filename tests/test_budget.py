"""Tests for the analytical budget calculator (see #43)."""

import math
from copy import deepcopy

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from helpers import CFG, REPO_ROOT
from pyofdr.analysis.budget import compute_budget, print_budget
from pyofdr.core.config import load_config
from pyofdr.utils.constants import C


class TestBudget:

    def test_runs_on_basic_yaml(self):
        cfg = load_config(REPO_ROOT / "configs" / "ofdr_basic.yaml")
        b = compute_budget(cfg)
        # sanity: all the expected keys and they are finite
        for k in ("P_laser", "P_ref_arm", "P_to_fiber", "P_back_near",
                  "P_back_far", "I_dc_ref", "sigma_total", "nep_total",
                  "dynamic_range_dB"):
            assert k in b
            assert math.isfinite(b[k])

    def test_splitter_halves_power(self):
        cfg = dict(CFG)
        cfg["optics"] = {"splitting_ratio": 0.5}
        b = compute_budget(cfg)
        # 50/50 split, no circulator loss configured -> each arm gets half
        assert b["P_ref_arm"] == pytest.approx(0.5 * b["P_laser"])

    def test_far_end_backscatter_is_attenuated(self):
        # 10 km with 0.2 dB/km -> round-trip attenuation 4 dB
        cfg = dict(CFG)
        cfg["fiber"] = {"length": 10_000.0, "n_core": 1.4682,
                        "rayleigh_coefficient_dB": -82.0,
                        "attenuation_dB_per_km": 0.2}
        # stretch the sweep so we don't trip the Nyquist validator
        cfg["source"] = {"center_wavelength": 1550e-9, "sweep_range": 40e-9,
                         "sweep_duration": 1.0, "power": 10e-3}
        cfg["adc"] = {"sample_rate": 10e9, "bits": 16,
                      "voltage_range": 2.0, "input_impedance": 50.0}
        b = compute_budget(cfg)
        ratio_dB = 10 * math.log10(b["P_back_far"] / b["P_back_near"])
        assert ratio_dB == pytest.approx(-4.0, abs=0.05)

    def test_lossless_fiber_near_equals_far(self):
        cfg = dict(CFG)
        cfg["fiber"] = {"length": 1.0, "n_core": 1.4682,
                        "rayleigh_coefficient_dB": -82.0,
                        "attenuation_dB_per_km": 0.0}
        b = compute_budget(cfg)
        assert b["P_back_near"] == pytest.approx(b["P_back_far"])

    def test_total_noise_is_rss(self):
        cfg = load_config(REPO_ROOT / "configs" / "ofdr_basic.yaml")
        cfg["adc"]["enob"] = 12.0
        b = compute_budget(cfg)
        rss = math.sqrt(b["sigma_shot"]**2 + b["sigma_thermal"]**2
                        + b["sigma_dark"]**2 + b["sigma_rin"]**2
                        + b["sigma_quant"]**2 + b["sigma_adc_extra"]**2)
        assert b["sigma_total"] == pytest.approx(rss)

    def test_rin_adds_to_noise_with_known_beat(self):
        cfg = dict(CFG)
        b_no_rin = compute_budget(cfg)
        cfg2 = dict(CFG)
        cfg2["source"] = dict(CFG["source"])
        cfg2["source"]["rin_dB_per_Hz"] = -140.0
        b_rin = compute_budget(cfg2, beat_rms_current=1e-4)
        assert b_no_rin["sigma_rin"] == 0.0
        assert b_rin["sigma_rin"] > 0.0
        assert b_rin["sigma_total"] > b_no_rin["sigma_total"]

    def test_rin_requires_beat_rms_instead_of_dc_current(self, capsys):
        cfg = deepcopy(CFG)
        cfg["source"]["rin_dB_per_Hz"] = -140.0
        b = compute_budget(cfg)
        for key in ("sigma_rin", "sigma_analog", "sigma_total", "nep_total",
                    "dynamic_range_dB"):
            assert math.isnan(b[key])
        assert math.isfinite(b["sigma_receiver"])
        print_budget(cfg)
        assert "supply beat_rms_current" in capsys.readouterr().out

    @pytest.mark.parametrize("balanced", [False, True])
    def test_rin_is_multiplicative_in_both_modes(self, balanced):
        cfg = deepcopy(CFG)
        cfg["detection"]["balanced"] = balanced
        cfg["source"]["rin_dB_per_Hz"] = -120.0
        zero = compute_budget(cfg, beat_rms_current=0.0)
        weak = compute_budget(cfg, beat_rms_current=1e-5)
        strong = compute_budget(cfg, beat_rms_current=2e-5)
        assert zero["sigma_rin"] == 0.0
        assert strong["sigma_rin"] == pytest.approx(2 * weak["sigma_rin"])

    @pytest.mark.parametrize("rms", [-1.0, math.nan, math.inf])
    def test_invalid_beat_rms_rejected(self, rms):
        with pytest.raises(ValueError, match="beat_rms_current"):
            compute_budget(CFG, beat_rms_current=rms)

    def test_receiver_modes_and_shot_switch(self):
        cfg = deepcopy(CFG)
        single = compute_budget(cfg)
        cfg["detection"]["balanced"] = True
        balanced = compute_budget(cfg)
        assert single["I_dc_pd"] == pytest.approx(single["I_dc_ref"] / 2)
        assert balanced["I_dc_pd"] == single["I_dc_pd"]
        for noise in ("sigma_shot", "sigma_dark"):
            assert balanced[noise] == pytest.approx(math.sqrt(2) * single[noise])
        assert balanced["sigma_thermal"] == single["sigma_thermal"]
        cfg["detection"]["shot_noise"] = False
        assert compute_budget(cfg)["sigma_shot"] == 0.0

    @pytest.mark.parametrize("order, fraction", [(1, 0.001), (1, 0.25),
                                                (4, 0.1), (8, 0.8)])
    def test_noise_bandwidth_matches_filter_impulse_energy(self, order, fraction):
        # Parseval: integral_0^Nyquist |H|^2 df = fs/2 * sum(h[n]^2).
        # This tests the analytical integration independently of its formula.
        cfg = deepcopy(CFG)
        fs = cfg["adc"]["sample_rate"]
        cfg["detection"].update(bandwidth=fraction * fs, filter_order=order)
        b = compute_budget(cfg)
        cutoff = min(fraction * fs, 0.99 * fs / 2)
        impulse = np.zeros(100_000)
        impulse[0] = 1.0
        h = sosfilt(butter(order, cutoff / (fs / 2), output="sos"), impulse)
        assert b["filter_cutoff"] == cutoff
        assert b["noise_bandwidth"] == pytest.approx(fs / 2 * np.sum(h * h), rel=1e-7)
        assert 0 < b["noise_bandwidth"] < fs / 2

    def test_adc_noise_is_not_filtered_by_receiver(self):
        cfg = deepcopy(CFG)
        cfg["detection"].update(shot_noise=False, thermal_nep=0, dark_current=0,
                                bandwidth=1e6)
        b = compute_budget(cfg)
        expected = 2.0 / (2**16 * math.sqrt(12) * 50.0)
        assert b["sigma_quant"] == pytest.approx(expected)
        assert b["sigma_total"] == b["sigma_quant"]
        cfg["adc"]["enob"] = 12.0
        b = compute_budget(cfg)
        assert b["sigma_total"] == pytest.approx(16 * expected)
        assert b["sigma_adc_extra"] > 0

    def test_custom_material_coefficients(self):
        cfg = deepcopy(CFG)
        cfg["strain"] = {"photoelastic_coefficient": 0.4}
        cfg["temperature"] = {"thermal_expansion": 1e-6, "thermo_optic": 8e-6}
        b = compute_budget(cfg)
        nu = C / cfg["source"]["center_wavelength"]
        assert b["d_nu_d_eps"] == pytest.approx(-0.6 * nu)
        assert b["d_nu_d_T"] == pytest.approx(-9e-6 * nu)
        assert b["eps_max"] == pytest.approx(b["delta_nu"] / (2 * 0.6 * nu))
        cfg["temperature"] = {"thermal_expansion": 0.0, "thermo_optic": 0.0}
        assert compute_budget(cfg)["d_nu_d_T"] == 0.0

    def test_phase_noise_zero_for_coherent_source(self):
        cfg = dict(CFG)
        cfg["source"] = dict(CFG["source"])
        cfg["source"]["linewidth"] = 0.0
        b = compute_budget(cfg)
        assert b["sigma_phi_far"] == 0.0

    def test_phase_noise_scales_with_sqrt_length(self):
        # same config, two lengths -- sigma_phi ~ sqrt(L)
        base = dict(CFG)
        base["source"] = dict(CFG["source"])
        base["source"]["linewidth"] = 1e5  # 100 kHz

        cfg1 = dict(base); cfg1["fiber"] = dict(base["fiber"]); cfg1["fiber"]["length"] = 1.0
        cfg2 = dict(base); cfg2["fiber"] = dict(base["fiber"]); cfg2["fiber"]["length"] = 4.0
        b1 = compute_budget(cfg1)
        b2 = compute_budget(cfg2)
        assert b2["sigma_phi_far"] / b1["sigma_phi_far"] == pytest.approx(2.0, rel=1e-6)

    def test_strain_sensitivity_silica_1550(self):
        # (1 - p_e) * nu_c with p_e=0.22 at 1550 nm ~ 150.8 MHz/ustrain
        b = compute_budget(CFG)
        assert b["d_nu_d_eps"] < 0
        per_ustrain = abs(b["d_nu_d_eps"]) * 1e-6   # per microstrain, in Hz
        assert per_ustrain == pytest.approx(150.8e6, rel=1e-3)

    def test_temperature_sensitivity_silica_1550(self):
        # (alpha_L + xi) * nu_c ~ 1.36 GHz/K at 1550 nm
        b = compute_budget(CFG)
        assert b["d_nu_d_T"] < 0
        assert abs(b["d_nu_d_T"]) == pytest.approx(1.36e9, rel=5e-3)

    def test_max_strain_positive_finite(self):
        b = compute_budget(CFG)
        assert b["eps_max"] > 0
        assert math.isfinite(b["eps_max"])
        # sanity: for a 40 nm sweep at 1550 nm the ceiling is ~1.6%
        assert 5e-3 < b["eps_max"] < 5e-2

    def test_budget_exposes_geometric_quantities(self):
        b = compute_budget(CFG)
        for k in ("dz", "f_beat_max", "f_nyquist", "delta_nu"):
            assert k in b and math.isfinite(b[k])

    def test_print_budget_runs(self, capsys):
        print_budget(CFG)
        out = capsys.readouterr().out
        assert "PyOFDR" in out
        assert "Dynamic range" in out
        assert "Strain sensitivity" in out
        assert "Max |strain|" in out
