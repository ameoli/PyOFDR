"""Tests for the analytical budget calculator (see #43)."""

import math

import pytest

from helpers import CFG, REPO_ROOT
from pyofdr.analysis.budget import compute_budget, print_budget
from pyofdr.core.config import load_config


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
        b = compute_budget(cfg)
        rss = math.sqrt(b["sigma_shot"]**2 + b["sigma_thermal"]**2
                        + b["sigma_dark"]**2 + b["sigma_rin"]**2
                        + b["sigma_quant"]**2)
        assert b["sigma_total"] == pytest.approx(rss)

    def test_rin_adds_to_noise(self):
        cfg = dict(CFG)
        b_no_rin = compute_budget(cfg)
        cfg2 = dict(CFG)
        cfg2["source"] = dict(CFG["source"])
        cfg2["source"]["rin_dB_per_Hz"] = -140.0
        b_rin = compute_budget(cfg2)
        assert b_no_rin["sigma_rin"] == 0.0
        assert b_rin["sigma_rin"] > 0.0
        assert b_rin["sigma_total"] > b_no_rin["sigma_total"]

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
