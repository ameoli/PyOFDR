"""Tests for the analytical budget calculator (see #43)."""

import math

import pytest

from helpers import CFG, REPO_ROOT
from analysis.budget import compute_budget, print_budget
from core.config import load_config


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
                        + b["sigma_dark"]**2 + b["sigma_quant"]**2)
        assert b["sigma_total"] == pytest.approx(rss)

    def test_print_budget_runs(self, capsys):
        print_budget(CFG)
        out = capsys.readouterr().out
        assert "PyOFDR" in out
        assert "Dynamic range" in out
