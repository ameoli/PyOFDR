"""Tests for config validation and unit parsing."""

import numpy as np
import pytest

from helpers import CFG, REPO_ROOT
from core.config import load_config
from core.config_models import RootConfig


class TestConfigValidation:

    def test_load_basic_config(self):
        cfg = load_config(REPO_ROOT / "configs" / "ofdr_basic.yaml")
        assert cfg["fiber"]["length"] == 10.0
        assert cfg["simulation"]["backend"] == "numpy"

    def test_typo_in_field_is_rejected(self):
        with pytest.raises(Exception):
            RootConfig(fiber={"lenght": 10.0})

    def test_negative_length_is_rejected(self):
        with pytest.raises(Exception):
            RootConfig(fiber={"length": -1.0})

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(Exception):
            RootConfig(simulation={"backend": "tensorflow"})

    def test_empty_config_uses_defaults(self):
        cfg = RootConfig().model_dump()
        assert cfg["adc"]["bits"] == 16
        assert cfg["source"]["power"] == 10e-3

    def test_nyquist_violation_is_rejected(self):
        # 1 km fiber + 1 MHz ADC -> beat blows past Nyquist
        with pytest.raises(Exception):
            RootConfig(
                fiber={"length": "1 km"},
                adc={"sample_rate": "1 MHz"},
            )


class TestUnitParsing:

    def test_wavelength_in_nm(self):
        cfg = RootConfig(source={"center_wavelength": "1550 nm"})
        assert cfg.source.center_wavelength == pytest.approx(1.55e-6)

    def test_sweep_range_in_nm(self):
        cfg = RootConfig(source={"sweep_range": "40 nm"})
        assert cfg.source.sweep_range == pytest.approx(4e-8)

    def test_sweep_duration_in_ms(self):
        cfg = RootConfig(source={"sweep_duration": "10 ms"})
        assert cfg.source.sweep_duration == pytest.approx(0.01)

    def test_power_in_mW(self):
        cfg = RootConfig(source={"power": "10 mW"})
        assert cfg.source.power == pytest.approx(0.01)

    def test_sample_rate_in_MHz(self):
        cfg = RootConfig(adc={"sample_rate": "200 MHz"})
        assert cfg.adc.sample_rate == pytest.approx(2e8)

    def test_fiber_length_in_km(self):
        # bump sample rate so we don't trip the Nyquist validator
        cfg = RootConfig(
            fiber={"length": "1 km"},
            adc={"sample_rate": "10 GHz"},
        )
        assert cfg.fiber.length == pytest.approx(1000.0)

    def test_bare_float_still_works(self):
        # backwards compat: existing YAMLs use bare numbers
        cfg = RootConfig(source={"center_wavelength": 1.55e-6})
        assert cfg.source.center_wavelength == 1.55e-6

    def test_wrong_dimension_is_rejected(self):
        # mass instead of length
        with pytest.raises(Exception):
            RootConfig(fiber={"length": "10 kg"})
