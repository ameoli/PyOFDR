"""Tests for ADC, jitter, and ENOB."""

import numpy as np
import pytest

from helpers import CFG
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.campaign import run_campaign
from pyofdr.core.config_models import RootConfig
from pyofdr.fiber.profile import FiberGenerator
from pyofdr.source.swept_laser import SweptLaser
from pyofdr.optics.mach_zehnder import MachZehnder
from pyofdr.detection.detector import Detector
from pyofdr.detection.filter import AntiAliasFilter
from pyofdr.digitizer.adc import ADC


class TestADC:

    def _run_full(self):
        acq = Acquisition()
        for cls in [FiberGenerator, SweptLaser, MachZehnder, Detector,
                    AntiAliasFilter, ADC]:
            acq = cls(CFG).process(acq)
        return acq

    def test_digital_is_int16(self):
        acq = self._run_full()
        assert acq.digital_main.dtype == np.int16

    def test_digital_in_range(self):
        acq = self._run_full()
        assert np.all(acq.digital_main >= -32768)
        assert np.all(acq.digital_main <= 32767)

    def test_wide_adc_does_not_wrap(self):
        # 18-bit code 117964 used to come out as -13108 through int16 (#89)
        cfg = {**CFG, "adc": {**CFG["adc"], "bits": 18}}
        adc = ADC(cfg)
        acq = Acquisition()
        acq.analog_main = np.full((1, 64), 0.9 * adc.v_max)
        out = adc.process(acq)
        assert out.digital_main.dtype == np.int32
        assert np.all(out.digital_main == 117964)


class TestADCJitter:

    def _cfg(self, **adc_overrides):
        cfg = {**CFG, "adc": {**CFG["adc"], **adc_overrides}}
        cfg["detection"] = {**CFG["detection"],  "shot_noise": False,
                             "thermal_nep": 0, "dark_current": 0}
        return cfg

    def test_zero_jitter_matches_baseline(self):
        a = run_campaign(self._cfg())[-1]
        b = run_campaign(self._cfg(jitter_rms=0.0))[-1]
        np.testing.assert_array_equal(a.digital_main, b.digital_main)

    def test_jitter_increases_noise(self):
        # need strong signal so dV/dt * sigma_j >> quantization step
        cfg_base = {**self._cfg(), "source": {**CFG["source"], "power": 1.0}}
        cfg_jit  = {**cfg_base,  "adc": {**cfg_base["adc"], "jitter_rms": 1e-9}}
        clean = run_campaign(cfg_base)[-1]
        noisy = run_campaign(cfg_jit)[-1]
        assert np.var(noisy.digital_main.astype(np.float64)) > \
               np.var(clean.digital_main.astype(np.float64))

    def test_more_jitter_means_more_noise(self):
        v_lo = np.var(run_campaign(self._cfg(jitter_rms=10e-12 ))[-1].digital_main.astype(np.float64))
        v_hi = np.var(run_campaign(self._cfg(jitter_rms=500e-12))[-1].digital_main.astype(np.float64))
        assert v_hi > v_lo

    def test_jitter_with_dc_input_is_silent(self):
        # dV/dt=0 for constant signal -> jitter should add nothing
        cfg = self._cfg(jitter_rms=1e-9)
        acq = Acquisition()
        acq.dt = 1.0 / cfg["adc"]["sample_rate"]
        acq.analog_main = np.ones((1, 50000), dtype=np.float64) * 0.1
        acq.sweep_index = 0
        snap = acq.analog_main.copy()
        ADC(cfg).process(acq)

        acq2 = Acquisition()
        acq2.dt  = acq.dt
        acq2.analog_main = snap
        acq2.sweep_index = 0
        ADC(self._cfg(jitter_rms=0)).process(acq2)
        np.testing.assert_array_equal(acq.digital_main, acq2.digital_main)

    def test_jitter_unit_string_accepted(self):
        from pyofdr.core.config_models import RootConfig
        cfg = RootConfig(adc={"jitter_rms": "50 ps"})
        assert cfg.adc.jitter_rms == pytest.approx(50e-12)


class TestADCEnob:

    def _cfg(self, **adc_overrides):
        cfg = {**CFG, "adc": {**CFG["adc"], **adc_overrides}}
        # turn off all detector noise so the only noise floor we measure
        # is whatever the ADC adds
        cfg["detection"] = {**CFG["detection"], "shot_noise": False,
                             "thermal_nep": 0, "dark_current": 0}
        return cfg

    def test_enob_unset_matches_legacy(self):
        a = run_campaign(self._cfg())[-1]
        b = run_campaign(self._cfg(enob=None))[-1]
        np.testing.assert_array_equal(a.digital_main, b.digital_main)

    def test_enob_equal_to_bits_is_noop(self):
        a = run_campaign(self._cfg())[-1]
        b = run_campaign(self._cfg(enob=16))[-1]
        np.testing.assert_array_equal(a.digital_main, b.digital_main)

    def test_enob_below_bits_increases_noise(self):
        # quiet baseline (no detector noise) -> any extra variance is from ENOB
        clean = run_campaign(self._cfg())[-1]
        noisy = run_campaign(self._cfg(enob=10))[-1]
        assert np.var(noisy.digital_main.astype(np.float64)) > \
               np.var(clean.digital_main.astype(np.float64))

    def test_enob_lower_means_more_noise(self):
        v8  = np.var(run_campaign(self._cfg(enob=8 ))[-1].digital_main.astype(np.float64))
        v12 = np.var(run_campaign(self._cfg(enob=12))[-1].digital_main.astype(np.float64))
        assert v8 > v12

    def test_enob_above_bits_is_rejected(self):
        with pytest.raises(ValueError):
            ADC(self._cfg(bits=12, enob=14))

    def test_enob_zero_is_rejected(self):
        with pytest.raises(ValueError):
            ADC(self._cfg(enob=0))

    def test_enob_noise_floor_matches_theory(self):
        # feed a known DC voltage, see whether the digital noise variance
        # matches sigma_total^2 = (V_range / (2^enob * sqrt(12)))^2
        cfg = self._cfg(enob=10)
        acq = Acquisition()
        acq.dt = 1.0 / cfg["adc"]["sample_rate"]
        # 1 core, 50000 samples, 0V analog input
        acq.analog_main = np.zeros((1, 50000), dtype=np.float64)
        acq.sweep_index = 0
        ADC(cfg).process(acq)
        # convert digital code back to voltage
        v_lsb = cfg["adc"]["voltage_range"] / 2 ** cfg["adc"]["bits"]
        v = acq.digital_main.astype(np.float64) * v_lsb
        sigma_obs = np.std(v)
        sigma_th  = cfg["adc"]["voltage_range"] / (2 ** 10 * np.sqrt(12))
        np.testing.assert_allclose(sigma_obs, sigma_th, rtol=0.05)

    def test_pydantic_rejects_enob_above_bits(self):
        with pytest.raises(Exception):
            RootConfig(adc={"bits": 12, "enob": 14})


class TestADCNonlinearity:

    def _cfg(self, **adc_overrides):
        cfg = {**CFG, "adc": {**CFG["adc"], **adc_overrides}}
        cfg["detection"] = {**CFG["detection"], "shot_noise": False,
                             "thermal_nep": 0, "dark_current": 0}
        return cfg

    def test_zero_nl_matches_baseline(self):
        a = run_campaign(self._cfg())[-1]
        b = run_campaign(self._cfg(dnl_rms_lsb=0.0, inl_peak_lsb=0.0))[-1]
        np.testing.assert_array_equal(a.digital_main, b.digital_main)

    def test_inl_changes_output(self):
        clean = run_campaign(self._cfg())[-1]
        bent  = run_campaign(self._cfg(inl_peak_lsb=20.0))[-1]
        assert not np.array_equal(clean.digital_main, bent.digital_main)

    def test_dnl_changes_output(self):
        clean = run_campaign(self._cfg())[-1]
        bent  = run_campaign(self._cfg(dnl_rms_lsb=0.5))[-1]
        assert not np.array_equal(clean.digital_main, bent.digital_main)

    def test_nl_is_deterministic(self):
        # DNL realisation is seeded from sim seed, so two runs match
        a = run_campaign(self._cfg(dnl_rms_lsb=0.5))[-1]
        b = run_campaign(self._cfg(dnl_rms_lsb=0.5))[-1]
        np.testing.assert_array_equal(a.digital_main, b.digital_main)

    def test_digital_still_in_range(self):
        acq = run_campaign(self._cfg(dnl_rms_lsb=1.0, inl_peak_lsb=50.0))[-1]
        assert np.all(acq.digital_main >= -32768)
        assert np.all(acq.digital_main <= 32767)
