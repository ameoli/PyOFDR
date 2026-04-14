"""Tests for detector, balanced detector, and anti-alias filter."""

import numpy as np
import pytest

from helpers import CFG
from core.acquisition import Acquisition
from core.campaign import run_campaign
from fiber.profile import FiberGenerator
from source.swept_laser import SweptLaser
from optics.mach_zehnder import MachZehnder
from detection.detector import Detector
from detection.filter import AntiAliasFilter


class TestDetector:

    def _run_detector(self, **det_overrides):
        cfg = {**CFG, "detection": {**CFG["detection"], **det_overrides}}
        acq = Acquisition()
        for step_cls in [FiberGenerator, SweptLaser, MachZehnder, Detector]:
            acq = step_cls(cfg).process(acq)
        return acq

    def test_analog_output_exists(self):
        acq = self._run_detector()
        assert acq.analog_main is not None

    def test_shot_noise_increases_variance(self):
        """Signal with shot noise should have more variance than without."""
        acq_quiet = self._run_detector(shot_noise=False, thermal_nep=0, dark_current=0)
        acq_noisy = self._run_detector(shot_noise=True, thermal_nep=0, dark_current=0)
        assert np.var(acq_noisy.analog_main) > np.var(acq_quiet.analog_main)

    def test_thermal_noise_increases_variance(self):
        acq_quiet = self._run_detector(shot_noise=False, thermal_nep=0, dark_current=0)
        acq_noisy = self._run_detector(shot_noise=False, thermal_nep=1e-11, dark_current=0)
        assert np.var(acq_noisy.analog_main) > np.var(acq_quiet.analog_main)

    def test_no_noise_gives_deterministic_output(self):
        """With all noise off, same seed should give identical output."""
        a = self._run_detector(shot_noise=False, thermal_nep=0, dark_current=0)
        b = self._run_detector(shot_noise=False, thermal_nep=0, dark_current=0)
        np.testing.assert_array_equal(a.analog_main, b.analog_main)

    def test_toggling_shot_noise_does_not_disturb_thermal_stream(self):
        # regression for #22: each noise type has its own rng now, so flipping
        # shot_noise off must not reshuffle the thermal samples.
        def thermal_only(shot_flag):
            cfg = {**CFG, "detection": {"responsivity": 1.0,
                                         "shot_noise": shot_flag,
                                         "thermal_nep": 1e-11,
                                         "dark_current": 0}}
            acq = Acquisition()
            acq = FiberGenerator(cfg).process(acq)
            acq = SweptLaser(cfg).process(acq)
            acq = MachZehnder(cfg).process(acq)
            acq.photocurrent_main = np.zeros_like(acq.photocurrent_main)
            return Detector(cfg).process(acq).analog_main

        np.testing.assert_array_equal(thermal_only(False), thermal_only(True))

    def test_dark_current_adds_noise_even_with_zero_signal(self):
        """Dark current noise should be present even if photocurrent is zero."""
        cfg = {**CFG, "detection": {"responsivity": 1.0, "shot_noise": False,
                                     "thermal_nep": 0, "dark_current": 1e-6}}
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        acq = SweptLaser(cfg).process(acq)
        acq = MachZehnder(cfg).process(acq)
        # zero out the photocurrent to isolate dark current
        acq.photocurrent_main = np.zeros_like(acq.photocurrent_main)
        acq = Detector(cfg).process(acq)
        # should not be all zeros -- dark current adds noise
        assert np.any(acq.analog_main != 0)


class TestBalancedDetector:

    def _run(self, balanced, **det_kw):
        cfg = {**CFG, "detection": {**CFG["detection"],
                                     "balanced": balanced, **det_kw}}
        acq = Acquisition()
        for cls in [FiberGenerator, SweptLaser, MachZehnder, Detector]:
            acq = cls(cfg).process(acq)
        return acq

    def test_balanced_false_matches_legacy(self):
        a = self._run(False, shot_noise=False, thermal_nep=0, dark_current=0)
        b = self._run(False, shot_noise=False, thermal_nep=0, dark_current=0)
        np.testing.assert_array_equal(a.analog_main, b.analog_main)

    def test_balanced_noiseless_same_signal(self):
        # no noise -> balanced and single should give identical signal
        a = self._run(False, shot_noise=False, thermal_nep=0, dark_current=0)
        b = self._run(True,  shot_noise=False, thermal_nep=0, dark_current=0)
        np.testing.assert_array_equal(a.analog_main, b.analog_main)

    def test_balanced_reduces_thermal_noise(self):
        # balanced halves each noise realisation (a-b)/2 -> variance is 1/2
        single = self._run(False, shot_noise=False,
                            thermal_nep=1e-9, dark_current=0)
        bal    = self._run(True, shot_noise=False,
                            thermal_nep=1e-9, dark_current=0)
        clean  = self._run(False, shot_noise=False,
                            thermal_nep=0, dark_current=0)

        noise_single = single.analog_main  - clean.analog_main
        noise_bal    = bal.analog_main - clean.analog_main
        assert np.var(noise_bal) < np.var(noise_single)

    def test_balanced_shot_noise_runs(self):
        # smoke test -- balanced shot noise uses DC current, just make sure
        # it doesn't crash
        bal = self._run(True, shot_noise=True, thermal_nep=0, dark_current=0)
        assert bal.analog_main is not None

    def test_balanced_full_pipeline(self):
        cfg = {**CFG, "detection": {**CFG["detection"], "balanced": True}}
        acq = run_campaign(cfg)[-1]
        assert acq.digital_main is not None


class TestAntiAliasFilter:

    def _run_up_to_filter(self, bandwidth=1e8):
        cfg = {**CFG, "detection": {**CFG["detection"], "bandwidth": bandwidth,
                                     "shot_noise": False, "thermal_nep": 0,
                                     "dark_current": 0}}
        acq = Acquisition()
        for cls in [FiberGenerator, SweptLaser, MachZehnder, Detector]:
            acq = cls(cfg).process(acq)
        return acq, cfg

    def test_filter_does_not_change_shape(self):
        acq, cfg = self._run_up_to_filter()
        n_before = len(acq.analog_main)
        acq = AntiAliasFilter(cfg).process(acq)
        assert len(acq.analog_main) == n_before

    def test_filter_reduces_high_freq_content(self):
        """Narrow bandwidth should cut high frequency noise."""
        acq_wide, cfg_wide = self._run_up_to_filter(bandwidth=1e8)
        acq_narrow, cfg_narrow = self._run_up_to_filter(bandwidth=1e7)

        acq_wide = AntiAliasFilter(cfg_wide).process(acq_wide)
        acq_narrow = AntiAliasFilter(cfg_narrow).process(acq_narrow)

        # narrower filter -> less high freq content -> smaller variance
        assert np.var(acq_narrow.analog_main) < np.var(acq_wide.analog_main)


class TestSaturation:

    def _run(self, sat=None, power=10e-3, **det_kw):
        cfg = {**CFG,
               "source":    {**CFG["source"], "power": power},
               "detection": {**CFG["detection"],
                              "shot_noise": False, "thermal_nep": 0,
                              "dark_current": 0,
                              "saturation_current": sat, **det_kw}}
        acq = Acquisition()
        for cls in [FiberGenerator, SweptLaser, MachZehnder, Detector]:
            acq = cls(cfg).process(acq)
        return acq

    def test_no_sat_matches_baseline(self):
        a = self._run(None)
        b = self._run(None)
        np.testing.assert_array_equal(a.analog_main, b.analog_main)

    def test_high_sat_is_noop(self):
        # clamp well above anything we'd ever see
        a = self._run(None)
        b = self._run(1.0)    # 1 A, nothing close
        np.testing.assert_array_equal(a.analog_main, b.analog_main)

    def test_low_sat_clips_voltage(self):
        # strong signal + tight clamp -> output should be bounded
        acq = self._run(sat=1e-6, power=1.0)
        Z = CFG["adc"]["input_impedance"]
        # V = I * Z, |I| <= I_sat
        assert np.max(np.abs(acq.analog_main)) <= 1e-6 * Z + 1e-12

    def test_sat_changes_signal(self):
        clean  = self._run(None,   power=1.0)
        capped = self._run(1e-6,   power=1.0)
        assert not np.array_equal(clean.analog_main, capped.analog_main)
