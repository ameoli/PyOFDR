"""Basic tests for the v0.1 pipeline.

Just checks that each step runs and produces reasonable output.
More rigorous analytical tests (beat frequency, Rayleigh statistics,
attenuation slope, etc) will come later once the physics is more
complete.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.acquisition import Acquisition
from core.campaign import run_campaign
from core.config import load_config
from core.config_models import RootConfig
from fiber.attenuation import round_trip_attenuation, dB_per_km_to_neper_per_m
from fiber.profile import FiberGenerator
from source.swept_laser import SweptLaser
from optics.mach_zehnder import MachZehnder
from detection.detector import Detector
from detection.filter import AntiAliasFilter
from digitizer.adc import ADC

REPO_ROOT = Path(__file__).parent.parent

# short fiber for fast tests
CFG = {
    "simulation": {"seed": 42},
    "fiber": {"length": 1.0, "n_core": 1.4682, "rayleigh_coefficient_dB": -82.0},
    "source": {"center_wavelength": 1550e-9, "sweep_range": 40e-9,
               "sweep_duration": 0.01, "power": 10e-3},
    "optics": {"splitting_ratio": 0.5},
    "detection": {"responsivity": 1.0},
    "adc": {"bits": 16, "sample_rate": 200e6, "voltage_range": 2.0,
            "input_impedance": 50.0},
}


class TestFiberGenerator:

    def test_profile_is_created(self):
        acq = FiberGenerator(CFG).process(Acquisition())
        assert acq.fiber_profile is not None

    def test_profile_is_complex(self):
        acq = FiberGenerator(CFG).process(Acquisition())
        assert acq.fiber_profile.dtype == np.complex128

    def test_z_starts_at_zero(self):
        acq = FiberGenerator(CFG).process(Acquisition())
        assert acq.z[0] == 0.0

    def test_spatial_resolution_positive(self):
        acq = FiberGenerator(CFG).process(Acquisition())
        assert acq.dz > 0

    def test_same_seed_same_profile(self):
        """Same seed should give identical results."""
        a = FiberGenerator(CFG).process(Acquisition())
        b = FiberGenerator(CFG).process(Acquisition())
        np.testing.assert_array_equal(a.fiber_profile, b.fiber_profile)

    def test_profile_has_correct_length(self):
        acq = FiberGenerator(CFG).process(Acquisition())
        assert len(acq.fiber_profile) == len(acq.z)

    def test_attenuation_envelope_exists(self):
        acq = FiberGenerator(CFG).process(Acquisition())
        assert acq.attenuation_envelope is not None
        assert len(acq.attenuation_envelope) == len(acq.z)

    def test_attenuation_starts_at_one(self):
        cfg = {**CFG, "fiber": {**CFG["fiber"], "attenuation_dB_per_km": 0.18}}
        acq = FiberGenerator(cfg).process(Acquisition())
        np.testing.assert_allclose(acq.attenuation_envelope[0], 1.0)

    def test_attenuation_decays(self):
        cfg = {**CFG, "fiber": {**CFG["fiber"], "attenuation_dB_per_km": 0.18}}
        acq = FiberGenerator(cfg).process(Acquisition())
        # should decrease monotonically
        assert np.all(np.diff(acq.attenuation_envelope) < 0)

    def test_zero_attenuation_gives_flat_envelope(self):
        cfg = {**CFG, "fiber": {**CFG["fiber"], "attenuation_dB_per_km": 0.0}}
        acq = FiberGenerator(cfg).process(Acquisition())
        np.testing.assert_array_equal(acq.attenuation_envelope,
                                       np.ones_like(acq.attenuation_envelope))


class TestSweptLaser:

    def _make_acq(self):
        acq = FiberGenerator(CFG).process(Acquisition())
        return SweptLaser(CFG).process(acq)

    def test_time_axis_exists(self):
        acq = self._make_acq()
        assert acq.t is not None
        assert acq.n_samples > 0

    def test_field_is_complex(self):
        acq = self._make_acq()
        assert acq.E_source.dtype == np.complex128

    def test_frequency_increases_monotonically(self):
        acq = self._make_acq()
        dnu = np.diff(acq.nu_inst)
        assert np.all(dnu > 0)

    def test_optical_power_matches_config(self):
        acq = self._make_acq()
        P = np.mean(np.abs(acq.E_source) ** 2)
        np.testing.assert_allclose(P, 10e-3, rtol=0.01)


class TestMachZehnder:

    def _make_acq(self):
        acq = Acquisition()
        acq = FiberGenerator(CFG).process(acq)
        acq = SweptLaser(CFG).process(acq)
        return MachZehnder(CFG).process(acq)

    def test_photocurrent_exists(self):
        acq = self._make_acq()
        assert acq.photocurrent_main is not None
        assert len(acq.photocurrent_main) == acq.n_samples

    def test_photocurrent_is_real(self):
        acq = self._make_acq()
        # should be float, not complex
        assert acq.photocurrent_main.dtype == np.float64


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

    def test_nyquist_violation_is_rejected(self):
        # 1 km fiber + 1 MHz ADC -> beat blows past Nyquist
        with pytest.raises(Exception):
            RootConfig(
                fiber={"length": "1 km"},
                adc={"sample_rate": "1 MHz"},
            )


class TestEndToEnd:

    def test_run_campaign(self):
        acq = run_campaign(CFG)
        assert acq.digital_main is not None
        assert acq.z is not None

    def test_reflectogram_has_energy_in_fiber_region(self):
        """The FFT should show most energy in the first N_z bins."""
        acq = run_campaign(CFG)
        spectrum = np.fft.fft(acq.digital_main.astype(np.float64))
        n_half = len(spectrum) // 2
        mag = np.abs(spectrum[:n_half])

        n_z = len(acq.z)
        energy_fiber = np.sum(mag[:n_z] ** 2)
        energy_total = np.sum(mag ** 2)

        # most of the energy should be in the fiber region
        assert energy_fiber / energy_total > 0.9
