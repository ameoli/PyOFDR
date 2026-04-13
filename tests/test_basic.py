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
from strain_transfer import CoxShearLag, IdealTransfer
from utils.seeding import derive_seed
from fiber.attenuation import round_trip_attenuation, dB_per_km_to_neper_per_m
from fiber.profile import FiberGenerator
from fiber.reflectors import apply_connector_losses, inject_reflectors
from fiber.strain import StrainPerturbation
from source.swept_laser import SweptLaser
from optics.mach_zehnder import MachZehnder
from optics.components import Circulator
from detection.detector import Detector
from detection.filter import AntiAliasFilter
from digitizer.adc import ADC
from output.hdf5_writer import HDF5Writer

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
        assert acq.fiber_profile.shape[-1] == len(acq.z)

    def test_profile_is_2d_with_core_axis(self):
        # leading axis is the core index, n_cores = 1 for now (#14)
        acq = FiberGenerator(CFG).process(Acquisition())
        assert acq.fiber_profile.ndim == 2
        assert acq.fiber_profile.shape[0] == 1

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

    def test_second_call_populates_fresh_acq(self):
        """Regression: _done cache used to early-return without setting
        fields on the new Acquisition, breaking multi-sweep (#20)."""
        gen = FiberGenerator(CFG)
        acq1 = gen.process(Acquisition())
        acq2 = gen.process(Acquisition())  # fresh acq, same step
        assert acq2.fiber_profile is not None
        assert acq2.z is not None
        np.testing.assert_array_equal(acq1.fiber_profile, acq2.fiber_profile)


class TestDiscreteReflectors:

    def test_no_reflectors_unchanged(self):
        """Empty reflector list should not change the profile."""
        acq = FiberGenerator(CFG).process(Acquisition())
        cfg2 = {**CFG, "fiber": {**CFG["fiber"], "reflectors": []}}
        acq2 = FiberGenerator(cfg2).process(Acquisition())
        np.testing.assert_array_equal(acq.fiber_profile, acq2.fiber_profile)

    def test_reflector_adds_peak(self):
        ref_z = 0.5   # midpoint of the 1m fiber
        R = 0.04      # 4% Fresnel reflection
        cfg = {**CFG, "fiber": {**CFG["fiber"],
               "reflectors": [{"z": ref_z, "R": R}]}}
        acq_ref = FiberGenerator(cfg).process(Acquisition())
        acq_bare = FiberGenerator(CFG).process(Acquisition())

        # power at the reflector bin should be much larger
        idx = int(round(ref_z / acq_ref.dz))
        power_ref = np.abs(acq_ref.fiber_profile[0, idx]) ** 2
        power_bare = np.abs(acq_bare.fiber_profile[0, idx]) ** 2
        assert power_ref > power_bare

    def test_reflector_outside_fiber_ignored(self):
        """Reflector beyond the fiber length should not crash."""
        cfg = {**CFG, "fiber": {**CFG["fiber"],
               "reflectors": [{"z": 999.0, "R": 0.01}]}}
        acq = FiberGenerator(cfg).process(Acquisition())
        assert acq.fiber_profile is not None

    def test_reflector_amplitude(self):
        """Check that injected amplitude matches sqrt(R)."""
        R = 0.09
        z_pos = 0.3
        cfg = {**CFG, "fiber": {**CFG["fiber"],
               "reflectors": [{"z": z_pos, "R": R}],
               "rayleigh_coefficient_dB": -300}}  # negligible Rayleigh
        acq = FiberGenerator(cfg).process(Acquisition())
        idx = int(round(z_pos / acq.dz))
        amp = np.abs(acq.fiber_profile[0, idx])
        np.testing.assert_allclose(amp, np.sqrt(R), atol=1e-10)

    def test_multiple_reflectors(self):
        refs = [{"z": 0.2, "R": 0.01}, {"z": 0.6, "R": 0.05}]
        cfg = {**CFG, "fiber": {**CFG["fiber"], "reflectors": refs,
               "rayleigh_coefficient_dB": -300}}
        acq = FiberGenerator(cfg).process(Acquisition())
        for ref in refs:
            idx = int(round(ref["z"] / acq.dz))
            amp = np.abs(acq.fiber_profile[0, idx])
            np.testing.assert_allclose(amp, np.sqrt(ref["R"]), atol=1e-10)

    def test_connector_loss_creates_step(self):
        """A connector with insertion loss should reduce the attenuation
        envelope beyond its position."""
        loss = 0.5  # dB one-way
        cfg_bare = {**CFG, "fiber": {**CFG["fiber"],
                    "attenuation_dB_per_km": 0.0}}
        cfg_loss = {**CFG, "fiber": {**CFG["fiber"],
                    "attenuation_dB_per_km": 0.0,
                    "reflectors": [{"z": 0.5, "R": 0.0, "loss_dB": loss}]}}
        acq_bare = FiberGenerator(cfg_bare).process(Acquisition())
        acq_loss = FiberGenerator(cfg_loss).process(Acquisition())

        idx = int(round(0.5 / acq_loss.dz))
        # before connector: same
        np.testing.assert_allclose(
            acq_loss.attenuation_envelope[:idx],
            acq_bare.attenuation_envelope[:idx])
        # after connector: step down
        expected_factor = 10.0 ** (-loss / 10.0)
        np.testing.assert_allclose(
            acq_loss.attenuation_envelope[idx:],
            acq_bare.attenuation_envelope[idx:] * expected_factor)

    def test_two_connectors_cascade(self):
        """Two lossy connectors should accumulate."""
        loss = 0.3
        refs = [{"z": 0.3, "R": 0.0, "loss_dB": loss},
                {"z": 0.7, "R": 0.0, "loss_dB": loss}]
        cfg = {**CFG, "fiber": {**CFG["fiber"],
               "attenuation_dB_per_km": 0.0, "reflectors": refs}}
        acq = FiberGenerator(cfg).process(Acquisition())
        factor = 10.0 ** (-loss / 10.0)
        # after 2nd connector: two steps
        idx2 = int(round(0.7 / acq.dz))
        np.testing.assert_allclose(
            acq.attenuation_envelope[idx2], factor ** 2, rtol=1e-10)

    def test_zero_loss_is_noop(self):
        """loss_dB=0 should not change anything."""
        cfg = {**CFG, "fiber": {**CFG["fiber"],
               "reflectors": [{"z": 0.5, "R": 0.04, "loss_dB": 0.0}]}}
        acq = FiberGenerator(cfg).process(Acquisition())
        cfg_bare = {**CFG, "fiber": {**CFG["fiber"],
                    "reflectors": [{"z": 0.5, "R": 0.04}]}}
        acq_bare = FiberGenerator(cfg_bare).process(Acquisition())
        np.testing.assert_array_equal(
            acq.attenuation_envelope, acq_bare.attenuation_envelope)


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


class TestPhaseNoise:

    def _run(self, linewidth):
        cfg = {**CFG, "source": {**CFG["source"], "linewidth": linewidth}}
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        return SweptLaser(cfg).process(acq)

    def test_zero_linewidth_matches_noiseless(self):
        # backwards compat: linewidth=0 must give the old result
        a = self._run(0.0)
        b = SweptLaser(CFG).process(FiberGenerator(CFG).process(Acquisition()))
        np.testing.assert_array_equal(a.E_source, b.E_source)

    def test_nonzero_linewidth_changes_field(self):
        a = self._run(0.0)
        b = self._run(1e5)
        assert not np.array_equal(a.E_source, b.E_source)

    def test_power_unchanged_by_phase_noise(self):
        # |E|^2 must stay equal to P regardless of phi noise
        acq = self._run(1e6)
        P = np.mean(np.abs(acq.E_source) ** 2)
        np.testing.assert_allclose(P, 10e-3, rtol=0.01)

    def test_phase_increment_variance_matches_theory(self):
        # var(d phi_noise) ~ 2*pi*lw*dt. Cancel the optical carrier by
        # dividing the noisy field by the noiseless one, then unwrap.
        lw = 1e6
        a = self._run(lw)
        b = self._run(0.0)
        phi_noise = np.unwrap(np.angle(a.E_source / b.E_source))
        dphi = np.diff(phi_noise)
        expected = 2.0 * np.pi * lw * a.dt
        np.testing.assert_allclose(np.var(dphi), expected, rtol=0.05)

    def test_same_seed_same_noise(self):
        a = self._run(1e5)
        b = self._run(1e5)
        np.testing.assert_array_equal(a.E_source, b.E_source)


class TestRIN:

    def _run(self, rin_dB=None):
        cfg = {**CFG, "source": {**CFG["source"], "rin_dB_per_Hz": rin_dB}}
        acq = FiberGenerator(cfg).process(Acquisition())
        return SweptLaser(cfg).process(acq)

    def test_no_rin_constant_power(self):
        acq = self._run(None)
        P = np.abs(acq.E_source)**2
        np.testing.assert_allclose(P, 10e-3, rtol=1e-10)

    def test_rin_changes_amplitude(self):
        clean = self._run(None)
        noisy = self._run(-120.0)
        assert not np.allclose(np.abs(clean.E_source)**2,
                               np.abs(noisy.E_source)**2)

    def test_rin_mean_power_unchanged(self):
        # RIN is zero-mean, so average power ~ P0
        acq = self._run(-130.0)
        P = np.abs(acq.E_source)**2
        np.testing.assert_allclose(np.mean(P), 10e-3, rtol=0.01)

    def test_rin_variance_matches_theory(self):
        # var(P) / P0^2 should equal RIN_linear * BW
        rin_dB = -130.0
        acq = self._run(rin_dB)
        P  = np.abs(acq.E_source)**2
        P0 = 10e-3
        rin_lin = 10.0**(rin_dB / 10.0)
        bw = 200e6 / 2.0
        expected = P0**2 * rin_lin * bw
        np.testing.assert_allclose(np.var(P), expected, rtol=0.05)

    def test_rin_reproducible(self):
        a = self._run(-130.0)
        b = self._run(-130.0)
        np.testing.assert_array_equal(a.E_source, b.E_source)

    def test_rin_does_not_affect_phase_noise(self):
        # RIN and phase noise use different sub-seeds, so toggling RIN
        # must not change the phase realisation
        cfg_ph   = {**CFG, "source": {**CFG["source"],
                                       "linewidth": 1e5, "rin_dB_per_Hz": None}}
        cfg_both = {**CFG, "source": {**CFG["source"],
                                       "linewidth": 1e5, "rin_dB_per_Hz": -130.0}}
        a = SweptLaser(cfg_ph).process(FiberGenerator(cfg_ph).process(Acquisition()))
        b = SweptLaser(cfg_both).process(FiberGenerator(cfg_both).process(Acquisition()))
        phi_a = np.unwrap(np.angle(a.E_source))
        phi_b = np.unwrap(np.angle(b.E_source))
        np.testing.assert_allclose(phi_a, phi_b,  atol=1e-10)


class TestMachZehnder:

    def _make_acq(self):
        acq = Acquisition()
        acq = FiberGenerator(CFG).process(acq)
        acq = SweptLaser(CFG).process(acq)
        return MachZehnder(CFG).process(acq)

    def test_photocurrent_exists(self):
        acq = self._make_acq()
        assert acq.photocurrent_main is not None
        assert acq.photocurrent_main.shape[-1] == acq.n_samples

    def test_photocurrent_is_real(self):
        acq = self._make_acq()
        # should be float, not complex
        assert acq.photocurrent_main.dtype == np.float64


class TestCirculator:

    def test_zero_loss_is_unity(self):
        c = Circulator(insertion_loss_dB=0.0)
        assert c.insertion_loss == pytest.approx(1.0)
        assert c.round_trip_transmission == pytest.approx(1.0)

    def test_insertion_loss_correct(self):
        c = Circulator(insertion_loss_dB=0.7)
        expected = 10 ** (-0.7 / 20.0)
        assert c.insertion_loss == pytest.approx(expected)

    def test_round_trip_is_squared(self):
        c = Circulator(insertion_loss_dB=1.0)
        il = c.insertion_loss
        assert c.round_trip_transmission == pytest.approx(il ** 2)

    def test_circulator_reduces_signal(self):
        """MZI output with circulator should be weaker than without."""
        cfg_no_circ = {**CFG, "optics": {**CFG["optics"],
                       "circulator": {"insertion_loss_dB": 0.0}}}
        cfg_circ = {**CFG, "optics": {**CFG["optics"],
                    "circulator": {"insertion_loss_dB": 3.0}}}
        acq0 = Acquisition()
        acq0 = FiberGenerator(cfg_no_circ).process(acq0)
        acq0 = SweptLaser(cfg_no_circ).process(acq0)
        acq0 = MachZehnder(cfg_no_circ).process(acq0)

        acq1 = Acquisition()
        acq1 = FiberGenerator(cfg_circ).process(acq1)
        acq1 = SweptLaser(cfg_circ).process(acq1)
        acq1 = MachZehnder(cfg_circ).process(acq1)

        # signal power should be lower with circulator loss
        assert np.var(acq1.photocurrent_main) < np.var(acq0.photocurrent_main)


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
        acq = run_campaign(cfg)
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


class TestADCJitter:

    def _cfg(self, **adc_overrides):
        cfg = {**CFG, "adc": {**CFG["adc"], **adc_overrides}}
        cfg["detection"] = {**CFG["detection"],  "shot_noise": False,
                             "thermal_nep": 0, "dark_current": 0}
        return cfg

    def test_zero_jitter_matches_baseline(self):
        a = run_campaign(self._cfg())
        b = run_campaign(self._cfg(jitter_rms=0.0))
        np.testing.assert_array_equal(a.digital_main, b.digital_main)

    def test_jitter_increases_noise(self):
        # need strong signal so dV/dt * sigma_j >> quantization step
        cfg_base = {**self._cfg(), "source": {**CFG["source"], "power": 1.0}}
        cfg_jit  = {**cfg_base,  "adc": {**cfg_base["adc"], "jitter_rms": 1e-9}}
        clean = run_campaign(cfg_base)
        noisy = run_campaign(cfg_jit)
        assert np.var(noisy.digital_main.astype(np.float64)) > \
               np.var(clean.digital_main.astype(np.float64))

    def test_more_jitter_means_more_noise(self):
        v_lo = np.var(run_campaign(self._cfg(jitter_rms=10e-12 )).digital_main.astype(np.float64))
        v_hi = np.var(run_campaign(self._cfg(jitter_rms=500e-12)).digital_main.astype(np.float64))
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
        from core.config_models import RootConfig
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
        a = run_campaign(self._cfg())
        b = run_campaign(self._cfg(enob=None))
        np.testing.assert_array_equal(a.digital_main, b.digital_main)

    def test_enob_equal_to_bits_is_noop(self):
        a = run_campaign(self._cfg())
        b = run_campaign(self._cfg(enob=16))
        np.testing.assert_array_equal(a.digital_main, b.digital_main)

    def test_enob_below_bits_increases_noise(self):
        # quiet baseline (no detector noise) -> any extra variance is from ENOB
        clean = run_campaign(self._cfg())
        noisy = run_campaign(self._cfg(enob=10))
        assert np.var(noisy.digital_main.astype(np.float64)) > \
               np.var(clean.digital_main.astype(np.float64))

    def test_enob_lower_means_more_noise(self):
        v8  = np.var(run_campaign(self._cfg(enob=8 )).digital_main.astype(np.float64))
        v12 = np.var(run_campaign(self._cfg(enob=12)).digital_main.astype(np.float64))
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


class TestMulticore:

    def _mc_cfg(self, n_cores):
        return {**CFG, "fiber": {**CFG["fiber"], "n_cores": n_cores}}

    def test_profile_shape_with_n_cores(self):
        acq = FiberGenerator(self._mc_cfg(4)).process(Acquisition())
        assert acq.fiber_profile.shape[0] == 4

    def test_cores_have_independent_profiles(self):
        acq = FiberGenerator(self._mc_cfg(4)).process(Acquisition())
        for i in range(4):
            for j in range(i + 1, 4):
                assert not np.array_equal(
                    acq.fiber_profile[i], acq.fiber_profile[j]
                )

    def test_pipeline_through_multicore(self):
        acq = run_campaign(self._mc_cfg(4))
        assert acq.digital_main.shape == (4, acq.n_samples)
        assert acq.photocurrent_main.shape == (4, acq.n_samples)

    def test_multicore_reproducibility(self):
        a = run_campaign(self._mc_cfg(4))
        b = run_campaign(self._mc_cfg(4))
        np.testing.assert_array_equal(a.digital_main, b.digital_main)

    def test_laser_field_is_shared_across_cores(self):
        # E_source is the optical field from a single laser,
        # must stay 1D regardless of n_cores
        acq = Acquisition()
        acq = FiberGenerator(self._mc_cfg(7)).process(acq)
        acq = SweptLaser(self._mc_cfg(7)).process(acq)
        assert acq.E_source.ndim == 1

    def test_per_core_detector_noise_is_independent(self):
        # with shot noise off and dark off, only thermal noise.
        # different cores should see different noise samples.
        cfg = {**self._mc_cfg(2), "detection": {**CFG["detection"],
               "shot_noise": False, "thermal_nep": 1e-9, "dark_current": 0}}
        acq = run_campaign(cfg)
        # subtract the (shared) deterministic beat to isolate noise
        # easier: just check the two analog traces are not equal
        assert not np.array_equal(acq.analog_main[0], acq.analog_main[1])


class TestSeeding:

    def test_components_get_distinct_seeds(self):
        s = 42
        seeds = {
            derive_seed(s, component="fiber"),
            derive_seed(s, component="laser",    sweep=0),
            derive_seed(s, component="detector", sweep=0),
            derive_seed(s, component="adc",      sweep=0),
        }
        assert len(seeds) == 4

    def test_core_stride_keeps_components_separated(self):
        # detector core 0 sweep 999_999 should not collide with detector core 1 sweep 0
        a = derive_seed(42, component="detector", core=0, sweep=999_999)
        b = derive_seed(42, component="detector", core=1, sweep=0)
        assert a != b

    def test_laser_seed_matches_legacy_offset(self):
        # behavior must be backwards compatible with the +1000+sweep convention
        assert derive_seed(42, component="laser", sweep=7) == 42 + 1000 + 7

    def test_detector_seed_matches_legacy_offset(self):
        assert derive_seed(42, component="detector", sweep=7) == 42 + 2000 + 7


class TestStrainTransfer:

    def test_ideal_constant_inside_segment(self):
        z = np.linspace(0, 1, 1001)
        segs = [{"start": 0.2, "end": 0.5, "epsilon": 1e-4}]
        eps = IdealTransfer().apply(segs, z)
        inside = (z >= 0.2) & (z <= 0.5)
        np.testing.assert_allclose(eps[inside], 1e-4)
        np.testing.assert_array_equal(eps[~inside], 0.0)

    def test_ideal_no_segments(self):
        z = np.linspace(0, 1, 50)
        eps = IdealTransfer().apply([], z)
        np.testing.assert_array_equal(eps, np.zeros_like(z))

    def test_ideal_zero_strain(self):
        z = np.linspace(0, 1, 50)
        eps = IdealTransfer().apply([{"start": 0.0, "end": 1.0, "epsilon": 0.0}], z)
        np.testing.assert_array_equal(eps, np.zeros_like(z))

    def test_ideal_overlapping_segments_add(self):
        z = np.linspace(0, 1, 1001)
        segs = [{"start": 0.2, "end": 0.6, "epsilon": 1e-4},
                {"start": 0.4, "end": 0.8, "epsilon": 2e-4}]
        eps = IdealTransfer().apply(segs, z)
        overlap = (z >= 0.4) & (z <= 0.6)
        np.testing.assert_allclose(eps[overlap], 3e-4)

    def test_cox_zero_at_segment_edges(self):
        z = np.linspace(0, 1, 1001)
        segs = [{"start": 0.2, "end": 0.5, "epsilon": 1e-4}]
        eps = CoxShearLag(beta=20.0).apply(segs, z)
        i0 = int(np.argmin(np.abs(z - 0.2)))
        i1 = int(np.argmin(np.abs(z - 0.5)))
        assert abs(eps[i0]) < 1e-10
        assert abs(eps[i1]) < 1e-10

    def test_cox_max_at_segment_centre(self):
        z = np.linspace(0, 1, 1001)
        segs = [{"start": 0.2, "end": 0.5, "epsilon": 1e-4}]
        eps = CoxShearLag(beta=50.0).apply(segs, z)
        ic = int(np.argmin(np.abs(z - 0.35)))
        assert eps[ic] == np.max(eps)
        # for beta * half >> 1 the centre approaches eps_host
        assert eps[ic] > 0.9 * 1e-4

    def test_cox_long_bond_approaches_ideal_in_middle(self):
        z = np.linspace(0, 1, 1001)
        segs = [{"start": 0.1, "end": 0.9, "epsilon": 1e-4}]
        ideal = IdealTransfer().apply(segs, z)
        cox = CoxShearLag(beta=200.0).apply(segs, z)
        middle = (z > 0.3) & (z < 0.7)
        np.testing.assert_allclose(cox[middle], ideal[middle], rtol=1e-4)

    def test_cox_short_bond_loses_transfer(self):
        # beta * half = 0.05 -> very poor coupling
        z = np.linspace(0, 1, 1001)
        segs = [{"start": 0.45, "end": 0.55, "epsilon": 1e-4}]
        eps = CoxShearLag(beta=1.0).apply(segs, z)
        assert np.max(eps) < 1e-5

    def test_cox_outside_segment_is_zero(self):
        z = np.linspace(0, 1, 1001)
        segs = [{"start": 0.3, "end": 0.6, "epsilon": 1e-4}]
        eps = CoxShearLag(beta=30.0).apply(segs, z)
        outside = (z < 0.3) | (z > 0.6)
        np.testing.assert_array_equal(eps[outside], 0.0)

    def test_cox_invalid_beta(self):
        with pytest.raises(ValueError):
            CoxShearLag(beta=0.0)
        with pytest.raises(ValueError):
            CoxShearLag(beta=-1.0)


class TestStrainPerturbation:

    def _strain_cfg(self, segments, p_e=0.22, **extra):
        cfg = {**CFG, "strain": {"segments": segments,
                                  "photoelastic_coefficient": p_e}}
        cfg.update(extra)
        return cfg

    def _unstrained(self, cfg=None):
        return FiberGenerator(cfg or CFG).process(Acquisition())

    def _strained(self, cfg):
        acq = FiberGenerator(cfg).process(Acquisition())
        return StrainPerturbation(cfg).process(acq)

    def test_no_segments_is_noop(self):
        cfg = self._strain_cfg([])
        a = self._unstrained()
        b = self._strained(cfg)
        np.testing.assert_array_equal(a.fiber_profile, b.fiber_profile)

    def test_zero_strain_is_noop(self):
        cfg = self._strain_cfg([{"start": 0.2, "end": 0.5, "epsilon": 0.0}])
        a = self._unstrained()
        b = self._strained(cfg)
        np.testing.assert_allclose(a.fiber_profile, b.fiber_profile)

    def test_strain_field_is_stored(self):
        cfg = self._strain_cfg([{"start": 0.2, "end": 0.5, "epsilon": 1e-4}])
        acq = self._strained(cfg)
        assert acq.strain_field is not None
        assert acq.strain_field.shape == acq.z.shape

    def test_no_strain_leaves_field_none(self):
        cfg = self._strain_cfg([])
        acq = self._strained(cfg)
        assert acq.strain_field is None

    def test_strain_preserves_amplitude(self):
        cfg = self._strain_cfg([{"start": 0.2, "end": 0.5, "epsilon": 1e-4}])
        a = self._unstrained()
        b = self._strained(cfg)
        np.testing.assert_allclose(np.abs(a.fiber_profile),
                                    np.abs(b.fiber_profile))

    def test_strain_phase_shift_matches_theory(self):
        eps = 1e-4
        z0, z1 = 0.2, 0.5
        p_e = 0.22
        cfg = self._strain_cfg([{"start": z0, "end": z1, "epsilon": eps}], p_e=p_e)
        a = self._unstrained()
        b = self._strained(cfg)
        # well past the segment the cumulative phase should be
        #   2 k0 n (1 - p_e) eps L
        ratio = b.fiber_profile[0, -1] / a.fiber_profile[0, -1]
        k0 = 2 * np.pi / 1550e-9
        expected = 2 * k0 * 1.4682 * (1 - p_e) * eps * (z1 - z0)
        # compare on the unit circle to avoid 2pi wraps
        # off by < dz of phase due to the cumsum discretization
        np.testing.assert_allclose(np.exp(1j * np.angle(ratio)),
                                    np.exp(1j * expected), atol=1e-2)

    def test_region_before_segment_unchanged(self):
        cfg = self._strain_cfg([{"start": 0.5, "end": 0.8, "epsilon": 1e-4}])
        a = self._unstrained()
        b = self._strained(cfg)
        before = a.z < 0.5
        np.testing.assert_allclose(a.fiber_profile[:, before],
                                    b.fiber_profile[:, before])

    def test_pipeline_runs_with_strain(self):
        cfg = self._strain_cfg([{"start": 0.3, "end": 0.6, "epsilon": 1e-5}])
        acq = run_campaign(cfg)
        assert acq.digital_main is not None

    def test_strain_does_not_double_apply_on_second_call(self):
        # mirror FiberGenerator: once applied, subsequent process()
        # calls on the same instance must be no-ops (matters for #4)
        cfg = self._strain_cfg([{"start": 0.3, "end": 0.6, "epsilon": 1e-4}])
        step = StrainPerturbation(cfg)
        acq = FiberGenerator(cfg).process(Acquisition())
        acq = step.process(acq)
        snap = acq.fiber_profile.copy()
        acq = step.process(acq)
        np.testing.assert_array_equal(acq.fiber_profile, snap)

    def test_strain_with_cox_transfer_runs(self):
        cfg = self._strain_cfg([{"start": 0.3, "end": 0.6, "epsilon": 1e-4}])
        cfg["strain"]["transfer"] = "cox"
        cfg["strain"]["cox"] = {"beta": 100.0}
        b = self._strained(cfg)
        assert b.fiber_profile is not None

    def test_cox_gives_smaller_total_phase_than_ideal(self):
        # cox rolls off at the edges -> integrated strain < ideal
        # -> total accumulated phase past the segment is smaller
        segs = [{"start": 0.3, "end": 0.6, "epsilon": 1e-4}]
        cfg_ideal = self._strain_cfg(segs)
        cfg_cox = {**self._strain_cfg(segs)}
        cfg_cox["strain"] = {**cfg_cox["strain"],
                              "transfer": "cox", "cox": {"beta": 30.0}}
        a = self._unstrained()
        b_ideal = self._strained(cfg_ideal)
        b_cox = self._strained(cfg_cox)
        # unwrap relative phase along z to compare totals without 2pi wraps
        phi_ideal = np.unwrap(np.angle(b_ideal.fiber_profile[0] / a.fiber_profile[0]))
        phi_cox = np.unwrap(np.angle(b_cox.fiber_profile[0] / a.fiber_profile[0]))
        assert abs(phi_cox[-1]) < abs(phi_ideal[-1])

    def test_cox_without_params_is_rejected(self):
        cfg = self._strain_cfg([{"start": 0.3, "end": 0.6, "epsilon": 1e-4}])
        cfg["strain"]["transfer"] = "cox"
        with pytest.raises(ValueError):
            StrainPerturbation(cfg)

    def test_unknown_transfer_is_rejected(self):
        cfg = self._strain_cfg([{"start": 0.3, "end": 0.6, "epsilon": 1e-4}])
        cfg["strain"]["transfer"] = "magic"
        with pytest.raises(ValueError):
            StrainPerturbation(cfg)

    def test_cox_config_validates(self):
        # missing cox block when transfer=cox -> Pydantic should reject
        with pytest.raises(Exception):
            RootConfig(strain={"transfer": "cox", "segments": []})
        # bad segment ordering
        with pytest.raises(Exception):
            RootConfig(strain={"segments": [{"start": 0.5, "end": 0.2, "epsilon": 1e-4}]})

    def test_multicore_strain_is_shared(self):
        # axial strain is geometric -> every core sees the same phase shift
        base = {**CFG, "fiber": {**CFG["fiber"], "n_cores": 3}}
        cfg = {**base, "strain": {"segments": [{"start": 0.3, "end": 0.6,
                                                  "epsilon": 1e-5}]}}
        acq = StrainPerturbation(cfg).process(
            FiberGenerator(cfg).process(Acquisition()))
        ref = FiberGenerator(cfg).process(Acquisition())
        ratio0 = acq.fiber_profile[0] / ref.fiber_profile[0]
        ratio1 = acq.fiber_profile[1] / ref.fiber_profile[1]
        np.testing.assert_allclose(ratio0, ratio1)

    def test_second_call_populates_fresh_acq(self):
        """Regression for #20: strain cache must re-attach profile."""
        cfg = {**CFG, "strain": {"segments": [{"start": 0.2, "end": 0.5,
                                                "epsilon": 1e-5}]}}
        gen = FiberGenerator(cfg)
        strain = StrainPerturbation(cfg)
        acq1 = strain.process(gen.process(Acquisition()))
        acq2 = strain.process(gen.process(Acquisition()))
        assert acq2.fiber_profile is not None
        np.testing.assert_array_equal(acq1.fiber_profile, acq2.fiber_profile)


class TestHDF5Writer:

    def test_write_and_read_back(self, tmp_path):
        acq = run_campaign(CFG)
        from core.config import compute_derived
        derived = compute_derived(CFG)
        path = tmp_path / "test_output.h5"
        with HDF5Writer(path) as w:
            w.write_config(CFG, derived)
            w.write_fiber(acq)
            w.write_sweep(acq, 0)

        import h5py, json
        with h5py.File(path, "r") as f:
            assert "config" in f.attrs
            cfg_back = json.loads(f.attrs["config"])
            assert cfg_back["fiber"]["length"] == CFG["fiber"]["length"]

            assert "fiber/z" in f
            assert "sweeps/0000/digital_main" in f
            assert "sweeps/0000/analog_main" in f

            dm = f["sweeps/0000/digital_main"][:]
            assert dm.shape == acq.digital_main.shape

    def test_no_output_path_skips_writing(self, tmp_path):
        """run_campaign with no output path should not create files."""
        acq = run_campaign(CFG)
        # no h5 files should exist
        assert list(tmp_path.glob("*.h5")) == []

    def test_strain_field_is_saved(self, tmp_path):
        cfg = {**CFG, "strain": {"segments": [{"start": 0.2, "end": 0.5,
                                                "epsilon": 1e-5}]}}
        acq = run_campaign(cfg)
        from core.config import compute_derived
        derived = compute_derived(cfg)
        path = tmp_path / "strain_out.h5"
        with HDF5Writer(path) as w:
            w.write_config(cfg, derived)
            w.write_fiber(acq)
            w.write_sweep(acq, 0)

        import h5py
        with h5py.File(path, "r") as f:
            assert "fiber/strain_field" in f
            sf = f["fiber/strain_field"][:]
            assert sf.shape == acq.z.shape

    def test_campaign_writes_hdf5(self, tmp_path):
        """run_campaign should write to disk when output.path is set."""
        path = tmp_path / "campaign.h5"
        cfg = {**CFG, "output": {"path": str(path)}}
        run_campaign(cfg)

        import h5py
        with h5py.File(path, "r") as f:
            assert "sweeps/0000/digital_main" in f
            assert "config" in f.attrs


class TestEndToEnd:

    def test_run_campaign(self):
        acq = run_campaign(CFG)
        assert acq.digital_main is not None
        assert acq.z is not None

    def test_reflectogram_has_energy_in_fiber_region(self):
        """The FFT should show most energy in the first N_z bins."""
        acq = run_campaign(CFG)
        # take core 0
        spectrum = np.fft.fft(acq.digital_main[0].astype(np.float64))
        n_half = len(spectrum) // 2
        mag = np.abs(spectrum[:n_half])

        n_z = len(acq.z)
        energy_fiber = np.sum(mag[:n_z] ** 2)
        energy_total = np.sum(mag ** 2)

        # most of the energy should be in the fiber region
        assert energy_fiber / energy_total > 0.9
