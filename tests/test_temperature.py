"""Tests for temperature perturbation and strain-T cross-sensitivity (#75)."""

import numpy as np
import pytest

from helpers import CFG
from core.acquisition import Acquisition
from core.campaign import run_campaign
from core.config_models import RootConfig
from fiber.profile import FiberGenerator
from fiber.strain import StrainPerturbation
from fiber.temperature import TemperaturePerturbation


# silica defaults at 1550 nm -- mirror the values in TemperatureConfig
ALPHA_L = 5.5e-7
XI      = 6.5e-6
P_E     = 0.22
N_CORE  = 1.4682
WL      = 1550e-9


class TestTemperaturePerturbation:

    def _temp_cfg(self, segments, **extra):
        cfg = {**CFG, "temperature": {"segments": segments}}
        cfg.update(extra)
        return cfg

    def _bare(self):
        return FiberGenerator(CFG).process(Acquisition())

    def _heated(self, cfg):
        acq = FiberGenerator(cfg).process(Acquisition())
        return TemperaturePerturbation(cfg).process(acq)

    def test_no_segments_is_noop(self):
        cfg = self._temp_cfg([])
        a = self._bare()
        b = self._heated(cfg)
        np.testing.assert_array_equal(a.fiber_profile, b.fiber_profile)

    def test_zero_dT_is_noop(self):
        cfg = self._temp_cfg([{"start": 0.2, "end": 0.5, "delta_T": 0.0}])
        a = self._bare()
        b = self._heated(cfg)
        np.testing.assert_allclose(a.fiber_profile, b.fiber_profile)

    def test_temperature_field_is_stored(self):
        cfg = self._temp_cfg([{"start": 0.2, "end": 0.5, "delta_T": 5.0}])
        acq = self._heated(cfg)
        assert acq.temperature_field is not None
        assert acq.temperature_field.shape == acq.z.shape

    def test_no_temperature_leaves_field_none(self):
        cfg = self._temp_cfg([])
        acq = self._heated(cfg)
        assert acq.temperature_field is None

    def test_temperature_preserves_amplitude(self):
        cfg = self._temp_cfg([{"start": 0.2, "end": 0.5, "delta_T": 5.0}])
        a = self._bare()
        b = self._heated(cfg)
        np.testing.assert_allclose(np.abs(a.fiber_profile),
                                    np.abs(b.fiber_profile))

    def test_temperature_phase_shift_matches_theory(self):
        dT = 5.0
        z0, z1 = 0.2, 0.5
        cfg = self._temp_cfg([{"start": z0, "end": z1, "delta_T": dT}])
        a = self._bare()
        b = self._heated(cfg)
        ratio = b.fiber_profile[0, -1] / a.fiber_profile[0, -1]
        k0 = 2 * np.pi / WL
        expected = 2 * k0 * N_CORE * (ALPHA_L + XI) * dT * (z1 - z0)
        # compare on the unit circle to avoid 2pi wraps
        np.testing.assert_allclose(np.exp(1j * np.angle(ratio)),
                                    np.exp(1j * expected), atol=1e-2)

    def test_region_before_segment_unchanged(self):
        cfg = self._temp_cfg([{"start": 0.5, "end": 0.8, "delta_T": 5.0}])
        a = self._bare()
        b = self._heated(cfg)
        before = a.z < 0.5
        np.testing.assert_allclose(a.fiber_profile[:, before],
                                    b.fiber_profile[:, before])

    def test_pipeline_runs_with_temperature(self):
        cfg = self._temp_cfg([{"start": 0.3, "end": 0.6, "delta_T": 2.0}])
        acq = run_campaign(cfg)[-1]
        assert acq.digital_main is not None

    def test_multicore_temperature_is_shared(self):
        # uniform axial dT affects every core the same way
        base = {**CFG, "fiber": {**CFG["fiber"], "n_cores": 3}}
        cfg = {**base, "temperature": {"segments": [{"start": 0.3, "end": 0.6,
                                                      "delta_T": 1.0}]}}
        acq = TemperaturePerturbation(cfg).process(
            FiberGenerator(cfg).process(Acquisition()))
        ref = FiberGenerator(cfg).process(Acquisition())
        ratio0 = acq.fiber_profile[0] / ref.fiber_profile[0]
        ratio1 = acq.fiber_profile[1] / ref.fiber_profile[1]
        np.testing.assert_allclose(ratio0, ratio1)

    def test_static_cache_idempotent(self):
        cfg = self._temp_cfg([{"start": 0.3, "end": 0.6, "delta_T": 2.0}])
        gen = FiberGenerator(cfg)
        step = TemperaturePerturbation(cfg)
        acq1 = step.process(gen.process(Acquisition()))
        snap = acq1.fiber_profile.copy()
        # second sweep: gen re-attaches its cache, step takes the cached path
        acq2 = step.process(gen.process(Acquisition()))
        np.testing.assert_array_equal(acq2.fiber_profile, snap)

    def test_dynamic_temperature_changes_per_sweep(self):
        # 25 Hz at sweep_duration=0.01s -> sweep 1 lands on +peak, sweep 3 on -peak
        cfg = self._temp_cfg([{"start": 0.3, "end": 0.6, "delta_T": 0.0,
                                "motion": {"kind": "harmonic",
                                           "amplitude": 2.0,
                                           "frequency": 25.0}}])
        gen = FiberGenerator(cfg)
        step = TemperaturePerturbation(cfg)
        bare = gen.process(Acquisition()).fiber_profile[0, -1]

        a_pos = step.process(gen.process(Acquisition(sweep_index=1)))
        a_neg = step.process(gen.process(Acquisition(sweep_index=3)))

        ang_pos = np.angle(a_pos.fiber_profile[0, -1] / bare)
        ang_neg = np.angle(a_neg.fiber_profile[0, -1] / bare)
        # phase signs flip across a half period
        assert ang_pos * ang_neg < 0


class TestStrainTemperatureCrossSensitivity:
    """The whole point of #75: the simulator should reproduce that an OFDR
    system can't tell strain from temperature without an extra channel.
    """

    def test_temperature_and_apparent_strain_produce_identical_phase(self):
        # apparent strain that mimics dT in the Rayleigh shift
        dT = 10.0
        eps_app = (ALPHA_L + XI) / (1.0 - P_E) * dT     # ~9.04e-5

        z0, z1 = 0.3, 0.7

        cfg_T = {**CFG, "temperature": {"segments":
                                          [{"start": z0, "end": z1, "delta_T": dT}]}}
        cfg_eps = {**CFG, "strain": {"segments":
                                       [{"start": z0, "end": z1, "epsilon": eps_app}],
                                     "photoelastic_coefficient": P_E}}

        bare_T = FiberGenerator(cfg_T).process(Acquisition())
        a = TemperaturePerturbation(cfg_T).process(
            FiberGenerator(cfg_T).process(Acquisition()))

        bare_eps = FiberGenerator(cfg_eps).process(Acquisition())
        b = StrainPerturbation(cfg_eps).process(
            FiberGenerator(cfg_eps).process(Acquisition()))

        phi_T = np.unwrap(np.angle(a.fiber_profile[0] / bare_T.fiber_profile[0]))
        phi_eps = np.unwrap(np.angle(b.fiber_profile[0] / bare_eps.fiber_profile[0]))

        # both built from the same cumsum*dz integration -> exact match
        np.testing.assert_allclose(phi_T, phi_eps, atol=1e-10)

    def test_strain_and_temperature_phases_add_linearly(self):
        eps = 5e-5
        dT = 3.0
        z0, z1 = 0.2, 0.6

        cfg = {**CFG,
               "strain":      {"segments": [{"start": z0, "end": z1, "epsilon": eps}]},
               "temperature": {"segments": [{"start": z0, "end": z1, "delta_T": dT}]}}

        bare = FiberGenerator(cfg).process(Acquisition())
        acq = TemperaturePerturbation(cfg).process(
            StrainPerturbation(cfg).process(
                FiberGenerator(cfg).process(Acquisition())))
        phi_obs = np.unwrap(np.angle(acq.fiber_profile[0] / bare.fiber_profile[0]))

        z = bare.z
        dz = bare.dz
        in_seg = ((z >= z0) & (z <= z1)).astype(float)
        cum = np.cumsum(in_seg) * dz
        k0 = 2 * np.pi / WL
        phi_th = 2 * k0 * N_CORE * ((1 - P_E) * eps + (ALPHA_L + XI) * dT) * cum

        np.testing.assert_allclose(phi_obs, phi_th, atol=1e-3)


class TestTemperatureConfigValidation:

    def test_bad_segment_ordering_rejected(self):
        with pytest.raises(Exception):
            RootConfig(temperature={"segments":
                                      [{"start": 0.5, "end": 0.2, "delta_T": 1.0}]})

    def test_negative_alpha_L_rejected(self):
        with pytest.raises(Exception):
            RootConfig(temperature={"thermal_expansion": -1.0})

    def test_default_is_silica(self):
        cfg = RootConfig().model_dump()
        assert cfg["temperature"]["thermal_expansion"] == pytest.approx(ALPHA_L)
        assert cfg["temperature"]["thermo_optic"]      == pytest.approx(XI)

    def test_custom_coefficients_take_effect(self):
        # use a synthetic xi 10x silica to verify it propagates through
        z0, z1 = 0.3, 0.6
        dT = 1.0
        xi_big = XI * 10.0
        cfg = {**CFG, "temperature":
                       {"segments": [{"start": z0, "end": z1, "delta_T": dT}],
                        "thermo_optic": xi_big}}
        bare = FiberGenerator(cfg).process(Acquisition())
        a = TemperaturePerturbation(cfg).process(
            FiberGenerator(cfg).process(Acquisition()))
        ratio = a.fiber_profile[0, -1] / bare.fiber_profile[0, -1]
        k0 = 2 * np.pi / WL
        expected = 2 * k0 * N_CORE * (ALPHA_L + xi_big) * dT * (z1 - z0)
        np.testing.assert_allclose(np.exp(1j * np.angle(ratio)),
                                    np.exp(1j * expected), atol=1e-2)
