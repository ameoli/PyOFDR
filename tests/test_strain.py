"""Tests for strain transfer models and strain perturbation."""

import numpy as np
import pytest

from helpers import CFG
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.campaign import run_campaign
from pyofdr.core.config_models import RootConfig
from pyofdr.strain_transfer import CoxShearLag, IdealTransfer
from pyofdr.fiber.profile import FiberGenerator
from pyofdr.fiber.strain import StrainPerturbation


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
        acq = run_campaign(cfg)[-1]
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
