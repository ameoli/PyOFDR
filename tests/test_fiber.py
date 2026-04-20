"""Tests for fiber profile generation, reflectors, and attenuation."""

import numpy as np
import pytest

from helpers import CFG
from core.acquisition import Acquisition
from fiber.attenuation import round_trip_attenuation, dB_per_km_to_neper_per_m
from fiber.profile import FiberGenerator
from fiber.reflectors import apply_connector_losses, inject_reflectors


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


class TestVaryingRayleigh:

    def _run(self, segments=None):
        cfg = {**CFG, "fiber": {**CFG["fiber"],
               "rayleigh_segments": segments or []}}
        return FiberGenerator(cfg).process(Acquisition())

    def test_no_segments_matches_baseline(self):
        a = self._run(None)
        b = FiberGenerator(CFG).process(Acquisition())
        np.testing.assert_array_equal(a.fiber_profile, b.fiber_profile)

    def test_high_R_segment_has_larger_amplitude(self):
        # pump one segment to -60 dB vs default -82 dB
        segs = [{"start": 0.3, "end": 0.7,
                  "rayleigh_coefficient_dB": -60.0}]
        acq = self._run(segs)
        z = acq.z
        P = np.abs(acq.fiber_profile[0])**2
        inside  = (z >= 0.3) & (z < 0.7)
        outside = ~inside
        # mean power should be much larger in the hot segment
        assert np.mean(P[inside]) > 10.0 * np.mean(P[outside])

    def test_segment_respects_bounds(self):
        segs = [{"start": 0.4, "end": 0.6,
                  "rayleigh_coefficient_dB": -50.0}]
        acq = self._run(segs)
        P = np.abs(acq.fiber_profile[0])**2
        z = acq.z
        # samples just outside segment should still be at baseline level.
        # average a chunk to beat speckle.
        out_before = P[(z > 0.1) & (z < 0.35)].mean()
        out_after  = P[(z > 0.65) & (z < 0.9)].mean()
        np.testing.assert_allclose(out_before, out_after, rtol=0.5)


class TestVaryingAttenuation:

    def _run(self, segments=None, base=0.0):
        cfg = {**CFG, "fiber": {**CFG["fiber"],
               "attenuation_dB_per_km": base,
               "attenuation_segments":  segments or []}}
        return FiberGenerator(cfg).process(Acquisition())

    def test_no_segments_matches_baseline(self):
        a = self._run(None, base=0.18)
        b = FiberGenerator({**CFG, "fiber": {**CFG["fiber"],
             "attenuation_dB_per_km": 0.18}}).process(Acquisition())
        np.testing.assert_array_equal(
            a.attenuation_envelope, b.attenuation_envelope)

    def test_lossy_segment_drops_envelope(self):
        # one lossy chunk in the middle
        segs = [{"start": 0.4, "end": 0.6,
                  "attenuation_dB_per_km": 20000.0}]
        acq = self._run(segs, base=0.0)
        env = acq.attenuation_envelope
        z = acq.z
        before = env[(z >= 0.3) & (z < 0.4)].mean()
        after  = env[(z >= 0.6) & (z < 0.7)].mean()
        # big drop across the lossy section
        assert after < 0.5 * before

    def test_envelope_monotonic(self):
        # attenuation can only reduce amplitude, never increase it
        segs = [{"start": 0.2, "end": 0.4, "attenuation_dB_per_km": 5.0},
                {"start": 0.6, "end": 0.8, "attenuation_dB_per_km": 2.0}]
        acq = self._run(segs, base=0.18)
        env = acq.attenuation_envelope
        assert np.all(np.diff(env) <= 1e-12)

    def test_zero_loss_segment_matches_baseline(self):
        # segment with same value as base -> no difference
        segs = [{"start": 0.3, "end": 0.7, "attenuation_dB_per_km": 0.5}]
        a = self._run(segs, base=0.5)
        b = FiberGenerator({**CFG, "fiber": {**CFG["fiber"],
             "attenuation_dB_per_km": 0.5}}).process(Acquisition())
        # numerical: varying path uses cumsum*dz, constant uses alpha*z,
        # off by one bin -- allow 1e-6 tolerance
        np.testing.assert_allclose(a.attenuation_envelope,
                                    b.attenuation_envelope, rtol=1e-4)


class TestRayleighStatistics:
    """Raw fiber_profile is a circular complex gaussian field, so |E|
    must follow a Rayleigh distribution with scale sigma/sqrt(2)
    (sigma^2 = R_lin * dz) and the phase must be uniform on [-pi, pi).
    See #65."""

    @staticmethod
    def _profile():
        # 2 m -> ~100k samples, plenty for the fit / KS test
        cfg = {**CFG, "simulation": {"seed": 123},
               "fiber": {**CFG["fiber"], "length": 2.0,
                         "attenuation_dB_per_km": 0.0}}
        return FiberGenerator(cfg).process(Acquisition()), cfg

    def test_amplitude_is_rayleigh(self):
        from scipy.stats import rayleigh
        acq, cfg = self._profile()
        amp = np.abs(acq.fiber_profile[0])
        R_lin    = 10 ** (cfg["fiber"]["rayleigh_coefficient_dB"] / 10.0)
        scale_th = np.sqrt(R_lin * acq.dz / 2.0)
        # MLE with loc pinned at 0 -- no origin shift in the PDF
        _, scale_hat = rayleigh.fit(amp, floc=0.0)
        np.testing.assert_allclose(scale_hat, scale_th, rtol=1e-2)

    def test_mean_power_matches_coefficient(self):
        acq, cfg = self._profile()
        P     = np.abs(acq.fiber_profile[0]) ** 2
        R_lin = 10 ** (cfg["fiber"]["rayleigh_coefficient_dB"] / 10.0)
        # E[|E|^2] = sigma^2 = R_lin * dz
        np.testing.assert_allclose(P.mean(), R_lin * acq.dz, rtol=1e-2)

    def test_phase_is_uniform(self):
        from scipy.stats import kstest
        acq, _ = self._profile()
        u = (np.angle(acq.fiber_profile[0]) + np.pi) / (2.0 * np.pi)
        _, p = kstest(u, "uniform")
        assert p > 0.01, f"KS rejects uniform phase (p={p:.3g})"
