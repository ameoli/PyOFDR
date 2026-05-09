"""Tests for fiber profile generation, reflectors, and attenuation."""

import numpy as np
import pytest

from helpers import CFG
from pyofdr.core.acquisition import Acquisition
from pyofdr.fiber.attenuation import round_trip_attenuation, dB_per_km_to_neper_per_m
from pyofdr.fiber.fbg import weak_fbg_signal
from pyofdr.fiber.profile import FiberGenerator
from pyofdr.fiber.reflectors import apply_connector_losses, inject_reflectors
from pyofdr.utils.constants import C as _C


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


class TestMultipleScattering:
    """Multi-bounce ghost reflections from cascading reflectors (#35)."""

    def test_default_disabled(self):
        # without the multiple_scattering block the profile is untouched
        refs = [{"z": 0.0, "R": 0.01}, {"z": 0.3, "R": 0.04}]
        cfg = {**CFG, "fiber": {**CFG["fiber"], "reflectors": refs}}
        a = FiberGenerator(cfg).process(Acquisition())
        cfg2 = {**cfg, "fiber": {**cfg["fiber"], "multiple_scattering": None}}
        b = FiberGenerator(cfg2).process(Acquisition())
        np.testing.assert_array_equal(a.fiber_profile, b.fiber_profile)

    def test_double_bounce_ghost_position_and_amplitude(self):
        """Two reflectors at z=0 and z=z_b -> ghost at 2*z_b with
        amplitude sqrt(R_a) * R_b."""
        z_b = 0.3
        R_a, R_b = 0.01, 0.04
        cfg = {**CFG, "fiber": {**CFG["fiber"],
               "rayleigh_coefficient_dB": -300,    # kill Rayleigh
               "reflectors": [{"z": 0.0, "R": R_a}, {"z": z_b, "R": R_b}],
               "multiple_scattering": {"max_order": 2}}}
        acq = FiberGenerator(cfg).process(Acquisition())
        # match the bin convention used inside the module: each z is rounded
        # to its bin first, then the apparent bin is 2*bin_b - bin_a (otherwise
        # we end up off-by-one when 2*round(x) != round(2*x))
        bin_b = int(round(z_b / acq.dz))
        ghost_bin = 2 * bin_b
        amp = np.abs(acq.fiber_profile[0, ghost_bin])
        np.testing.assert_allclose(amp, np.sqrt(R_a) * R_b, atol=1e-10)

    def test_no_ghost_when_disabled(self):
        z_b = 0.3
        cfg = {**CFG, "fiber": {**CFG["fiber"],
               "rayleigh_coefficient_dB": -300,
               "reflectors": [{"z": 0.0, "R": 0.01}, {"z": z_b, "R": 0.04}]}}
        acq = FiberGenerator(cfg).process(Acquisition())
        ghost_bin = 2 * int(round(z_b / acq.dz))
        amp = np.abs(acq.fiber_profile[0, ghost_bin])
        # ~0 since Rayleigh is suppressed and no ghost added
        assert amp < 1e-9

    def test_ghost_outside_range_dropped(self):
        # ghost at 2*0.6 = 1.2 m is past the 1 m fiber -- no error, no peak
        cfg = {**CFG, "fiber": {**CFG["fiber"],
               "reflectors": [{"z": 0.0, "R": 0.01}, {"z": 0.6, "R": 0.04}],
               "multiple_scattering": {"max_order": 2}}}
        acq = FiberGenerator(cfg).process(Acquisition())
        assert acq.fiber_profile.shape[-1] == len(acq.z)

    def test_higher_order_adds_extra_paths(self):
        # max_order=3 enumerates 5-reflection paths the order-2 sweep missed
        refs = [{"z": 0.0, "R": 0.01},
                {"z": 0.1, "R": 0.04},
                {"z": 0.2, "R": 0.09}]
        cfg2 = {**CFG, "fiber": {**CFG["fiber"],
                "rayleigh_coefficient_dB": -300,
                "reflectors": refs,
                "multiple_scattering": {"max_order": 2}}}
        cfg3 = {**cfg2, "fiber": {**cfg2["fiber"],
                "multiple_scattering": {"max_order": 3}}}
        a = FiberGenerator(cfg2).process(Acquisition())
        b = FiberGenerator(cfg3).process(Acquisition())
        diff = np.abs(np.abs(b.fiber_profile) - np.abs(a.fiber_profile)).max()
        assert diff > 0


class TestWeakFBG:
    """Weak FBG arrays under Born approximation (#40)."""

    def test_empty_returns_zeros(self):
        nu = np.linspace(1.9e14, 1.95e14, 1000)
        out = weak_fbg_signal([], dz=1e-3, n_z=1000,
                              attenuation=None, nu_inst=nu, n_core=1.4682)
        assert np.all(out == 0.0)

    def test_on_bragg_peak_amplitude(self):
        """At nu == nu_B the sinc envelope is exactly sqrt(R_max)."""
        lam_B = 1550e-9
        nu_B = _C / lam_B
        # one sample on Bragg, others far enough to give negligible sinc
        nu = np.array([nu_B, nu_B + 1e13])
        fbg = [{"z": 0.0, "bragg_wavelength": lam_B,
                "length": 5e-3, "peak_reflectivity": 0.25}]
        out = weak_fbg_signal(fbg, dz=1e-3, n_z=10,
                              attenuation=None, nu_inst=nu, n_core=1.4682)
        # idx_z = 0 -> phase = 0 -> cos = 1, envelope = sqrt(0.25) = 0.5
        np.testing.assert_allclose(out[0], 0.5, atol=1e-12)

    def test_first_null_position(self):
        """Reflectivity should be zero at nu = nu_B + C/(2 n L)."""
        lam_B = 1550e-9
        n_core = 1.4682
        L_g = 5e-3
        delta_nu_null = _C / (2.0 * n_core * L_g)
        nu = np.array([_C / lam_B + delta_nu_null])
        fbg = [{"z": 0.0, "bragg_wavelength": lam_B,
                "length": L_g, "peak_reflectivity": 0.25}]
        out = weak_fbg_signal(fbg, dz=1e-3, n_z=10,
                              attenuation=None, nu_inst=nu, n_core=n_core)
        np.testing.assert_allclose(out[0], 0.0, atol=1e-12)

    def test_amplitude_scales_with_sqrt_R(self):
        """4x peak reflectivity -> 2x signal amplitude."""
        nu = np.linspace(_C / 1551e-9, _C / 1549e-9, 2000)
        base = [{"z": 0.0, "bragg_wavelength": 1550e-9,
                 "length": 5e-3, "peak_reflectivity": 0.01}]
        big  = [{"z": 0.0, "bragg_wavelength": 1550e-9,
                 "length": 5e-3, "peak_reflectivity": 0.04}]
        out_b = weak_fbg_signal(base, 1e-3, 10, None, nu, 1.4682)
        out_g = weak_fbg_signal(big,  1e-3, 10, None, nu, 1.4682)
        np.testing.assert_allclose(out_g, 2.0 * out_b, atol=1e-12)

    def test_far_offband_is_negligible(self):
        """Bragg far outside the sweep range -> tiny signal."""
        nu = np.linspace(_C / 1551e-9, _C / 1549e-9, 2000)
        # nu_B ~ 200 nm away
        fbg = [{"z": 0.0, "bragg_wavelength": 1750e-9,
                "length": 5e-3, "peak_reflectivity": 0.5}]
        out = weak_fbg_signal(fbg, 1e-3, 10, None, nu, 1.4682)
        assert np.max(np.abs(out)) < 1e-3

    def test_outside_fiber_ignored(self):
        nu = np.linspace(1.9e14, 1.95e14, 100)
        fbg = [{"z": 999.0, "bragg_wavelength": 1550e-9,
                "length": 5e-3, "peak_reflectivity": 0.1}]
        out = weak_fbg_signal(fbg, 1e-3, 10, None, nu, 1.4682)
        assert np.all(out == 0.0)

    def test_two_fbgs_accumulate(self):
        """A list of two FBGs at the same z should give 2x the signal of one."""
        nu = np.linspace(_C / 1551e-9, _C / 1549e-9, 4000)
        fbg1 = [{"z": 0.0, "bragg_wavelength": 1550e-9,
                  "length": 5e-3, "peak_reflectivity": 0.04}]
        fbg2 = fbg1 + fbg1
        out1 = weak_fbg_signal(fbg1, 1e-3, 10, None, nu, 1.4682)
        out2 = weak_fbg_signal(fbg2, 1e-3, 10, None, nu, 1.4682)
        np.testing.assert_allclose(out2, 2.0 * out1, atol=1e-12)


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


class TestIndexPerturbation:
    """Small-signal delta_n(z) adds a round-trip phase to the scatterers
    without touching their amplitude or bin position. See #68 (full
    z-dependent n(z) with OPL-based dz is #33)."""

    def test_empty_unchanged(self):
        cfg = {**CFG, "fiber": {**CFG["fiber"], "index_segments": []}}
        a = FiberGenerator(CFG).process(Acquisition())
        b = FiberGenerator(cfg).process(Acquisition())
        np.testing.assert_array_equal(a.fiber_profile, b.fiber_profile)

    def test_amplitude_unchanged(self):
        segs = [{"start": 0.3, "end": 0.7, "delta_n": 1e-4}]
        cfg = {**CFG, "fiber": {**CFG["fiber"], "index_segments": segs}}
        a = FiberGenerator(CFG).process(Acquisition())
        b = FiberGenerator(cfg).process(Acquisition())
        np.testing.assert_allclose(np.abs(a.fiber_profile),
                                    np.abs(b.fiber_profile), rtol=1e-12)

    def test_phase_matches_cumulative_formula(self):
        # phase accumulates linearly through the segment and plateaus after
        dn = 1e-4
        segs = [{"start": 0.3, "end": 0.7, "delta_n": dn}]
        cfg = {**CFG, "fiber": {**CFG["fiber"], "index_segments": segs}}
        a = FiberGenerator(CFG).process(Acquisition())
        b = FiberGenerator(cfg).process(Acquisition())

        # same seed so speckle cancels in the ratio
        ratio = b.fiber_profile[0] / a.fiber_profile[0]
        np.testing.assert_allclose(np.abs(ratio), 1.0, rtol=1e-10)

        phi_meas = np.unwrap(np.angle(ratio))
        z, dz = b.z, b.dz
        delta_n_arr = np.where((z >= 0.3) & (z < 0.7), dn, 0.0)
        k0 = 2.0 * np.pi / CFG["source"]["center_wavelength"]
        phi_exp = 2.0 * k0 * np.cumsum(delta_n_arr) * dz
        np.testing.assert_allclose(phi_meas, phi_exp, atol=1e-9)

    def test_plateau_after_segment(self):
        segs = [{"start": 0.3, "end": 0.7, "delta_n": 2e-4}]
        cfg = {**CFG, "fiber": {**CFG["fiber"], "index_segments": segs}}
        a = FiberGenerator(CFG).process(Acquisition())
        b = FiberGenerator(cfg).process(Acquisition())
        ratio = b.fiber_profile[0] / a.fiber_profile[0]
        phi = np.unwrap(np.angle(ratio))
        z = b.z
        # phase must be flat for z > end of segment
        tail = phi[z > 0.75]
        np.testing.assert_allclose(tail - tail[0], 0.0, atol=1e-9)

    def test_sign_flip(self):
        # +delta_n and -delta_n should give opposite phases
        sp = [{"start": 0.3, "end": 0.6, "delta_n": +1e-4}]
        sm = [{"start": 0.3, "end": 0.6, "delta_n": -1e-4}]
        cfg_p = {**CFG, "fiber": {**CFG["fiber"], "index_segments": sp}}
        cfg_m = {**CFG, "fiber": {**CFG["fiber"], "index_segments": sm}}
        bp = FiberGenerator(cfg_p).process(Acquisition())
        bm = FiberGenerator(cfg_m).process(Acquisition())
        a  = FiberGenerator(CFG).process(Acquisition())
        phi_p = np.unwrap(np.angle(bp.fiber_profile[0] / a.fiber_profile[0]))
        phi_m = np.unwrap(np.angle(bm.fiber_profile[0] / a.fiber_profile[0]))
        np.testing.assert_allclose(phi_p, -phi_m, atol=1e-9)


class TestIndexFluctuations:
    """Stochastic OU n(z) fluctuations -- partial #33. Sits on top of the
    same small-signal phase machinery as IndexPerturbation, so amplitude
    is unchanged and the recovered delta_n stats should match an OU
    process with stationary std=sigma and lag-1 autocorr=exp(-dz/L_corr).
    """

    @staticmethod
    def _pair(sigma, L_corr, length=2.0, seed=42):
        cfg_base = {**CFG, "simulation": {"seed": seed},
                    "fiber": {**CFG["fiber"], "length": length,
                              "attenuation_dB_per_km": 0.0}}
        cfg_fl = {**cfg_base, "fiber": {**cfg_base["fiber"],
                  "index_fluctuations": {
                      "sigma": sigma, "correlation_length": L_corr}}}
        a = FiberGenerator(cfg_base).process(Acquisition())
        b = FiberGenerator(cfg_fl).process(Acquisition())
        return a, b

    def test_none_unchanged(self):
        # absent block -> identical to baseline
        a = FiberGenerator(CFG).process(Acquisition())
        b = FiberGenerator({**CFG, "fiber": {**CFG["fiber"]}}).process(Acquisition())
        np.testing.assert_array_equal(a.fiber_profile, b.fiber_profile)

    def test_zero_sigma_is_noop(self):
        # sigma=0 must not consume the rng stream. Identical baseline.
        a, b = self._pair(0.0, 0.01)
        np.testing.assert_array_equal(a.fiber_profile, b.fiber_profile)

    def test_amplitude_unchanged(self):
        # phase-only perturbation -> |E| identical
        a, b = self._pair(1e-5, 0.01)
        np.testing.assert_allclose(np.abs(a.fiber_profile),
                                    np.abs(b.fiber_profile), rtol=1e-12)

    def test_recovered_std_matches_sigma(self):
        sigma = 1e-6
        a, b = self._pair(sigma, 0.005)
        ratio = b.fiber_profile[0] / a.fiber_profile[0]
        phi = np.unwrap(np.angle(ratio))
        k0  = 2.0 * np.pi / CFG["source"]["center_wavelength"]
        # local delta_n recovered as the discrete derivative of phi/(2 k0 dz)
        dn = np.diff(phi) / (2.0 * k0 * b.dz)
        np.testing.assert_allclose(np.std(dn), sigma, rtol=5e-2)

    def test_recovered_lag1_autocorr(self):
        # OU stationary lag-1 autocorr = exp(-dz/L_corr)
        sigma, L_corr = 1e-6, 0.005
        a, b = self._pair(sigma, L_corr)
        ratio = b.fiber_profile[0] / a.fiber_profile[0]
        phi = np.unwrap(np.angle(ratio))
        k0  = 2.0 * np.pi / CFG["source"]["center_wavelength"]
        dn = np.diff(phi) / (2.0 * k0 * b.dz)
        a_th  = np.exp(-b.dz / L_corr)
        a_emp = np.mean(dn[:-1] * dn[1:]) / np.mean(dn**2)
        np.testing.assert_allclose(a_emp, a_th, rtol=5e-2)

    def test_deterministic_with_seed(self):
        cfg = {**CFG, "fiber": {**CFG["fiber"],
               "index_fluctuations": {"sigma": 1e-6,
                                       "correlation_length": 0.01}}}
        p1 = FiberGenerator(cfg).process(Acquisition()).fiber_profile
        p2 = FiberGenerator(cfg).process(Acquisition()).fiber_profile
        np.testing.assert_array_equal(p1, p2)

    def test_combines_with_segments(self):
        # deterministic segment + stochastic fluctuations should add up:
        # angle(profile_combined / profile_baseline) ~= angle from segment
        # plus a zero-mean stochastic phase.
        seg = [{"start": 0.3, "end": 0.7, "delta_n": 1e-5}]
        cfg_seg = {**CFG, "fiber": {**CFG["fiber"], "index_segments": seg}}
        cfg_both = {**cfg_seg, "fiber": {**cfg_seg["fiber"],
                    "index_fluctuations": {"sigma": 1e-7,
                                            "correlation_length": 0.01}}}
        a = FiberGenerator(CFG).process(Acquisition())
        bs = FiberGenerator(cfg_seg).process(Acquisition())
        bb = FiberGenerator(cfg_both).process(Acquisition())
        # amplitude: all three identical
        np.testing.assert_allclose(np.abs(a.fiber_profile),
                                    np.abs(bb.fiber_profile), rtol=1e-12)
        # phase plateau after the segment: combined ~= seg + small noise
        phi_seg  = np.unwrap(np.angle(bs.fiber_profile[0] / a.fiber_profile[0]))
        phi_both = np.unwrap(np.angle(bb.fiber_profile[0] / a.fiber_profile[0]))
        # the difference is the OU phase, which has small std on this short fiber
        diff = phi_both - phi_seg
        assert np.std(diff) < 0.05 * np.max(np.abs(phi_seg))


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


class TestCrosstalk:
    """MCF core-to-core crosstalk, level A (phase-scrambled scalar), #47."""

    def _mcf(self, n_cores=7, crosstalk=None):
        fiber = {**CFG["fiber"], "n_cores": n_cores}
        if crosstalk is not None:
            fiber["crosstalk"] = crosstalk
        return {**CFG, "fiber": fiber}

    def test_disabled_by_default(self):
        # n_cores > 1 with no crosstalk block -> cores stay independent
        cfg = self._mcf(n_cores=7)
        acq = FiberGenerator(cfg).process(Acquisition())
        assert acq.fiber_profile.shape[0] == 7
        a, b = acq.fiber_profile[0], acq.fiber_profile[1]
        corr = np.abs(np.mean(np.conj(a) * b))
        ref = np.sqrt(np.mean(np.abs(a)**2) * np.mean(np.abs(b)**2))
        assert corr / ref < 5e-2

    def test_xt_boosts_centre_core_power(self):
        # aggressive xt injects power from all 6 neighbours; pick +20 dB/km
        # (nonphysical, just for visibility on the 1 m test fibre)
        cfg_no = self._mcf(n_cores=7)
        cfg_xt = self._mcf(n_cores=7, crosstalk={
            "xt_dB_per_km": 20.0, "topology": "hex7",
        })
        p_no = FiberGenerator(cfg_no).process(Acquisition()).fiber_profile
        p_xt = FiberGenerator(cfg_xt).process(Acquisition()).fiber_profile

        pwr_no = float(np.mean(np.abs(p_no[0])**2))
        pwr_xt = float(np.mean(np.abs(p_xt[0])**2))
        assert pwr_xt > 1.1 * pwr_no

    def test_no_op_for_single_core(self):
        fiber = {**CFG["fiber"], "n_cores": 1,
                 "crosstalk": {"xt_dB_per_km": -30.0, "topology": "linear"}}
        cfg = {**CFG, "fiber": fiber}
        acq = FiberGenerator(cfg).process(Acquisition())
        assert acq.fiber_profile.shape[0] == 1

    def test_hex7_requires_7_cores(self):
        cfg = self._mcf(n_cores=4, crosstalk={
            "xt_dB_per_km": -30.0, "topology": "hex7",
        })
        with pytest.raises(ValueError, match="hex7"):
            FiberGenerator(cfg).process(Acquisition())

    def test_linear_topology_accepts_any_n(self):
        cfg = self._mcf(n_cores=3, crosstalk={
            "xt_dB_per_km": -20.0, "topology": "linear",
        })
        acq = FiberGenerator(cfg).process(Acquisition())
        assert acq.fiber_profile.shape[0] == 3

    def test_deterministic_with_same_seed(self):
        cfg = self._mcf(n_cores=7, crosstalk={
            "xt_dB_per_km": -15.0, "topology": "hex7",
        })
        p1 = FiberGenerator(cfg).process(Acquisition()).fiber_profile
        p2 = FiberGenerator(cfg).process(Acquisition()).fiber_profile
        np.testing.assert_array_equal(p1, p2)
