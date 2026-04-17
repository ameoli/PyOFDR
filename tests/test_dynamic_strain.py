"""Tests for dynamic (inter-sweep) strain perturbations, issue #41.

Only harmonic motion is implemented here -- propagating / thermal /
impulsive models come later. The sweep-rate sampling assumption is that
strain is effectively constant during a single sweep (sweep_duration <<
1/f_vib).
"""

import warnings

import numpy as np
import pytest

from helpers import CFG
from core.campaign import run_campaign
from strain_transfer import evaluate_motion, realize_segments


def _dyn_cfg(segments, n_sweeps=1):
    return {
        **CFG,
        "simulation": {"seed": 42, "n_sweeps": n_sweeps},
        "strain": {
            "segments": segments,
            "photoelastic_coefficient": 0.22,
            "transfer": "ideal",
        },
    }


class TestEvaluateMotion:

    def test_none_returns_zero(self):
        assert evaluate_motion(None, 0.0) == 0.0
        assert evaluate_motion(None, 1.23) == 0.0

    def test_harmonic_zero_at_start(self):
        m = {"kind": "harmonic", "amplitude": 1e-4, "frequency": 100.0, "phase": 0.0}
        assert abs(evaluate_motion(m, 0.0)) < 1e-20

    def test_harmonic_peak_at_quarter_period(self):
        m = {"kind": "harmonic", "amplitude": 5e-5, "frequency": 100.0, "phase": 0.0}
        t_peak = 1.0 / (4 * 100.0)   # T/4
        assert abs(evaluate_motion(m, t_peak) - 5e-5) < 1e-10

    def test_harmonic_phase_shift(self):
        # phase = pi/2 -> sin starts at 1
        m = {"kind": "harmonic", "amplitude": 1e-4, "frequency": 20.0,
             "phase": np.pi / 2}
        assert abs(evaluate_motion(m, 0.0) - 1e-4) < 1e-12

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown motion kind"):
            evaluate_motion({"kind": "gibberish"}, 0.0)


class TestRealizeSegments:

    def test_passthrough_for_static(self):
        segs = [{"start": 0.1, "end": 0.2, "epsilon": 1e-4, "motion": None}]
        out = realize_segments(segs, 1.23)
        assert len(out) == 1
        assert out[0]["epsilon"] == 1e-4
        assert out[0]["motion"] is None

    def test_harmonic_adds_on_top_of_static(self):
        segs = [{"start": 0.0, "end": 1.0, "epsilon": 2e-4,
                 "motion": {"kind": "harmonic", "amplitude": 1e-4,
                             "frequency": 100.0, "phase": np.pi / 2}}]
        out = realize_segments(segs, 0.0)
        # static 2e-4 + harmonic(0) with phase=pi/2 -> 1e-4 -> total 3e-4
        assert abs(out[0]["epsilon"] - 3e-4) < 1e-12
        assert out[0]["motion"] is None   # stripped so callers don't re-apply

    def test_multiple_segments_independent(self):
        segs = [
            {"start": 0.0, "end": 0.3, "epsilon": 0.0,
             "motion": {"kind": "harmonic", "amplitude": 1e-4,
                         "frequency": 10.0, "phase": 0.0}},
            {"start": 0.5, "end": 0.8, "epsilon": 5e-5, "motion": None},
        ]
        out = realize_segments(segs, 1.0 / 40.0)   # T/4 of 10 Hz -> peak
        assert abs(out[0]["epsilon"] - 1e-4) < 1e-10
        assert out[1]["epsilon"] == 5e-5


class TestCampaignWithHarmonicMotion:

    def test_strain_trace_matches_expected_sinusoid(self):
        """Run a bunch of sweeps with a harmonic motion; the strain at an
        inside index should trace out sin(2*pi*f*t). T_sweep=10 ms, f=20 Hz
        -> 5 sweeps per period (sample rate 100 Hz, well below Nyquist)."""
        amp = 1e-4
        freq = 20.0      # Hz
        n_sweeps = 15
        cfg = _dyn_cfg([
            {"start": 0.3, "end": 0.5, "epsilon": 0.0,
             "motion": {"kind": "harmonic", "amplitude": amp,
                         "frequency": freq, "phase": 0.0}},
        ], n_sweeps=n_sweeps)

        acqs = run_campaign(cfg)
        z = acqs[0].z
        idx = int(np.argmin(np.abs(z - 0.4)))

        trace = np.array([a.strain_field[idx] for a in acqs])
        T_sweep = cfg["source"]["sweep_duration"]
        t = np.arange(n_sweeps) * T_sweep
        expected = amp * np.sin(2 * np.pi * freq * t)
        # sanity: trace isn't trivially zero
        assert np.max(np.abs(trace)) > 0.5 * amp
        np.testing.assert_allclose(trace, expected, atol=1e-12)

    def test_dynamic_does_not_affect_unrelated_z(self):
        """Positions outside the motion segment see zero strain every sweep."""
        cfg = _dyn_cfg([
            {"start": 0.3, "end": 0.5, "epsilon": 0.0,
             "motion": {"kind": "harmonic", "amplitude": 1e-4,
                         "frequency": 20.0, "phase": 0.0}},
        ], n_sweeps=5)

        acqs = run_campaign(cfg)
        z = acqs[0].z
        outside_idx = int(np.argmin(np.abs(z - 0.1)))
        for a in acqs:
            assert abs(a.strain_field[outside_idx]) < 1e-12

    def test_static_plus_motion_offsets(self):
        """Static epsilon + motion should offset the sine around static."""
        cfg = _dyn_cfg([
            {"start": 0.3, "end": 0.5, "epsilon": 2e-4,
             "motion": {"kind": "harmonic", "amplitude": 1e-4,
                         "frequency": 20.0, "phase": 0.0}},
        ], n_sweeps=10)

        acqs = run_campaign(cfg)
        z = acqs[0].z
        idx = int(np.argmin(np.abs(z - 0.4)))
        trace = np.array([a.strain_field[idx] for a in acqs])
        # mean should be ~ the static offset. with 10 sweeps at 20 Hz and
        # T_sweep=0.01 we cover 2 full periods -> mean(sin) ~ 0.
        assert abs(trace.mean() - 2e-4) < 5e-6

    def test_fiber_profile_differs_between_sweeps(self):
        """Dynamic strain -> the strained profile must change each sweep."""
        cfg = _dyn_cfg([
            {"start": 0.3, "end": 0.5, "epsilon": 0.0,
             "motion": {"kind": "harmonic", "amplitude": 5e-4,
                         "frequency": 20.0, "phase": 0.0}},
        ], n_sweeps=4)

        acqs = run_campaign(cfg)
        p0 = acqs[0].fiber_profile
        p1 = acqs[1].fiber_profile
        assert not np.allclose(p0, p1)

    def test_pure_static_path_unchanged(self):
        """No motion -> behaviour is identical to before (cache used, same profile)."""
        cfg = _dyn_cfg([
            {"start": 0.3, "end": 0.5, "epsilon": 1e-4},
        ], n_sweeps=3)
        acqs = run_campaign(cfg)
        # strain is time-invariant -> all sweeps see the same field
        for a in acqs[1:]:
            np.testing.assert_array_equal(a.strain_field, acqs[0].strain_field)

    def test_epsilon_defaults_to_zero_for_motion_only(self):
        """Motion-only segment (no epsilon specified) should work."""
        cfg = _dyn_cfg([
            {"start": 0.3, "end": 0.5,
             "motion": {"kind": "harmonic", "amplitude": 1e-4,
                         "frequency": 20.0, "phase": 0.0}},
        ], n_sweeps=2)
        acqs = run_campaign(cfg)
        assert acqs[0].strain_field is not None

    def test_nyquist_warning_at_or_above_half_sweep_rate(self):
        """Motion freq >= 1/(2*T_sweep) should emit a UserWarning."""
        # T_sweep = 10 ms -> Nyquist = 50 Hz
        cfg = _dyn_cfg([
            {"start": 0.3, "end": 0.5, "epsilon": 0.0,
             "motion": {"kind": "harmonic", "amplitude": 1e-4,
                         "frequency": 60.0, "phase": 0.0}},
        ], n_sweeps=2)
        with pytest.warns(UserWarning, match="Nyquist"):
            run_campaign(cfg)

    def test_nyquist_warning_at_exact_boundary(self):
        """Exactly at Nyquist is the classic zero-crossing trap."""
        cfg = _dyn_cfg([
            {"start": 0.3, "end": 0.5, "epsilon": 0.0,
             "motion": {"kind": "harmonic", "amplitude": 1e-4,
                         "frequency": 50.0, "phase": 0.0}},
        ], n_sweeps=2)
        with pytest.warns(UserWarning, match="Nyquist"):
            run_campaign(cfg)

    def test_no_warning_below_nyquist(self):
        cfg = _dyn_cfg([
            {"start": 0.3, "end": 0.5, "epsilon": 0.0,
             "motion": {"kind": "harmonic", "amplitude": 1e-4,
                         "frequency": 20.0, "phase": 0.0}},
        ], n_sweeps=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")   # turn warnings into errors
            run_campaign(cfg)                # should not raise

    def test_reflectogram_shows_motion(self):
        """End-to-end: a strained segment under harmonic motion produces
        a phase response that changes between sweeps in the reflectogram."""
        from analysis.demodulation import fft_reflectogram
        cfg = _dyn_cfg([
            {"start": 0.3, "end": 0.5, "epsilon": 0.0,
             "motion": {"kind": "harmonic", "amplitude": 5e-4,
                         "frequency": 20.0, "phase": 0.0}},
        ], n_sweeps=4)
        # put a reflector past the strained segment to pick up the phase
        cfg["fiber"] = {**cfg["fiber"],
                        "reflectors": [{"z": 0.8, "R": 0.01}]}
        cfg["detection"] = {**cfg["detection"], "shot_noise": False,
                             "thermal_nep": 0.0, "dark_current": 0.0}

        acqs = run_campaign(cfg)
        phases = []
        for a in acqs:
            H, z = fft_reflectogram(
                a.digital_main[0].astype(np.float64), a.dz)
            idx = int(np.argmin(np.abs(z - 0.8)))
            phases.append(np.angle(H[idx]))

        # phases should not all be equal (motion changes the strain shift)
        phases = np.array(phases)
        assert np.ptp(phases) > 0.01
