"""Tests for macrobending loss (see #36)."""

import math

import numpy as np
import pytest

from helpers import CFG
from core.acquisition import Acquisition
from fiber.bends import bend_loss_dB
from fiber.profile import FiberGenerator


class TestBendLossFormula:

    def test_loss_scales_linearly_with_turns(self):
        a = bend_loss_dB(radius=8e-3, turns=1)
        b = bend_loss_dB(radius=8e-3, turns=10)
        assert b == pytest.approx(10 * a)

    def test_large_radius_negligible(self):
        # R >> R_c -> practically zero loss
        assert bend_loss_dB(radius=100e-3, turns=10) < 1e-5

    def test_tight_radius_significant(self):
        # R = R_c -> loss_per_turn = A / e ~ 36.8 dB
        loss = bend_loss_dB(radius=5e-3, turns=1)
        assert loss == pytest.approx(100.0 / math.e, rel=1e-6)

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError):
            bend_loss_dB(radius=0.0, turns=1)

    def test_custom_constants(self):
        loss = bend_loss_dB(radius=10e-3, turns=1,
                             A_dB_per_turn=50.0, R_c=10e-3)
        assert loss == pytest.approx(50.0 / math.e, rel=1e-6)


class TestBendsInProfile:

    def _run(self, bends=None):
        cfg = {**CFG, "fiber": {**CFG["fiber"],
               "attenuation_dB_per_km": 0.0,
               "bends": bends or []}}
        return FiberGenerator(cfg).process(Acquisition())

    def test_no_bends_matches_baseline(self):
        a = self._run(None)
        b = FiberGenerator(CFG).process(Acquisition())
        np.testing.assert_array_equal(a.fiber_profile, b.fiber_profile)
        np.testing.assert_allclose(a.attenuation_envelope,
                                    b.attenuation_envelope, rtol=1e-6)

    def test_large_radius_bend_negligible(self):
        # 100 mm radius bend -> basically no effect
        bends = [{"start": 0.3, "end": 0.5, "radius": 100e-3, "turns": 5}]
        acq = self._run(bends)
        bare = FiberGenerator({**CFG, "fiber": {**CFG["fiber"],
               "attenuation_dB_per_km": 0.0}}).process(Acquisition())
        np.testing.assert_allclose(acq.attenuation_envelope,
                                    bare.attenuation_envelope, atol=1e-4)

    def test_tight_bend_drops_envelope(self):
        # 15 mm, 1 turn -> ~5 dB one-way, 10 dB round-trip
        # amplitude envelope convention is exp(-alpha_np*z) which, squared,
        # gives round-trip power loss -> amplitude factor ~ 10^(-10/20) = 0.32
        bends = [{"start": 0.4, "end": 0.5,
                  "radius": 15e-3, "turns": 1}]
        acq = self._run(bends)
        env = acq.attenuation_envelope
        z = acq.z
        before = env[(z >= 0.3) & (z < 0.4)].mean()
        after  = env[(z >= 0.5) & (z < 0.6)].mean()
        ratio = after / before
        # expected: 10^(-2*bend_loss_dB/20) = 10^(-bend_loss/10)
        expected = 10 ** (-bend_loss_dB(15e-3, 1) / 10.0)
        assert abs(ratio - expected) / expected < 0.05

    def test_envelope_still_monotonic(self):
        bends = [{"start": 0.2, "end": 0.3, "radius": 10e-3, "turns": 2},
                 {"start": 0.6, "end": 0.7, "radius": 8e-3,  "turns": 1}]
        acq = self._run(bends)
        env = acq.attenuation_envelope
        assert np.all(np.diff(env) <= 1e-12)

    def test_more_turns_more_loss(self):
        b1 = self._run([{"start": 0.4, "end": 0.5, "radius": 7e-3, "turns": 1}])
        b5 = self._run([{"start": 0.4, "end": 0.5, "radius": 7e-3, "turns": 5}])
        z = b1.z
        idx = np.argmin(np.abs(z - 0.7))
        assert b5.attenuation_envelope[idx] < b1.attenuation_envelope[idx]
