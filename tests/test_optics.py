"""Tests for Mach-Zehnder interferometer and circulator."""

import math

import numpy as np
import pytest

from helpers import CFG
from core.acquisition import Acquisition
from fiber.profile import FiberGenerator
from source.swept_laser import SweptLaser
from optics.mach_zehnder import MachZehnder
from optics.components import Circulator
from utils.constants import C


class TestMachZehnder:

    def _make_acq(self, cfg=None):
        cfg = cfg or CFG
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        acq = SweptLaser(cfg).process(acq)
        return MachZehnder(cfg).process(acq)

    def test_photocurrent_exists(self):
        acq = self._make_acq()
        assert acq.photocurrent_main is not None
        assert acq.photocurrent_main.shape[-1] == acq.n_samples

    def test_photocurrent_is_real(self):
        acq = self._make_acq()
        # should be float, not complex
        assert acq.photocurrent_main.dtype == np.float64


class TestCoherenceRolloff:
    """Lorentzian coherence roll-off with finite laser linewidth (#62)."""

    def test_zero_linewidth_matches_no_rolloff(self):
        # with linewidth=0 the visibility is identically 1, so the beat
        # must be bit-for-bit identical to the no-rolloff path
        cfg0 = {**CFG, "source": {**CFG["source"], "linewidth": 0.0}}
        acq = Acquisition()
        acq = FiberGenerator(cfg0).process(acq)
        acq = SweptLaser(cfg0).process(acq)
        acq = MachZehnder(cfg0).process(acq)
        assert acq.log[-1]["coherence_rolloff"] is False

    def test_visibility_matches_lorentzian(self):
        # single reflector, compare FFT magnitude at its expected bin with
        # vs without linewidth. Ratio should equal exp(-pi * dnu * tau).
        # Linewidth picked to land visibility near 0.5 -- far from the
        # Rayleigh/numerical floor.
        length = 5.0
        reflector_z = 4.5
        linewidth = 5.0e6
        cfg_base = {**CFG,
                    "fiber":  {**CFG["fiber"], "length": length,
                               "reflectors": [{"z": reflector_z, "R": 1e-4}],
                               "rayleigh_coefficient_dB": -200.0},
                    "source": {**CFG["source"], "linewidth": 0.0}}
        cfg_lw = {**cfg_base, "source": {**CFG["source"], "linewidth": linewidth}}

        def _peak(cfg):
            acq = Acquisition()
            acq = FiberGenerator(cfg).process(acq)
            acq = SweptLaser(cfg).process(acq)
            acq = MachZehnder(cfg).process(acq)
            fft = np.abs(np.fft.rfft(acq.photocurrent_main[0]))
            return fft.max()

        p0 = _peak(cfg_base)
        p1 = _peak(cfg_lw)
        n_core = CFG["fiber"]["n_core"]
        tau = 2.0 * n_core * reflector_z / C
        expected = math.exp(-math.pi * linewidth * tau)
        assert p1 / p0 == pytest.approx(expected, rel=0.05)

    def test_log_flag_on(self):
        cfg = {**CFG, "source": {**CFG["source"], "linewidth": 1.0e4}}
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        acq = SweptLaser(cfg).process(acq)
        acq = MachZehnder(cfg).process(acq)
        assert acq.log[-1]["coherence_rolloff"] is True


class TestPowerModulation:
    """Source power variations (edge droop, RIN) must propagate to the beat."""

    @staticmethod
    def _reflector_cfg(**src_overrides):
        # single strong reflector, rayleigh pushed way down, no coherence
        # roll-off -> beat is a clean sinusoid whose analytic envelope tracks P(t).
        return {**CFG,
                "fiber":  {**CFG["fiber"], "length": 5.0,
                           "reflectors": [{"z": 4.5, "R": 1e-4}],
                           "rayleigh_coefficient_dB": -200.0},
                "source": {**CFG["source"], "linewidth": 0.0,
                           "rin_dB_per_Hz": None,
                           "power_envelope_edge_dB": 0.0,
                           **src_overrides}}

    def _run(self, cfg):
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        acq = SweptLaser(cfg).process(acq)
        return MachZehnder(cfg).process(acq)

    def test_envelope_droop_reaches_beat(self):
        # 6 dB parabolic droop on source power must show up as 6 dB drop
        # of the beat's analytic envelope at the sweep edges.
        from scipy.signal import hilbert
        edge_dB = 6.0
        cfg = self._reflector_cfg(power_envelope_edge_dB=edge_dB)
        acq = self._run(cfg)
        beat = acq.photocurrent_main[0]
        env = np.abs(hilbert(beat))
        n = len(env)
        # peak at center, edge samples at t=0 and t~T
        env_center = env[n // 2]
        env_edge   = 0.5 * (env[0] + env[-1])
        drop_dB = 10.0 * np.log10(env_edge / env_center)
        np.testing.assert_allclose(drop_dB, -edge_dB, atol=0.3)

    def test_rin_increases_beat_variance(self):
        # turning RIN on must increase var(beat) by ~sigma_rin^2 (relative).
        # at -85 dB/Hz with BW=100 MHz, sigma_rin^2 ~ 0.316 -> ratio ~1.3
        rin_dB = -85.0
        clean = self._run(self._reflector_cfg())
        noisy = self._run(self._reflector_cfg(rin_dB_per_Hz=rin_dB))
        bw = CFG["adc"]["sample_rate"] / 2.0
        expected_ratio = 1.0 + 10.0 ** (rin_dB / 10.0) * bw
        ratio = np.var(noisy.photocurrent_main) / np.var(clean.photocurrent_main)
        np.testing.assert_allclose(ratio, expected_ratio, rtol=0.1)

    def test_flat_power_scales_linearly_with_P0(self):
        # constant P: doubling the source power must double the beat.
        # guards against a regression where a stale mean-power scalar
        # would decouple the beat from the current P0.
        cfg_1 = self._reflector_cfg()
        cfg_2 = {**cfg_1, "source": {**cfg_1["source"],
                                     "power": 2.0 * cfg_1["source"]["power"]}}
        a1 = self._run(cfg_1)
        a2 = self._run(cfg_2)
        # compare where beat_1 is safely away from zero crossings
        b1 = a1.photocurrent_main[0]
        mask = np.abs(b1) > 0.5 * np.abs(b1).max()
        ratio = a2.photocurrent_main[0][mask] / b1[mask]
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-10)


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
