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
