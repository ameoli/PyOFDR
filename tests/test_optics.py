"""Tests for Mach-Zehnder interferometer and circulator."""

import numpy as np
import pytest

from helpers import CFG
from core.acquisition import Acquisition
from fiber.profile import FiberGenerator
from source.swept_laser import SweptLaser
from optics.mach_zehnder import MachZehnder
from optics.components import Circulator


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
