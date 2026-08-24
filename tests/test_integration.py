"""End-to-end integration tests."""

import numpy as np

from helpers import CFG
from pyofdr.core.campaign import run_campaign


class TestEndToEnd:

    def test_run_campaign(self):
        acq = run_campaign(CFG)[-1]
        assert acq.digital_main is not None
        assert acq.z is not None

    def test_reflectogram_has_energy_in_fiber_region(self):
        """The FFT should show most energy in the first N_z bins."""
        # noise off: with realistic shot noise (#91) the white floor spread
        # over the full band eats into the ratio -- this test is about the
        # signal geometry, not the noise
        cfg = {**CFG, "detection": {**CFG["detection"], "shot_noise": False}}
        acq = run_campaign(cfg)[-1]
        # take core 0
        spectrum = np.fft.fft(acq.digital_main[0].astype(np.float64))
        n_half = len(spectrum) // 2
        mag = np.abs(spectrum[:n_half])

        n_z = len(acq.z)
        energy_fiber = np.sum(mag[:n_z] ** 2)
        energy_total = np.sum(mag ** 2)

        # most of the energy should be in the fiber region
        assert energy_fiber / energy_total > 0.9
