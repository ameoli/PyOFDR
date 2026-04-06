"""Basic tests for the v0.1 pipeline.

Just checks that each step runs and produces reasonable output.
More rigorous analytical tests (beat frequency, Rayleigh statistics,
attenuation slope, etc) will come later once the physics is more
complete.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.acquisition import Acquisition
from core.campaign import run_campaign
from fiber.profile import FiberGenerator
from source.swept_laser import SweptLaser
from optics.mach_zehnder import MachZehnder
from detection.detector import Detector
from digitizer.adc import ADC

# short fiber for fast tests
CFG = {
    "simulation": {"seed": 42},
    "fiber": {"length": 1.0, "n_core": 1.4682, "rayleigh_coefficient_dB": -82.0},
    "source": {"center_wavelength": 1550e-9, "sweep_range": 40e-9,
               "sweep_duration": 0.01, "power": 10e-3},
    "optics": {"splitting_ratio": 0.5},
    "detection": {"responsivity": 1.0},
    "adc": {"bits": 16, "sample_rate": 200e6, "voltage_range": 2.0,
            "input_impedance": 50.0},
}


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
        assert len(acq.fiber_profile) == len(acq.z)


class TestSweptLaser:

    def _make_acq(self):
        acq = FiberGenerator(CFG).process(Acquisition())
        return SweptLaser(CFG).process(acq)

    def test_time_axis_exists(self):
        acq = self._make_acq()
        assert acq.t is not None
        assert acq.n_samples > 0

    def test_field_is_complex(self):
        acq = self._make_acq()
        assert acq.E_source.dtype == np.complex128

    def test_frequency_increases_monotonically(self):
        acq = self._make_acq()
        dnu = np.diff(acq.nu_inst)
        assert np.all(dnu > 0)

    def test_optical_power_matches_config(self):
        acq = self._make_acq()
        P = np.mean(np.abs(acq.E_source) ** 2)
        np.testing.assert_allclose(P, 10e-3, rtol=0.01)


class TestMachZehnder:

    def _make_acq(self):
        acq = Acquisition()
        acq = FiberGenerator(CFG).process(acq)
        acq = SweptLaser(CFG).process(acq)
        return MachZehnder(CFG).process(acq)

    def test_photocurrent_exists(self):
        acq = self._make_acq()
        assert acq.photocurrent_main is not None
        assert len(acq.photocurrent_main) == acq.n_samples

    def test_photocurrent_is_real(self):
        acq = self._make_acq()
        # should be float, not complex
        assert acq.photocurrent_main.dtype == np.float64


class TestDetector:

    def test_analog_output_exists(self):
        acq = Acquisition()
        for step_cls in [FiberGenerator, SweptLaser, MachZehnder, Detector]:
            acq = step_cls(CFG).process(acq)
        assert acq.analog_main is not None


class TestADC:

    def _run_full(self):
        acq = Acquisition()
        for cls in [FiberGenerator, SweptLaser, MachZehnder, Detector, ADC]:
            acq = cls(CFG).process(acq)
        return acq

    def test_digital_is_int16(self):
        acq = self._run_full()
        assert acq.digital_main.dtype == np.int16

    def test_digital_in_range(self):
        acq = self._run_full()
        assert np.all(acq.digital_main >= -32768)
        assert np.all(acq.digital_main <= 32767)


class TestEndToEnd:

    def test_run_campaign(self):
        acq = run_campaign(CFG)
        assert acq.digital_main is not None
        assert acq.z is not None

    def test_reflectogram_has_energy_in_fiber_region(self):
        """The FFT should show most energy in the first N_z bins."""
        acq = run_campaign(CFG)
        spectrum = np.fft.fft(acq.digital_main.astype(np.float64))
        n_half = len(spectrum) // 2
        mag = np.abs(spectrum[:n_half])

        n_z = len(acq.z)
        energy_fiber = np.sum(mag[:n_z] ** 2)
        energy_total = np.sum(mag ** 2)

        # most of the energy should be in the fiber region
        assert energy_fiber / energy_total > 0.9
