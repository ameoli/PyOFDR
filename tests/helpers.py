"""Shared config and constants for the test suite."""

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# short fiber for fast tests
CFG = {
    "simulation": {"seed": 42},
    "fiber": {"length": 1.0, "n_core": 1.4682, "rayleigh_coefficient_dB": -82.0},
    "source": {"center_wavelength": 1550e-9, "sweep_range": 40e-9,
               "sweep_duration": 0.01, "power": 10e-3},
    # iso/RL pushed effectively off so the z=0 bin stays clean for tests
    # that look at fft.max() or hilbert envelopes. specific circulator
    # tests override these.
    "optics": {"splitting_ratio": 0.5,
               "circulator": {"isolation_dB": 400.0, "return_loss_dB": 400.0}},
    "detection": {"responsivity": 1.0},
    "adc": {"bits": 16, "sample_rate": 200e6, "voltage_range": 2.0,
            "input_impedance": 50.0},
}
