"""Compare the warped beat with the optical Fourier sum, before detection."""

import numpy as np
import pytest

from helpers import CFG
from pyofdr.core.acquisition import Acquisition
from pyofdr.fiber.profile import FiberGenerator
from pyofdr.optics.mach_zehnder import MachZehnder
from pyofdr.source.swept_laser import SweptLaser
from pyofdr.utils.constants import C


def case(count, **source):
    span = C * 40e-9 / (1550e-9)**2
    dz = C / (2 * 1.4682 * span)
    cfg = {**CFG,
           "fiber": {**CFG["fiber"], "length": .4999 * count * dz, "n_cores": 2},
           "source": {**CFG["source"], "sweep_duration": count / 200e6, **source},
           "optics": {"splitting_ratio": .5,
                      "circulator": {"insertion_loss_dB": 0,
                                     "return_loss_dB": 400, "isolation_dB": 400}}}
    acq = SweptLaser(cfg).process(FiberGenerator(cfg).process(Acquisition()))
    return cfg, acq, span


@pytest.mark.parametrize("count", [4000, 4000.4])
@pytest.mark.parametrize("bin_index", [8, 800, 1600, 1998])
@pytest.mark.parametrize("offset", [-.5, .37, .5])
def test_shifted_tone_preserves_amplitude_and_phase(count, bin_index, offset, record_property):
    cfg, acq, span = case(count, flicker_noise_Hz=1)
    T = cfg["source"]["sweep_duration"]
    # A constant frequency offset gives a known fractional-sample time shift.
    acq.nu_noise = np.full(acq.n_samples, offset * acq.dt * span / T)
    acq.fiber_profile[:] = 0
    amplitudes = np.array([.7 + .2j, -.3 + .9j])
    acq.fiber_profile[:, bin_index] = amplitudes
    s = acq.t + offset * acq.dt
    expected = amplitudes[:, None] * np.exp(2j*np.pi*bin_index*s/T)
    actual = MachZehnder(cfg).process(acq).photocurrent_main / np.abs(acq.E_source)**2
    record_property("max_field_error", float(np.max(np.abs(actual-expected.real))))
    np.testing.assert_allclose(actual, expected.real, atol=2e-7, rtol=0)


@pytest.mark.parametrize("count", [128, 128.4])
@pytest.mark.parametrize("kind", ["polynomial", "ripple", "dispersion", "colored"])
def test_warped_multicore_profile_matches_direct_sum(count, kind, record_property):
    cfg, acq, span = case(count)
    T = cfg["source"]["sweep_duration"]
    t = acq.t
    gamma = span / T
    if kind == "polynomial":
        cfg["source"].update(sweep_nonlinearity_a2=.02*span/T**2,
                             sweep_nonlinearity_a3=-.03*span/T**3)
        deviation = .02*span*(t/T)**2 - .03*span*(t/T)**3
    elif kind == "ripple":
        cfg["source"].update(sweep_ripple_amplitude=.001*span, sweep_ripple_period=T/3)
        deviation = .001*span*np.sin(6*np.pi*t/T)
    elif kind == "dispersion":
        cfg["fiber"]["dispersion_D"] = 1e-3
        beta2 = -(1550e-9)**2 * 1e-3 / (2*np.pi*C)
        deviation = np.pi*beta2*gamma**2*C/1.4682*(t-T/2)**2
    else:
        cfg["source"]["flicker_noise_Hz"] = 1
        deviation = span * (.001*np.cos(2*np.pi*t/T) + .0005*np.sin(4*np.pi*t/T))
        acq.nu_noise = deviation
    rng = np.random.default_rng(317)
    shape = acq.fiber_profile.shape
    profile = .01*(rng.normal(size=shape) + 1j*rng.normal(size=shape))
    acq.fiber_profile = profile.copy()
    acq.attenuation_envelope = np.linspace(1, .3, shape[1])
    acq.ghost_profile = .003*(rng.normal(size=shape) + 1j*rng.normal(size=shape))
    weighted = profile*acq.attenuation_envelope + acq.ghost_profile
    s = t + deviation/gamma
    expected = sum(weighted[:, k, None]*np.exp(2j*np.pi*k*s/T)
                   for k in range(shape[1]))
    actual = MachZehnder(cfg).process(acq).photocurrent_main / np.abs(acq.E_source)**2
    error = np.linalg.norm(actual-expected.real) / np.linalg.norm(expected.real)
    record_property("relative_rms_error", float(error))
    assert error < 2e-7
    np.testing.assert_array_equal(acq.fiber_profile, profile)


@pytest.mark.parametrize("amplitude", [0, .7 + .2j])
def test_single_bin_profile(amplitude):
    cfg, acq, _ = case(1.5, sweep_nonlinearity_a2=1)
    assert acq.fiber_profile.shape[1] == 1
    acq.fiber_profile[:] = amplitude
    actual = MachZehnder(cfg).process(acq).photocurrent_main / np.abs(acq.E_source)**2
    np.testing.assert_allclose(actual, np.real(amplitude), atol=1e-14, rtol=0)
