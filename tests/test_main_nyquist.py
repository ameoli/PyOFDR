"""Check the unwrapped main-beat phase before sampling the warped fiber sum."""

import numpy as np
import pytest

from pyofdr.core.acquisition import Acquisition
from pyofdr.core.config_models import RootConfig
from pyofdr.fiber.profile import FiberGenerator
from pyofdr.optics.mach_zehnder import MachZehnder
from pyofdr.source.swept_laser import SweptLaser
from pyofdr.utils.constants import C


def config(**source):
    # Five spatial bins: the last tone is 4 Hz, ADC Nyquist is 8 Hz.
    return {"fiber": {"length": 4.5*C/(2*1.4682*16), "n_cores": 2},
            "source": {"center_wavelength": 1., "sweep_range": 16/C,
                       "sweep_duration": 1., "linewidth": 0., **source},
            "adc": {"sample_rate": 16.},
            "optics": {"circulator": {"insertion_loss_dB": 0.,
                                       "return_loss_dB": 400., "isolation_dB": 400.}}}


def acquisition(cfg):
    return SweptLaser(cfg).process(FiberGenerator(cfg).process(Acquisition()))


@pytest.mark.parametrize("count", [2000, 2000.4])
def test_fast_ripple_from_review_rejected_with_aux_disabled(count):
    span = C*4e-9/(1550e-9)**2
    cfg = {"fiber": {"length": .15},
           "source": {"sweep_range": 4e-9, "sweep_duration": count/20e6,
                      "linewidth": 0., "sweep_ripple_amplitude": .003*span,
                      "sweep_ripple_period": 1e-6},
           "adc": {"sample_rate": 20e6}}
    assert not RootConfig(**cfg).optics.aux_mzi.enabled
    acq = acquisition(cfg)
    with pytest.raises(ValueError, match="main.*Nyquist"):
        MachZehnder(cfg).process(acq)
    assert acq.photocurrent_main is None


@pytest.mark.parametrize("slope", [2., 3., -2., -3.])
@pytest.mark.parametrize("noise_kind", ["flicker_noise_Hz", "random_walk_noise_Hz"])
def test_at_or_above_nyquist_rejected_in_both_directions(slope, noise_kind):
    cfg = config(**{noise_kind: 1.})
    acq = acquisition(cfg)
    acq.nu_noise = 16*(slope-1)*acq.t
    with pytest.raises(ValueError, match="main.*Nyquist"):
        MachZehnder(cfg).process(acq)


@pytest.mark.parametrize("slope", [1.999, -1.999, 0., 1.])
def test_safe_warp_preserves_tone_amplitude_and_phase(slope):
    cfg = config(flicker_noise_Hz=1.)
    acq = acquisition(cfg)
    # A constant offset avoids an out-of-sweep warning for reversed time.
    offset = (1-slope)*acq.t[-1]/2
    acq.nu_noise = 16*((slope-1)*acq.t+offset)
    acq.fiber_profile[:] = 0
    acq.fiber_profile[:, 4] = [.3+.2j, -.1+.5j]
    expected = np.array([.3+.2j, -.1+.5j])[:, None]*np.exp(2j*np.pi*4*(slope*acq.t+offset))
    actual = MachZehnder(cfg).process(acq).photocurrent_main/np.abs(acq.E_source)**2
    np.testing.assert_allclose(actual, expected.real, atol=2e-7, rtol=0)


@pytest.mark.parametrize("source", [
    {"sweep_nonlinearity_a2": 16.},
    {"sweep_nonlinearity_a2": -32.},
    {"sweep_nonlinearity_a3": 16.},
    {"sweep_ripple_amplitude": 16., "sweep_ripple_period": .5},
])
def test_deterministic_sweep_excursions_rejected(source):
    cfg = config(**source)
    with pytest.raises(ValueError, match="main.*Nyquist"):
        MachZehnder(cfg).process(acquisition(cfg))


@pytest.mark.parametrize("dispersion", [-1., 1.])
def test_dispersion_is_included_in_sampling_check(dispersion):
    cfg = config()
    cfg["fiber"]["dispersion_D"] = dispersion
    with pytest.raises(ValueError, match="main.*Nyquist"):
        MachZehnder(cfg).process(acquisition(cfg))


def test_local_excursion_rejected_even_if_average_slope_is_safe():
    cfg = config(flicker_noise_Hz=1.)
    acq = acquisition(cfg)
    acq.nu_noise[:] = 0
    acq.nu_noise[8] = 4.
    assert np.mean(1+np.diff(acq.nu_noise)/(16*acq.dt)) == 1.
    with pytest.raises(ValueError, match="main.*Nyquist"):
        MachZehnder(cfg).process(acq)


def test_uses_acquisition_clock_and_includes_ghosts():
    cfg = config(flicker_noise_Hz=1.)
    acq = acquisition(cfg)
    # Keep the sweep range but use a slower acquisition than the config.
    acq.dt = .125
    acq.t = np.arange(acq.n_samples)*acq.dt
    acq.nu_noise[:] = 0
    acq.fiber_profile[:] = 0
    acq.ghost_profile = np.zeros_like(acq.fiber_profile)
    acq.ghost_profile[1, 4] = 1.
    with pytest.raises(ValueError, match="main.*Nyquist"):
        MachZehnder(cfg).process(acq)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_warp_rejected(value):
    cfg = config(flicker_noise_Hz=1.)
    acq = acquisition(cfg)
    acq.nu_noise[:] = 0
    acq.nu_noise[8] = value
    with pytest.raises(ValueError, match="finite"):
        MachZehnder(cfg).process(acq)


def test_unwarped_linear_sweep_matches_direct_fourier_sum():
    cfg = config()
    acq = acquisition(cfg)
    weighted = acq.fiber_profile*acq.attenuation_envelope
    expected = sum(weighted[:, k, None]*np.exp(2j*np.pi*k*acq.t)
                   for k in range(weighted.shape[1]))
    actual = MachZehnder(cfg).process(acq).photocurrent_main/np.abs(acq.E_source)**2
    np.testing.assert_allclose(actual, expected.real, atol=1e-14, rtol=0)
