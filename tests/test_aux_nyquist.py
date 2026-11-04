"""Sampling limits of the auxiliary interferometer."""

import numpy as np
import pytest

from pyofdr.core.acquisition import Acquisition
from pyofdr.core.config_models import RootConfig
from pyofdr.optics.aux_mzi import AuxMZI
from pyofdr.source.swept_laser import SweptLaser
from pyofdr.utils.constants import C


def config(gamma=8., delay=.25, enabled=True):
    # Binary sample times make the exact Nyquist boundary unambiguous.
    return {
        "source": {"center_wavelength": 1., "sweep_range": gamma/C,
                   "sweep_duration": 1., "linewidth": 0.},
        "adc": {"sample_rate": 8.},
        "optics": {"aux_mzi": {"enabled": enabled, "delay": delay}},
    }


def acquisition(nu, dt=.125):
    return Acquisition(nu_inst=np.asarray(nu, dtype=float), dt=dt, n_samples=len(nu))


def test_reported_499_mhz_case_rejected_before_simulation():
    with pytest.raises(ValueError, match="aux.*Nyquist"):
        RootConfig(optics={"aux_mzi": {"enabled": True, "delay": "1 us"}})


@pytest.mark.parametrize("gamma", [16., 20.])
def test_nominal_at_or_above_nyquist_is_rejected(gamma):
    with pytest.raises(ValueError, match="aux.*Nyquist"):
        RootConfig(**config(gamma=gamma))


def test_nominal_just_below_nyquist_is_accepted():
    RootConfig(**config(gamma=16*(1-1e-6)))


@pytest.mark.parametrize("delay, rejected", [(.26, False), (.36, True)])
def test_nominal_check_uses_rounded_delay(delay, rejected):
    # Requested delay * gamma is >4 Hz for both. Rounded lag is 2 or 3.
    cfg = config(gamma=15.5, delay=delay)
    if rejected:
        with pytest.raises(ValueError, match="aux.*Nyquist"):
            RootConfig(**cfg)
    else:
        RootConfig(**cfg)


def test_disabled_aux_does_not_reject_unused_delay():
    cfg = config(gamma=20., enabled=False)
    acq = acquisition(np.arange(8)*100.)
    assert AuxMZI(cfg).process(acq).aux_signal is None


@pytest.mark.parametrize("step", [2., 3., -2., -3.])
def test_runtime_rejects_at_or_above_nyquist_in_either_direction(step):
    aux = AuxMZI(config())  # nominal beat 2 Hz, Nyquist 4 Hz
    with pytest.raises(ValueError, match="aux.*Nyquist"):
        aux.process(acquisition(np.arange(8)*step))


@pytest.mark.parametrize("step", [1.999, -1.999])
def test_runtime_accepts_below_nyquist_without_changing_signal(step):
    nu = np.arange(8)*step
    acq = AuxMZI(config()).process(acquisition(nu))
    phase = 2*np.pi*np.array([sum(nu[j-1:j+1]) for j in range(2, 8)])*.125
    np.testing.assert_allclose(acq.aux_signal[2:], np.cos(phase), atol=1e-13)


def test_runtime_uses_acquisition_sampling_rate():
    # The same delay is 4 samples here; the actual Nyquist is 8 Hz.
    with pytest.raises(ValueError, match="aux.*Nyquist"):
        AuxMZI(config()).process(acquisition(np.arange(16)*2., dt=.0625))


@pytest.mark.parametrize("source_change", [
    {"sweep_nonlinearity_a2": 8.},
    {"sweep_nonlinearity_a3": 8.},
    {"sweep_ripple_amplitude": 4., "sweep_ripple_period": .5},
    {"flicker_noise_Hz": 100.},
    {"random_walk_noise_Hz": 100.},
    {"linewidth": 100.},
])
def test_runtime_checks_realized_nonlinearity_and_frequency_noise(source_change):
    cfg = config()
    cfg["source"].update(source_change)
    RootConfig(**cfg)  # the nominal linear beat alone is safe
    acq = SweptLaser(cfg).process(Acquisition())
    with pytest.raises(ValueError, match="aux.*Nyquist"):
        AuxMZI(cfg).process(acq)


def test_local_excursion_is_rejected_even_when_mean_beat_is_safe():
    nu = np.arange(8, dtype=float)
    nu[4] += 10
    assert abs(np.mean(nu[2:]-nu[:-2])) < 4
    with pytest.raises(ValueError, match="aux.*Nyquist"):
        AuxMZI(config()).process(acquisition(nu))


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_beat_cannot_pass_the_check(value):
    nu = np.arange(8, dtype=float)
    nu[4] = value
    with pytest.raises(ValueError, match="finite"):
        AuxMZI(config()).process(acquisition(nu))


def test_delay_must_leave_two_valid_samples_at_config_time():
    with pytest.raises(ValueError, match="two valid samples"):
        RootConfig(**config(gamma=1., delay=.875))


def test_delay_must_leave_two_valid_samples_at_runtime():
    with pytest.raises(ValueError, match="two valid samples"):
        AuxMZI(config()).process(acquisition([0., 1., 2.]))
