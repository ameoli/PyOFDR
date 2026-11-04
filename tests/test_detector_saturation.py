"""Per-photodiode saturation, before balanced subtraction or AC coupling."""

import numpy as np
import pytest

from helpers import CFG
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.campaign import run_campaign
from pyofdr.core.config_models import RootConfig
from pyofdr.detection.detector import Detector
from pyofdr.utils.constants import E_CHARGE


def config(balanced=False, limit=.003, **detection):
    return {**CFG, "detection": {**CFG["detection"], "balanced": balanced,
                                 "saturation_current": limit, "shot_noise": False,
                                 "thermal_nep": 0, "dark_current": 0, **detection}}


def current(beat, cfg, sweep=0):
    detector = Detector(cfg)
    beat = np.atleast_2d(np.asarray(beat, dtype=float))
    acq = Acquisition(photocurrent_main=beat/detector.responsivity,
                      dt=1/200e6, sweep_index=sweep)
    detector.process(acq)
    return acq.analog_main/detector.impedance


@pytest.mark.parametrize("balanced,limit,beat_mA,expected_mA", [
    (False, .003, [-1, 0, .25, 1], [-1, 0, .25, .5]),
    (False, .001, [-2, -1, 0, .1], [-.5, 0, 0, 0]),
    (True, .003, [-2, -.5, 0, .5, 2], [-1.5, -.5, 0, .5, 1.5]),
    (True, .001, [-.1, 0, .1], [0, 0, 0]),
    (False, .003, [-4, -3, -2.5], [-2.5, -2.5, -2.5]),
])
def test_clips_total_current_in_each_photodiode(balanced, limit, beat_mA, expected_mA):
    # Default reference gives 2.5 mA DC per diode, even with a tiny AC beat.
    actual = current(np.array(beat_mA)*1e-3, config(balanced, limit))
    np.testing.assert_allclose(actual[0], np.array(expected_mA)*1e-3, rtol=0, atol=1e-17)


@pytest.mark.parametrize("balanced,expected_mA", [(False, [-.6, 0, 0]),
                                                 (True, [-.6, 0, .6])])
def test_polynomial_response_cannot_exceed_saturation(balanced, expected_mA):
    cfg = config(balanced, .003, nonlinearity_coefficients=[400])
    beat = np.array([-2, 0, 2] if balanced else [-1, 0, 1])*1e-3
    actual = current(beat, cfg)
    np.testing.assert_allclose(actual[0], np.array(expected_mA)*1e-3, rtol=0, atol=1e-17)


@pytest.mark.parametrize("power,responsivity,eta,expected_mA", [
    (.004, 1, .5, .5), (.01, .5, .5, .25), (.01, 1, .2, .5),
])
def test_dc_headroom_follows_power_responsivity_and_splitter(power, responsivity, eta, expected_mA):
    cfg = config(False, .0015, responsivity=responsivity)
    cfg["source"] = {**cfg["source"], "power": power}
    cfg["optics"] = {**cfg["optics"], "splitting_ratio": eta}
    actual = current([.00075], cfg)
    np.testing.assert_allclose(actual, expected_mA*1e-3, rtol=0, atol=1e-17)


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("coefficients", [[], [0, 0], [10, 1000]])
def test_inactive_limit_preserves_noiseless_response(balanced, coefficients):
    beat = np.array([[0, 1e-10, -1e-10, 1e-5], [1e-12, -1e-12, 0, -1e-5]])
    uncapped = current(beat, config(balanced, None, nonlinearity_coefficients=coefficients))
    capped = current(beat, config(balanced, 1, nonlinearity_coefficients=coefficients))
    np.testing.assert_array_equal(capped, uncapped)


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("limit", [.001, .0025, .01])
def test_shot_noise_uses_collected_current(balanced, limit):
    cfg = config(balanced, limit, shot_noise=True)
    noise = current(np.zeros((2, 200000)), cfg)
    n_pd = 2 if balanced else 1
    expected = 2*E_CHARGE*n_pd*min(.0025, limit)*100e6
    np.testing.assert_allclose(np.var(noise, axis=1), expected, rtol=.02)
    assert abs(np.corrcoef(noise)[0, 1]) < .02
    np.testing.assert_array_equal(noise, current(np.zeros((2, 200000)), cfg))
    assert not np.array_equal(noise, current(np.zeros((2, 200000)), cfg, sweep=1))


@pytest.mark.parametrize("balanced,beat_mA,mean_mA,noise_current_mA", [
    (False, 1, .5, 3), (True, 2, 1.5, 4.5),
])
def test_shot_noise_when_only_one_branch_clips(balanced, beat_mA, mean_mA, noise_current_mA):
    signal = np.full(200000, beat_mA*1e-3)
    output = current(signal, config(balanced, .003, shot_noise=True))[0]
    noise = output - mean_mA*1e-3
    expected = 2*E_CHARGE*noise_current_mA*1e-3*100e6
    assert np.var(noise) == pytest.approx(expected, rel=.02)


@pytest.mark.parametrize("balanced", [False, True])
def test_thermal_and_dark_noise_remain_after_saturation(balanced):
    beat = np.zeros(100000)
    for source in [{"thermal_nep": 1e-10}, {"dark_current": 1e-6}]:
        overdriven = current(beat, config(balanced, .001, **source))
        linear = current(beat, config(balanced, None, **source))
        np.testing.assert_array_equal(overdriven, linear)
        assert np.std(overdriven) > 0


@pytest.mark.parametrize("balanced", [False, True])
def test_saturation_in_full_pipeline(balanced):
    cfg = config(balanced, .001)
    cfg["source"] = {**cfg["source"], "sweep_duration": 1e-4, "sweep_range": 1e-9}
    acq = run_campaign(cfg)[0]
    np.testing.assert_array_equal(acq.analog_main, 0)
    np.testing.assert_array_equal(acq.digital_main, 0)


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf"), "-1 mA"])
def test_saturation_limit_must_be_positive_and_finite(limit):
    with pytest.raises(ValueError):
        RootConfig(**config(limit=limit))


def test_saturation_limit_accepts_units():
    cfg = RootConfig(**config(limit="1 mA"))
    assert cfg.detection.saturation_current == .001
