"""FBG envelopes and propagation phases must follow the same laser sweep."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from pyofdr.core.acquisition import Acquisition
from pyofdr.fiber.profile import FiberGenerator
from pyofdr.optics.mach_zehnder import MachZehnder
from pyofdr.source.swept_laser import SweptLaser
from pyofdr.utils.constants import C


def case(count=2000, kind="linear", sweep=0):
    wl, n_core = 1550e-9, 1.4682
    span = C*4e-9/wl**2
    T = count/20e6
    dz = C/(2*n_core*span)
    cfg = {"fiber": {"length": .15, "n_cores": 2, "fbg_arrays": [
               {"z": 400*dz, "length": 1e-4, "bragg_wavelength": wl, "peak_reflectivity": .01}]},
           "source": {"sweep_range": 4e-9, "sweep_duration": T, "linewidth": 0.},
           "adc": {"sample_rate": 20e6},
           "optics": {"circulator": {"insertion_loss_dB": 0.,
                                      "return_loss_dB": 400., "isolation_dB": 400.}}}
    if kind == "quadratic":
        cfg["source"]["sweep_nonlinearity_a2"] = .005*span/T**2
    elif kind == "cubic":
        cfg["source"]["sweep_nonlinearity_a3"] = -.005*span/T**3
    elif kind == "ripple":
        cfg["source"].update(sweep_ripple_amplitude=.002*span, sweep_ripple_period=T/3)
    elif kind in ["dispersion_pos", "dispersion_neg"]:
        cfg["fiber"]["dispersion_D"] = .01 if kind == "dispersion_pos" else -.01
    elif kind == "colored":
        cfg["source"].update(flicker_noise_Hz=1e7, random_walk_noise_Hz=1e7)
    acq = SweptLaser(cfg).process(FiberGenerator(cfg).process(Acquisition(sweep_index=sweep)))
    acq.fiber_profile[:] = 0.
    t = acq.t
    # Optical frequency deviation entering propagation phase (Hz).
    delta = np.zeros_like(t)
    if kind == "quadratic":
        delta = .005*span*(t/T)**2
    elif kind == "cubic":
        delta = -.005*span*(t/T)**3
    elif kind == "ripple":
        delta = .002*span*np.sin(6*np.pi*t/T)
    elif kind.startswith("dispersion"):
        beta2 = -wl**2*cfg["fiber"]["dispersion_D"]/(2*np.pi*C)
        delta = np.pi*beta2*(span/T)**2*C/n_core*(t-T/2)**2
    elif kind == "colored":
        delta = acq.nu_noise.copy()
    return cfg, acq, span, delta


def expected_signal(cfg, acq, span, delta):
    n = 1.4682
    T = cfg["source"]["sweep_duration"]
    signal = np.zeros(acq.n_samples)
    for fbg in cfg["fiber"]["fbg_arrays"]:
        k = min(round(fbg["z"]/acq.dz), len(acq.z)-1)
        shift = (0 if acq.strain_field is None else .78*acq.strain_field[k])
        shift += (0 if acq.temperature_field is None else 7.05e-6*acq.temperature_field[k])
        nu_B = C/(fbg["bragg_wavelength"]*(1+shift))
        tau = 2*n*(k*acq.dz)/C
        envelope = np.sqrt(fbg["peak_reflectivity"])*acq.attenuation_envelope[k]
        envelope *= np.exp(-np.pi*cfg["source"]["linewidth"]*tau)
        envelope = envelope*np.sinc(2*n*fbg["length"]*(acq.nu_inst-nu_B)/C)
        phase = 2*np.pi*tau*(span/T*acq.t+delta)
        signal += envelope*np.cos(phase)
    return signal*np.abs(acq.E_source)**2


@pytest.mark.parametrize("count", [2000, 2000.4])
@pytest.mark.parametrize("kind", ["linear", "quadratic", "cubic", "ripple",
                                  "dispersion_pos", "dispersion_neg", "colored"])
def test_fbg_matches_analytic_phase_and_envelope(count, kind):
    cfg, acq, span, delta = case(count, kind)
    expected = expected_signal(cfg, acq, span, delta)
    actual = MachZehnder(cfg).process(acq).photocurrent_main
    for core in actual:
        np.testing.assert_allclose(core, expected, atol=2e-14, rtol=0)
    if kind == "quadratic":
        # The review case had nearly two missing phase cycles at the far end.
        assert 400*delta[-1]/span > 1.99


def test_nonlinear_fbg_matches_a_reflector_with_its_spectral_envelope():
    cfg, acq, span, delta = case(kind="quadratic")
    with_fbg = MachZehnder(cfg).process(acq).photocurrent_main
    reflector_cfg = deepcopy(cfg)
    reflector_cfg["fiber"]["fbg_arrays"] = []
    reflector_acq = replace(acq, fiber_profile=np.zeros_like(acq.fiber_profile), log=[])
    reflector_acq.fiber_profile[:, 400] = .1
    reflector = MachZehnder(reflector_cfg).process(reflector_acq).photocurrent_main
    envelope = np.sinc(2*1.4682*1e-4*(acq.nu_inst-C/1550e-9)/C)
    np.testing.assert_allclose(with_fbg, reflector*envelope, atol=2e-10, rtol=0)


def test_warp_preserves_attenuation_coherence_and_resonance_shifts():
    cfg, acq, span, delta = case(kind="ripple")
    cfg["source"]["linewidth"] = 1e8
    cfg["fiber"]["fbg_arrays"].append({"z": cfg["fiber"]["length"], "length": 2e-4,
                                        "bragg_wavelength": 1551e-9, "peak_reflectivity": .04})
    acq.attenuation_envelope = np.linspace(1., .3, len(acq.z))
    acq.strain_field = np.linspace(-1e-4, 1e-4, len(acq.z))
    acq.temperature_field = np.linspace(-5., 5., len(acq.z))
    expected = expected_signal(cfg, acq, span, delta)
    actual = MachZehnder(cfg).process(acq).photocurrent_main
    for core in actual:
        np.testing.assert_allclose(core, expected, atol=2e-14, rtol=0)


def test_warp_is_recomputed_for_each_sweep():
    cfg, _, _, _ = case(kind="colored")
    optics = MachZehnder(cfg)
    signals = []
    for sweep in [0, 1, 0]:
        _, acq, span, delta = case(kind="colored", sweep=sweep)
        expected = expected_signal(cfg, acq, span, delta)
        signal = optics.process(acq).photocurrent_main.copy()
        np.testing.assert_allclose(signal[0], expected, atol=2e-14, rtol=0)
        signals.append(signal)
    assert not np.array_equal(signals[0], signals[1])
    np.testing.assert_array_equal(signals[0], signals[2])


def test_fbg_only_warp_cannot_bypass_main_nyquist_check():
    cfg, _, _, _ = case(kind="ripple")
    cfg["source"]["sweep_ripple_period"] = 1e-6
    acq = SweptLaser(cfg).process(FiberGenerator(cfg).process(Acquisition()))
    acq.fiber_profile[:] = 0.
    with pytest.raises(ValueError, match="main.*Nyquist"):
        MachZehnder(cfg).process(acq)


@pytest.mark.parametrize("kind", ["linear", "quadratic"])
def test_white_fm_does_not_enter_propagation_warp(kind):
    cfg, _, span, delta = case(kind=kind)
    cfg["source"]["linewidth"] = 1e5
    acq = SweptLaser(cfg).process(FiberGenerator(cfg).process(Acquisition()))
    acq.fiber_profile[:] = 0.
    expected = expected_signal(cfg, acq, span, delta)
    actual = MachZehnder(cfg).process(acq).photocurrent_main
    np.testing.assert_allclose(actual[0], expected, atol=2e-14, rtol=0)
