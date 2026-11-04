"""Check ADC transition levels and code widths from the conversion output."""

import numpy as np
import pytest

from pyofdr.core.acquisition import Acquisition
from pyofdr.core.config_models import RootConfig
from pyofdr.digitizer.adc import ADC


def convert(adc, voltage, sweep=0):
    acq = Acquisition(analog_main=np.atleast_2d(voltage), sweep_index=sweep, dt=1/200e6)
    return adc.process(acq).digital_main


def measure_transitions(adc):
    # Find each transition by asking the ADC for its code, not reading its table.
    target = np.arange(1, adc.n_levels) - adc.n_levels//2
    lo = np.full(len(target), -adc.v_max)
    hi = np.full(len(target), adc.v_max)
    for _ in range(42):
        mid = (lo+hi)/2
        below = convert(adc, mid)[0] < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return (lo+hi)/2


@pytest.mark.parametrize("bits", [4, 8, 16])
@pytest.mark.parametrize("kind", ["inl_peak_lsb", "dnl_rms_lsb"])
def test_tiny_errors_do_not_move_code_centres(bits, kind):
    adc = ADC({"adc": {"bits": bits, kind: 1e-6}})
    codes = np.arange(-adc.n_levels//2, adc.n_levels//2)
    voltage = (codes+.5)*adc.voltage_range/adc.n_levels
    np.testing.assert_array_equal(convert(adc, voltage)[0], codes)


@pytest.mark.parametrize("voltage_range", [2., 3.3])
def test_inl_moves_transitions_in_the_specified_direction(voltage_range):
    adc = ADC({"adc": {"bits": 4, "voltage_range": voltage_range, "inl_peak_lsb": .25}})
    lsb = voltage_range/16
    # Quarter-scale sine peaks move these transitions by +/-0.25 LSB.
    inputs = np.array([-3.8, -3.75, -3.7, 3.7, 3.75, 3.8])*lsb
    np.testing.assert_array_equal(convert(adc, inputs), [[-5, -4, -4, 3, 4, 4]])


@pytest.mark.parametrize("amplitude", [1e-6, .25, 3.])
def test_measured_inl_matches_sinusoidal_threshold_displacement(amplitude):
    adc = ADC({"adc": {"bits": 8, "inl_peak_lsb": amplitude}})
    k = np.arange(1, 256)
    measured = measure_transitions(adc)/(2/256) - (k-128)
    expected = amplitude*np.sin(2*np.pi*k/256)
    np.testing.assert_allclose(measured, expected, atol=1e-9, rtol=0)


def test_code_density_matches_sinusoidal_code_widths():
    adc = ADC({"adc": {"bits": 8, "inl_peak_lsb": 3}})
    n = 256*4096
    voltage = -1 + 2*(np.arange(n)+.5)/n
    codes = convert(adc, voltage)[0].astype(int) + 128
    widths = np.bincount(codes, minlength=256)/4096
    k = np.arange(257)
    expected = 1 + 3*np.diff(np.sin(2*np.pi*k/256))
    np.testing.assert_allclose(widths, expected, atol=2/4096, rtol=0)
    assert np.all(np.diff(codes) >= 0)


@pytest.mark.parametrize("rms", [.01, .2, .5, 1.])
def test_measured_dnl_rms_and_positive_widths(rms):
    adc = ADC({"adc": {"bits": 12, "dnl_rms_lsb": rms}})
    edges = np.r_[-1, measure_transitions(adc), 1]
    widths = np.diff(edges)/(2/4096)
    assert np.all(widths > 0)
    assert np.mean(widths) == pytest.approx(1, abs=1e-12)
    # RMS is a population parameter; one finite chip has sampling scatter.
    assert np.std(widths-1) == pytest.approx(rms, rel=.08)


def test_thresholds_are_fixed_across_sweeps_and_shared_by_cores():
    cfg = {"simulation": {"seed": 42}, "adc": {"bits": 8, "dnl_rms_lsb": .2}}
    adc = ADC(cfg)
    voltage = np.tile(np.linspace(-1, 1, 10000), (2, 1))
    first = convert(adc, voltage)
    np.testing.assert_array_equal(first[0], first[1])
    np.testing.assert_array_equal(first, convert(adc, voltage, sweep=1))
    np.testing.assert_array_equal(first, convert(ADC(cfg), voltage))
    other = ADC({**cfg, "simulation": {"seed": 43}})
    assert not np.array_equal(first, convert(other, voltage))


@pytest.mark.parametrize("bits", [1, 8, 18])
def test_nonlinear_full_scale_and_output_dtype(bits):
    adc = ADC({"adc": {"bits": bits, "dnl_rms_lsb": .2, "inl_peak_lsb": .1}})
    half = 2**(bits-1)
    result = convert(adc, [-2., -1., 1., 2.])
    np.testing.assert_array_equal(result, [[-half, -half, half-1, half-1]])
    assert result.dtype == (np.int16 if bits <= 16 else np.int32)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("parameter", ["inl_peak_lsb", "dnl_rms_lsb"])
def test_nonfinite_nonlinearity_rejected(parameter, value):
    with pytest.raises(ValueError):
        RootConfig(adc={parameter: value})


def test_crossed_thresholds_are_rejected():
    with pytest.raises(ValueError, match="threshold"):
        ADC({"adc": {"bits": 4, "inl_peak_lsb": 3e9}})
