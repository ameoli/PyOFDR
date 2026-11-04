"""High-resolution ADCs must not allocate a table with billions of entries."""

import tracemalloc

import numpy as np
import pytest

from pyofdr.cli import main
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.config_models import RootConfig
from pyofdr.digitizer.adc import ADC


@pytest.fixture(autouse=True)
def guard_large_table(monkeypatch):
    # Fail a regression before it can exhaust the machine's memory.
    arange = np.arange

    def bounded_arange(*args, **kwargs):
        stop = args[0] if len(args) == 1 else args[1]
        start = 0 if len(args) == 1 else args[0]
        step = args[2] if len(args) > 2 else 1
        assert (stop-start)/step <= 2**21+1, "unexpected full ADC table"
        return arange(*args, **kwargs)

    monkeypatch.setattr(np, "arange", bounded_arange)


def convert(adc, voltage, sweep=0):
    acq = Acquisition(analog_main=np.atleast_2d(voltage), dt=1/200e6, sweep_index=sweep)
    return adc.process(acq).digital_main


@pytest.mark.parametrize("bits", [21, 24, 32])
@pytest.mark.parametrize("dnl", [1e-6, .2])
def test_high_resolution_dnl_rejected_during_config_validation(bits, dnl):
    with pytest.raises(ValueError, match="DNL.*20"):
        RootConfig(adc={"bits": bits, "dnl_rms_lsb": dnl})


def test_dnl_rejected_before_attempting_any_table(monkeypatch):
    def unexpected_table(self):
        pytest.fail("table allocation reached before validation")

    monkeypatch.setattr(ADC, "_build_thresholds", unexpected_table)
    with pytest.raises(ValueError, match="DNL.*20"):
        ADC({"adc": {"bits": 32, "dnl_rms_lsb": .1, "inl_peak_lsb": .1}})


def test_cli_validate_explains_dnl_limit(tmp_path, capsys):
    path = tmp_path / "adc.yaml"
    path.write_text("adc:\n  bits: 32\n  dnl_rms_lsb: 0.1\n")
    assert main(["validate", str(path)]) == 1
    assert "DNL" in capsys.readouterr().err


def test_dnl_at_20_bits_still_supported():
    tracemalloc.start()
    try:
        adc = ADC({"adc": {"bits": 20, "dnl_rms_lsb": .2, "inl_peak_lsb": .1}})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 96*1024**2
    out = convert(adc, np.linspace(-2, 2, 1000))[0]
    assert np.all(np.diff(out.astype(np.int64)) >= 0)
    assert (out[0], out[-1]) == (-2**19, 2**19-1)


@pytest.mark.parametrize("bits", [21, 24, 32])
@pytest.mark.parametrize("amplitude", [1e-6, .25, 1e5])
def test_high_resolution_inl_has_correct_rails_and_order(bits, amplitude):
    adc = ADC({"adc": {"bits": bits, "inl_peak_lsb": amplitude}})
    out = convert(adc, np.linspace(-2, 2, 10001))[0]
    assert out.dtype == np.int32
    assert np.all(np.diff(out.astype(np.int64)) >= 0)
    assert (out[0], out[-1]) == (-2**(bits-1), 2**(bits-1)-1)


@pytest.mark.parametrize("amplitude", [1e-6, .25, 20000.])
def test_analytic_conversion_matches_explicit_21_bit_table(amplitude):
    n = 2**21
    k = np.arange(n+1, dtype=float)
    error = amplitude*np.sin(2*np.pi*k/n)
    error[[0, n//2, -1]] = 0.
    thresholds = (k+error-n//2)*(3.3/n)
    selected = thresholds[[1, 2, n//4, n//2-1, n//2, n//2+1, 3*n//4, n-2, n-1]]
    voltage = np.r_[np.linspace(-2, 2, 2000), selected,
                    np.nextafter(selected, -np.inf), np.nextafter(selected, np.inf)]
    expected = np.searchsorted(thresholds[1:-1], voltage, side="right")-n//2
    adc = ADC({"adc": {"bits": 21, "voltage_range": 3.3, "inl_peak_lsb": amplitude}})
    np.testing.assert_array_equal(convert(adc, voltage)[0], expected)


def test_32_bit_quarter_scale_transitions_and_ties():
    adc = ADC({"adc": {"bits": 32, "inl_peak_lsb": .25}})
    quarter = 2**30
    codes = np.array([-quarter+.15, -quarter+.25, -quarter+.35,
                     quarter-.35, quarter-.25, quarter-.15, 0.])
    expected = [-quarter-1, -quarter, -quarter, quarter-1, quarter, quarter, 0]
    np.testing.assert_array_equal(convert(adc, codes*(2/2**32))[0], expected)


def test_32_bit_inl_memory_and_chunk_independence():
    voltage = np.tile(np.linspace(-1, 1, 100001), (2, 1))
    tracemalloc.start()
    try:
        adc = ADC({"adc": {"bits": 32, "inl_peak_lsb": 50.}})
        first = convert(adc, voltage)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 32*1024**2
    np.testing.assert_array_equal(first[0], first[1])
    np.testing.assert_array_equal(first, convert(adc, voltage, sweep=1))
    pieces = [convert(adc, voltage[:, :12345]), convert(adc, voltage[:, 12345:])]
    np.testing.assert_array_equal(first, np.concatenate(pieces, axis=1))


@pytest.mark.parametrize("amplitude", [2**32, 1e300])
def test_large_inl_rejected_without_allocating_table(amplitude):
    with pytest.raises(ValueError, match="threshold"):
        ADC({"adc": {"bits": 32, "inl_peak_lsb": amplitude}})


@pytest.mark.parametrize("bits", [21, 32])
def test_nearly_touching_thresholds_rejected_before_search(bits):
    amplitude = (1-1e-10)/np.sin(2*np.pi/2**bits)
    with pytest.raises(ValueError, match="threshold"):
        ADC({"adc": {"bits": bits, "inl_peak_lsb": amplitude}})


@pytest.mark.parametrize("bits", [21, 32])
def test_compressed_but_resolvable_central_codes(bits):
    n = 2**bits
    amplitude = .999/np.sin(2*np.pi/n)
    k = n//2 + np.arange(-1000, 1002)
    error = amplitude*np.sin(2*np.pi*k/n)
    error[k == n//2] = 0.
    edges = (k+error-n//2)*(2/n)
    voltage = (edges[:-1]+edges[1:])/2
    adc = ADC({"adc": {"bits": bits, "inl_peak_lsb": amplitude}})
    np.testing.assert_array_equal(convert(adc, voltage)[0], k[:-1]-n//2)


def test_high_resolution_ideal_and_enob_remain_available():
    voltage = np.linspace(-.5, .5, 1000)
    ideal = convert(ADC({"adc": {"bits": 32}}), voltage)
    expected = np.floor(voltage*2**31).astype(np.int32)
    np.testing.assert_array_equal(ideal[0], expected)
    cfg = {"adc": {"bits": 32, "enob": 20., "jitter_rms": 1e-12, "inl_peak_lsb": .25}}
    noisy = ADC(cfg)
    first = convert(noisy, voltage)
    np.testing.assert_array_equal(first, convert(ADC(cfg), voltage))
    assert not np.array_equal(first, ideal)
    assert not np.array_equal(first, convert(noisy, voltage, sweep=1))
