"""A fractional sample count must not stretch the optical sweep."""

import math

import h5py
import numpy as np
import pytest

from pyofdr.analysis.demodulation import (
    fft_reflectogram, reflectogram_from_acq, windowed_xcorr_strain,
)
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.campaign import run_campaign
from pyofdr.core.config import compute_derived
from pyofdr.core.config_models import RootConfig
from pyofdr.fiber.profile import FiberGenerator
from pyofdr.optics.aux_mzi import AuxMZI
from pyofdr.optics.mach_zehnder import MachZehnder
from pyofdr.source.swept_laser import SweptLaser
from pyofdr.utils.constants import C
from pyofdr.utils.sampling import sweep_sample_count


def reflector_config(count=4096.4, bin_index=64, cores=1):
    wavelength = 1550e-9
    span = C * 1e-9 / wavelength**2
    dz = C / (2 * 1.4682 * span)
    return RootConfig(**{
        "simulation": {"seed": 42},
        "source": {"center_wavelength": wavelength, "sweep_range": 1e-9,
                   "sweep_duration": 1e-4, "power": 0.01, "linewidth": 0},
        "fiber": {"length": (bin_index + 8) * dz, "n_core": 1.4682,
                  "n_cores": cores, "rayleigh_coefficient_dB": -300,
                  "attenuation_dB_per_km": 0,
                  "reflectors": [{"z": bin_index * dz, "R": 1e-4}]},
        "optics": {"splitting_ratio": 0.4, "circulator": {
            "insertion_loss_dB": 0, "isolation_dB": 300, "return_loss_dB": 300}},
        "detection": {"balanced": True, "responsivity": 0.9,
                      "shot_noise": False, "thermal_nep": 0, "dark_current": 0},
        "adc": {"sample_rate": count / 1e-4, "input_impedance": 50},
    }).model_dump()


def optical_acquisition(config):
    acq = FiberGenerator(config).process(Acquisition())
    acq = SweptLaser(config).process(acq)
    return MachZehnder(config).process(acq)


@pytest.mark.parametrize("count, expected", [
    (4096.0, 4096), (np.nextafter(4096.0, np.inf), 4096),
    (np.nextafter(4096.0, 0), 4096), (4096.000001, 4097),
    (4096.4, 4097), (4095.9, 4096), (0.4, 1),
])
def test_sample_count_roundoff_policy(count, expected):
    assert sweep_sample_count(1.0, count) == expected


@pytest.mark.parametrize("count", [512.0, 512.4, 4096.000001, 4096.4])
@pytest.mark.parametrize("bin_index", [8, 64, 200])
def test_reflector_matches_physical_chirp(count, bin_index):
    cfg = reflector_config(count, bin_index)
    acq = optical_acquisition(cfg)
    T = cfg["source"]["sweep_duration"]
    fs = cfg["adc"]["sample_rate"]
    span = C * cfg["source"]["sweep_range"] / cfg["source"]["center_wavelength"]**2
    position = cfg["fiber"]["reflectors"][0]["z"]
    frequency = 2 * cfg["fiber"]["n_core"] * position * span / (C * T)
    amplitude = 2 * 0.01 * np.sqrt(0.4 * 0.6 * 1e-4)
    phase = np.angle(acq.fiber_profile[0, bin_index])
    expected = amplitude * np.cos(2 * np.pi * frequency * acq.t + phase)
    np.testing.assert_allclose(acq.photocurrent_main[0], expected, rtol=0, atol=amplitude * 1e-9)
    np.testing.assert_allclose(acq.t, np.arange(acq.n_samples) / fs, rtol=5e-16, atol=0)
    assert acq.t[-1] < T
    np.testing.assert_allclose(acq.z[bin_index], position, rtol=0, atol=1e-15)


@pytest.mark.parametrize("count", [512.0, 512.4, 4096.4])
def test_complex_multicore_profile_matches_direct_sum(count):
    cfg = reflector_config(count, cores=2)
    fiber = FiberGenerator(cfg)
    source = SweptLaser(cfg)
    optics = MachZehnder(cfg)
    rng = np.random.default_rng(321)
    for sweep in [0, 1]:
        acq = source.process(fiber.process(Acquisition(sweep_index=sweep)))
        profile = 0.01 * (rng.normal(size=acq.fiber_profile.shape)
                          + 1j * rng.normal(size=acq.fiber_profile.shape))
        acq.fiber_profile = profile.copy()
        weighted = profile * acq.attenuation_envelope
        expected = np.zeros((2, acq.n_samples), dtype=complex)
        for k in range(profile.shape[1]):
            expected += weighted[:, k, None] * np.exp(2j * np.pi * k * acq.t / 1e-4)
        prefactor = 2 * np.sqrt(0.4 * 0.6) * np.abs(acq.E_source)**2
        expected = prefactor * expected.real
        acq = optics.process(acq)
        error = np.linalg.norm(acq.photocurrent_main - expected) / np.linalg.norm(expected)
        assert error < 1e-9
        np.testing.assert_array_equal(acq.fiber_profile, profile)


def test_integer_sweep_retains_ifft_path():
    cfg = reflector_config(4096)
    acq = SweptLaser(cfg).process(FiberGenerator(cfg).process(Acquisition()))
    weighted = acq.fiber_profile * acq.attenuation_envelope
    weighted[:, 0] += 10**(-300 / 20)
    h = np.pad(weighted, ((0, 0), (0, acq.n_samples - weighted.shape[1])))
    expected = np.fft.ifft(h, axis=-1) * acq.n_samples
    power = np.abs(acq.E_source)**2
    expected = 2 * math.sqrt(0.4 * 0.6) * power * expected.real
    expected += 2 * math.sqrt(0.4 * 0.6) * 10**(-300 / 20) * power
    actual = MachZehnder(cfg).process(acq).photocurrent_main
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("count", [512.4, 4096.4])
def test_fbg_uses_actual_sample_times(count):
    cfg = reflector_config(count)
    position = cfg["fiber"]["reflectors"][0]["z"]
    cfg["fiber"]["reflectors"] = []
    cfg["fiber"]["fbg_arrays"] = [{"z": position, "bragg_wavelength": 1550e-9,
                                    "length": 1e-4, "peak_reflectivity": 1e-4}]
    acq = optical_acquisition(cfg)
    envelope = 0.01 * np.sinc(2 * 1.4682 * 1e-4 * (acq.nu_inst - C / 1550e-9) / C)
    expected = 2 * np.sqrt(0.4 * 0.6) * 0.01 * envelope * np.cos(2 * np.pi * 64 * acq.t / 1e-4)
    np.testing.assert_allclose(acq.photocurrent_main[0], expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("a2", [-1e17, 1e17])
def test_nonlinear_warp_uses_physical_time_grid(a2):
    cfg = reflector_config(4096.4, bin_index=16)
    cfg["source"]["sweep_nonlinearity_a2"] = a2
    acq = optical_acquisition(cfg)
    T = 1e-4
    span = C * 1e-9 / (1550e-9)**2
    s = acq.t + a2 * acq.t**2 / (span / T)
    amplitude = 2 * 0.01 * np.sqrt(0.4 * 0.6 * 1e-4)
    # Separate the sampling correction from the existing linear interpolation error.
    phase = np.angle(acq.fiber_profile[0, 16])
    expected_interpolated = np.interp(
        s, acq.t, amplitude * np.cos(2 * np.pi * 16 * acq.t / T + phase))
    np.testing.assert_allclose(acq.photocurrent_main[0], expected_interpolated,
                               rtol=0, atol=amplitude * 1e-9)
    inside = (s >= acq.t[0]) & (s <= acq.t[-1])
    exact = amplitude * np.cos(2 * np.pi * 16 * s[inside] / T + phase)
    rms = np.sqrt(np.mean((acq.photocurrent_main[0, inside] - exact)**2)) / amplitude
    assert rms < 1e-4


def test_auxiliary_frequency_matches_requested_chirp():
    cfg = reflector_config(4096.4)
    cfg["source"]["sweep_range"] = 1e-12
    delay = 16 / cfg["adc"]["sample_rate"]
    cfg["optics"]["aux_mzi"].update(enabled=True, delay=delay)
    acq = SweptLaser(cfg).process(Acquisition())
    acq = AuxMZI(cfg).process(acq)
    span = C * 1e-12 / (1550e-9)**2
    expected = span / 1e-4 * delay
    measured = acq.log[-1]["mean_beat_Hz"]
    assert measured == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("padding", [None, 32768])
def test_fft_axis_uses_adc_clock(padding):
    cfg = reflector_config(512.4)
    acq = optical_acquisition(cfg)
    # A clean tone on a known FFT bin isolates axis calibration from peak fitting.
    acq.digital_main = np.cos(2 * np.pi * 64 * np.arange(acq.n_samples) / acq.n_samples)[None, :]
    H, z = reflectogram_from_acq(acq, n_pad=padding)
    N = padding if padding is not None else acq.n_samples
    frequency_bins = np.fft.fftfreq(N, d=acq.dt)[:(N + 1) // 2]
    gamma = (C * 1e-9 / (1550e-9)**2) / 1e-4
    expected_z = C * frequency_bins / (2 * 1.4682 * gamma)
    np.testing.assert_allclose(z, expected_z, rtol=1e-14, atol=1e-16)
    assert acq.fft_dz != acq.dz
    _, direct_z = fft_reflectogram(acq.digital_main[0], acq.fft_dz, n_pad=padding)
    np.testing.assert_array_equal(z, direct_z)


def test_sampling_calibration_survives_hdf5(tmp_path):
    cfg = reflector_config(512.4)
    cfg["output"]["path"] = str(tmp_path / "fractional.h5")
    acq = run_campaign(cfg)[0]
    derived = compute_derived(cfg)
    assert derived["n_t"] == acq.n_samples == 513
    for name in ["frequency_step_hz", "fft_sweep_range_hz", "fft_dz"]:
        assert getattr(acq, name) == pytest.approx(derived[name], rel=1e-14)
    with h5py.File(cfg["output"]["path"], "r") as stored:
        for name in ["frequency_step_hz", "fft_sweep_range_hz", "fft_dz"]:
            assert stored["derived"].attrs[name] == pytest.approx(getattr(acq, name), rel=1e-14)
            assert stored["sweeps/0000"].attrs[name] == getattr(acq, name)
        assert stored["sweeps/0000"].attrs["n_samples"] == acq.n_samples
        assert stored["sweeps/0000"].attrs["dt"] == acq.dt


@pytest.mark.parametrize("N", [9, 513])
def test_odd_fft_keeps_last_positive_frequency(N):
    last_bin = N // 2
    beat = np.cos(2 * np.pi * last_bin * np.arange(N) / N)
    spectrum, z = fft_reflectogram(beat, dz=1.0)
    assert len(spectrum) == last_bin + 1
    assert z[np.argmax(np.abs(spectrum))] == last_bin


@pytest.mark.parametrize("lag", [-0.7, 0.7])
def test_fractional_record_recovers_known_spectral_shift(lag):
    cfg = reflector_config(4096.4, bin_index=700)
    cfg["fiber"]["reflectors"] = []
    source = SweptLaser(cfg)
    fiber = FiberGenerator(cfg)
    reference = source.process(fiber.process(Acquisition()))
    measured = source.process(fiber.process(Acquisition()))
    rng = np.random.default_rng(5)
    profile = rng.normal(size=reference.fiber_profile.shape) + 1j * rng.normal(size=reference.fiber_profile.shape)
    profile[:, :128] = 0
    W = 64
    reference.fiber_profile = profile
    measured.fiber_profile = profile * np.exp(-2j * np.pi * lag * np.arange(profile.shape[1]) / W)
    reference = MachZehnder(cfg).process(reference)
    measured = MachZehnder(cfg).process(measured)
    H_ref, _ = fft_reflectogram(reference.photocurrent_main[0], reference.fft_dz)
    H_meas, _ = fft_reflectogram(measured.photocurrent_main[0], measured.fft_dz)
    z, strain = windowed_xcorr_strain(
        H_meas, H_ref, reference.fft_dz, W * reference.fft_dz, W * reference.fft_dz,
        reference.fft_sweep_range_hz, center_freq=1, p_e=0,
    )
    inside = (z > 200 * reference.dz) & (z < 600 * reference.dz)
    span = C * cfg["source"]["sweep_range"] / (1550e-9)**2
    expected_shift = lag * span / W
    # Finite fractional records leak between FFT bins, even without noise.
    np.testing.assert_allclose(-strain[inside], expected_shift, rtol=0.01)
