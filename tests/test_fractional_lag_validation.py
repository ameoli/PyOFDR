"""Validation of the shipped continuous estimator, including issue #105.

Known phase ramps test numerical accuracy without a second peak estimator
as the oracle. Campaign pairs keep the fiber seed fixed and select different
sweeps so receiver noise cannot cancel between reference and measurement.
"""

from copy import deepcopy

import numpy as np
import pytest

from helpers import CFG
from pyofdr.analysis.demodulation import (
    reflectogram_from_acq,
    windowed_xcorr_strain,
)
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.campaign import run_campaign
from pyofdr.detection.detector import Detector
from pyofdr.detection.filter import AntiAliasFilter
from pyofdr.utils.constants import C


@pytest.mark.parametrize("seed", [7, 42, 105])
@pytest.mark.parametrize("window", [64, 488])
def test_continuous_accuracy_across_fractional_lags(seed, window, record_property):
    """Fixed 8x localization + continuous refinement recovers known lags.

    This validates the selected algorithm instead of exposing a padding-factor
    option just to implement the alternative originally proposed in #105.
    """
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=window) + 1j * rng.normal(size=window)
    lags = np.r_[np.linspace(-0.95, 0.95, 39),
                 -window / 2 - 0.03, -window / 2 + 0.03,
                 window / 2 - 0.03, window / 2 + 0.03]
    errors = {"continuous": [], "parabolic": []}
    for lag in lags:
        measured = reference * np.exp(
            -2j * np.pi * lag * np.arange(window) / window)
        for method in errors:
            _, strain = windowed_xcorr_strain(
                measured, reference, dz=1.0, gauge_length=window,
                stride=window, sweep_range_hz=window, center_freq=1.0,
                p_e=0.0, peak_method=method,
            )
            recovered_lag = -strain[0]
            assert -window / 2 <= recovered_lag < window / 2
            error = (recovered_lag - lag + window / 2) % window - window / 2
            errors[method].append(abs(error))

    record_property("max_continuous_error_bins", max(errors["continuous"]))
    record_property("max_parabolic_interior_error_bins", max(errors["parabolic"][:39]))
    assert max(errors["continuous"]) < 1e-7
    # The regression must exercise the original bias, not only easy grid peaks.
    assert max(errors["parabolic"][:39]) > 0.05


def _campaign_config(seed, balanced):
    cfg = deepcopy(CFG)
    cfg["simulation"] = {"seed": seed, "n_sweeps": 2}
    # 131072 samples: enough to resolve the 1 m fiber without Nyquist aliasing,
    # but much cheaper than the default two million samples per sweep.
    cfg["source"].update(sweep_duration=131072 / 20e6,
                         linewidth=0.0, rin_dB_per_Hz=None)
    cfg["detection"].update(
        balanced=balanced, bandwidth=8e6, filter_order=4,
        shot_noise=True, thermal_nep=1e-12, dark_current=1e-9,
    )
    cfg["adc"].update(sample_rate=20e6, bits=18, voltage_range=0.2,
                      input_impedance=1000.0)
    cfg["strain"] = {"photoelastic_coefficient": 0.22, "segments": []}
    cfg["temperature"] = {"thermal_expansion": 5.5e-7,
                           "thermo_optic": 6.5e-6, "segments": []}
    return cfg


def _receiver_noise(acq, cfg):
    """Subtract the matching noiseless receiver output, before ADC rounding."""
    quiet = deepcopy(cfg)
    quiet["detection"].update(shot_noise=False, thermal_nep=0.0,
                              dark_current=0.0)
    baseline = Acquisition(dt=acq.dt, photocurrent_main=acq.photocurrent_main)
    baseline = Detector(quiet).process(baseline)
    baseline = AntiAliasFilter(quiet).process(baseline)
    residual = (acq.analog_main - baseline.analog_main)[0]
    # Exclude filter settling and edges when checking noise correlation.
    return residual[len(residual) // 10:-len(residual) // 10]


@pytest.mark.parametrize("seed", [7, 105])
@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("kind,magnitude", [("strain", 17e-6),
                                            ("temperature", 5.0)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_recovery_with_independent_acquisition_noise(seed, balanced,
                                                    kind, magnitude, sign,
                                                    record_property):
    cfg = _campaign_config(seed, balanced)
    perturbed = deepcopy(cfg)
    imposed = sign * magnitude
    field = "epsilon" if kind == "strain" else "delta_T"
    perturbed[kind]["segments"] = [{"start": 0.2, "end": 0.8, field: imposed}]

    # Same fiber seed, different sweep indices. Changing the global seed would
    # also change the Rayleigh fingerprint and would test a different problem.
    reference = run_campaign(cfg)[0]
    measured = run_campaign(perturbed)[1]
    assert (reference.sweep_index, measured.sweep_index) == (0, 1)

    p_e = cfg["strain"]["photoelastic_coefficient"]
    thermal = (cfg["temperature"]["thermal_expansion"]
               + cfg["temperature"]["thermo_optic"])
    coefficient = 1 - p_e if kind == "strain" else thermal
    applied_field = (measured.strain_field if kind == "strain"
                     else measured.temperature_field)
    wavelength = cfg["source"]["center_wavelength"]
    phase = (4 * np.pi * cfg["fiber"]["n_core"] / wavelength
             * coefficient * np.cumsum(applied_field) * measured.dz)
    # Undo only the known physical perturbation: the complex fingerprint,
    # not just its magnitude, must match the reference.
    np.testing.assert_allclose(
        measured.fiber_profile * np.exp(-1j * phase),
        reference.fiber_profile, rtol=1e-12, atol=0,
    )

    noise_ref = _receiver_noise(reference, cfg)
    noise_meas = _receiver_noise(measured, perturbed)
    assert np.std(noise_ref) > 0
    assert np.std(noise_meas) > 0
    # The original same-sweep setup produces almost identical receiver noise;
    # this assertion ensures it cannot silently return to these regressions.
    noise_correlation = np.corrcoef(noise_ref, noise_meas)[0, 1]
    record_property("receiver_noise_correlation", noise_correlation)
    assert abs(noise_correlation) < 0.03

    for acq in (reference, measured):
        # Reject clipping as an explanation for estimator errors.
        assert np.max(np.abs(acq.analog_main)) < cfg["adc"]["voltage_range"] / 2
    H_ref, _ = reflectogram_from_acq(reference)
    H_meas, _ = reflectogram_from_acq(measured)
    gauge = 0.01
    centers, apparent_strain = windowed_xcorr_strain(
        H_meas, H_ref, reference.fft_dz, gauge_length=gauge, stride=gauge,
        sweep_range_hz=reference.fft_sweep_range_hz,
        center_freq=C / wavelength, p_e=p_e,
    )
    recovered = (apparent_strain if kind == "strain"
                 else apparent_strain * (1 - p_e) / thermal)
    # Non-overlapping windows wholly within the perturbed segment; exclude
    # its edges where one gauge would average perturbed and quiet material.
    inside = (centers > 0.2 + gauge) & (centers < 0.8 - gauge)
    outside = (centers > 0.03) & (centers < 0.2 - gauge)
    assert inside.sum() >= 40
    assert outside.sum() >= 10
    assert np.isfinite(recovered[inside | outside]).all()
    errors = recovered[inside] - imposed
    relative_bias = abs(np.median(errors)) / magnitude
    relative_rmse = np.sqrt(np.mean(errors**2)) / magnitude
    quiet_offset = abs(np.median(recovered[outside])) / magnitude
    record_property("imposed_value", imposed)
    record_property("recovered_median", np.median(recovered[inside]))
    record_property("relative_median_bias", relative_bias)
    record_property("relative_window_rmse", relative_rmse)
    record_property("relative_quiet_offset", quiet_offset)
    # Acceptance limits for this stated receiver/SNR regime, not universal
    # uncertainty guarantees: median bias <5%, window RMSE <10% of input.
    assert relative_bias < 0.05
    assert relative_rmse < 0.10
    assert quiet_offset < 0.05
