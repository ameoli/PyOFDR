"""Temperature ground truth must follow each saved sweep."""

import h5py
import numpy as np
import pytest

from helpers import CFG
from pyofdr.core.campaign import run_campaign


def config():
    return {**CFG,
            "simulation": {"seed": 42, "n_sweeps": 4},
            "fiber": {**CFG["fiber"], "n_cores": 2},
            "source": {**CFG["source"], "sweep_duration": .001, "sweep_range": .5e-9},
            "adc": {**CFG["adc"], "sample_rate": 20e6},
            "strain": {"segments": [{"start": .1, "end": .4, "epsilon": 1e-5}]}}


@pytest.mark.parametrize("keep", [True, False])
@pytest.mark.parametrize("motion", [
    {"kind": "harmonic", "amplitude": 3., "frequency": 250.},
    {"kind": "thermal", "amplitude": 3., "tau": .01},
])
def test_dynamic_temperature_saved_for_each_sweep(tmp_path, keep, motion):
    cfg = config()
    cfg["temperature"] = {"segments": [
        {"start": .2, "end": .7, "delta_T": 0., "motion": motion},
        {"start": .6, "end": .9, "delta_T": -.5},
    ]}
    expected = run_campaign(cfg)
    assert not np.array_equal(expected[0].temperature_field, expected[1].temperature_field)
    path = tmp_path / "temperature.h5"
    cfg["output"] = {"path": str(path)}
    actual = run_campaign(cfg, keep_acquisitions=keep)
    assert len(actual) == (4 if keep else 0)

    with h5py.File(path) as f:
        z = f["fiber/z"][:]
        np.testing.assert_array_equal(f["fiber/temperature_field"][:], expected[0].temperature_field)
        for i, acq in enumerate(expected):
            sweep = f[f"sweeps/{i:04d}"]
            saved = sweep["temperature_field"][:]
            assert saved.shape == z.shape
            assert saved.dtype == acq.temperature_field.dtype
            np.testing.assert_array_equal(saved, acq.temperature_field)
            time = i*.001
            value = (3*np.sin(2*np.pi*250*time) if motion["kind"] == "harmonic"
                     else 3*(1-np.exp(-time/.01)))
            field = value*((z >= .2) & (z <= .7)) - .5*((z >= .6) & (z <= .9))
            np.testing.assert_allclose(saved, field, atol=1e-14, rtol=0)
            np.testing.assert_array_equal(sweep["strain_field"][:], acq.strain_field)
            np.testing.assert_array_equal(sweep["digital_main"][:], acq.digital_main)
            if keep:
                np.testing.assert_array_equal(saved, actual[i].temperature_field)


@pytest.mark.parametrize("delta_T", [0., -2.])
def test_static_temperature_including_zero_is_saved_per_sweep(tmp_path, delta_T):
    cfg = config()
    cfg["temperature"] = {"segments": [{"start": .2, "end": .7, "delta_T": delta_T}]}
    path = tmp_path / "static.h5"
    cfg["output"] = {"path": str(path)}
    assert run_campaign(cfg, keep_acquisitions=False) == []
    with h5py.File(path) as f:
        z = f["fiber/z"][:]
        expected = delta_T*((z >= .2) & (z <= .7))
        np.testing.assert_array_equal(f["fiber/temperature_field"][:], expected)
        for i in range(4):
            np.testing.assert_array_equal(f[f"sweeps/{i:04d}/temperature_field"][:], expected)


@pytest.mark.parametrize("keep", [True, False])
def test_no_temperature_does_not_create_a_field(tmp_path, keep):
    cfg = config()
    path = tmp_path / "no_temperature.h5"
    cfg["output"] = {"path": str(path)}
    run_campaign(cfg, keep_acquisitions=keep)
    with h5py.File(path) as f:
        assert "temperature_field" not in f["fiber"]
        for i in range(4):
            assert "temperature_field" not in f[f"sweeps/{i:04d}"]
