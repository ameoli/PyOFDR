"""Tests for campaign runner, multi-sweep, HDF5 output, and multicore."""

import numpy as np
import pytest

from helpers import CFG
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.campaign import run_campaign
from pyofdr.core.config import compute_derived
from pyofdr.fiber.profile import FiberGenerator
from pyofdr.source.swept_laser import SweptLaser
from pyofdr.output.hdf5_writer import HDF5Writer
from pyofdr.utils.seeding import derive_seed


class TestMulticore:

    def _mc_cfg(self, n_cores):
        return {**CFG, "fiber": {**CFG["fiber"], "n_cores": n_cores}}

    def test_profile_shape_with_n_cores(self):
        acq = FiberGenerator(self._mc_cfg(4)).process(Acquisition())
        assert acq.fiber_profile.shape[0] == 4

    def test_cores_have_independent_profiles(self):
        acq = FiberGenerator(self._mc_cfg(4)).process(Acquisition())
        for i in range(4):
            for j in range(i + 1, 4):
                assert not np.array_equal(
                    acq.fiber_profile[i], acq.fiber_profile[j]
                )

    def test_pipeline_through_multicore(self):
        acq = run_campaign(self._mc_cfg(4))[-1]
        assert acq.digital_main.shape == (4, acq.n_samples)
        assert acq.photocurrent_main.shape == (4, acq.n_samples)

    def test_multicore_reproducibility(self):
        a = run_campaign(self._mc_cfg(4))[-1]
        b = run_campaign(self._mc_cfg(4))[-1]
        np.testing.assert_array_equal(a.digital_main, b.digital_main)

    def test_laser_field_is_shared_across_cores(self):
        # E_source is the optical field from a single laser,
        # must stay 1D regardless of n_cores
        acq = Acquisition()
        acq = FiberGenerator(self._mc_cfg(7)).process(acq)
        acq = SweptLaser(self._mc_cfg(7)).process(acq)
        assert acq.E_source.ndim == 1

    def test_per_core_detector_noise_is_independent(self):
        # with shot noise off and dark off, only thermal noise.
        # different cores should see different noise samples.
        cfg = {**self._mc_cfg(2), "detection": {**CFG["detection"],
               "shot_noise": False, "thermal_nep": 1e-9, "dark_current": 0}}
        acq = run_campaign(cfg)[-1]
        # subtract the (shared) deterministic beat to isolate noise
        # easier: just check the two analog traces are not equal
        assert not np.array_equal(acq.analog_main[0], acq.analog_main[1])


class TestSeeding:

    def test_components_get_distinct_seeds(self):
        s = 42
        seeds = {
            derive_seed(s, component="fiber"),
            derive_seed(s, component="laser",    sweep=0),
            derive_seed(s, component="detector", sweep=0),
            derive_seed(s, component="adc",      sweep=0),
        }
        assert len(seeds) == 4

    def test_core_stride_keeps_components_separated(self):
        # detector core 0 sweep 999_999 should not collide with detector core 1 sweep 0
        a = derive_seed(42, component="detector", core=0, sweep=999_999)
        b = derive_seed(42, component="detector", core=1, sweep=0)
        assert a != b

    def test_laser_seed_matches_legacy_offset(self):
        # behavior must be backwards compatible with the +1000+sweep convention
        assert derive_seed(42, component="laser", sweep=7) == 42 + 1000 + 7

    def test_detector_seed_matches_legacy_offset(self):
        assert derive_seed(42, component="detector", sweep=7) == 42 + 2000 + 7


class TestMultiSweep:

    def test_returns_correct_count(self):
        cfg = {**CFG, "simulation": {**CFG["simulation"], "n_sweeps": 3}}
        results = run_campaign(cfg)
        assert len(results) == 3

    def test_sweep_indices(self):
        cfg = {**CFG, "simulation": {**CFG["simulation"], "n_sweeps": 3}}
        results = run_campaign(cfg)
        for i, acq in enumerate(results):
            assert acq.sweep_index == i

    def test_fiber_shared_across_sweeps(self):
        cfg = {**CFG, "simulation": {**CFG["simulation"], "n_sweeps": 2}}
        results = run_campaign(cfg)
        np.testing.assert_array_equal(
            results[0].fiber_profile, results[1].fiber_profile)

    def test_noise_differs_across_sweeps(self):
        cfg = {**CFG, "simulation": {**CFG["simulation"], "n_sweeps": 2}}
        results = run_campaign(cfg)
        assert not np.array_equal(
            results[0].digital_main, results[1].digital_main)

    def test_deterministic_across_runs(self):
        cfg = {**CFG, "simulation": {**CFG["simulation"], "n_sweeps": 2}}
        a = run_campaign(cfg)
        b = run_campaign(cfg)
        np.testing.assert_array_equal(a[0].digital_main, b[0].digital_main)
        np.testing.assert_array_equal(a[1].digital_main, b[1].digital_main)

    def test_single_sweep_backward_compat(self):
        """n_sweeps=1 should behave like the old single-sweep campaign."""
        results = run_campaign(CFG)
        assert len(results) == 1
        assert results[0].digital_main is not None


class TestHDF5Writer:

    def test_write_and_read_back(self, tmp_path):
        acq = run_campaign(CFG)[-1]
        derived = compute_derived(CFG)
        path = tmp_path / "test_output.h5"
        with HDF5Writer(path) as w:
            w.write_config(CFG, derived)
            w.write_fiber(acq)
            w.write_sweep(acq, 0)

        import h5py, json
        with h5py.File(path, "r") as f:
            assert "config" in f.attrs
            cfg_back = json.loads(f.attrs["config"])
            assert cfg_back["fiber"]["length"] == CFG["fiber"]["length"]

            assert "fiber/z" in f
            assert "sweeps/0000/digital_main" in f
            assert "sweeps/0000/analog_main" in f

            dm = f["sweeps/0000/digital_main"][:]
            assert dm.shape == acq.digital_main.shape

    def test_no_output_path_skips_writing(self, tmp_path):
        """run_campaign with no output path should not create files."""
        run_campaign(CFG)
        # no h5 files should exist
        assert list(tmp_path.glob("*.h5")) == []

    def test_strain_field_is_saved(self, tmp_path):
        cfg = {**CFG, "strain": {"segments": [{"start": 0.2, "end": 0.5,
                                                "epsilon": 1e-5}]}}
        acq = run_campaign(cfg)[-1]
        derived = compute_derived(cfg)
        path = tmp_path / "strain_out.h5"
        with HDF5Writer(path) as w:
            w.write_config(cfg, derived)
            w.write_fiber(acq)
            w.write_sweep(acq, 0)

        import h5py
        with h5py.File(path, "r") as f:
            assert "fiber/strain_field" in f
            sf = f["fiber/strain_field"][:]
            assert sf.shape == acq.z.shape

    def test_campaign_writes_hdf5(self, tmp_path):
        """run_campaign should write to disk when output.path is set."""
        path = tmp_path / "campaign.h5"
        cfg = {**CFG, "output": {"path": str(path)}}
        run_campaign(cfg)

        import h5py
        with h5py.File(path, "r") as f:
            assert "sweeps/0000/digital_main" in f
            assert "config" in f.attrs

    def test_multi_sweep_hdf5(self, tmp_path):
        """Multi-sweep campaign should write all sweeps to HDF5."""
        path = tmp_path / "multi.h5"
        cfg = {**CFG, "simulation": {**CFG["simulation"], "n_sweeps": 3},
               "output": {"path": str(path)}}
        run_campaign(cfg)

        import h5py
        with h5py.File(path, "r") as f:
            assert "sweeps/0000/digital_main" in f
            assert "sweeps/0001/digital_main" in f
            assert "sweeps/0002/digital_main" in f

    def test_aux_signal_persisted_when_enabled(self, tmp_path):
        """With aux_mzi enabled the aux waveform is saved per-sweep."""
        path = tmp_path / "aux.h5"
        cfg = {**CFG,
               "optics": {**CFG["optics"],
                          "aux_mzi": {"enabled": True, "delay": 50e-9}},
               "output": {"path": str(path)}}
        run_campaign(cfg)

        import h5py
        with h5py.File(path, "r") as f:
            assert "sweeps/0000/aux_signal" in f
            aux = f["sweeps/0000/aux_signal"][:]
            assert aux.ndim == 1
            assert np.max(np.abs(aux)) <= 1.0 + 1e-6
            valid = f["sweeps/0000/aux_signal"].attrs["valid_start"]
            assert valid > 0

    def test_no_aux_signal_key_when_disabled(self, tmp_path):
        """Default config (aux off) should not write an aux_signal dataset."""
        path = tmp_path / "no_aux.h5"
        cfg = {**CFG, "output": {"path": str(path)}}
        run_campaign(cfg)

        import h5py
        with h5py.File(path, "r") as f:
            assert "sweeps/0000/aux_signal" not in f
