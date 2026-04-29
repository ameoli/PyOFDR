"""Tests for sweep nonlinearity (issue #23, part 1: no k-clock yet)."""

import warnings

import numpy as np
import pytest

from helpers import CFG
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.campaign import run_campaign
from pyofdr.source.swept_laser import SweptLaser
from pyofdr.fiber.profile import FiberGenerator
from pyofdr.analysis.demodulation import fft_reflectogram
from pyofdr.analysis.spatial_metrics import measure_resolution


def _nl_cfg(a2=0.0, a3=0.0, ripple_amp=0.0, ripple_period=0.0, **extra):
    """Build a config with sweep nonlinearity and a discrete reflector."""
    cfg = {
        **CFG,
        "source": {
            **CFG["source"],
            "sweep_nonlinearity_a2": a2,
            "sweep_nonlinearity_a3": a3,
            "sweep_ripple_amplitude": ripple_amp,
            "sweep_ripple_period": ripple_period,
            "linewidth": 0.0,
        },
        # need a reflector to measure resolution
        "fiber": {
            **CFG["fiber"],
            "reflectors": [{"z": 0.5, "R": 0.01}],
        },
        # no noise so the peak is clean
        "detection": {
            **CFG["detection"],
            "shot_noise": False,
            "thermal_nep": 0.0,
            "dark_current": 0.0,
        },
    }
    cfg.update(extra)
    return cfg


class TestSweepNonlinearityOnLaser:

    def test_zero_coefficients_is_linear(self):
        """Default a2=a3=0 gives monotonically increasing nu_inst."""
        cfg = _nl_cfg()
        acq = SweptLaser(cfg).process(
            FiberGenerator(cfg).process(Acquisition()))
        dnu = np.diff(acq.nu_inst)
        assert np.all(dnu > 0)

    def test_a2_changes_nu_inst(self):
        cfg_lin = _nl_cfg()
        cfg_nl = _nl_cfg(a2=1e18)
        acq_lin = SweptLaser(cfg_lin).process(
            FiberGenerator(cfg_lin).process(Acquisition()))
        acq_nl = SweptLaser(cfg_nl).process(
            FiberGenerator(cfg_nl).process(Acquisition()))
        # nu_inst should differ
        assert not np.allclose(acq_lin.nu_inst, acq_nl.nu_inst)

    def test_ripple_adds_oscillation(self):
        cfg = _nl_cfg(ripple_amp=1e9, ripple_period=1e-3)
        acq_nl = SweptLaser(cfg).process(
            FiberGenerator(cfg).process(Acquisition()))
        acq_lin = SweptLaser(_nl_cfg()).process(
            FiberGenerator(_nl_cfg()).process(Acquisition()))
        # ripple makes nu_inst deviate from the linear ramp
        residual = acq_nl.nu_inst - acq_lin.nu_inst
        assert np.max(np.abs(residual)) > 0.5e9  # close to the amplitude


class TestSweepNonlinearityOnBeat:

    def test_zero_nl_matches_baseline(self):
        """No nonlinearity -> same beat as before."""
        cfg = _nl_cfg()
        acq = run_campaign(cfg)[-1]
        H, z = fft_reflectogram(acq.digital_main[0].astype(np.float64), acq.dz)
        # reflector at 0.5 m should be sharp
        r = measure_resolution(H, z, 0.5)
        # should be close to the theoretical dz
        assert r["resolution"] < 0.01   # < 1 cm

    def test_a2_broadens_reflector_peak(self):
        """Quadratic nonlinearity smears the reflector peak."""
        cfg_clean = _nl_cfg()
        cfg_nl = _nl_cfg(a2=5e18)

        acq_clean = run_campaign(cfg_clean)[-1]
        acq_nl = run_campaign(cfg_nl)[-1]

        H_c, z = fft_reflectogram(
            acq_clean.digital_main[0].astype(np.float64), acq_clean.dz)
        H_n, _ = fft_reflectogram(
            acq_nl.digital_main[0].astype(np.float64), acq_nl.dz)

        r_clean = measure_resolution(H_c, z, 0.5)
        r_nl = measure_resolution(H_n, z, 0.5)
        # nonlinear peak should be wider
        assert r_nl["resolution"] > r_clean["resolution"] * 1.5

    def test_a3_broadens_reflector_peak(self):
        cfg_clean = _nl_cfg()
        cfg_nl = _nl_cfg(a3=5e23)

        acq_clean = run_campaign(cfg_clean)[-1]
        acq_nl = run_campaign(cfg_nl)[-1]

        H_c, z = fft_reflectogram(
            acq_clean.digital_main[0].astype(np.float64), acq_clean.dz)
        H_n, _ = fft_reflectogram(
            acq_nl.digital_main[0].astype(np.float64), acq_nl.dz)

        r_clean = measure_resolution(H_c, z, 0.5)
        r_nl = measure_resolution(H_n, z, 0.5)
        assert r_nl["resolution"] > r_clean["resolution"] * 1.5

    def test_ripple_spreads_reflector_energy(self):
        # ripple creates sidelobes that spread energy away from the
        # central peak -- check that the peak amplitude drops.
        cfg_clean = _nl_cfg()
        cfg_nl = _nl_cfg(ripple_amp=1e12, ripple_period=2e-3)

        acq_clean = run_campaign(cfg_clean)[-1]
        acq_nl = run_campaign(cfg_nl)[-1]

        H_c, z = fft_reflectogram(
            acq_clean.digital_main[0].astype(np.float64), acq_clean.dz)
        H_n, _ = fft_reflectogram(
            acq_nl.digital_main[0].astype(np.float64), acq_nl.dz)

        # peak amplitude in the reflector region
        mask = (z > 0.45) & (z < 0.55)
        peak_clean = np.max(np.abs(H_c[mask]))
        peak_nl = np.max(np.abs(H_n[mask]))
        # ripple spreads energy -> peak drops by several dB
        drop_dB = 20 * np.log10(peak_clean / peak_nl)
        assert drop_dB > 3.0

    def test_existing_configs_unaffected(self):
        """Existing configs without nonlinearity fields still work."""
        acq = run_campaign(CFG)[-1]
        assert acq.digital_main is not None


class TestTimeWarpBounds:
    """np.interp clamps past the grid, so extreme excursion extrapolates."""

    def test_mild_nonlinearity_silent(self):
        # excursion ~0.2% of sweep -> below the 5% warning threshold
        cfg = _nl_cfg(a2=1e14)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_campaign(cfg)
        assert not any("time-warp" in str(x.message) for x in w)

    def test_extreme_nonlinearity_warns(self):
        cfg = _nl_cfg(a2=5e18)
        with pytest.warns(UserWarning, match="time-warp"):
            run_campaign(cfg)
