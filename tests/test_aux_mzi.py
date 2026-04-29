"""Tests for the auxiliary MZI and k-clock resampling (issue #23, part 2)."""

import numpy as np
import pytest

from helpers import CFG
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.campaign import run_campaign
from pyofdr.source.swept_laser import SweptLaser
from pyofdr.fiber.profile import FiberGenerator
from pyofdr.optics.aux_mzi import AuxMZI
from pyofdr.analysis.demodulation import (
    fft_reflectogram, kclock_resample, _strict_increasing_mask,
)
from pyofdr.analysis.spatial_metrics import measure_resolution


def _aux_cfg(delay=50e-9, enabled=True, a2=0.0, a3=0.0,
             ripple_amp=0.0, ripple_period=0.0, **extra):
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
        "optics": {
            **CFG["optics"],
            "aux_mzi": {"enabled": enabled, "delay": delay},
        },
        "fiber": {
            **CFG["fiber"],
            "reflectors": [{"z": 0.5, "R": 0.01}],
        },
        "detection": {
            **CFG["detection"],
            "shot_noise": False,
            "thermal_nep": 0.0,
            "dark_current": 0.0,
        },
    }
    cfg.update(extra)
    return cfg


def _build_acq_for_aux(cfg):
    """Run just enough pipeline for the aux MZI step."""
    acq = FiberGenerator(cfg).process(Acquisition())
    acq = SweptLaser(cfg).process(acq)
    acq = AuxMZI(cfg).process(acq)
    return acq


class TestAuxMZIStep:

    def test_disabled_leaves_acq_untouched(self):
        cfg = _aux_cfg(enabled=False)
        acq = _build_acq_for_aux(cfg)
        assert acq.aux_signal is None

    def test_enabled_produces_signal_of_correct_shape(self):
        cfg = _aux_cfg()
        acq = _build_acq_for_aux(cfg)
        assert acq.aux_signal is not None
        assert acq.aux_signal.shape == (acq.n_samples,)
        assert acq.aux_valid_start > 0

    def test_aux_is_bounded_cosine(self):
        cfg = _aux_cfg()
        acq = _build_acq_for_aux(cfg)
        # cos(...) in [-1, 1]
        assert np.max(np.abs(acq.aux_signal)) <= 1.0 + 1e-12

    def test_linear_sweep_gives_clean_sinusoid(self):
        """With no nonlinearity, aux is a pure tone at gamma*tau."""
        cfg = _aux_cfg(delay=100e-9)
        acq = _build_acq_for_aux(cfg)
        aux = acq.aux_signal[acq.aux_valid_start:]
        dt = acq.dt

        # expected beat frequency
        src = cfg["source"]
        from pyofdr.utils.units import wavelength_range_to_freq_range
        gamma = wavelength_range_to_freq_range(
            src["center_wavelength"], src["sweep_range"]
        ) / src["sweep_duration"]
        f_expected = gamma * 100e-9

        # power spectrum peak should be at f_expected
        N = len(aux)
        spec = np.abs(np.fft.rfft(aux * np.hanning(N)))
        freqs = np.fft.rfftfreq(N, dt)
        f_peak = freqs[np.argmax(spec)]
        assert abs(f_peak - f_expected) / f_expected < 0.01   # within 1%

    def test_nonlinear_sweep_modulates_aux_frequency(self):
        """With a2 != 0 the aux inst freq drifts. Compare peak widths in the PSD."""
        cfg_lin = _aux_cfg(delay=100e-9)
        cfg_nl  = _aux_cfg(delay=100e-9, a2=1e19)

        aux_lin = _build_acq_for_aux(cfg_lin).aux_signal[_build_acq_for_aux(cfg_lin).aux_valid_start:]
        acq_nl = _build_acq_for_aux(cfg_nl)
        aux_nl = acq_nl.aux_signal[acq_nl.aux_valid_start:]

        N = min(len(aux_lin), len(aux_nl))
        spec_lin = np.abs(np.fft.rfft(aux_lin[:N] * np.hanning(N)))
        spec_nl  = np.abs(np.fft.rfft(aux_nl[:N]  * np.hanning(N)))

        # nonlinear spectrum is broader -> peak amplitude drops
        assert spec_nl.max() < spec_lin.max() * 0.9

    def test_delay_too_short_raises(self):
        cfg = _aux_cfg(delay=1e-12)   # 1 ps at fs=200MHz -> n_tau = 0
        with pytest.raises(ValueError, match="too short"):
            _build_acq_for_aux(cfg)

    def test_delay_too_long_raises(self):
        cfg = _aux_cfg(delay=1.0)   # 1 s -- much longer than sweep
        with pytest.raises(ValueError, match="exceeds"):
            _build_acq_for_aux(cfg)

    def test_enabled_requires_positive_delay(self):
        # pydantic check from the OpticsConfig validator
        cfg = _aux_cfg(delay=0.0, enabled=True)
        with pytest.raises(ValueError, match="positive delay"):
            AuxMZI(cfg)


class TestKClockResampling:

    def test_preserves_length_by_default(self):
        cfg = _aux_cfg()
        acq = run_campaign(cfg)[-1]
        beat = acq.digital_main[0].astype(np.float64)
        out = kclock_resample(beat, acq.aux_signal, trim_start=acq.aux_valid_start)
        assert out.shape == (len(beat) - acq.aux_valid_start,)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            kclock_resample(np.zeros(100), np.zeros(200))

    def test_linear_sweep_roundtrip_is_near_identity(self):
        """Linear sweep -> aux phase linear in t -> resampling = identity (up
        to interpolation error). Peak location should be preserved."""
        cfg = _aux_cfg()
        acq = run_campaign(cfg)[-1]
        beat = acq.digital_main[0].astype(np.float64)
        beat_rs = kclock_resample(beat, acq.aux_signal,
                                   trim_start=acq.aux_valid_start)

        # uniform-nu resampling preserves dz = c/(2n*delta_nu)
        H_orig, z_orig = fft_reflectogram(beat, acq.dz)
        H_rs, z_rs = fft_reflectogram(beat_rs, acq.dz)

        zpk_orig = z_orig[np.argmax(np.abs(H_orig))]
        zpk_rs   = z_rs[np.argmax(np.abs(H_rs))]
        assert abs(zpk_orig - zpk_rs) < 1e-3

    def test_nonlinear_sweep_recovers_sharp_peak(self):
        """With a2 on, the reflector peak is smeared. Resampling on the
        aux phase should bring it back close to the clean resolution."""
        cfg_clean = _aux_cfg()
        cfg_nl    = _aux_cfg(a2=5e16)   # moderate, keeps aux below Nyquist

        acq_clean = run_campaign(cfg_clean)[-1]
        acq_nl    = run_campaign(cfg_nl)[-1]

        # raw nonlinear reflectogram
        H_nl, z_nl = fft_reflectogram(
            acq_nl.digital_main[0].astype(np.float64), acq_nl.dz)
        r_nl = measure_resolution(H_nl, z_nl, 0.5)

        # k-clock corrected
        beat_nl = acq_nl.digital_main[0].astype(np.float64)
        beat_rs = kclock_resample(beat_nl, acq_nl.aux_signal,
                                   trim_start=acq_nl.aux_valid_start)
        H_rs, z_rs = fft_reflectogram(beat_rs, acq_nl.dz)
        r_rs = measure_resolution(H_rs, z_rs, 0.5)

        # clean baseline
        H_c, z_c = fft_reflectogram(
            acq_clean.digital_main[0].astype(np.float64), acq_clean.dz)
        r_c = measure_resolution(H_c, z_c, 0.5)

        # sanity: the uncorrected peak was actually smeared
        assert r_nl["resolution"] > r_c["resolution"] * 1.5
        # k-clock shrinks it by a lot
        assert r_rs["resolution"] < r_nl["resolution"] * 0.5
        # and lands within ~2x of the clean reference
        assert r_rs["resolution"] < r_c["resolution"] * 2.5

    def test_ripple_correction(self):
        """Small ripple (keeps the sweep monotonic); k-clock should restore
        the peak amplitude that was spread by the ripple sidelobes."""
        # amp*2pi/period = 3.14e13 << gamma=5e14 -> sweep stays monotonic
        cfg = _aux_cfg(ripple_amp=1e10, ripple_period=2e-3)
        acq = run_campaign(cfg)[-1]

        H_nl, z_nl = fft_reflectogram(
            acq.digital_main[0].astype(np.float64), acq.dz)
        mask = (z_nl > 0.45) & (z_nl < 0.55)
        peak_nl = np.max(np.abs(H_nl[mask]))

        beat = acq.digital_main[0].astype(np.float64)
        beat_rs = kclock_resample(beat, acq.aux_signal,
                                   trim_start=acq.aux_valid_start)
        H_rs, z_rs = fft_reflectogram(beat_rs, acq.dz)
        mask_rs = (z_rs > 0.45) & (z_rs < 0.55)
        peak_rs = np.max(np.abs(H_rs[mask_rs]))

        # peak should come back up (energy re-concentrates)
        assert peak_rs > peak_nl * 1.5


class TestKClockMonotonicity:
    """#64 -- kclock_resample was passing non-monotonic phi to np.interp."""

    def test_mask_keeps_strictly_increasing_samples(self):
        phi = np.array([0.0, 1.0, 0.9, 1.1, 1.2, 1.15, 1.3])
        keep = _strict_increasing_mask(phi)
        # drop 0.9 (< 1.0) and 1.15 (< 1.2); first sample always kept
        expected = np.array([True, True, False, True, True, False, True])
        np.testing.assert_array_equal(keep, expected)

    def test_mask_noop_for_monotonic_phi(self):
        phi = np.linspace(0.0, 10.0, 100)
        assert _strict_increasing_mask(phi).all()

    def test_mask_after_initial_overshoot(self):
        # first sample is a big outlier: nothing ever exceeds it until the very
        # end -> everything in between must be dropped.
        phi = np.array([10.0, 5.0, 6.0, 7.0, 8.0, 11.0])
        keep = _strict_increasing_mask(phi)
        np.testing.assert_array_equal(keep, [True, False, False, False, False, True])

    def test_raises_on_fully_broken_aux(self):
        # random aux -> Hilbert unwrap is noise, way above the 1% threshold
        rng = np.random.default_rng(0)
        aux = rng.standard_normal(4096)
        beat = np.zeros(4096)
        with pytest.raises(ValueError, match="non-monotonic"):
            kclock_resample(beat, aux)

    def test_small_glitches_do_not_silently_distort(self):
        # build a clean aux + beat pair via the real pipeline, then splice a
        # handful of old samples into the middle of aux. After the fix the
        # filtered path must stay close to the clean-aux output; before the
        # fix np.interp would see a non-monotonic xp and distort the result.
        cfg = _aux_cfg()
        acq = run_campaign(cfg)[-1]
        beat = acq.digital_main[0].astype(np.float64)
        aux_clean = acq.aux_signal.astype(np.float64)
        trim = acq.aux_valid_start

        out_clean = kclock_resample(beat, aux_clean, trim_start=trim)

        # splice ~0.2% of samples back in time -> local hilbert-phase back-step
        aux_bad = aux_clean.copy()
        n = len(aux_bad)
        src = n // 3
        dst = 2 * n // 3
        k = max(10, n // 500)
        aux_bad[dst : dst + k] = aux_clean[src : src + k]
        out_bad = kclock_resample(beat, aux_bad, trim_start=trim)

        # the bulk of the resampled beat should still agree with the clean case
        corr = np.corrcoef(out_bad, out_clean)[0, 1]
        assert corr > 0.99
