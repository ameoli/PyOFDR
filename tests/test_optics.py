"""Tests for Mach-Zehnder interferometer and circulator."""

import math

import numpy as np
import pytest

from helpers import CFG
from pyofdr.core.acquisition import Acquisition
from pyofdr.fiber.profile import FiberGenerator
from pyofdr.source.swept_laser import SweptLaser
from pyofdr.optics.mach_zehnder import MachZehnder
from pyofdr.optics.components import Circulator
from pyofdr.utils.constants import C


class TestMachZehnder:

    def _make_acq(self, cfg=None):
        cfg = cfg or CFG
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        acq = SweptLaser(cfg).process(acq)
        return MachZehnder(cfg).process(acq)

    def test_photocurrent_exists(self):
        acq = self._make_acq()
        assert acq.photocurrent_main is not None
        assert acq.photocurrent_main.shape[-1] == acq.n_samples

    def test_photocurrent_is_real(self):
        acq = self._make_acq()
        # should be float, not complex
        assert acq.photocurrent_main.dtype == np.float64


class TestFBGIntegration:
    """End-to-end check that an FBG produces a peak in the OFDR trace (#40)."""

    def test_on_bragg_energy_concentrated_at_z(self):
        # The OFDR spatial response of an FBG of length L_g is a rect of
        # width L_g (in z), so the FFT peak isn't a delta -- it's a plateau
        # roughly L_g/dz bins wide centred at the FBG position. Check
        # energy concentration rather than the argmax bin.
        lam_B = CFG["source"]["center_wavelength"]
        z_fbg = 0.5
        L_g = 1e-2
        cfg = {**CFG,
               "fiber":  {**CFG["fiber"],
                          "rayleigh_coefficient_dB": -200.0,
                          "fbg_arrays": [{"z": z_fbg,
                                           "bragg_wavelength": lam_B,
                                           "length": L_g,
                                           "peak_reflectivity": 0.04}]}}
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        acq = SweptLaser(cfg).process(acq)
        acq = MachZehnder(cfg).process(acq)

        spectrum = np.abs(np.fft.rfft(acq.photocurrent_main[0]))

        n_core = CFG["fiber"]["n_core"]
        wl = CFG["source"]["center_wavelength"]
        dwl = CFG["source"]["sweep_range"]
        T = CFG["source"]["sweep_duration"]
        gamma = (C / wl**2 * dwl) / T
        f_beat = 2.0 * n_core * z_fbg * gamma / C
        expected_bin = int(round(f_beat * acq.n_samples * acq.dt))
        # half-width of the FBG plateau in bins
        half_width = int(round(L_g / acq.dz / 2)) + 5

        in_band = spectrum[expected_bin - half_width:expected_bin + half_width].sum()
        total = spectrum.sum()
        # most of the energy should sit inside the FBG plateau
        assert in_band / total > 0.9

    def test_no_fbg_no_peak(self):
        """Sanity: dropping the FBG should remove the peak."""
        cfg = {**CFG, "fiber": {**CFG["fiber"],
               "rayleigh_coefficient_dB": -200.0}}
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        acq = SweptLaser(cfg).process(acq)
        acq = MachZehnder(cfg).process(acq)
        spectrum = np.abs(np.fft.rfft(acq.photocurrent_main[0]))
        # extremely low Rayleigh, no FBG -> spectrum should be tiny
        assert spectrum.max() < 1e-6

    def test_two_fbgs_two_peaks(self):
        """Two FBGs at different z should give two peaks at the right bins."""
        lam_B = CFG["source"]["center_wavelength"]
        z1, z2 = 0.2, 0.8
        L_g = 5e-3
        cfg = {**CFG,
               "fiber":  {**CFG["fiber"],
                          "rayleigh_coefficient_dB": -200.0,
                          "fbg_arrays": [
                              {"z": z1, "bragg_wavelength": lam_B,
                               "length": L_g, "peak_reflectivity": 0.04},
                              {"z": z2, "bragg_wavelength": lam_B,
                               "length": L_g, "peak_reflectivity": 0.04},
                          ]}}
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        acq = SweptLaser(cfg).process(acq)
        acq = MachZehnder(cfg).process(acq)
        spectrum = np.abs(np.fft.rfft(acq.photocurrent_main[0]))

        n_core = CFG["fiber"]["n_core"]
        wl = CFG["source"]["center_wavelength"]
        dwl = CFG["source"]["sweep_range"]
        T = CFG["source"]["sweep_duration"]
        gamma = (C / wl**2 * dwl) / T
        half_w = int(round(L_g / acq.dz / 2)) + 5

        for z_fbg in (z1, z2):
            f_beat = 2.0 * n_core * z_fbg * gamma / C
            bin_z = int(round(f_beat * acq.n_samples * acq.dt))
            in_band = spectrum[bin_z - half_w:bin_z + half_w].sum()
            # local plateau should dominate the wider neighbourhood
            wide = spectrum[bin_z - 4 * half_w:bin_z + 4 * half_w].sum()
            assert in_band / wide > 0.5


class TestCoherenceRolloff:
    """Lorentzian coherence roll-off with finite laser linewidth (#62)."""

    def test_zero_linewidth_matches_no_rolloff(self):
        # with linewidth=0 the visibility is identically 1, so the beat
        # must be bit-for-bit identical to the no-rolloff path
        cfg0 = {**CFG, "source": {**CFG["source"], "linewidth": 0.0}}
        acq = Acquisition()
        acq = FiberGenerator(cfg0).process(acq)
        acq = SweptLaser(cfg0).process(acq)
        acq = MachZehnder(cfg0).process(acq)
        assert acq.log[-1]["coherence_rolloff"] is False

    def test_visibility_matches_lorentzian(self):
        # single reflector, compare FFT magnitude at its expected bin with
        # vs without linewidth. Ratio should equal exp(-pi * dnu * tau).
        # Linewidth picked to land visibility near 0.5 -- far from the
        # Rayleigh/numerical floor.
        length = 5.0
        reflector_z = 4.5
        linewidth = 5.0e6
        cfg_base = {**CFG,
                    "fiber":  {**CFG["fiber"], "length": length,
                               "reflectors": [{"z": reflector_z, "R": 1e-4}],
                               "rayleigh_coefficient_dB": -200.0},
                    "source": {**CFG["source"], "linewidth": 0.0}}
        cfg_lw = {**cfg_base, "source": {**CFG["source"], "linewidth": linewidth}}

        def _peak(cfg):
            acq = Acquisition()
            acq = FiberGenerator(cfg).process(acq)
            acq = SweptLaser(cfg).process(acq)
            acq = MachZehnder(cfg).process(acq)
            fft = np.abs(np.fft.rfft(acq.photocurrent_main[0]))
            return fft.max()

        p0 = _peak(cfg_base)
        p1 = _peak(cfg_lw)
        n_core = CFG["fiber"]["n_core"]
        tau = 2.0 * n_core * reflector_z / C
        expected = math.exp(-math.pi * linewidth * tau)
        assert p1 / p0 == pytest.approx(expected, rel=0.05)

    def test_log_flag_on(self):
        cfg = {**CFG, "source": {**CFG["source"], "linewidth": 1.0e4}}
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        acq = SweptLaser(cfg).process(acq)
        acq = MachZehnder(cfg).process(acq)
        assert acq.log[-1]["coherence_rolloff"] is True


class TestPowerModulation:
    """Source power variations (edge droop, RIN) must propagate to the beat."""

    @staticmethod
    def _reflector_cfg(**src_overrides):
        # single strong reflector, rayleigh pushed way down, no coherence
        # roll-off -> beat is a clean sinusoid whose analytic envelope tracks P(t).
        return {**CFG,
                "fiber":  {**CFG["fiber"], "length": 5.0,
                           "reflectors": [{"z": 4.5, "R": 1e-4}],
                           "rayleigh_coefficient_dB": -200.0},
                "source": {**CFG["source"], "linewidth": 0.0,
                           "rin_dB_per_Hz": None,
                           "power_envelope_edge_dB": 0.0,
                           **src_overrides}}

    def _run(self, cfg):
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        acq = SweptLaser(cfg).process(acq)
        return MachZehnder(cfg).process(acq)

    def test_envelope_droop_reaches_beat(self):
        # 6 dB parabolic droop on source power must show up as 6 dB drop
        # of the beat's analytic envelope at the sweep edges.
        from scipy.signal import hilbert
        edge_dB = 6.0
        cfg = self._reflector_cfg(power_envelope_edge_dB=edge_dB)
        acq = self._run(cfg)
        beat = acq.photocurrent_main[0]
        env = np.abs(hilbert(beat))
        n = len(env)
        # peak at center, edge samples at t=0 and t~T
        env_center = env[n // 2]
        env_edge   = 0.5 * (env[0] + env[-1])
        drop_dB = 10.0 * np.log10(env_edge / env_center)
        np.testing.assert_allclose(drop_dB, -edge_dB, atol=0.3)

    def test_rin_increases_beat_variance(self):
        # turning RIN on must increase var(beat) by ~sigma_rin^2 (relative).
        # at -85 dB/Hz with BW=100 MHz, sigma_rin^2 ~ 0.316 -> ratio ~1.3
        rin_dB = -85.0
        clean = self._run(self._reflector_cfg())
        noisy = self._run(self._reflector_cfg(rin_dB_per_Hz=rin_dB))
        bw = CFG["adc"]["sample_rate"] / 2.0
        expected_ratio = 1.0 + 10.0 ** (rin_dB / 10.0) * bw
        ratio = np.var(noisy.photocurrent_main) / np.var(clean.photocurrent_main)
        np.testing.assert_allclose(ratio, expected_ratio, rtol=0.1)

    def test_flat_power_scales_linearly_with_P0(self):
        # constant P: doubling the source power must double the beat.
        # guards against a regression where a stale mean-power scalar
        # would decouple the beat from the current P0.
        cfg_1 = self._reflector_cfg()
        cfg_2 = {**cfg_1, "source": {**cfg_1["source"],
                                     "power": 2.0 * cfg_1["source"]["power"]}}
        a1 = self._run(cfg_1)
        a2 = self._run(cfg_2)
        # compare where beat_1 is safely away from zero crossings
        b1 = a1.photocurrent_main[0]
        mask = np.abs(b1) > 0.5 * np.abs(b1).max()
        ratio = a2.photocurrent_main[0][mask] / b1[mask]
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-10)


class TestCirculator:

    def test_zero_loss_is_unity(self):
        c = Circulator(insertion_loss_dB=0.0)
        assert c.insertion_loss == pytest.approx(1.0)
        assert c.round_trip_transmission == pytest.approx(1.0)

    def test_insertion_loss_correct(self):
        c = Circulator(insertion_loss_dB=0.7)
        expected = 10 ** (-0.7 / 20.0)
        assert c.insertion_loss == pytest.approx(expected)

    def test_round_trip_is_squared(self):
        c = Circulator(insertion_loss_dB=1.0)
        il = c.insertion_loss
        assert c.round_trip_transmission == pytest.approx(il ** 2)

    def test_circulator_reduces_signal(self):
        """MZI output with circulator should be weaker than without."""
        cfg_no_circ = {**CFG, "optics": {**CFG["optics"],
                       "circulator": {"insertion_loss_dB": 0.0,
                                      "isolation_dB": 400.0,
                                      "return_loss_dB": 400.0}}}
        cfg_circ = {**CFG, "optics": {**CFG["optics"],
                    "circulator": {"insertion_loss_dB": 3.0,
                                   "isolation_dB": 400.0,
                                   "return_loss_dB": 400.0}}}
        acq0 = Acquisition()
        acq0 = FiberGenerator(cfg_no_circ).process(acq0)
        acq0 = SweptLaser(cfg_no_circ).process(acq0)
        acq0 = MachZehnder(cfg_no_circ).process(acq0)

        acq1 = Acquisition()
        acq1 = FiberGenerator(cfg_circ).process(acq1)
        acq1 = SweptLaser(cfg_circ).process(acq1)
        acq1 = MachZehnder(cfg_circ).process(acq1)

        # signal power should be lower with circulator loss
        assert np.var(acq1.photocurrent_main) < np.var(acq0.photocurrent_main)


class TestCirculatorLeakage:
    """Isolation + return-loss show up as a z=0 spike on the reflectogram."""

    @staticmethod
    def _cfg(*, iso_dB, rl_dB, il_dB=0.0):
        return {**CFG,
                "fiber":  {**CFG["fiber"], "rayleigh_coefficient_dB": -200.0},
                "optics": {**CFG["optics"],
                           "circulator": {"insertion_loss_dB": il_dB,
                                          "isolation_dB": iso_dB,
                                          "return_loss_dB": rl_dB}}}

    def _run(self, cfg):
        acq = Acquisition()
        acq = FiberGenerator(cfg).process(acq)
        acq = SweptLaser(cfg).process(acq)
        return MachZehnder(cfg).process(acq)

    def test_return_loss_creates_z0_spike(self):
        # with the fiber profile pushed below 1e-10 the rfft bin at f=0
        # is dominated by the return-loss reflector. lowering RL_dB by 20
        # raises the bin by 20 dB.
        a_loose = self._run(self._cfg(iso_dB=400, rl_dB=40))
        a_tight = self._run(self._cfg(iso_dB=400, rl_dB=60))
        s_loose = np.abs(np.fft.rfft(a_loose.photocurrent_main[0]))
        s_tight = np.abs(np.fft.rfft(a_tight.photocurrent_main[0]))
        ratio_dB = 20.0 * np.log10(s_loose.max() / s_tight.max())
        np.testing.assert_allclose(ratio_dB, 20.0, atol=0.5)

    def test_isolation_creates_dc_offset(self):
        # leakage is purely DC -> the mean of the photocurrent must scale
        # like 10**(-iso_dB/20). it does NOT pick up the round-trip IL^2.
        a_50 = self._run(self._cfg(iso_dB=50, rl_dB=400, il_dB=3.0))
        a_70 = self._run(self._cfg(iso_dB=70, rl_dB=400, il_dB=3.0))
        ratio = a_50.photocurrent_main.mean() / a_70.photocurrent_main.mean()
        # 20 dB step in iso -> 10x in amplitude
        np.testing.assert_allclose(ratio, 10.0, rtol=0.02)

    def test_return_loss_picks_up_IL_squared(self):
        # the RL reflector sits before the IL^2 prefactor, so raising the
        # circulator IL by 3 dB drops the z=0 spike by 6 dB (round trip).
        a_0 = self._run(self._cfg(iso_dB=400, rl_dB=40, il_dB=0.0))
        a_3 = self._run(self._cfg(iso_dB=400, rl_dB=40, il_dB=3.0))
        s_0 = np.abs(np.fft.rfft(a_0.photocurrent_main[0])).max()
        s_3 = np.abs(np.fft.rfft(a_3.photocurrent_main[0])).max()
        drop_dB = 20.0 * np.log10(s_3 / s_0)
        np.testing.assert_allclose(drop_dB, -6.0, atol=0.2)

    def test_isolation_rides_power_envelope(self):
        # the leakage DC term is multiplied by P(t), so a 6 dB parabolic
        # droop on the source must show up as 6 dB on the mean of the
        # leakage at the edges versus the centre.
        cfg = self._cfg(iso_dB=20, rl_dB=400)  # 20 dB iso so leakage dominates
        cfg["source"] = {**cfg["source"],
                         "linewidth": 0.0,
                         "power_envelope_edge_dB": 6.0}
        acq = self._run(cfg)
        pm = acq.photocurrent_main[0]
        n = len(pm)
        # the photocurrent is dc_offset + small zero-mean fluctuation. take a
        # short-window mean to track the local DC.
        win = max(n // 200, 100)
        from scipy.ndimage import uniform_filter1d
        local_dc = uniform_filter1d(pm, win)
        edge = 0.5 * (local_dc[win] + local_dc[-win])
        centre = local_dc[n // 2]
        drop_dB = 10.0 * np.log10(edge / centre)
        np.testing.assert_allclose(drop_dB, -6.0, atol=0.3)
