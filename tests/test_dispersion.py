"""Chromatic dispersion (GVD) on the OFDR beat signal -- issue #34.

A scatterer at z picks up a roundtrip phase 4*pi^2 * beta2 * z * (nu-nu_0)^2.
The chirp factors out as phi(z,t) = z*K(t), so the IFFT trick still holds:
the GVD warp is z-independent and reuses the sweep-nonlinearity machinery.
The peak broadening is proportional to z (further reflectors get smeared
more), and to the sweep bandwidth (richer chirp, more accumulated phase).
"""

import numpy as np

from helpers import CFG
from pyofdr.analysis.demodulation import fft_reflectogram
from pyofdr.analysis.spatial_metrics import measure_resolution
from pyofdr.core.campaign import run_campaign
from pyofdr.utils.constants import C as _C


def _disp_cfg(D=0.0, z=0.5, **extra):
    """CFG with chromatic dispersion + a single reflector."""
    cfg = {
        **CFG,
        "fiber": {
            **CFG["fiber"],
            "dispersion_D": D,
            "reflectors": [{"z": z, "R": 0.01}],
        },
        # noiseless so the peak is clean and FWHM is meaningful
        "detection": {
            **CFG["detection"],
            "shot_noise": False,
            "thermal_nep": 0.0,
            "dark_current": 0.0,
        },
    }
    cfg.update(extra)
    return cfg


class TestChromaticDispersion:

    def test_existing_configs_unaffected(self):
        # default config has D=0 -> no dispersion warp triggered
        acq = run_campaign(CFG)[-1]
        assert acq.digital_main is not None

    def test_zero_D_matches_baseline(self):
        # explicit D=0 must be byte-identical to absent dispersion_D
        cfg_a = _disp_cfg(D=0.0)
        cfg_b = {**CFG, "fiber": {**CFG["fiber"],
                  "reflectors": [{"z": 0.5, "R": 0.01}]},
                  "detection": {**CFG["detection"], "shot_noise": False,
                                "thermal_nep": 0.0, "dark_current": 0.0}}
        a = run_campaign(cfg_a)[-1]
        b = run_campaign(cfg_b)[-1]
        np.testing.assert_array_equal(a.digital_main, b.digital_main)

    def test_dispersion_broadens_peak(self):
        # ~60x SMF dispersion -> ~mm peak width vs ~20 um baseline
        cfg_clean = _disp_cfg(D=0.0)
        cfg_disp  = _disp_cfg(D=1e-3)
        a_c = run_campaign(cfg_clean)[-1]
        a_d = run_campaign(cfg_disp)[-1]

        H_c, z = fft_reflectogram(a_c.digital_main[0].astype(np.float64), a_c.dz)
        H_d, _ = fft_reflectogram(a_d.digital_main[0].astype(np.float64), a_d.dz)

        r_c = measure_resolution(H_c, z, 0.5)
        r_d = measure_resolution(H_d, z, 0.5)
        assert r_d["resolution"] > 5.0 * r_c["resolution"]

    def test_FWHM_matches_GVD_formula(self):
        # quantitative: spread = 2*pi*|beta2|*BW*c*z/n
        # for D=1e-3, BW=5 THz, z=0.5, n=1.4682 -> ~4 mm
        D = 1e-3
        z0 = 0.5
        cfg = _disp_cfg(D=D, z=z0)
        acq = run_campaign(cfg)[-1]
        H, z = fft_reflectogram(acq.digital_main[0].astype(np.float64), acq.dz)
        r = measure_resolution(H, z, z0)

        wl = CFG["source"]["center_wavelength"]
        dwl = CFG["source"]["sweep_range"]
        BW = _C / wl**2 * dwl
        n_core = CFG["fiber"]["n_core"]
        beta2 = -wl**2 * D / (2 * np.pi * _C)
        dz_predict = 2 * np.pi * abs(beta2) * BW * _C * z0 / n_core
        # -6 dB width of a chirp-broadened peak ~ predicted spread,
        # within ~30% (loose to allow for window/leakage shaping)
        np.testing.assert_allclose(r["resolution"], dz_predict, rtol=0.3)

    def test_negative_D_also_broadens(self):
        # broadening depends on |beta2|, sign of D shouldn't matter
        cfg_pos = _disp_cfg(D=+1e-3)
        cfg_neg = _disp_cfg(D=-1e-3)
        a_p = run_campaign(cfg_pos)[-1]
        a_n = run_campaign(cfg_neg)[-1]
        H_p, z = fft_reflectogram(a_p.digital_main[0].astype(np.float64), a_p.dz)
        H_n, _ = fft_reflectogram(a_n.digital_main[0].astype(np.float64), a_n.dz)
        r_p = measure_resolution(H_p, z, 0.5)
        r_n = measure_resolution(H_n, z, 0.5)
        # symmetric to a few percent
        np.testing.assert_allclose(r_p["resolution"], r_n["resolution"], rtol=0.05)

    def test_broadening_scales_linearly_with_z(self):
        # GVD spread scales linearly with the scatterer position
        cfg_a = _disp_cfg(D=5e-4, z=0.3)
        cfg_b = _disp_cfg(D=5e-4, z=0.6)
        a = run_campaign(cfg_a)[-1]
        b = run_campaign(cfg_b)[-1]
        H_a, z = fft_reflectogram(a.digital_main[0].astype(np.float64), a.dz)
        H_b, _ = fft_reflectogram(b.digital_main[0].astype(np.float64), b.dz)
        r_a = measure_resolution(H_a, z, 0.3)
        r_b = measure_resolution(H_b, z, 0.6)
        # 2x the distance -> 2x the broadening, allow 30% slack
        np.testing.assert_allclose(r_b["resolution"] / r_a["resolution"],
                                    2.0, rtol=0.3)
