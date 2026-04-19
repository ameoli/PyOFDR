"""Analytical OFDR budget calculator.

Pure algebra from the config -- no simulation. Covers:
  - geometric quantities (dz, max beat frequency, Nyquist headroom)
  - optical power chain (laser -> splitter -> circulator -> fiber)
  - Rayleigh backscatter at near/far end
  - receiver noise floor (shot, thermal, dark, RIN, quantization)
  - NEP referred to optical input
  - dynamic range
  - phase-noise RMS at the far end (kept separate from the current RSS)
  - strain / temperature sensitivity + max unambiguous strain

Silica constants (p_e, alpha_L, dn/dT) are hardcoded -- per-fiber overrides
can come later.

See issue #43.
"""

from __future__ import annotations

import math
from typing import Any

from core.config import compute_derived
from core.config_models import RootConfig
from utils.constants import C, E_CHARGE
from utils.units import dB_to_linear


# silica photoelastic coefficient
_P_E = 0.22
# thermal expansion of silica [1/K]
_ALPHA_L = 5.5e-7
# thermo-optic (1/n * dn/dT) for silica near 1550 nm [1/K]
_XI = 6.5e-6


def compute_budget(cfg: dict[str, Any]) -> dict[str, float]:
    """Return a dict of budget quantities (all SI) from a config dict."""
    cfg = RootConfig(**cfg).model_dump()

    src_ = cfg["source"]
    opt  = cfg["optics"]
    fib  = cfg["fiber"]
    det  = cfg["detection"]
    adc  = cfg["adc"]

    P_laser = src_["power"]
    wl      = src_["center_wavelength"]
    lw      = src_["linewidth"]
    rin     = src_["rin_dB_per_Hz"]
    eta     = opt["splitting_ratio"]
    IL_circ = opt["circulator"]["insertion_loss_dB"]

    # use the homogeneous terms only -- segments are a refinement that
    # this first pass doesn't try to account for
    alpha_dB_km = fib["attenuation_dB_per_km"]
    R_dB        = fib["rayleigh_coefficient_dB"]
    L           = fib["length"]
    n_core      = fib["n_core"]

    R_resp = det["responsivity"]
    NEP    = det["thermal_nep"]
    I_dark = det["dark_current"]
    B      = det["bandwidth"]
    I_sat  = det["saturation_current"]

    bits = adc["bits"]
    V_fs = adc["voltage_range"]
    Z    = adc["input_impedance"]
    fs   = adc["sample_rate"]

    # --- optical power chain (all linear [W]) --------------------------
    il_circ_lin = dB_to_linear(-IL_circ)
    P_ref_arm   = eta * P_laser
    P_to_fiber  = (1.0 - eta) * P_laser * il_circ_lin

    # one-way power attenuation coefficient [1/m]
    alpha_m   = alpha_dB_km * math.log(10.0) / 10.0 / 1000.0
    R_per_m   = dB_to_linear(R_dB)       # backscattered power per metre of fiber

    def _back_from(z):
        # power backscattered by a 1-m slab at z, through the circulator back
        return P_to_fiber * R_per_m * math.exp(-2.0 * alpha_m * z) * il_circ_lin

    P_back_near = _back_from(0.0)
    P_back_far  = _back_from(L)

    # --- receiver -------------------------------------------------------
    # reference arm DC photocurrent dominates in heterodyne
    I_dc = R_resp * P_ref_arm

    sigma_shot    = math.sqrt(2.0 * E_CHARGE * I_dc * B)
    sigma_thermal = R_resp * NEP * math.sqrt(B)
    sigma_dark    = math.sqrt(2.0 * E_CHARGE * I_dark * B) if I_dark > 0 else 0.0

    # RIN converts directly into photocurrent noise at the reference arm.
    # Worst case (single-ended); balanced detection suppresses the common
    # mode, but that depends on matching and is not captured here.
    if rin is not None:
        rin_lin   = 10.0 ** (rin / 10.0)
        sigma_rin = I_dc * math.sqrt(rin_lin * B)
    else:
        sigma_rin = 0.0

    # quantization: V_fs / (2^bits * sqrt(12)) total RMS, spread over [0, fs/2]
    sigma_q_V_tot = V_fs / (2 ** bits) / math.sqrt(12.0)
    sigma_q_V     = sigma_q_V_tot * math.sqrt(B / (fs / 2.0))
    sigma_quant   = sigma_q_V / Z

    sigma_total = math.sqrt(
        sigma_shot ** 2 + sigma_thermal ** 2 + sigma_dark ** 2
        + sigma_rin ** 2 + sigma_quant ** 2
    )

    nep_total = sigma_total / R_resp     # W RMS over B

    # dynamic range: saturation if declared, else the reference DC swing
    I_max = I_sat if I_sat is not None else I_dc
    dynamic_range_dB = (
        20.0 * math.log10(I_max / sigma_total) if sigma_total > 0 else float("inf")
    )

    # --- phase noise at the far end ------------------------------------
    # Lorentzian source, round-trip delay tau = 2 n L / c:
    # <Dphi^2> = 2*pi*lw*tau.   Meaningful only when tau << 1/lw.
    tau_far   = 2.0 * n_core * L / C
    sigma_phi = math.sqrt(2.0 * math.pi * lw * tau_far) if lw > 0 else 0.0

    # --- strain / temperature sensitivity ------------------------------
    # Rayleigh spectral shift (Froggatt-Moore):
    #   Dnu / nu = - ((1-p_e)*eps + (alpha_L + xi)*DT)
    nu_c       = C / wl
    d_nu_d_eps = -(1.0 - _P_E) * nu_c       # Hz per unit strain
    d_nu_d_T   = -(_ALPHA_L + _XI) * nu_c   # Hz per K

    # --- max unambiguous strain ----------------------------------------
    # Absolute ceiling from the spectral shift method: the cross-correlation
    # window can track a shift up to +/- delta_nu/2. Processing with a finite
    # gauge length makes this tighter in practice.
    derived  = compute_derived(cfg)
    delta_nu = derived["delta_nu"]
    eps_max  = (delta_nu / 2.0) / abs(d_nu_d_eps)

    out = {
        "P_laser":          P_laser,
        "P_ref_arm":        P_ref_arm,
        "P_to_fiber":       P_to_fiber,
        "P_back_near":      P_back_near,
        "P_back_far":       P_back_far,
        "I_dc_ref":         I_dc,
        "sigma_shot":       sigma_shot,
        "sigma_thermal":    sigma_thermal,
        "sigma_dark":       sigma_dark,
        "sigma_rin":        sigma_rin,
        "sigma_quant":      sigma_quant,
        "sigma_total":      sigma_total,
        "nep_total":        nep_total,
        "dynamic_range_dB": dynamic_range_dB,
        "bandwidth":        B,
        "sigma_phi_far":    sigma_phi,
        "tau_far":          tau_far,
        "d_nu_d_eps":       d_nu_d_eps,
        "d_nu_d_T":         d_nu_d_T,
        "eps_max":          eps_max,
    }
    out.update(derived)
    return out


def _dBm(p):
    return 10.0 * math.log10(p * 1000.0) if p > 0 else float("-inf")


def print_budget(cfg: dict[str, Any]) -> None:
    """Pretty-print the budget for a config dict."""
    b = compute_budget(cfg)
    print("PyOFDR power & noise budget:")
    print(f"  Spatial resolution:   {b['dz']*1e3:.3f} mm")
    print(f"  Max beat freq:        {b['f_beat_max']*1e-6:.1f} MHz "
          f"(Nyquist {b['f_nyquist']*1e-6:.0f} MHz)")
    print(f"  Laser power:          {_dBm(b['P_laser']):+.1f} dBm")
    print(f"  Reference arm:        {_dBm(b['P_ref_arm']):+.1f} dBm")
    print(f"  Into fiber:           {_dBm(b['P_to_fiber']):+.1f} dBm")
    print(f"  Backscatter /m near:  {_dBm(b['P_back_near']):+.1f} dBm")
    print(f"  Backscatter /m far:   {_dBm(b['P_back_far']):+.1f} dBm")
    print(f"  Ref photocurrent:     {b['I_dc_ref']*1e3:.3f} mA")
    print(f"  Noise floor over {b['bandwidth']*1e-6:.1f} MHz:")
    print(f"    shot:               {b['sigma_shot']*1e9:.3f} nA")
    print(f"    thermal:            {b['sigma_thermal']*1e9:.3f} nA")
    print(f"    dark:               {b['sigma_dark']*1e9:.3f} nA")
    print(f"    RIN:                {b['sigma_rin']*1e9:.3f} nA")
    print(f"    quantization:       {b['sigma_quant']*1e9:.3f} nA")
    print(f"    total:              {b['sigma_total']*1e9:.3f} nA")
    print(f"  NEP (total, over B):  {b['nep_total']*1e9:.2f} nW")
    print(f"  Dynamic range:        {b['dynamic_range_dB']:.1f} dB")
    print(f"  Phase noise (far):    {b['sigma_phi_far']*1e3:.2f} mrad RMS")
    print(f"  Strain sensitivity:   {b['d_nu_d_eps']*1e-12:.2f} MHz/ustrain")
    print(f"  Temp sensitivity:     {b['d_nu_d_T']*1e-9:.3f} GHz/K")
    print(f"  Max |strain|:         {b['eps_max']*1e6:.0f} ustrain")
