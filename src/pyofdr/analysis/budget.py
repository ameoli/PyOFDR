"""Analytical OFDR budget calculator.

Config-based estimates -- no simulation. Covers:
  - geometric quantities (dz, max beat frequency, Nyquist headroom)
  - optical power chain (laser -> splitter -> circulator -> fiber)
  - Rayleigh backscatter at near/far end
  - receiver noise floor (shot, thermal, dark, RIN, quantization)
  - NEP referred to optical input
  - dynamic range
  - phase-noise RMS at the far end (kept separate from the current RSS)
  - strain / temperature sensitivity + max unambiguous strain

Noise quantities are input-referred current standard deviations. Receiver
noise is integrated over the digital Butterworth noise bandwidth; ADC noise
is added afterwards over the full Nyquist band. RIN in the current pipeline
is multiplicative on the beat, so its estimate needs the noiseless pre-filter
beat RMS. Missing RIN input yields NaN rather than a reference-arm DC estimate.

See issue #43.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from scipy.integrate import quad

from pyofdr.core.config import compute_derived
from pyofdr.core.config_models import RootConfig
from pyofdr.utils.constants import C, E_CHARGE
from pyofdr.utils.units import dB_to_linear


@lru_cache(maxsize=128)
def _noise_bandwidth(fs: float, bandwidth: float, order: int) -> float:
    """One-sided integral of the digital Butterworth |H(f)|^2, in Hz.

    Use the same 0.99-Nyquist cutoff clamp as AntiAliasFilter. The bilinear
    substitution x = tan(pi*f/fs)/tan(pi*fc/fs) resolves even narrow filters
    without a frequency grid tied to fs. Split at x=1 (the cutoff).
    """
    cutoff = min(bandwidth, 0.99 * fs / 2.0)
    a = math.tan(math.pi * cutoff / fs)

    def integrand(x):
        if x <= 1.0:
            gain_sq = 1.0 / (1.0 + x ** (2 * order))
        else:
            inverse_power = x ** (-2 * order)
            gain_sq = inverse_power / (1.0 + inverse_power)
        return gain_sq / (1.0 + (a * x) ** 2)

    area = quad(integrand, 0.0, 1.0, epsabs=1e-10)[0]
    area += quad(integrand, 1.0, math.inf, epsabs=1e-10)[0]
    return fs * a / math.pi * area


def compute_budget(
    cfg: dict[str, Any], *, beat_rms_current: float | None = None,
) -> dict[str, float]:
    """Return budget quantities in SI for the configured receiver mode.

    ``beat_rms_current`` is sqrt(mean(I_beat**2)) in A, before the receiver
    filter, from a noiseless linear run (including any optical leakage).
    It is required only when source RIN is enabled. Then sigma_rin, sigma_analog,
    sigma_total, nep_total and dynamic_range_dB are NaN if it is omitted;
    sigma_receiver still reports the known additive receiver contribution.

    Estimates assume reference-dominated shot noise, a linear unsaturated
    detector, weak unclipped white RIN, and settled filter statistics. ADC
    quantization uses the uniform-error approximation; ENOB extra noise is
    included, but jitter and DNL/INL are not. No post-ADC filtering is assumed.
    Phase-noise RMS is a separate theoretical quantity, not a prediction of
    the MZI's stochastic white-FM pedestal (which is not simulated).
    """
    if beat_rms_current is not None and (
        not math.isfinite(beat_rms_current) or beat_rms_current < 0
    ):
        raise ValueError("beat_rms_current must be finite and nonnegative (A)")
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
    noise_bandwidth = _noise_bandwidth(fs, B, det["filter_order"])

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
    # Keep I_dc_ref as the pre-recombiner reference-arm equivalent current.
    # The 50/50 recombiner sends half to each PD, in both receiver modes.
    I_dc = R_resp * P_ref_arm
    I_dc_pd = 0.5 * I_dc
    n_pd = 2 if det["balanced"] else 1

    sigma_shot = (
        math.sqrt(2.0 * E_CHARGE * n_pd * I_dc_pd * noise_bandwidth)
        if det["shot_noise"] else 0.0
    )
    # One TIA after the subtraction node; dark current is specified per PD.
    sigma_thermal = R_resp * NEP * math.sqrt(noise_bandwidth)
    sigma_dark = math.sqrt(2.0 * E_CHARGE * n_pd * I_dark * noise_bandwidth)

    # The simulator propagates RIN on the beat in BOTH modes. Reference-arm
    # common-mode DC RIN is not propagated, so I_dc*sqrt(RIN*B) is not its floor.
    if rin is not None:
        rin_lin   = 10.0 ** (rin / 10.0)
        sigma_rin = (
            beat_rms_current * math.sqrt(rin_lin * noise_bandwidth)
            if beat_rms_current is not None else math.nan
        )
    else:
        sigma_rin = 0.0

    # ADC is downstream of the analog filter: its error has NOT been filtered.
    sigma_q_V_tot = V_fs / (2 ** bits) / math.sqrt(12.0)
    sigma_quant = sigma_q_V_tot / Z
    enob = adc["enob"]
    sigma_adc_extra = 0.0
    if enob is not None and enob < bits:
        sigma_enob = V_fs / (2.0 ** enob * math.sqrt(12.0)) / Z
        sigma_adc_extra = math.sqrt(sigma_enob ** 2 - sigma_quant ** 2)

    sigma_receiver = math.sqrt(sigma_shot ** 2 + sigma_thermal ** 2 + sigma_dark ** 2)
    sigma_analog = math.hypot(sigma_receiver, sigma_rin)
    sigma_total = math.sqrt(sigma_analog ** 2 + sigma_quant ** 2 + sigma_adc_extra ** 2)

    nep_total = sigma_total / R_resp     # integrated sampled noise, W RMS

    # A reference-current proxy unless a saturation limit is supplied;
    # this does not account for ADC clipping or nonlinear detector gain.
    I_max = I_sat if I_sat is not None else n_pd * I_dc_pd
    if math.isnan(sigma_total):
        dynamic_range_dB = math.nan
    elif sigma_total > 0:
        dynamic_range_dB = 20.0 * math.log10(I_max / sigma_total)
    else:
        dynamic_range_dB = math.inf

    # --- phase noise at the far end ------------------------------------
    # Lorentzian source, round-trip delay tau = 2 n L / c:
    # <Dphi^2> = 2*pi*lw*tau.   Meaningful only when tau << 1/lw.
    tau_far   = 2.0 * n_core * L / C
    sigma_phi = math.sqrt(2.0 * math.pi * lw * tau_far) if lw > 0 else 0.0

    # --- strain / temperature sensitivity ------------------------------
    # Rayleigh spectral shift (Froggatt-Moore):
    #   Dnu / nu = - ((1-p_e)*eps + (alpha_L + xi)*DT)
    nu_c       = C / wl
    p_e = cfg["strain"]["photoelastic_coefficient"]
    alpha_L = cfg["temperature"]["thermal_expansion"]
    xi = cfg["temperature"]["thermo_optic"]
    d_nu_d_eps = -(1.0 - p_e) * nu_c       # Hz per unit strain
    d_nu_d_T   = -(alpha_L + xi) * nu_c    # Hz per K

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
        "I_dc_pd":          I_dc_pd,
        "sigma_shot":       sigma_shot,
        "sigma_thermal":    sigma_thermal,
        "sigma_dark":       sigma_dark,
        "sigma_rin":        sigma_rin,
        "sigma_quant":      sigma_quant,
        "sigma_adc_extra":  sigma_adc_extra,
        "sigma_receiver":   sigma_receiver,
        "sigma_analog":     sigma_analog,
        "sigma_total":      sigma_total,
        "nep_total":        nep_total,
        "dynamic_range_dB": dynamic_range_dB,
        "bandwidth":        B,
        "filter_cutoff":    min(B, 0.99 * fs / 2.0),
        "noise_bandwidth":  noise_bandwidth,
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


def print_budget(cfg: dict[str, Any], *, beat_rms_current: float | None = None) -> None:
    """Pretty-print the budget for a config dict."""
    b = compute_budget(cfg, beat_rms_current=beat_rms_current)
    print("PyOFDR power & noise budget:")
    print(f"  Spatial resolution:   {b['dz']*1e3:.3f} mm")
    print(f"  Max beat freq:        {b['f_beat_max']*1e-6:.1f} MHz "
          f"(Nyquist {b['f_nyquist']*1e-6:.0f} MHz)")
    print(f"  Laser power:          {_dBm(b['P_laser']):+.1f} dBm")
    print(f"  Reference arm:        {_dBm(b['P_ref_arm']):+.1f} dBm")
    print(f"  Into fiber:           {_dBm(b['P_to_fiber']):+.1f} dBm")
    print(f"  Backscatter /m near:  {_dBm(b['P_back_near']):+.1f} dBm")
    print(f"  Backscatter /m far:   {_dBm(b['P_back_far']):+.1f} dBm")
    print(f"  Ref current per PD:   {b['I_dc_pd']*1e3:.3f} mA")
    print(f"  Filter cutoff:        {b['filter_cutoff']*1e-6:.3f} MHz")
    print(f"  Receiver noise bandwidth: {b['noise_bandwidth']*1e-6:.3f} MHz")
    print("  Noise estimates (input-referred current):")
    print(f"    shot:               {b['sigma_shot']*1e9:.3f} nA")
    print(f"    thermal:            {b['sigma_thermal']*1e9:.3f} nA")
    print(f"    dark:               {b['sigma_dark']*1e9:.3f} nA")
    if math.isnan(b["sigma_rin"]):
        print("    RIN:                unavailable; supply beat_rms_current (A)")
    else:
        print(f"    RIN (beat):         {b['sigma_rin']*1e9:.3f} nA")
    print(f"    receiver additive:  {b['sigma_receiver']*1e9:.3f} nA")
    print(f"    quantization:       {b['sigma_quant']*1e9:.3f} nA")
    print(f"    ADC ENOB extra:     {b['sigma_adc_extra']*1e9:.3f} nA")
    if math.isnan(b["sigma_total"]):
        print("    total / NEP / Dynamic range: unavailable without beat RMS")
    else:
        print(f"    total (sampled):    {b['sigma_total']*1e9:.3f} nA")
        print(f"  NEP (sampled total):  {b['nep_total']*1e9:.2f} nW")
        print(f"  Dynamic range (proxy): {b['dynamic_range_dB']:.1f} dB")
    print(f"  Phase noise (far, theoretical): {b['sigma_phi_far']*1e3:.2f} mrad RMS")
    print(f"  Strain sensitivity:   {b['d_nu_d_eps']*1e-12:.2f} MHz/ustrain")
    print(f"  Temp sensitivity:     {b['d_nu_d_T']*1e-9:.3f} GHz/K")
    print(f"  Max |strain|:         {b['eps_max']*1e6:.0f} ustrain")
