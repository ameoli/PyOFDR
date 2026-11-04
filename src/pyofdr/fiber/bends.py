"""Macrobending loss models.

Three ways to get the loss of a bent section (picked per-bend via the
`model` field in the YAML, see #90):

  exponential (default) -- Marcuse-style empirical form for single-mode
  fiber at 1550 nm:

      alpha_per_turn(R) = A * exp(-R / R_c)       [dB/turn]

  A and R_c are ballparks for SMF-28 (A ~ 100 dB/turn, R_c ~ 5 mm).
  Tweak them in the YAML if you're simulating a different fiber.

  tabulated -- interpolate a list of (radius, dB_per_turn) points, the
  format datasheets and lab measurements come in. Interpolation is
  linear in log(loss) vs R, since the loss is exponential in radius --
  linear interp between two datasheet points would be off by orders of
  magnitude in between.

  marcuse -- LP01 radiation loss for a weakly guiding step-index fiber,
  evaluated at the source centre wavelength. Core geometry and phase
  indices must be supplied. The effective-radius correction defaults to
  1.28 for silica; use 1.0 for the uncorrected formula.

No transition loss, coating reflections or bend-induced birefringence (#57).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, k0, k1


_LP11_CUTOFF = 2.404825557695773


def lp01_mode(core_radius, core_index, cladding_index, wavelength):
    """Return (U, W, beta) for the straight step-index fiber.

    Lengths are in metres, beta in rad/m. Both indices are phase indices.
    Only the single-mode, weak-guidance regime is supported here.
    """
    if any(not math.isfinite(x) or x <= 0 for x in
           (core_radius, core_index, cladding_index, wavelength)):
        raise ValueError("Marcuse fiber parameters must be finite and positive")
    if not core_index > cladding_index >= 1.0:
        raise ValueError("Marcuse needs core_index > cladding_index >= 1")
    index_difference = (core_index - cladding_index) * (core_index + cladding_index)
    delta = 0.5 * ((core_index - cladding_index) / core_index) * (1.0 + cladding_index / core_index)
    if delta > 0.01:
        raise ValueError("Marcuse weak-guidance model requires delta <= 0.01")
    wave_number = 2.0 * math.pi / wavelength
    V = wave_number * core_radius * math.sqrt(index_difference)
    if not math.isfinite(V) or not 0 < V < _LP11_CUTOFF:
        raise ValueError("Marcuse single-mode model requires 0 < V < 2.4048255577")

    def residual(log_w):
        w = math.exp(log_w)
        u = math.sqrt(max(0.0, (V - w) * (V + w)))
        return u * j1(u) / j0(u) - w * k1(w) / k0(w)

    # Solving in log(W) also resolves weakly confined modes with W << V.
    lower = -300.0
    if math.log(V) <= lower or residual(lower) <= 0:
        raise ValueError("LP01 confinement is too weak for this numerical solver")
    log_w = brentq(residual, lower, math.log(V), xtol=1e-12)
    w = math.exp(log_w)
    u = math.sqrt((V - w) * (V + w))
    beta = math.hypot(wave_number * cladding_index, w / core_radius)
    return u, w, beta


def marcuse_bend_loss_dB(radius, turns, wavelength, core_radius,
                         core_index, cladding_index, radius_correction=1.28):
    """One-way loss [dB] from Marcuse's LP01 curvature formula.

    R_eff changes the radiation coefficient, not the physical arc length.
    The returned loss uses 2*pi*radius*turns. See docs/theory.md for the
    power/field convention and the assumptions behind this approximation.
    """
    if any(not math.isfinite(x) or x <= 0 for x in
           (radius, turns, radius_correction)):
        raise ValueError("Marcuse radius, turns and correction must be finite and positive")
    u, w, beta = lp01_mode(core_radius, core_index, cladding_index, wavelength)
    if radius <= core_radius:
        raise ValueError("Marcuse bend radius must exceed the core radius")
    effective_radius = radius * radius_correction
    if not math.isfinite(effective_radius) or effective_radius <= core_radius:
        raise ValueError("Marcuse effective radius must be finite and exceed the core radius")
    V = math.hypot(u, w)
    kappa = u / core_radius
    gamma = w / core_radius

    # This is 2*alpha, the power decay coefficient [1/m], evaluated in log form.
    exponent = (2.0 / 3.0) * (gamma / beta)**2 * gamma * effective_radius
    log_rate = (0.5 * math.log(math.pi) + 2.0 * math.log(kappa)
                - math.log(2.0) - 1.5 * math.log(gamma) - 2.0 * math.log(V)
                - 0.5 * math.log(effective_radius) - 2.0 * math.log(k1(w))
                - exponent)
    log_loss = (log_rate + math.log(10.0 / math.log(10.0))
                + math.log(2.0 * math.pi) + math.log(radius) + math.log(turns))
    try:
        loss = math.exp(log_loss)
    except OverflowError as exc:
        raise ValueError("Marcuse loss is too large to represent") from exc
    if not math.isfinite(loss):
        raise ValueError("Marcuse loss is outside the numerical range")
    return loss


def bend_loss_dB(radius, turns, A_dB_per_turn=100.0, R_c=5e-3):
    """Total macrobending loss across a bent section [dB].

    radius, R_c in metres. turns is unitless (can be fractional).
    """
    if radius <= 0:
        raise ValueError("bend radius must be > 0")
    per_turn = A_dB_per_turn * math.exp(-radius / R_c)
    return per_turn * turns


def tabulated_bend_loss_dB(radius, turns, table):
    """Total loss [dB] interpolated from (radius, dB_per_turn) points.

    Outside the table range the end values are clamped (with a warning),
    no exponential extrapolation into the unknown.
    """
    if radius <= 0:
        raise ValueError("bend radius must be > 0")
    pts = sorted(table, key=lambda p: p["radius"])
    radii   = np.array([p["radius"] for p in pts])
    logloss = np.log([p["dB_per_turn"] for p in pts])
    if radius < radii[0] or radius > radii[-1]:
        warnings.warn(f"bend radius {radius:g} m outside table range "
                      f"[{radii[0]:g}, {radii[-1]:g}] m, clamping to the edge")
    per_turn = float(np.exp(np.interp(radius, radii, logloss)))
    return per_turn * turns


def bend_loss_total_dB(bend, wavelength=None):
    """Dispatch on bend['model'] (dict straight from the validated config)."""
    model = bend.get("model", "exponential")
    if model == "marcuse":
        if wavelength is None:
            raise ValueError("the Marcuse bend model needs the source wavelength")
        return marcuse_bend_loss_dB(bend["radius"], bend["turns"], wavelength,
                                    **bend["marcuse"])
    if model == "tabulated":
        return tabulated_bend_loss_dB(bend["radius"], bend["turns"], bend["table"])
    if model != "exponential":
        raise ValueError(f"unknown bend model: {model!r}")
    return bend_loss_dB(bend["radius"], bend["turns"],
                        bend["A_dB_per_turn"], bend["R_c"])
