"""Macrobending loss models.

Two ways to get the loss of a bent section (picked per-bend via the
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

Intentionally still no wavelength dependence, no transition loss,
no whispering-gallery ripples, no bend-induced birefringence. See
#57 for those, and #90 for the analytic Marcuse model (TODO).
"""

from __future__ import annotations

import math
import warnings

import numpy as np


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


def bend_loss_total_dB(bend):
    """Dispatch on bend['model'] (dict straight from the validated config)."""
    if bend.get("model", "exponential") == "tabulated":
        return tabulated_bend_loss_dB(bend["radius"], bend["turns"], bend["table"])
    return bend_loss_dB(bend["radius"], bend["turns"],
                        bend["A_dB_per_turn"], bend["R_c"])
