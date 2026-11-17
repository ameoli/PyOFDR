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
NumPy calls keep validation errors and table-range warnings. Native parameters
follow PyRayleigh's domain rules; unsupported Marcuse modes return NaN.
"""

from __future__ import annotations

import warnings

import numpy as np
from pyrayleigh import SpatialProfile
from pyrayleigh.backends import real_arrays
from pyrayleigh.modes import (exponential_bend_loss_db, marcuse_bend_loss_db,
                              solve_lp01_mode, tabulated_bend_loss_db)
from pyrayleigh.units import wavelength_to_frequency


def _scalar_arrays(*values, xp=None):
    xp, values = real_arrays(*values, xp=None if xp is np else xp)
    if any(value.ndim != 0 for value in values):
        raise ValueError("bend parameters must be scalars")
    return xp, values


def lp01_mode(core_radius, core_index, cladding_index, wavelength, *, xp=None):
    """Return (U, W, beta) for the straight step-index fiber.

    Lengths are in metres, beta in rad/m. Both indices are phase indices.
    Only the single-mode, weak-guidance regime is supported here.
    """
    xp, values = _scalar_arrays(core_radius, core_index, cladding_index, wavelength, xp=xp)
    core_radius, core_index, cladding_index, wavelength = values
    if isinstance(wavelength, np.ndarray) and any(not np.isfinite(value) or value <= 0. for value in values):
        raise ValueError("Marcuse fiber parameters must be finite and positive")
    frequency = wavelength_to_frequency(wavelength, xp=xp)
    mode = solve_lp01_mode(frequency, core_radius, core_index, cladding_index, xp=xp)
    if isinstance(mode.U, np.ndarray) and not np.isfinite(mode.U):
        raise ValueError("LP01 parameters or confinement are outside the supported range")
    return mode.U, mode.W, mode.propagation_constant


def marcuse_bend_loss_dB(radius, turns, wavelength, core_radius,
                         core_index, cladding_index, radius_correction=1.28, *, xp=None):
    """One-way loss [dB] from Marcuse's LP01 curvature formula.

    R_eff changes the radiation coefficient, not the physical arc length.
    The returned loss uses 2*pi*radius*turns. See docs/theory.md for the
    power/field convention and the assumptions behind this approximation.
    """
    xp, values = _scalar_arrays(radius, turns, wavelength, core_radius, core_index,
                                cladding_index, radius_correction, xp=xp)
    with np.errstate(over="ignore"):
        loss = marcuse_bend_loss_db(*values, xp=xp)
    if isinstance(loss, np.ndarray) and not np.isfinite(loss):
        raise ValueError("Marcuse parameters or loss are outside the supported range")
    return loss


def bend_loss_dB(radius, turns, A_dB_per_turn=100.0, R_c=5e-3, *, xp=None):
    """Total macrobending loss across a bent section [dB].

    radius, R_c in metres. turns is unitless (can be fractional).
    """
    xp, values = _scalar_arrays(radius, turns, A_dB_per_turn, R_c, xp=xp)
    if isinstance(values[0], np.ndarray) and values[0] <= 0:
        raise ValueError("bend radius must be > 0")
    return exponential_bend_loss_db(*values, xp=xp)


def tabulated_bend_loss_dB(radius, turns, table, *, xp=None):
    """Total loss [dB] interpolated from (radius, dB_per_turn) points.

    Outside the table range the end values are clamped (with a warning),
    no exponential extrapolation into the unknown.
    """
    pts = sorted(table, key=lambda p: p["radius"])
    radii = [p["radius"] for p in pts]
    xp, values = _scalar_arrays(radius, turns, *(p["dB_per_turn"] for p in pts), xp=xp)
    radius, turns = values[:2]
    if isinstance(radius, np.ndarray):
        if radius <= 0:
            raise ValueError("bend radius must be > 0")
        if radius < radii[0] or radius > radii[-1]:
            warnings.warn(f"bend radius {radius:g} m outside table range "
                          f"[{radii[0]:g}, {radii[-1]:g}] m, clamping to the edge")
    return tabulated_bend_loss_db(radius, turns, radii, xp.stack(values[2:]), xp=xp)


def bend_loss_to_positions(z, start, end, total_loss_dB, *, xp=None):
    """Cumulative one-way loss [dB] on a fixed bend interval, clipped at the input."""
    xp, (z, loss) = real_arrays(z, total_loss_dB, xp=None if xp is np else xp)
    length = end - start
    start = max(0., start)
    if end <= start:
        return xp.zeros_like(z)
    rate = loss / length
    profile = SpatialProfile([start, end], xp.stack((rate, rate)))
    return profile.integral(z, xp=xp)


def bend_loss_total_dB(bend, wavelength=None, *, xp=None):
    """Dispatch on bend['model'] (dict straight from the validated config)."""
    model = bend.get("model", "exponential")
    if model == "marcuse":
        if wavelength is None:
            raise ValueError("the Marcuse bend model needs the source wavelength")
        return marcuse_bend_loss_dB(bend["radius"], bend["turns"], wavelength,
                                    **bend["marcuse"], xp=xp)
    if model == "tabulated":
        return tabulated_bend_loss_dB(bend["radius"], bend["turns"], bend["table"], xp=xp)
    if model != "exponential":
        raise ValueError(f"unknown bend model: {model!r}")
    return bend_loss_dB(bend["radius"], bend["turns"],
                        bend["A_dB_per_turn"], bend["R_c"], xp=xp)
