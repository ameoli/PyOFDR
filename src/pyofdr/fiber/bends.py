"""Macrobending loss model.

Marcuse-style empirical form for single-mode fiber at 1550 nm:

    alpha_per_turn(R) = A * exp(-R / R_c)       [dB/turn]

A and R_c are ballparks for SMF-28 (A ~ 100 dB/turn, R_c ~ 5 mm).
Tweak them in the YAML if you're simulating a different fiber.

Intentionally simple: no wavelength dependence, no transition loss,
no whispering-gallery ripples, no bend-induced birefringence. See
the follow-up issue for those.
"""

from __future__ import annotations

import math


def bend_loss_dB(radius, turns, A_dB_per_turn=100.0, R_c=5e-3):
    """Total macrobending loss across a bent section [dB].

    radius, R_c in metres. turns is unitless (can be fractional).
    """
    if radius <= 0:
        raise ValueError("bend radius must be > 0")
    per_turn = A_dB_per_turn * math.exp(-radius / R_c)
    return per_turn * turns
