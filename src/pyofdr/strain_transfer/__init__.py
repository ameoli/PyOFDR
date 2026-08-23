"""Strain transfer from host structure to fiber.

The host strain is what the user specifies (e.g. from a structural
load case). The fiber sees a *transferred* strain that depends on
how the cable / coating couples to the host. Two models so far:

  IdealTransfer : eps_fiber == eps_host inside each bonded segment.

  CoxShearLag   : shear-lag profile a la Cox 1952, parameterized by
                  a single shape constant beta [1/m]. Inside a bonded
                  segment of half-length L_h, at distance s from the
                  centre,

                      eps_fiber(s) = eps_host (1 - cosh(beta s)/cosh(beta L_h))

                  so the strain rolls off near the segment edges and
                  approaches eps_host in the middle for long bonds.
                  beta lumps together bond shear modulus, fiber radius,
                  Young modulus etc -- I'm not deriving it from elastic
                  constants here, the user just passes a number.

Both models expose `apply(segments, z, xp=...)` and return an array
shaped like z. Used by the strain perturbation step in fiber/strain.py.

For dynamic (inter-sweep) perturbations, segments may carry a `motion`
field (harmonic / thermal / impulsive for now; see strain_transfer.motions
for the registry). realize_segments collapses such segments to static
ones at a given lab time so transfer.apply keeps working unchanged.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from pyofdr.strain_transfer.motions import check_motion_sampling, evaluate_motion


class StrainTransfer(Protocol):
    def apply(self, segments: list, z: np.ndarray, *, xp=np) -> np.ndarray: ...


def realize_segments(segments: list, t: float) -> list:
    """Freeze any dynamic motion at time t -> plain static segments."""
    out = []
    for seg in segments:
        eps = seg["epsilon"] + evaluate_motion(seg.get("motion"), t)
        # strip the motion field so downstream code doesn't re-apply it
        out.append({**seg, "epsilon": eps, "motion": None})
    return out


class IdealTransfer:
    """Lossless: eps_fiber == eps_host inside each segment, 0 outside."""

    def apply(self, segments, z, *, xp=np):
        eps = xp.zeros_like(z)
        for seg in segments:
            z0  = float(seg["start"])
            z1  = float(seg["end"])
            val = float(seg["epsilon"])
            mask = (z >= z0) & (z <= z1)
            # overlapping segments add up -- physically what you'd expect
            eps = xp.where(mask, eps + val, eps)
        return eps


class CoxShearLag:
    """Cox shear-lag transfer (1D), single shape constant beta [1/m].

    Long bonds (beta * L_h >> 1) -> the centre approaches eps_host and
    only the edges roll off.
    Short bonds (beta * L_h << 1) -> poor coupling, transfer is small
    everywhere.
    """

    def __init__(self, beta: float) -> None:
        if beta <= 0:
            raise ValueError("Cox beta must be positive")
        self.beta = float(beta)

    def apply(self, segments, z, *, xp=np):
        eps = xp.zeros_like(z)
        b = self.beta
        for seg in segments:
            z0  = float(seg["start"])
            z1  = float(seg["end"])
            val = float(seg["epsilon"])
            half   = 0.5 * (z1 - z0)
            centre = 0.5 * (z0 + z1)
            mask = (z >= z0) & (z <= z1)
            # distance from segment centre, clipped to the segment so points
            # far outside don't blow up the exponentials below
            s = xp.minimum(xp.abs(z - centre), half)
            # cosh(b s)/cosh(b half) written with exponents <= 0: the naive
            # ratio overflows for stiff bonds (beta*half > ~710, see #85)
            ratio = (xp.exp(b * (s - half)) + xp.exp(-b * (s + half))) \
                / (1.0 + math.exp(-2.0 * b * half))
            local = val * (1.0 - ratio)
            eps = xp.where(mask, eps + local, eps)
        return eps
