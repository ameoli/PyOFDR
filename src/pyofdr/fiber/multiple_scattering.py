"""Multi-bounce returns from discrete reflectors."""

from __future__ import annotations

import math

import numpy as np
from pyrayleigh import (lossless_reflector, point_path_amplitude, reciprocal_transmission_factors,
                        reflection_paths_from_positions)
from pyrayleigh.backends import real_arrays
from pyrayleigh.units import db_to_linear

from pyofdr.fiber.reflectors import reflector_bin


def ghost_transmission_factors(reflectors, bins, loss_to_z=None, *, forward_transmission=None, xp=None):
    """Input-side insertion factors and reciprocal travel on fixed reflector bins."""
    parameters = () if forward_transmission is None else (forward_transmission,)
    xp, values = real_arrays(0. if loss_to_z is None else loss_to_z,
                             *(value for ref in reflectors for value in
                               (ref["R"], ref.get("loss_dB", 0.))),
                             *parameters,
                             xp=None if xp is np else xp)
    field = values[0]
    if len(bins) != len(reflectors):
        raise ValueError("give one bin per reflector")
    if loss_to_z is not None and field.ndim != 1:
        raise ValueError("loss_to_z must be a vector")
    if forward_transmission is not None:
        forward_transmission = values[-1]
        if forward_transmission.shape != (len(reflectors),):
            raise ValueError("give one forward transmission per reflector")
    zero = xp.asarray(0., dtype=xp.float64)
    losses = xp.stack([zero] + [zero if loss_to_z is None else field[b] for b in bins])
    extra, transmission = [], []
    for i, ref in enumerate(reflectors):
        reflectance, loss = values[1 + 2*i:3 + 2*i]
        factor = db_to_linear(-loss/2., xp=xp)
        extra.append(factor)
        if forward_transmission is None:
            point = lossless_reflector(ref["z"], reflectance, xp=xp)
            transmission.append(point.t_left_to_right*factor)
        else:
            transmission.append(forward_transmission[i])
    extra = xp.stack(extra) if extra else xp.zeros((0,), dtype=xp.float64)
    transmission = xp.stack(transmission) if transmission else xp.zeros((0,), dtype=xp.float64)
    return extra, reciprocal_transmission_factors(losses, transmission, xp=xp)


def add_ghost_reflections(
    profile, dz, reflectors, *, max_order=2, reflection=None,
    loss_to_z=None, forward_transmission=None, xp=np,
):
    """Add ghosts with all path losses already included, in-place.

    Order N has 2N-1 reflections. Positions are snapped before forming
    the apparent bin; ghosts outside the array are dropped.
    loss_to_z is the cumulative one-way distributed loss in dB, without
    connector losses. reflection contains the left-side coefficients,
    with shape (n_cores, n_reflectors).
    forward_transmission optionally supplies real reciprocal one-way
    coefficients, including insertion loss, in the input reflector order.
    """
    if max_order < 2 or len(reflectors) < 2:
        return profile

    n_cores, n_z = profile.shape
    selected = sorted(
        (i for i, r in enumerate(reflectors)
         if 0 <= reflector_bin(r, dz, n_z) < n_z),
        key=lambda i: reflectors[i]["z"],
    )
    refs = [reflectors[i] for i in selected]
    bins = [reflector_bin(r, dz, n_z) for r in refs]
    if len(set(bins)) != len(bins):
        raise ValueError("discrete reflectors must occupy distinct spatial bins")
    if reflection is None:
        reflection = xp.tile(xp.asarray([math.sqrt(r["R"]) for r in refs]),
                             (n_cores, 1))
    else:
        reflection = reflection[:, selected]
    if forward_transmission is not None:
        _, (forward_transmission,) = real_arrays(forward_transmission, xp=None if xp is np else xp)
        if forward_transmission.shape != (len(reflectors),):
            raise ValueError("give one forward transmission per reflector")
        forward_transmission = forward_transmission[selected]

    extra, travel = ghost_transmission_factors(refs, bins, loss_to_z,
                                               forward_transmission=forward_transmission, xp=xp)
    r_left = reflection * extra**2
    r_right = -xp.conj(reflection)

    active = [i for i, r in enumerate(refs) if r["R"] > 0]
    paths = reflection_paths_from_positions([bins[i] for i in active], max_order=max_order)
    for route in paths:
        if len(route) == 1:
            continue
        path = tuple(active[i] for i in route)
        b = sum(bins[idx] if m % 2 == 0 else -bins[idx]
                for m, idx in enumerate(path))
        if not 0 <= b < n_z:
            continue

        amp = point_path_amplitude(path, r_left, r_right, travel, xp=None if xp is np else xp)
        profile[:, b] += amp

    return profile
