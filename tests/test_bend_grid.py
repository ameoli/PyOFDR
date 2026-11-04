"""Bend loss integrated over physical intervals, including partial grid cells."""

import math

import numpy as np
import pytest

from helpers import CFG
from pyofdr.core.acquisition import Acquisition
from pyofdr.fiber.profile import FiberGenerator


MODELS = [
    ({"model": "exponential", "A_dB_per_turn": math.e, "R_c": .01}, 1.0),
    ({"model": "tabulated", "table": [{"radius": .005, "dB_per_turn": 5},
                                       {"radius": .01, "dB_per_turn": 2.5}]}, 2.5),
    ({"model": "marcuse", "marcuse": {"core_radius": 4.1e-6,
                                       "core_index": 1.444745306273739,
                                       "cladding_index": 1.44}}, 3.356204768598211),
]


def config(bends=(), **fiber):
    return {**CFG, "fiber": {**CFG["fiber"], "bends": list(bends), **fiber}}


def bend(start, end, params, turns=1):
    return {"start": start, "end": end, "radius": .01, "turns": turns, **params}


@pytest.mark.parametrize("params,total_dB", MODELS)
@pytest.mark.parametrize("start,end", [(100.1, 100.5), (100.8, 101.2),
                                      (100, 101.5), (100, 103), (0, .4), (.2, 2.7)])
def test_loss_before_inside_and_after_bend(params, total_dB, start, end):
    bare = FiberGenerator(CFG).process(Acquisition())
    start, end = start*bare.dz, end*bare.dz
    generator = FiberGenerator(config([bend(start, end, params)]))
    acq = generator.process(Acquisition())
    loss = -10*np.log10(acq.attenuation_envelope)
    np.testing.assert_allclose(loss[acq.z <= start], 0, atol=1e-12)
    np.testing.assert_allclose(loss[acq.z >= end], total_dB, rtol=1e-12)
    inside = (acq.z > start) & (acq.z < end)
    expected = total_dB*(acq.z[inside]-start)/(end-start)
    np.testing.assert_allclose(loss[inside], expected, rtol=1e-12)
    np.testing.assert_array_equal(acq.fiber_profile, bare.fiber_profile)
    assert np.all(np.diff(acq.attenuation_envelope) <= 0)
    cached = generator.process(Acquisition(sweep_index=1))
    np.testing.assert_array_equal(cached.attenuation_envelope, acq.attenuation_envelope)


@pytest.mark.parametrize("params,total_dB", MODELS)
@pytest.mark.parametrize("sweep_range", [20e-9, 40e-9, 80e-9])
def test_same_physical_bend_has_same_total_loss_on_different_grids(params, total_dB, sweep_range):
    cfg = config([bend(.00205, .002059, params)])
    cfg["source"] = {**cfg["source"], "sweep_range": sweep_range}
    acq = FiberGenerator(cfg).process(Acquisition())
    np.testing.assert_allclose(acq.attenuation_envelope[acq.z >= .002059],
                               10**(-total_dB/10), rtol=1e-12)


def test_overlapping_bends_add_to_existing_attenuation():
    base = FiberGenerator(CFG).process(Acquisition())
    dz = base.dz
    options = {"attenuation_dB_per_km": 100,
               "attenuation_segments": [{"start": base.z[20], "end": base.z[30],
                                          "attenuation_dB_per_km": 300}]}
    bare = FiberGenerator(config(**options)).process(Acquisition())
    params, _ = MODELS[0]
    acq = FiberGenerator(config([bend(25.5*dz, 27.5*dz, params),
                                 bend(26.5*dz, 28.5*dz, params, turns=2)],
                                **options)).process(Acquisition())
    added_loss = -10*np.log10(acq.attenuation_envelope/bare.attenuation_envelope)
    # At bins 26, 27, 28: first bend contributes .25, .75, 1 dB;
    # the second contributes 0, .5, 1.5 dB.
    np.testing.assert_allclose(added_loss[[26, 27, 28]], [.25, 1.25, 2.5], atol=1e-12)
    np.testing.assert_allclose(added_loss[29:], 3, atol=1e-12)


@pytest.mark.parametrize("width_bins", [.1, 2.7])
def test_bend_at_fiber_end_only_attenuates_the_path_to_each_sample(width_bins):
    bare = FiberGenerator(CFG).process(Acquisition())
    end = CFG["fiber"]["length"]
    start = end - width_bins*bare.dz
    params, _ = MODELS[0]
    acq = FiberGenerator(config([bend(start, end, params)])).process(Acquisition())
    expected_loss = [0 if z <= start else (z-start)/(end-start) for z in acq.z]
    np.testing.assert_allclose(acq.attenuation_envelope, 10**(-np.array(expected_loss)/10),
                               rtol=1e-12)


def test_integration_starts_at_fiber_input():
    dz = FiberGenerator(CFG).process(Acquisition()).dz
    params, _ = MODELS[0]
    acq = FiberGenerator(config([bend(-.5*dz, .5*dz, params)])).process(Acquisition())
    # Bounds before z=0 are currently accepted, but that portion is not traversed.
    assert acq.attenuation_envelope[0] == 1
    np.testing.assert_allclose(acq.attenuation_envelope[1:], 10**(-.5/10), rtol=1e-12)


@pytest.mark.parametrize("params,total_dB", MODELS)
@pytest.mark.parametrize("start,end,passes", [(2.1, 2.5, 2), (12.2, 12.6, 4),
                                             (25.1, 25.5, 0), (4.6, 5.4, 3),
                                             (19.6, 20.4, 2)])
def test_ghost_loss_uses_full_and_partial_bend_crossings(params, total_dB, start, end, passes):
    dz = FiberGenerator(CFG).process(Acquisition()).dz
    options = {"reflectors": [{"z": 5*dz, "R": .04}, {"z": 20*dz, "R": .09}],
               "multiple_scattering": {"max_order": 2}}
    bare = FiberGenerator(config(**options)).process(Acquisition())
    acq = FiberGenerator(config([bend(start*dz, end*dz, params)], **options)).process(Acquisition())
    ratio = acq.ghost_profile[0, 35] / bare.ghost_profile[0, 35]
    assert ratio == pytest.approx(10**(-passes*total_dB/20), rel=1e-12, abs=1e-14)
