"""LP01 mode and Marcuse loss checks, including the OFDR attenuation convention."""

import math

import numpy as np
import pytest
from scipy.special import j0, j1, k0, k1

from helpers import CFG
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.config_models import RootConfig
from pyofdr.fiber.bends import bend_loss_total_dB, lp01_mode, marcuse_bend_loss_dB
from pyofdr.fiber.profile import FiberGenerator


# Martins et al. (Conftele 2009), Table I: a=4.1 um, n_clad=1.440, NA=0.117.
# n_core = sqrt(n_clad**2 + NA**2). These are example waveguide parameters,
# not defaults for every SMF-28 fiber.
FIBER = {"core_radius": 4.1e-6, "core_index": 1.444745306273739,
         "cladding_index": 1.44}

# Independently computed with mpmath at 70 decimal digits. Bisection in
# b=W**2/V**2 (240 iterations) used U*J1(U)*K0(W)-W*K1(W)*J0(U)=0.
# These are numerical reference values, not experimental loss measurements.
MODES = [
    (1310e-9, 1.6193640806061295, 1.6344193265961299, 6918202.195238498),
    (1550e-9, 1.5089422143265536, 1.226518140343568, 5844942.291210868),
    (1625e-9, 1.4759008044820001, 1.1233815646353687, 5574606.425580748),
]
LOSSES = [
    (1310e-9, .005, 1, 1.0, 46.19300056287784),
    (1310e-9, .010, 1, 1.28, .05920915994712783),
    (1310e-9, .020, 2.5, 1.28, .0000026041057348797894),
    (1550e-9, .005, 1, 1.0, 157.99011139659696),
    (1550e-9, .010, 1, 1.28, 3.356204768598211),
    (1550e-9, .020, 2.5, 1.28, .014797705525512475),
    (1625e-9, .005, 1, 1.0, 205.23729191688077),
    (1625e-9, .010, 1, 1.28, 8.210017522028247),
    (1625e-9, .020, 2.5, 1.28, .10227218252576663),
]


def loss(radius=.01, turns=1.0, wavelength=1550e-9, **params):
    return marcuse_bend_loss_dB(radius, turns, wavelength, **{**FIBER, **params})


def bend_config(**params):
    bend = {"start": .2, "end": .4, "radius": .01, "turns": 1,
            "model": "marcuse", "marcuse": {**FIBER, **params}}
    return {**CFG, "fiber": {**CFG["fiber"], "bends": [bend]}}


@pytest.mark.parametrize("wavelength,u,w,beta", MODES)
def test_mode_matches_high_precision_reference(wavelength, u, w, beta):
    actual = lp01_mode(wavelength=wavelength, **FIBER)
    np.testing.assert_allclose(actual, (u, w, beta), rtol=2e-12)


@pytest.mark.parametrize("wavelength,radius,turns,correction,expected", LOSSES)
def test_loss_matches_high_precision_reference(wavelength, radius, turns, correction, expected):
    actual = loss(radius, turns, wavelength, radius_correction=correction)
    assert actual == pytest.approx(expected, rel=2e-10)


@pytest.mark.parametrize("V", [.25, .5, 1.0, 2.0, 2.4048])
def test_lp01_field_and_derivative_are_continuous(V):
    a, n1, n2 = FIBER.values()
    wavelength = 2*math.pi*a*math.sqrt((n1-n2)*(n1+n2)) / V
    u, w, beta = lp01_mode(a, n1, n2, wavelength)
    assert 0 < w < V
    assert math.hypot(u, w) == pytest.approx(V, rel=2e-13)
    # Match fields at r=a, then compare their radial derivatives there.
    cladding_amplitude = j0(u) / k0(w)
    assert u*j1(u) == pytest.approx(cladding_amplitude*w*k1(w), rel=2e-12)
    assert beta >= 2*math.pi*n2/wavelength
    assert beta < 2*math.pi*n1/wavelength


def test_radius_correction_does_not_stretch_the_arc():
    corrected = loss(radius=.01, radius_correction=1.28)
    uncorrected = loss(radius=.0128, radius_correction=1.0)
    assert corrected == pytest.approx(uncorrected/1.28, rel=1e-12)


def test_turns_radius_and_wavelength_dependence():
    assert loss(turns=2.5) == pytest.approx(2.5*loss(), rel=1e-12)
    assert loss(radius=.005) > loss(radius=.01) > loss(radius=.02) > 0
    assert loss(wavelength=1310e-9) < loss(wavelength=1550e-9) < loss(wavelength=1625e-9)
    assert loss(radius=100.0) == 0.0


def test_uniform_length_scaling_leaves_db_per_turn_unchanged():
    for scale in (.001, 1000):
        scaled = loss(radius=.01*scale, wavelength=1550e-9*scale,
                      core_radius=FIBER["core_radius"]*scale)
        assert scaled == pytest.approx(loss(), rel=2e-12)


@pytest.mark.parametrize("bad", [
    {"radius": 0}, {"radius": float("inf")}, {"turns": -1},
    {"turns": float("nan")}, {"wavelength": 0},
    {"core_radius": 0}, {"core_index": 1.43}, {"cladding_index": 1.45},
    {"cladding_index": .9}, {"core_index": float("inf")},
    {"core_index": 1.6}, {"radius_correction": 0},
    {"radius_correction": float("nan")}, {"radius": 4e-6},
    {"radius_correction": 1e-5}, {"core_radius": 8e-6},
    {"wavelength": 1000e-9}, {"turns": 1e308},
])
def test_unsupported_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        loss(**bad)


def test_unresolved_weak_confinement_is_rejected():
    with pytest.raises(ValueError, match="confinement"):
        lp01_mode(wavelength=1e-3, **FIBER)


def test_config_requires_the_marcuse_parameters():
    cfg = bend_config()
    del cfg["fiber"]["bends"][0]["marcuse"]
    with pytest.raises(ValueError, match="needs core_radius"):
        RootConfig(**cfg)


def test_config_accepts_units_and_default_correction():
    cfg = RootConfig(**bend_config(core_radius="4.1 um")).model_dump()
    bend = cfg["fiber"]["bends"][0]
    assert bend["marcuse"]["core_radius"] == pytest.approx(4.1e-6)
    assert bend["marcuse"]["radius_correction"] == 1.28
    assert bend_loss_total_dB(bend, 1550e-9) == pytest.approx(3.356204768598211)
    with pytest.raises(ValueError, match="source wavelength"):
        bend_loss_total_dB(bend)


@pytest.mark.parametrize("params", [{"core_index": 1.43}, {"core_radius": "8 um"},
                                   {"radius_correction": float("inf")}])
def test_invalid_waveguide_is_caught_during_config_validation(params):
    with pytest.raises(ValueError):
        RootConfig(**bend_config(**params))


def test_marcuse_parameters_on_other_models_are_rejected():
    cfg = bend_config()
    cfg["fiber"]["bends"][0]["model"] = "exponential"
    with pytest.raises(ValueError, match="require model='marcuse'"):
        RootConfig(**cfg)


@pytest.mark.parametrize("wavelength", [1310e-9, 1550e-9, 1625e-9])
def test_profile_uses_source_wavelength_and_round_trip_field_loss(wavelength):
    cfg = bend_config()
    cfg["source"] = {**cfg["source"], "center_wavelength": wavelength}
    bare = FiberGenerator({**cfg, "fiber": {**cfg["fiber"], "bends": []}}).process(Acquisition())
    bend = cfg["fiber"]["bends"][0]
    # Align boundaries to the grid so the loss integral needs no edge tolerance.
    bend["start"], bend["end"] = bare.z[1000], bare.z[2000]
    acq = FiberGenerator(cfg).process(Acquisition())
    np.testing.assert_array_equal(acq.fiber_profile, bare.fiber_profile)
    np.testing.assert_array_equal(acq.attenuation_envelope[:1001], 1.0)
    one_way_dB = loss(wavelength=wavelength)
    np.testing.assert_allclose(acq.attenuation_envelope[2000:], 10**(-one_way_dB/10), rtol=1e-12)
    # Squaring the field must give twice the one-way power loss in dB.
    assert -10*np.log10(acq.attenuation_envelope[-1]**2) == pytest.approx(2*one_way_dB)


def test_ghost_counts_repeated_passes_through_analytic_bend():
    cfg = bend_config()
    grid = FiberGenerator(CFG).process(Acquisition())
    bend = cfg["fiber"]["bends"][0]
    bend["start"], bend["end"] = grid.z[1000], grid.z[2000]
    cfg["fiber"].update({"reflectors": [{"z": 0, "R": .04},
                                       {"z": grid.z[3000], "R": .09}],
                         "multiple_scattering": {"max_order": 2}})
    acq = FiberGenerator(cfg).process(Acquisition())
    expected = .96 * .2 * .09 * 10**(-4*loss()/20)
    assert abs(acq.ghost_profile[0, 6000]) == pytest.approx(expected, rel=1e-12)
