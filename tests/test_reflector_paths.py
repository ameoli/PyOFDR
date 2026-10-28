"""Small reflector networks with known complex returns."""

import numpy as np
import pytest

from helpers import CFG
from pyofdr.core.acquisition import Acquisition
from pyofdr.fiber.attenuation import cumulative_loss_dB, round_trip_attenuation_varying
from pyofdr.fiber.crosstalk import apply_crosstalk
from pyofdr.fiber.multiple_scattering import add_ghost_reflections
from pyofdr.fiber.profile import FiberGenerator
from pyofdr.fiber.strain import StrainPerturbation
from pyofdr.fiber.temperature import TemperaturePerturbation
from pyofdr.optics.mach_zehnder import MachZehnder
from pyofdr.source.swept_laser import SweptLaser
from pyofdr.utils.seeding import derive_seed


def fiber_config(refs, **fiber_options):
    return {**CFG, "fiber": {**CFG["fiber"], "reflectors": refs,
            "rayleigh_coefficient_dB": -300,
            "multiple_scattering": {"max_order": 3}, **fiber_options}}


def ghosts(refs, reflection, loss=None, order=2):
    out = np.zeros((1, 100), complex)
    add_ghost_reflections(out, 1.0, refs, reflection=np.array([reflection]),
                          loss_to_z=loss, max_order=order)
    return out[0]


def test_two_reflector_complex_amplitudes():
    a, b = 5, 20
    ra, rb = 0.2 * np.exp(0.7j), 0.3 * np.exp(-1.1j)
    refs = [{"z": a, "R": 0.04}, {"z": b, "R": 0.09}]
    out = ghosts(refs, [ra, rb], order=3)
    expected = np.zeros(100, complex)
    expected[2*b-a] = -(1 - 0.04) * ra.conjugate() * rb**2
    expected[3*b-2*a] = (1 - 0.04) * ra.conjugate()**2 * rb**3
    np.testing.assert_allclose(out, expected, atol=1e-16)


@pytest.mark.parametrize("loss_bin, passes", [(2, 2), (12, 4), (25, 0)])
def test_localized_loss_counts_real_crossings(loss_bin, passes):
    refs = [{"z": 5, "R": 0.04}, {"z": 20, "R": 0.09}]
    # A 3 dB step before, inside, or beyond the cavity.
    loss = np.where(np.arange(100) > loss_bin, 3.0, 0.0)
    actual = ghosts(refs, [0.2, 0.3], loss=loss)[35]
    expected = -0.96 * 0.2 * 0.3**2 * 10**(-passes * 3 / 20)
    np.testing.assert_allclose(actual, expected, atol=1e-16)


def test_uniform_loss_matches_apparent_distance():
    refs = [{"z": 5, "R": 0.04}, {"z": 20, "R": 0.09}]
    out = ghosts(refs, [0.2, 0.3], loss=0.1 * np.arange(100))
    np.testing.assert_allclose(out[35], -0.96 * 0.2 * 0.3**2 * 10**(-7 / 20))


@pytest.mark.parametrize("where, R, loss, passes", [
    (0, 0.16, 1.0, 2), (12, 0.16, 1.0, 4), (25, 1.0, 4000.0, 0),
    (12, 0.0, 3.0, 4), (12, 1.0, 0.0, 4),
])
def test_intermediate_interfaces(where, R, loss, passes):
    refs = [{"z": 5, "R": 0.04}, {"z": 20, "R": 0.09},
            {"z": where, "R": R, "loss_dB": loss}]
    out = ghosts(refs, [0.2, 0.3, np.sqrt(R)])
    t = np.sqrt(1 - R) * 10**(-loss / 20)
    expected = -0.96 * 0.2 * 0.3**2 * t**passes
    np.testing.assert_allclose(out[35], expected, atol=1e-16)
    assert np.all(np.isfinite(out))


def test_loss_at_bouncing_interfaces():
    refs = [{"z": 5, "R": 0.04, "loss_dB": 1.0},
            {"z": 20, "R": 0.09, "loss_dB": 2.0}]
    # Twice through the loss before a, four times through that before b.
    expected = -0.96 * 0.2 * 0.3**2 * 10**(-(2*1 + 4*2) / 20)
    np.testing.assert_allclose(ghosts(refs, [0.2, 0.3])[35], expected)


@pytest.mark.parametrize("R_a, R_b", [(0.0, 0.09), (1.0, 0.09), (0.04, 0.0)])
def test_no_accessible_cavity_gives_no_ghosts(R_a, R_b):
    refs = [{"z": 0, "R": R_a}, {"z": 20, "R": R_b}]
    out = ghosts(refs, [np.sqrt(R_a), np.sqrt(R_b)], order=3)
    np.testing.assert_array_equal(out, 0)


def test_seeded_phases_and_direct_transmission():
    refs = [{"z": 0.1, "R": 0.04}, {"z": 0.3, "R": 0.09}]
    cfg = fiber_config(refs, n_cores=2)
    generator = FiberGenerator(cfg)
    first = generator.process(Acquisition())
    again = generator.process(Acquisition(sweep_index=1))
    repeat = FiberGenerator(cfg).process(Acquisition())
    changed = FiberGenerator({**cfg, "simulation": {"seed": 43}}).process(Acquisition())
    for acq in (again, repeat):
        np.testing.assert_array_equal(first.fiber_profile, acq.fiber_profile)
        np.testing.assert_array_equal(first.ghost_profile, acq.ghost_profile)
    a, b = [round(r["z"] / first.dz) for r in refs]
    assert not np.allclose(first.fiber_profile[:, [a, b]].imag, 0)
    assert not np.allclose(first.fiber_profile[0, [a, b]], first.fiber_profile[1, [a, b]])
    assert not np.allclose(first.fiber_profile[:, [a, b]], changed.fiber_profile[:, [a, b]])
    np.testing.assert_allclose(np.abs(first.fiber_profile[:, b]) * first.attenuation_envelope[b],
                               0.96 * 0.3)
    ra, rb = first.fiber_profile[:, a], first.fiber_profile[:, b]
    np.testing.assert_allclose(first.ghost_profile[:, 2*b-a], -0.96 * ra.conj() * rb**2)
    reversed_cfg = fiber_config(list(reversed(refs)), n_cores=2)
    reversed_acq = FiberGenerator(reversed_cfg).process(Acquisition())
    np.testing.assert_array_equal(first.fiber_profile, reversed_acq.fiber_profile)
    np.testing.assert_array_equal(first.ghost_profile, reversed_acq.ghost_profile)
    bare = FiberGenerator(fiber_config([], n_cores=2)).process(Acquisition())
    mask = np.ones(len(first.z), bool)
    mask[[a, b]] = False
    np.testing.assert_array_equal(first.fiber_profile[:, mask], bare.fiber_profile[:, mask])


@pytest.mark.parametrize("kind", ["segment", "bend"])
@pytest.mark.parametrize("start,end,passes", [(0.02, 0.06, 2), (0.15, 0.20, 4),
                                             (0.35, 0.40, 0)])
def test_generated_ghost_uses_segment_and_bend_loss(kind, start, end, passes):
    refs = [{"z": 0.1, "R": 0.04}, {"z": 0.3, "R": 0.09}]
    options = {"multiple_scattering": {"max_order": 2}}
    if kind == "segment":
        options["attenuation_segments"] = [{"start": start, "end": end,
                                            "attenuation_dB_per_km": 1000.0}]
        loss_per_m = 1.0
    else:
        options["bends"] = [{"start": start, "end": end, "radius": 0.005,
                             "turns": 1, "A_dB_per_turn": np.e, "R_c": 0.005}]
        loss_per_m = 1.0 / (end - start)  # 1 dB in total
    acq = FiberGenerator(fiber_config(refs, **options)).process(Acquisition())
    a, b = [round(r["z"] / acq.dz) for r in refs]
    covered_length = np.count_nonzero((acq.z >= start) & (acq.z < end)) * acq.dz
    expected = 0.96 * 0.2 * 0.09 * 10**(-passes * loss_per_m * covered_length / 20)
    np.testing.assert_allclose(abs(acq.ghost_profile[0, 2*b-a]), expected, rtol=1e-12)


def test_index_phase_and_crosstalk_also_apply_to_ghosts():
    refs = [{"z": 0.0, "R": 0.04}, {"z": 0.2, "R": 0.09}]
    cfg = fiber_config(refs, n_cores=2)
    bare = FiberGenerator(cfg).process(Acquisition())
    index = {"start": 0.1, "end": 0.5, "delta_n": 1e-7}
    xt = {"xt_dB_per_km": -20.0, "topology": "linear"}
    cfg["fiber"] = {**cfg["fiber"], "index_segments": [index], "crosstalk": xt}
    actual = FiberGenerator(cfg).process(Acquisition())
    dn = np.where((bare.z >= 0.1) & (bare.z < 0.5), 1e-7, 0)
    phase = 4 * np.pi / cfg["source"]["center_wavelength"] * np.cumsum(dn) * bare.dz
    rng = np.random.default_rng(derive_seed(42, component="crosstalk"))
    expected = apply_crosstalk(bare.ghost_profile * np.exp(1j * phase), bare.z,
                               -20.0, "linear", rng)
    np.testing.assert_allclose(actual.ghost_profile, expected, atol=1e-15)


def test_full_mirror_blocks_direct_returns_but_not_earlier_cavity():
    refs = [{"z": 0.0, "R": 0.04}, {"z": 0.2, "R": 0.09},
            {"z": 0.317, "R": 1.0}]
    cfg = fiber_config(refs, multiple_scattering={"max_order": 2})
    acq = FiberGenerator(cfg).process(Acquisition())
    b = round(0.2 / acq.dz)
    mirror = round(0.317 / acq.dz)
    np.testing.assert_array_equal(acq.attenuation_envelope[mirror + 1:], 0)
    np.testing.assert_allclose(acq.attenuation_envelope[mirror], 0.96 * 0.91)
    np.testing.assert_allclose(abs(acq.ghost_profile[0, 2*b]), 0.96 * 0.2 * 0.09)


def test_loss_integral_starts_at_input():
    alpha = np.array([1000., 2000., 0., 4000.])
    loss = cumulative_loss_dB(alpha, 1.0)
    np.testing.assert_array_equal(loss, [0., 1., 3., 3.])
    envelope = round_trip_attenuation_varying(np.arange(4), alpha, 1.0)
    np.testing.assert_allclose(envelope, 10**(-loss / 10))


def test_unresolved_reflectors_are_rejected():
    refs = [{"z": 0.1, "R": 0.04}, {"z": 0.1, "R": 0.09}]
    with pytest.raises(ValueError, match="distinct spatial bins"):
        FiberGenerator(fiber_config(refs)).process(Acquisition())


def test_frequency_domain_cavity_reference():
    # Sum the infinite reflection series recursively in frequency space.
    # Weak mirrors make the omitted orders (>5 bounces) smaller than 1e-10.
    bins = [5, 13, 27]
    R = np.array([0.0004, 0.0009, 0.0006])
    r = np.sqrt(R) * np.exp(1j * np.array([0.7, -1.1, 2.0]))
    extra_dB = np.array([0.2, 0.7, 0.4])
    refs = [{"z": b, "R": power, "loss_dB": loss}
            for b, power, loss in zip(bins, R, extra_dB)]
    loss = np.cumsum(np.where(np.arange(100) < 18, 0.01, 0.08))
    loss -= loss[0]
    theta = 2 * np.pi * np.arange(512) / 512
    response = np.zeros(512, complex)
    for i in range(2, -1, -1):
        if i < 2:
            response *= np.exp(1j * theta * (bins[i+1] - bins[i]))
            response *= 10**(-(loss[bins[i+1]] - loss[bins[i]]) / 10)
        q2 = 10**(-extra_dB[i] / 10)
        response = q2 * (r[i] + (1 - R[i]) * response / (1 + r[i].conjugate() * response))
    response *= np.exp(1j * theta * bins[0]) * 10**(-loss[bins[0]] / 10)
    expected = np.fft.fft(response)[:100] / 512
    actual = ghosts(refs, r, loss=loss, order=3)
    upstream = 1.0
    for b, ri, power, extra in zip(bins, r, R, extra_dB):
        actual[b] += upstream * ri * 10**(-(loss[b] + extra) / 10)
        upstream *= (1 - power) * 10**(-extra / 10)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("dynamic", [False, True])
def test_perturbations_preserve_ghosts_across_sweeps(dynamic):
    cfg = fiber_config([{"z": 0.0, "R": 0.04}, {"z": 0.2, "R": 0.09}])
    segment = {"start": 0.1, "end": 0.5, "epsilon": 1e-6}
    if dynamic:
        segment["motion"] = {"kind": "harmonic", "amplitude": 1e-6, "frequency": 3.0}
    cfg["strain"] = {"segments": [segment]}
    cfg["temperature"] = {"segments": [{"start": 0.1, "end": 0.5, "delta_T": 0.2}]}
    fiber, strain, temp = FiberGenerator(cfg), StrainPerturbation(cfg), TemperaturePerturbation(cfg)
    base = fiber.process(Acquisition())
    ghost_bin = 2 * round(0.2 / base.dz)
    for sweep in range(3):
        acq = temp.process(strain.process(fiber.process(Acquisition(sweep_index=sweep))))
        phase = acq.fiber_profile[0, ghost_bin] / base.fiber_profile[0, ghost_bin]
        np.testing.assert_allclose(acq.ghost_profile[0, ghost_bin],
                                   base.ghost_profile[0, ghost_bin] * phase)


def test_optics_adds_ghosts_without_another_attenuation():
    cfg = fiber_config([{"z": 0.0, "R": 0.04}, {"z": 0.2, "R": 0.09},
                        {"z": 0.3, "R": 1.0}])
    cfg["source"] = {**cfg["source"], "sweep_duration": 1e-4,
                     "sweep_range": 1e-9, "linewidth": 0.0}
    cfg["adc"] = {**cfg["adc"], "sample_rate": 200e6}
    cfg["optics"] = {**cfg["optics"], "circulator": {
        **cfg["optics"]["circulator"], "insertion_loss_dB": 0.0}}
    acq = SweptLaser(cfg).process(FiberGenerator(cfg).process(Acquisition()))
    weighted = acq.fiber_profile * acq.attenuation_envelope + acq.ghost_profile
    expected = np.fft.ifft(weighted, n=acq.n_samples, axis=-1) * acq.n_samples
    expected = np.abs(acq.E_source)**2 * expected.real
    actual = MachZehnder(cfg).process(acq).photocurrent_main
    np.testing.assert_allclose(actual, expected, atol=1e-16)
