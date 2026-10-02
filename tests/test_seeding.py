"""Seed collisions and reproducibility (#83)."""

import numpy as np
import pytest

from pyofdr.utils.seeding import derive_seed


@pytest.mark.parametrize("a, b", [
    ({"component": "laser", "sweep": 1000}, {"component": "detector"}),
    ({"component": "detector", "sweep": 1000}, {"component": "adc"}),
    ({"component": "laser", "sweep": 100_000}, {"component": "laser", "sub": 1}),
    ({"component": "adc", "sweep": 200_000}, {"component": "adc", "sub": 2}),
    ({"component": "detector", "sweep": 1_000_000},
     {"component": "detector", "core": 1}),
    ({"component": "adc", "sub": 10}, {"component": "adc", "core": 1}),
])
def test_old_collisions(a, b):
    seed_a = derive_seed(42, **a)
    seed_b = derive_seed(42, **b)
    assert seed_a != seed_b
    a = np.random.default_rng(seed_a).standard_normal(32)
    b = np.random.default_rng(seed_b).standard_normal(32)
    assert not np.array_equal(a, b)


def test_neighboring_campaign_seeds_do_not_reuse_sweeps():
    a = derive_seed(42, component="laser", sweep=1)
    b = derive_seed(43, component="laser", sweep=0)
    assert a != b


def test_long_campaign_seeds_are_distinct():
    components = ["fiber", "laser", "detector", "adc", "crosstalk",
                  "index_fluctuations"]
    seeds = set()
    for component in components:
        for core in range(2):
            for sub in range(3):
                for sweep in range(1100):
                    seed = derive_seed(42, component=component, core=core,
                                       sweep=sweep, sub=sub)
                    assert seed not in seeds
                    seeds.add(seed)


def test_call_order_does_not_change_seed():
    a = derive_seed(42, component="adc", core=2, sweep=1001, sub=1)
    derive_seed(43, component="laser", sweep=50)
    b = derive_seed(42, component="adc", core=2, sweep=1001, sub=1)
    assert isinstance(a, int)
    assert a == b
    np.testing.assert_array_equal(np.random.default_rng(a).standard_normal(32),
                                  np.random.default_rng(b).standard_normal(32))


def test_seed_scheme_is_stable():
    # Fixed value: changing the encoding would change every simulation.
    seed = derive_seed(42, component="adc", core=2, sweep=1001, sub=1)
    assert seed == 0x3f4f9a1147e97b4e040cee161d41317df1e1b9eea0f5e708bdf00b408c2dfb0d


def test_large_indices_keep_their_boundaries():
    # large counters must not spill into the next field
    a = derive_seed(2**128 + 42, component="adc", core=2**32, sweep=0, sub=1)
    b = derive_seed(2**128 + 42, component="adc", core=0, sweep=1, sub=2**32)
    assert a != b


@pytest.mark.parametrize("field", ["base", "core", "sweep", "sub"])
def test_negative_values_rejected(field):
    args = {"base": 42, "component": "laser", "core": 0, "sweep": 0, "sub": 0}
    args[field] = -1
    with pytest.raises(ValueError, match="non-negative"):
        derive_seed(**args)


@pytest.mark.parametrize("field", ["base", "core", "sweep", "sub"])
def test_fractional_values_rejected(field):
    args = {"base": 42, "component": "laser", "core": 0, "sweep": 0, "sub": 0}
    args[field] = 1.5
    with pytest.raises(TypeError):
        derive_seed(**args)


def test_numpy_integers_work():
    a = derive_seed(np.int64(42), component="laser", sweep=np.int64(1001))
    b = derive_seed(42, component="laser", sweep=1001)
    assert a == b


def test_unknown_component_rejected():
    with pytest.raises(KeyError):
        derive_seed(42, component="lasre")
