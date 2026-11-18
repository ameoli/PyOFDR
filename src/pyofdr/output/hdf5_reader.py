"""Read prescribed PyRayleigh states from a simulation output."""

from dataclasses import fields
from pathlib import Path

import h5py
from pyrayleigh import FiberState, Field, FieldTerm, SpatialProfile, TimeHistory
from pyrayleigh.backends import real_arrays


def _flag(group, name):
    value = group.attrs[name]
    if value not in (True, False):
        raise ValueError(f"invalid {name} flag at {group.name}")
    return bool(value)


def _numbered(group):
    names = sorted(group, key=int)
    if names != [f"{index:04d}" for index in range(len(names))]:
        raise ValueError(f"nonconsecutive indices at {group.name}")
    return names


def _read_state(group, xp):
    if _flag(group, "reference"):
        if len(group):
            raise ValueError("a reference state must not contain field data")
        return None
    names = [item.name for item in fields(FiberState)]
    if set(group) != set(names):
        raise ValueError("saved state fields do not match FiberState")
    components = {}
    for name in names:
        terms = []
        for key in _numbered(group[name]):
            node = group[name][key]
            _, (values,) = real_arrays(node["values"][:], xp=xp)
            profile = SpatialProfile(node["positions"][:], values)
            history = None
            if "history" in node:
                _, (values,) = real_arrays(node["history/values"][:], xp=xp)
                history = TimeHistory(node["history/times"][:], values)
            terms.append(FieldTerm(profile, history))
        components[name] = Field(terms)
    return FiberState(**components)


def load_fiber_state(path: str | Path, *, xp=None) -> FiberState | tuple[FiberState | None, ...] | None:
    """Load the first sweep's prescribed state, optionally in a native namespace.

    A tuple follows core order and retains shared state identities. None means
    an explicitly stored reference state. The fiber model is supplied separately.
    """
    with h5py.File(path, "r") as source:
        if "fiber/state" not in source:
            raise ValueError("file has no saved PyRayleigh state")
        group = source["fiber/state"]
        if not _flag(group, "per_core"):
            return _read_state(group, xp)
        names = _numbered(group)
        if not names or len(names) != source["fiber"].attrs["n_cores"]:
            raise ValueError("saved state count does not match the fiber core count")
        states, saved = [], []
        for name in names:
            node = group[name]
            for previous, state in saved:
                if node.id == previous.id:
                    break
            else:
                state = _read_state(node, xp)
                saved.append((node, state))
            states.append(state)
        return tuple(states)
