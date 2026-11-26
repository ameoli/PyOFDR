"""Read recorded sweeps and prescribed states from a simulation output."""

import json
from dataclasses import fields
from numbers import Integral
from pathlib import Path

import h5py
import numpy as np
from pyrayleigh import (DiagnosticResponse, FiberState, Field, FieldTerm, MultiCoreResponse, Probe,
                       Response, SpatialProfile, TimeHistory)
from pyrayleigh.backends import real_arrays

from pyofdr.core.acquisition import Acquisition


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


def _read_states(group, xp):
    if not _flag(group, "per_core"):
        return _read_state(group, xp)
    names = _numbered(group)
    if not names or len(names) != group.parent.attrs["n_cores"]:
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


def load_fiber_state(path: str | Path, *, xp=None) -> FiberState | tuple[FiberState | None, ...] | None:
    """Load the first sweep's prescribed state, optionally in a native namespace.

    A tuple follows core order and retains shared state identities. None means
    an explicitly stored reference state. The fiber model is supplied separately.
    """
    with h5py.File(path, "r") as source:
        if "fiber/state" not in source:
            raise ValueError("file has no saved PyRayleigh state")
        return _read_states(source["fiber/state"], xp)


def load_sweep(path: str | Path, sweep_index=0) -> Acquisition:
    """Load recorded NumPy data, retaining the precision stored in the file.

    Nominal sample times come from the saved clock. Fields absent from the
    requested sweep stay unset. The fiber model and laser field are external.
    Legacy field snapshots from fiber metadata apply only to sweep zero.
    Saved total/direct fields reconstruct diagnostics without solving the fiber.
    Diagnostic metadata uses JSON, so sequences are read back as lists.
    """
    if isinstance(sweep_index, bool) or not isinstance(sweep_index, Integral) or sweep_index < 0:
        raise ValueError("sweep_index must be a nonnegative integer")
    with h5py.File(path, "r") as source:
        group = source[f"sweeps/{sweep_index:04d}"]
        count = group.attrs["n_samples"]
        dt = float(group.attrs["dt"])
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 1:
            raise ValueError("invalid saved sample count")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("invalid saved sample interval")
        acq = Acquisition(sweep_index=int(sweep_index), n_samples=int(count), dt=dt)
        metadata = {name: group.attrs[name] for name in
                    ("frequency_step_hz", "fft_sweep_range_hz", "fft_dz") if name in group.attrs}
        if "derived" in source and "dz" in source["derived"].attrs:
            metadata["dz"] = source["derived"].attrs["dz"]
        for name, value in metadata.items():
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid saved {name}")
            setattr(acq, name, value)
        for name, dataset in (("z", "fiber/z"), ("attenuation_envelope", "fiber/attenuation")):
            if dataset in source:
                setattr(acq, name, source[dataset][:])
        n_cores = None
        for name in ("digital_main", "analog_main"):
            if name in group:
                values = group[name][:]
                if values.ndim != 2 or values.shape[1] != count or values.shape[0] < 1:
                    raise ValueError(f"invalid saved {name} shape")
                if n_cores is not None and values.shape[0] != n_cores:
                    raise ValueError("saved channels have different core counts")
                n_cores = values.shape[0]
                setattr(acq, name, values)
        if "aux_signal" in group:
            acq.aux_signal = group["aux_signal"][:]
            start = group["aux_signal"].attrs["valid_start"]
            if acq.aux_signal.shape != (count,):
                raise ValueError("invalid saved auxiliary signal shape")
            if isinstance(start, (bool, np.bool_)) or not isinstance(start, Integral) or not 0 <= start <= count:
                raise ValueError("invalid saved auxiliary valid_start")
            acq.aux_valid_start = int(start)
        for name in ("strain_field", "temperature_field"):
            if name in group:
                setattr(acq, name, group[name][:])
            elif sweep_index == 0 and f"fiber/{name}" in source:
                setattr(acq, name, source[f"fiber/{name}"][:])
        if "fiber_response" in group:
            node = group["fiber_response"]
            probe = Probe(node["frequency"][:], node["time"][:] if "time" in node else None)
            if probe.shape != (count,):
                raise ValueError("saved probe does not match the sample count")
            reflection = node["reflection"][:]
            kind = Response if reflection.ndim == 1 else MultiCoreResponse
            acq.fiber_response = kind(probe, reflection=reflection, transmission=node["transmission"][:])
            if "diagnostics" in node:
                if kind is not Response:
                    raise ValueError("saved diagnostics require a scalar response")
                diagnostic = node["diagnostics"]
                metadata = json.loads(diagnostic.attrs["metadata"])
                if not isinstance(metadata, dict):
                    raise ValueError("saved diagnostic metadata must be an object")
                direct = Response(probe, reflection=diagnostic["direct/reflection"][:],
                                  transmission=diagnostic["direct/transmission"][:])
                acq.fiber_diagnostics = DiagnosticResponse(acq.fiber_response, direct, metadata)
            response_cores = 1 if kind is Response else acq.fiber_response.n_cores
            if n_cores is not None and n_cores != response_cores:
                raise ValueError("saved response and signals have different core counts")
            n_cores = response_cores
            acq.nu_inst = probe.frequency.copy()
        if "fiber" in source and "n_cores" in source["fiber"].attrs and n_cores is not None:
            if n_cores != source["fiber"].attrs["n_cores"]:
                raise ValueError("saved signals do not match the fiber core count")
        if "fiber/state" in source:
            acq.fiber_state = _read_states(source["fiber/state"], None)
        acq.log = json.loads(group.attrs.get("log", "[]"))
        if not isinstance(acq.log, list) or any(not isinstance(item, dict) for item in acq.log):
            raise ValueError("invalid saved sweep log")
        acq.t = np.arange(count)*dt
        return acq
