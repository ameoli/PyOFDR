"""HDF5 streaming writer for simulation output.

Writes each sweep as it completes so we don't need to hold
everything in RAM for long campaigns.

Layout:
    root attr seed_scheme   RNG seed derivation scheme
    /config                 JSON string of the validated config
    /derived                attrs with computed quantities
    /fiber/z                spatial axis [m]
    /fiber/attenuation      round-trip attenuation envelope
    /fiber/strain_field     first sweep's strain eps(z), if present
    /fiber/temperature_field first sweep's applied dT(z) [K], if present
    /fiber/state            prescribed native fields and histories from the first sweep
    /sweeps/0000/digital_main   (n_cores, n_t) int16 (int32 when adc.bits > 16)
    /sweeps/0000/analog_main    (n_cores, n_t) float32
    /sweeps/0000/aux_signal     (n_t,) float32, present iff aux MZI enabled
    /sweeps/0000/strain_field   (n_z,) applied strain eps(z), if present
    /sweeps/0000/temperature_field (n_z,) applied dT(z) [K], if present
    /sweeps/0000/fiber_response/diagnostics/direct  complex direct fields, if requested
    /sweeps/0000/log            JSON string
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import h5py
import numpy as np
from pyrayleigh import DiagnosticResponse, FiberState

from pyofdr.core.acquisition import Acquisition
from pyofdr.utils.seeding import SEED_SCHEME


def _write_state(group, state):
    group.attrs["reference"] = state is None
    if state is None:
        return
    for item in fields(state):
        component = group.create_group(item.name)
        for index, term in enumerate(getattr(state, item.name).terms):
            node = component.create_group(f"{index:04d}")
            for name in ("positions", "values"):
                node.create_dataset(name, data=np.asarray(getattr(term.profile, name)))
            if term.history is not None:
                history = node.create_group("history")
                for name in ("times", "values"):
                    history.create_dataset(name, data=np.asarray(getattr(term.history, name)))


class HDF5Writer:

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file: h5py.File | None = None

    # -- context manager --------------------------------------------------

    def __enter__(self):
        self._target = self.path.resolve()
        self._target.parent.mkdir(parents=True, exist_ok=True)
        # Close the temporary handle before h5py opens it on Windows.
        with NamedTemporaryFile(dir=self._target.parent, prefix=".pyofdr-",
                                suffix=".tmp", delete=False) as tmp:
            self._pending = Path(tmp.name)
        try:
            self._file = h5py.File(self._pending, "w")
        except BaseException:
            self._pending.unlink(missing_ok=True)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._file.close()
            if exc_type is None:
                self._pending.replace(self._target)
        finally:
            self._file = None
            self._pending.unlink(missing_ok=True)
        return False

    # -- writing helpers --------------------------------------------------

    def write_config(self, cfg: dict, derived: dict) -> None:
        f = self._file
        f.attrs["config"] = json.dumps(cfg)
        f.attrs["seed_scheme"] = SEED_SCHEME
        grp = f.create_group("derived")
        for k, v in derived.items():
            grp.attrs[k] = v

    def write_fiber(self, acq: Acquisition) -> None:
        """Write fiber data and first-sweep fields (call once)."""
        if "fiber" in self._file:
            return
        grp = self._file.create_group("fiber")
        if acq.fiber_response is not None and acq.fiber_model is not None:
            model = acq.fiber_model
            cores = getattr(model, "cores", (model,))
            grp.attrs["engine"] = "pyrayleigh"
            grp.attrs["fiber_ids"] = json.dumps([core.fiber_id for core in cores])
            grp.attrs["n_cores"] = len(cores)
            for name in ("length", "reference_temperature", "frequency_min", "frequency_max"):
                grp.attrs[name] = getattr(model, name)
            state_group = grp.create_group("state")
            shared = acq.fiber_state is None or isinstance(acq.fiber_state, FiberState)
            state_group.attrs["per_core"] = not shared
            if shared:
                _write_state(state_group, acq.fiber_state)
            else:
                saved = {}
                for core, state in enumerate(acq.fiber_state):
                    key = f"{core:04d}"
                    if id(state) in saved:
                        state_group[key] = saved[id(state)]
                    else:
                        node = state_group.create_group(key)
                        _write_state(node, state)
                        saved[id(state)] = node
        if acq.z is not None:
            grp.create_dataset("z", data=np.asarray(acq.z))
        if acq.attenuation_envelope is not None:
            grp.create_dataset("attenuation", data=np.asarray(acq.attenuation_envelope))
        if acq.strain_field is not None:
            grp.create_dataset("strain_field", data=np.asarray(acq.strain_field))
        if acq.temperature_field is not None:
            grp.create_dataset("temperature_field",
                               data=np.asarray(acq.temperature_field))

    def write_sweep(self, acq: Acquisition, sweep_index: int) -> None:
        """Write one sweep's data."""
        diagnostic = acq.fiber_diagnostics
        if diagnostic is not None and (not isinstance(diagnostic, DiagnosticResponse)
                                       or diagnostic.total is not acq.fiber_response):
            raise ValueError("fiber_diagnostics must use fiber_response as its total")
        name = f"sweeps/{sweep_index:04d}"
        grp = self._file.create_group(name)
        grp.attrs["n_samples"] = acq.n_samples
        grp.attrs["dt"] = acq.dt
        grp.attrs["frequency_step_hz"] = acq.frequency_step_hz
        grp.attrs["fft_sweep_range_hz"] = acq.fft_sweep_range_hz
        grp.attrs["fft_dz"] = acq.fft_dz

        if acq.fiber_response is not None:
            response = acq.fiber_response
            native = grp.create_group("fiber_response")
            for name in ("reflection", "transmission"):
                native.create_dataset(name, data=np.asarray(getattr(response, name)),
                                      compression="gzip", compression_opts=4)
            native.create_dataset("frequency", data=np.asarray(response.probe.frequency))
            if response.probe.time is not None:
                native.create_dataset("time", data=np.asarray(response.probe.time))
            if diagnostic is not None:
                node = native.create_group("diagnostics")
                node.attrs["metadata"] = json.dumps(diagnostic.metadata)
                direct = node.create_group("direct")
                for name in ("reflection", "transmission"):
                    direct.create_dataset(name, data=np.asarray(getattr(diagnostic.direct, name)),
                                          compression="gzip", compression_opts=4)

        if acq.strain_field is not None:
            grp.create_dataset("strain_field",  data=np.asarray(acq.strain_field))
        if acq.temperature_field is not None:
            grp.create_dataset("temperature_field",
                               data=np.asarray(acq.temperature_field))

        if acq.digital_main is not None:
            grp.create_dataset("digital_main", data=np.asarray(acq.digital_main),
                               compression="gzip", compression_opts=4)
        if acq.analog_main is not None:
            grp.create_dataset("analog_main", data=np.asarray(acq.analog_main, dtype=np.float32),
                               compression="gzip", compression_opts=4)
        if acq.aux_signal is not None:
            ds = grp.create_dataset("aux_signal",
                                     data=np.asarray(acq.aux_signal, dtype=np.float32),
                                     compression="gzip", compression_opts=4)
            ds.attrs["valid_start"] = acq.aux_valid_start

        grp.attrs["log"] = json.dumps(acq.log)
