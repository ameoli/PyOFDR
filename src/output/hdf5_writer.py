"""HDF5 streaming writer for simulation output.

Writes each sweep as it completes so we don't need to hold
everything in RAM for long campaigns.

Layout:
    /config                 JSON string of the validated config
    /derived                attrs with computed quantities
    /fiber/z                spatial axis [m]
    /fiber/attenuation      round-trip attenuation envelope
    /fiber/strain_field     applied strain eps(z), if present
    /sweeps/0000/digital_main   (n_cores, n_t) int16
    /sweeps/0000/analog_main    (n_cores, n_t) float32
    /sweeps/0000/aux_signal     (n_t,) float32, present iff aux MZI enabled
    /sweeps/0000/log            JSON string
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from core.acquisition import Acquisition


class HDF5Writer:

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file: h5py.File | None = None

    # -- context manager --------------------------------------------------

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
            self._file = None
        return False

    # -- writing helpers --------------------------------------------------

    def write_config(self, cfg: dict, derived: dict) -> None:
        f = self._file
        f.attrs["config"] = json.dumps(cfg)
        grp = f.create_group("derived")
        for k, v in derived.items():
            grp.attrs[k] = v

    def write_fiber(self, acq: Acquisition) -> None:
        """Write static fiber data (call once, after first sweep)."""
        if "fiber" in self._file:
            return
        grp = self._file.create_group("fiber")
        if acq.z is not None:
            grp.create_dataset("z", data=np.asarray(acq.z))
        if acq.attenuation_envelope is not None:
            grp.create_dataset("attenuation", data=np.asarray(acq.attenuation_envelope))
        if acq.strain_field is not None:
            grp.create_dataset("strain_field", data=np.asarray(acq.strain_field))

    def write_sweep(self, acq: Acquisition, sweep_index: int) -> None:
        """Write one sweep's data."""
        name = f"sweeps/{sweep_index:04d}"
        grp = self._file.create_group(name)

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
