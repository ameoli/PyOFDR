"""Attach the PyRayleigh fiber and its current state to a sweep."""

import math

import numpy as np
from pyrayleigh.backends import real_arrays

from pyofdr.core.pipeline import PipelineStep
from pyofdr.fiber.model import build_fiber, build_state
from pyofdr.utils.constants import C
from pyofdr.utils.units import wavelength_range_to_freq_range


class FiberModel(PipelineStep):
    name = "fiber_model"

    def __init__(self, config, *, reference_temperature=293.15):
        super().__init__(config)
        source = self.config["source"]
        frequency = C/source["center_wavelength"]
        width = wavelength_range_to_freq_range(source["center_wavelength"], source["sweep_range"])
        self.frequency_min = frequency - width/2.
        self.frequency_max = frequency + width/2.
        self.reference_temperature = reference_temperature
        self._model = None
        self._dz = None

    def process(self, acq):
        if acq.z is None or not math.isfinite(acq.dz) or acq.dz <= 0.:
            raise RuntimeError("FiberModel: z/dz not set (run FiberGenerator first)")
        xp, _ = real_arrays(acq.z)
        if self._model is None:
            self._model = build_fiber(self.config, cell_length=acq.dz,
                                      reference_temperature=self.reference_temperature,
                                      frequency_min=self.frequency_min, frequency_max=self.frequency_max, xp=xp)
            self._dz = acq.dz
        sample = self._model.cores[0].rayleigh
        positions = acq.z
        if positions.shape == (sample.cell_count + 1,) and positions[-1] == sample.extent:
            positions = positions[:-1]
        if acq.dz != self._dz or not np.array_equal(positions, sample.positions):
            raise ValueError("FiberModel: spatial grid does not match the cached fiber")
        state = build_state(self.config, sample.edges, acq.strain_field, acq.temperature_field, xp=xp)
        acq.fiber_model = self._model
        acq.fiber_state = state
        acq.add_log(self.name, n_cores=self._model.n_cores, n_z=sample.cell_count)
        return acq
