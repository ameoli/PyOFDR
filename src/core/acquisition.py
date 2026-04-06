
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Acquisition:
    sweep_index: int = 0

    z: np.ndarray | None = None       # positions along fiber [m]
    dz: float = 0.0                   # spatial resolution [m]

    fiber_profile: np.ndarray | None = None       # complex Rayleigh phasors
    attenuation_envelope: np.ndarray | None = None  # amplitude decay, float64

    t: np.ndarray | None = None       # time samples [s]
    dt: float = 0.0                   # sample interval [s]
    n_samples: int = 0

    E_source: np.ndarray | None = None    # complex optical field
    nu_inst: np.ndarray | None = None     # instanteneous frequency [Hz]

    photocurrent_main: np.ndarray | None = None   # main interferomter [W]

    analog_main: np.ndarray | None = None   # voltage [V]

    digital_main: np.ndarray | None = None  # quantized, int16

    log: list[dict] = field(default_factory=list)

    def add_log(self, step_name, **kwargs):
        self.log.append({"step": step_name, **kwargs})
