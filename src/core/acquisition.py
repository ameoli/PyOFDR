
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Acquisition:
    sweep_index: int = 0

    z: np.ndarray | None = None       # positions along fiber [m]
    dz: float = 0.0                   # spatial resolution [m]

    # per-core arrays have shape (n_cores, ...). single-core runs use (1, ...).
    fiber_profile: np.ndarray | None = None       # (n_cores, n_z), complex
    attenuation_envelope: np.ndarray | None = None  # (n_z,), float

    t: np.ndarray | None = None       # (n_t,), time samples [s]
    dt: float = 0.0                   # sample interval [s]
    n_samples: int = 0

    # laser is shared across cores -> 1D
    E_source: np.ndarray | None = None    # (n_t,), complex optical field
    nu_inst: np.ndarray | None = None     # (n_t,), instanteneous frequency [Hz]

    strain_field: np.ndarray | None = None         # (n_z,), applied strain eps(z)
    temperature_field: np.ndarray | None = None    # (n_z,), applied dT(z) [K]

    photocurrent_main: np.ndarray | None = None   # (n_cores, n_t), W
    analog_main: np.ndarray | None = None         # (n_cores, n_t), V
    digital_main: np.ndarray | None = None        # (n_cores, n_t), int16

    # auxiliary MZI (k-clock) -- single-channel, same length as main
    aux_signal: np.ndarray | None = None          # (n_t,), real, dimensionless
    aux_valid_start: int = 0                      # first valid sample (drop earlier)

    log: list[dict] = field(default_factory=list)

    def add_log(self, step_name, **kwargs):
        self.log.append({"step": step_name, **kwargs})
