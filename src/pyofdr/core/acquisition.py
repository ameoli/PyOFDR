
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np
from pyrayleigh import DiagnosticResponse, Fiber, FiberState, MultiCoreFiber, MultiCoreResponse, Probe, Response


@dataclass
class Acquisition:
    sweep_index: int = 0

    z: np.ndarray | None = None       # positions along fiber [m]
    dz: float = 0.0                   # fiber construction grid [m]

    # per-core arrays have shape (n_cores, ...). single-core runs use (1, ...).
    fiber_profile: np.ndarray | None = None       # (n_cores, n_z), complex
    attenuation_envelope: np.ndarray | None = None  # (n_z,), float
    ghost_profile: np.ndarray | None = None       # (n_cores, n_z), already attenuated

    fiber_model: Fiber | MultiCoreFiber | None = None
    fiber_state: FiberState | Sequence[FiberState | None] | None = None
    fiber_response: Response | MultiCoreResponse | None = None
    fiber_diagnostics: DiagnosticResponse | None = field(default=None, kw_only=True)

    t: np.ndarray | None = None       # (n_t,), time samples [s]
    dt: float = 0.0                   # sample interval [s]
    n_samples: int = 0
    pretrigger_samples: int = field(default=0, kw_only=True)  # pending history before the recorded sweep
    frequency_step_hz: float = 0.0   # nominal linear chirp increment per ADC sample
    fft_sweep_range_hz: float = 0.0  # N * frequency_step_hz, not (N-1) increments
    fft_dz: float = 0.0             # unpadded FFT spatial spacing [m]

    # laser is shared across cores -> 1D
    E_source: np.ndarray | None = None    # (n_t,), complex optical field
    nu_inst: np.ndarray | None = None     # (n_t,), instanteneous frequency [Hz]
    nu_noise: np.ndarray | None = None    # (n_t,), slow (1/f, 1/f^2) FM noise [Hz]

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

    def discard_pretrigger(self):
        """Keep the recorded samples after the causal receiver has processed its history."""
        count = self.pretrigger_samples
        if count == 0:
            return
        if count < 0 or count >= self.n_samples:
            raise ValueError("pretrigger must leave at least one recorded sample")
        response = self.fiber_response
        if response is not None:
            old = response.probe
            probe = Probe(old.frequency[count:], None if old.time is None else old.time[count:])

            def cropped(value):
                return type(value)(probe, reflection=value.reflection[..., count:],
                                   transmission=value.transmission[..., count:])

            if self.fiber_diagnostics is None:
                self.fiber_response = cropped(response)
            else:
                data = self.fiber_diagnostics
                self.fiber_diagnostics = DiagnosticResponse(cropped(data.total), cropped(data.direct),
                                                           metadata=data.metadata)
                self.fiber_response = self.fiber_diagnostics.total
        for name in ("t", "E_source", "nu_inst", "nu_noise", "photocurrent_main",
                     "analog_main", "digital_main", "aux_signal"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value[..., count:])
        self.aux_valid_start = max(0, self.aux_valid_start-count)
        self.n_samples -= count
        self.fft_sweep_range_hz = self.n_samples*self.frequency_step_hz
        self.pretrigger_samples = 0
        self.add_log("capture", pretrigger_samples=count, n_samples=self.n_samples)
