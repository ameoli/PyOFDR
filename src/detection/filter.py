"""Anti-alias low pass filter.

Goes between detector and ADC.
Without this, noise above nyquist folds back into the signal.
"""

from __future__ import annotations

from typing import Any

from core.acquisition import Acquisition
from core.pipeline import PipelineStep


class AntiAliasFilter(PipelineStep):

    name = "filter"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        det = self.config["detection"]
        self.bandwidth = det["bandwidth"]   # Hz
        self.order = det["filter_order"]

    def process(self, acq: Acquisition) -> Acquisition:
        if acq.analog_main is None:
            raise RuntimeError("Filter: analog_main not set")

        dt = acq.dt
        fs = 1.0 / dt
        nyq = fs / 2.0

        # cutoff cant exceed nyquist
        cutoff = min(self.bandwidth, nyq * 0.99)

        acq.analog_main = self.bk.sosfilt_lowpass(
            acq.analog_main, self.order, cutoff / nyq
        )

        acq.add_log("filter", bandwidth_MHz=self.bandwidth * 1e-6,
                     cutoff_MHz=cutoff * 1e-6)
        return acq
