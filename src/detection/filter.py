"""Anti-alias low pass filter.

Goes between detector and ADC. 4th order Butterworth,
cutoff at the detection bandwidth. Without this, noise
above nyquist folds back into the signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

from core.acquisition import Acquisition
from core.pipeline import PipelineStep


class AntiAliasFilter(PipelineStep):

    name = "filter"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        det = config.get("detection", {})
        self.bandwidth = det.get("bandwidth", 1.0e+8)  # Hz, default 100 MHz
        self.order = det.get("filter_order", 4)

    def process(self, acq: Acquisition) -> Acquisition:
        if acq.analog_main is None:
            raise RuntimeError("Filter: analog_main not set")

        dt = acq.dt
        fs = 1.0 / dt
        nyq = fs / 2.0

        # cutoff cant exceed nyquist
        cutoff = min(self.bandwidth, nyq * 0.99)

        sos = butter(self.order, cutoff / nyq, btype="low", output="sos")
        acq.analog_main = sosfilt(sos, acq.analog_main)

        acq.add_log("filter", bandwidth_MHz=self.bandwidth * 1e-6,
                     cutoff_MHz=cutoff * 1e-6)
        return acq
