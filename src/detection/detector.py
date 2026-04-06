"""
Later we need to add:
- shot noise (sigma^2 = 2*e*I*B)
- thermal noise (from NEP)
- dark current
- bandwidth limiting
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.acquisition import Acquisition
from core.pipeline import PipelineStep


class Detector(PipelineStep):

    name = "detection"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        det = config.get("detection", {})
        adc = config.get("adc", {})
        self.responsivity = det.get("responsivity", 1.0)      # A/W
        self.impedance = adc.get("input_impedance", 50.0)     # ohm

    def process(self, acq: Acquisition) -> Acquisition:
        if acq.photocurrent_main is None:
            raise RuntimeError("Detector: photocurrent not set")

        # V = I_photo * R * Z
        acq.analog_main = (acq.photocurrent_main
                           * self.responsivity
                           * self.impedance)

        acq.add_log("detection", type="ideal")
        return acq
