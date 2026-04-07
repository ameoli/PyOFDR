"""ADC model
TBD
"""

from __future__ import annotations

from typing import Any

from core.acquisition import Acquisition
from core.pipeline import PipelineStep


class ADC(PipelineStep):

    name = "adc"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        adc = config.get("adc", {})
        self.bits = adc.get("bits", 16)
        self.voltage_range = adc.get("voltage_range", 2.0)
        self.v_max = self.voltage_range / 2.0
        self.n_levels = 2 ** self.bits

    def process(self, acq: Acquisition) -> Acquisition:
        if acq.analog_main is None:
            raise RuntimeError("ADC: analog_main is None")

        xp = self.bk.xp
        half = self.n_levels // 2   # 32768 for 16-bit

        clipped = xp.clip(acq.analog_main, -self.v_max, self.v_max)
        normalized = clipped / self.v_max           # [-1, 1]
        digital = xp.floor(normalized * half).astype(xp.int32)
        digital = xp.clip(digital, -half, half - 1)

        acq.digital_main = digital.astype(xp.int16)

        acq.add_log("adc", bits=self.bits, lsb_uV=self.voltage_range / self.n_levels * 1e6)
        return acq
