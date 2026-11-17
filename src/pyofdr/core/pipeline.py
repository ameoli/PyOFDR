from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from pyofdr.backends import get_backend
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.config_models import InstrumentConfig, RootConfig

class PipelineStep(ABC):

    name: str = "unnamed"
    config_model = RootConfig

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = self.config_model(**config).model_dump()
        self.bk = get_backend(self.config["simulation"]["backend"])

    @abstractmethod
    def process(self, acq: Acquisition) -> Acquisition:
        """Process the acquisition and return it."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


class InstrumentStep(PipelineStep):
    config_model = InstrumentConfig

    def __init__(self, config: dict[str, Any] | InstrumentConfig) -> None:
        if isinstance(config, InstrumentConfig):
            config = config.model_dump(include=set(InstrumentConfig.model_fields))
        elif any(key in config for key in ("fiber", "strain", "temperature")):
            config = RootConfig(**config).instrument().model_dump()
        super().__init__(config)
