from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from pyofdr.backends import get_backend
from pyofdr.core.acquisition import Acquisition
from pyofdr.core.config_models import RootConfig

class PipelineStep(ABC):

    name: str = "unnamed"

    def __init__(self, config: dict[str, Any]) -> None:
        # validate up front so steps can read fields without re-specifying
        # defaults. tests that pass partial dicts still work because every
        # section in RootConfig has a default_factory.
        self.config = RootConfig(**config).model_dump()
        self.bk = get_backend(self.config["simulation"]["backend"])

    @abstractmethod
    def process(self, acq: Acquisition) -> Acquisition:
        """Process the acquisition and return it."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
