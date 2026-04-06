from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from core.acquisition import Acquisition

class PipelineStep(ABC):

    name: str = "unnamed"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def process(self, acq: Acquisition) -> Acquisition:
        """Process the acquisition and return it."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
