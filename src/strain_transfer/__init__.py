"""Strain transfer from host structure to fiber.

The host strain (e.g. from a structural model or analytical
load case) is what the user specifies. The fiber sees a
*transferred* strain that depends on the bond and the cable
mechanics. Only the ideal (full transfer) model is in here for
now; Cox shear lag will follow, see #15.

This module is standalone and not yet wired into the pipeline.
It will be picked up by the strain perturbation step (roadmap #5).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class StrainTransfer(Protocol):
    def __call__(self, strain_host: np.ndarray) -> np.ndarray: ...


class IdealTransfer:
    """Lossless transfer: eps_fiber == eps_host."""

    def __call__(self, strain_host: np.ndarray) -> np.ndarray:
        return np.asarray(strain_host)
