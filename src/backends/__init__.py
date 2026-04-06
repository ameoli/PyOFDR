"""Array backend abstraction.

For now just numpy, but the idea is to support cupy later
for GPU accleration.
"""

import numpy as np


def get_backend(name="numpy"):
    """Get the array backend. Only numpy for now."""
    if name == "numpy":
        return np
    raise ValueError(f"Unknown backend '{name}'. Only 'numpy' supported for now.")
