"""Sample counts for a finite sweep."""

import math


def sweep_sample_count(duration, sample_rate):
    """Count samples in [0, duration), ignoring roundoff near an integer."""
    requested = duration * sample_rate
    nearest = round(requested)
    # A multiplication residue should not add one whole ADC sample.
    if nearest >= 1 and abs(requested - nearest) <= 4 * math.ulp(requested):
        return nearest
    return math.ceil(requested)
