def phase_at_bins(field, dz, prefactor, xp):
    """Round-trip phase at z[k] from the field in cells 0 through k-1."""
    phase = xp.zeros_like(field, dtype=float)
    phase[1:] = prefactor * xp.cumsum(field[:-1]) * dz
    return phase
