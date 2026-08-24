"""Passive optical components.

Circulator: 3-port device that routes light in one direction.
Port 1 -> 2: laser to fiber (insertion loss)
Port 2 -> 3: fiber to detector (insertion loss)
Port 1 -> 3: direct leakage (limited by isolation)
"""


class Circulator:
    """Simple scalar circulator model.

    The signal passes through the circulator twice (out to fiber and
    back), so the round-trip field attenuation is IL^2 in linear units.
    """

    def __init__(self, insertion_loss_dB=0.7, isolation_dB=50.0,
                 return_loss_dB=55.0):
        self.insertion_loss_dB = insertion_loss_dB
        self.isolation_dB = isolation_dB
        self.return_loss_dB = return_loss_dB

    @property
    def insertion_loss(self):
        """Single-pass field transmission (linear, < 1)."""
        return 10 ** (-self.insertion_loss_dB / 20.0)

    @property
    def isolation(self):
        """Leakage from port 1 to port 3 (linear, very small)."""
        return 10 ** (-self.isolation_dB / 20.0)

    @property
    def return_loss(self):
        """Back-reflected field amplitude (linear)."""
        return 10 ** (-self.return_loss_dB / 20.0)

    @property
    def round_trip_transmission(self):
        """Two passes through the circulator (field amplitude)."""
        il = self.insertion_loss
        return il * il
