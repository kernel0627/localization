"""Private, deterministic noise for the frozen two-configuration simulator.

The controller receives only the six derived angles.  Fixed link offsets and
white ray noise are applied to the four simulator-held ray directions before
those angles are made, which preserves their shared-ray covariance.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from scripts.q1_1.localization import pairwise_angles


SCHEDULE = ((1, 4, 5), (1, 7, 8))


@dataclass(frozen=True)
class TwoConfigurationNoise:
    """A compatible ``observe``/``execute`` noise object for ``simulate``.

    The stream labels 0 and 1 deliberately match :class:`SimulationNoise` so
    the existing white-bearing and execution seed pairing remains available.
    Stream 2 is reserved for the receiver--transmitter fixed bias and has no
    interaction with either old stream.
    """

    bearing_std_deg: float = 0.0
    actuation_relative_std: float = 0.0
    link_bias_std_deg: float = 0.0
    seed: int = 20220905

    def __post_init__(self):
        values = (
            self.bearing_std_deg,
            self.actuation_relative_std,
            self.link_bias_std_deg,
        )
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("noise standard deviations must be finite and nonnegative")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    def generator(self, slot, receiver_id, stream):
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, slot, receiver_id, stream])
        )

    def link_bias_rad(self, receiver_id, transmitter_id):
        """Return this run's constant receiver--transmitter bearing offset."""
        if self.link_bias_std_deg == 0:
            return 0.0
        if receiver_id not in range(2, 10) or transmitter_id not in range(10):
            raise ValueError("receiver/transmitter IDs are outside the public schedule")
        return float(
            np.random.default_rng(
                np.random.SeedSequence([self.seed, receiver_id, transmitter_id, 2])
            ).normal(0, np.deg2rad(self.link_bias_std_deg))
        )

    @staticmethod
    def transmitter_ids(slot):
        """Public IDs for a two-slot schedule phase (one-based ``slot``)."""
        if not isinstance(slot, int) or slot < 1:
            raise ValueError("slot must be a positive integer")
        return (0, *SCHEDULE[(slot - 1) % len(SCHEDULE)])

    def observe(self, position, transmitters, slot, receiver_id):
        """Perturb ray directions and return geometry-consistent pair angles."""
        if (
            self.bearing_std_deg == 0
            and self.link_bias_std_deg == 0
        ):
            return pairwise_angles(position, transmitters)
        rays = np.asarray(transmitters, dtype=float) - np.asarray(position, dtype=float)
        if rays.shape != (4, 2) or not np.isfinite(rays).all():
            raise ValueError("the two-configuration simulator requires four finite rays")
        directions = np.arctan2(rays[:, 1], rays[:, 0])
        if self.bearing_std_deg:
            directions += self.generator(slot, receiver_id, 0).normal(
                0, np.deg2rad(self.bearing_std_deg), len(directions)
            )
        if self.link_bias_std_deg:
            directions += np.array(
                [self.link_bias_rad(receiver_id, tx) for tx in self.transmitter_ids(slot)]
            )
        return np.array(
            [
                abs(
                    np.arctan2(
                        np.sin(directions[a] - directions[b]),
                        np.cos(directions[a] - directions[b]),
                    )
                )
                for a, b in combinations(range(len(directions)), 2)
            ]
        )

    def execute(self, radial, angular, slot, receiver_id):
        if self.actuation_relative_std == 0:
            return radial, angular
        factors = 1 + self.generator(slot, receiver_id, 1).normal(
            0, self.actuation_relative_std, 2
        )
        return radial * factors[0], angular * factors[1]
