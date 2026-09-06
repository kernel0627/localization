"""Reproducible observation and actuation errors owned by the simulator."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from scripts.q1_1.localization import pairwise_angles


@dataclass(frozen=True)
class SimulationNoise:
    bearing_std_deg: float = 0.0
    actuation_relative_std: float = 0.0
    seed: int = 20220905

    def __post_init__(self):
        if any(
            not np.isfinite(value) or value < 0
            for value in (self.bearing_std_deg, self.actuation_relative_std)
        ):
            raise ValueError("noise standard deviations must be finite and nonnegative")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    def generator(self, slot, receiver_id, stream):
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, slot, receiver_id, stream])
        )

    def observe(self, position, transmitters, slot, receiver_id):
        if self.bearing_std_deg == 0:
            return pairwise_angles(position, transmitters)
        rays = np.asarray(transmitters) - position
        directions = np.arctan2(rays[:, 1], rays[:, 0])
        directions += self.generator(slot, receiver_id, 0).normal(
            0, np.deg2rad(self.bearing_std_deg), len(directions)
        )
        # A shared noisy ray appears in several angles; do not perturb six
        # redundant pair angles independently and violate their geometry.
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
