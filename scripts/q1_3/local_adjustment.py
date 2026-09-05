"""Receiver-local estimation and simultaneous radial/angular correction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

from scripts.q1_1.localization import (
    GeometryError,
    angle_jacobian,
    angle_residuals,
    fy_position,
    pairwise_angles,
)


@dataclass(frozen=True)
class LocalSettings:
    radius_m: float = 100.0
    gain: float = 0.5
    radial_tolerance_m: float = 1e-3
    angular_tolerance_rad: float = 1e-5
    consistency_tolerance_rad: float = 1e-5
    selection_scale_rad: float = 1e-3
    max_radial_step_m: float = 5.0
    max_angular_step_rad: float = np.deg2rad(2.0)

    def __post_init__(self):
        values = vars(self)
        if any(not np.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("all settings must be finite and positive")
        if self.gain > 1:
            raise ValueError("gain must not exceed 1")


@dataclass(frozen=True)
class BiasEstimate:
    pair: tuple[int, int]
    radial_bias_m: float
    angular_bias_rad: float
    fitted_residual_rad: float
    consistency_rad: float
    geometry_sigma_min: float
    score: float
    success: bool
    nfev: int


@dataclass(frozen=True)
class LocalDecision:
    receiver_id: int
    candidates: tuple[BiasEstimate, ...]
    selected: BiasEstimate | None
    radial_step_m: float
    angular_step_rad: float
    status: str


def public_schedule() -> tuple[tuple[int, int, int], ...]:
    """Fixed FY01 reference plus every auxiliary pair, in published ID order."""
    return tuple((1, a, b) for a, b in combinations(range(2, 10), 2))


def _polar_state(receiver_id, state, radius):
    # Optimize dr and R*dtheta in metres, so both variables have equal scale.
    r = radius + state[0]
    theta = np.deg2rad(40 * (receiver_id - 1)) + state[1] / radius
    radial = np.array([np.cos(theta), np.sin(theta)])
    tangent = np.array([-np.sin(theta), np.cos(theta)])
    return r * radial, np.column_stack((radial, r / radius * tangent))


@lru_cache(maxsize=None)
def _nominal_geometry(receiver_id, pair, radius):
    anchors = np.array([fy_position(i, radius) for i in (0, *pair)])
    sigma = float(
        np.linalg.svd(
            angle_jacobian(fy_position(receiver_id, radius), anchors), compute_uv=False
        )[-1]
    )
    return anchors, sigma


def estimate_bias(receiver_id, pair, observed_angles, *, radius=100.0):
    """Fit two local polar biases using the Q1(1) observation equation.

    Reference coordinates are their nominal values, so biased transmitters
    cause a biased estimate. This function receives this receiver's angles only.
    """
    anchors, sigma = _nominal_geometry(receiver_id, tuple(pair), radius)

    def residual(state):
        point, _ = _polar_state(receiver_id, state, radius)
        return angle_residuals(point, anchors, observed_angles)

    def jacobian(state):
        point, derivative = _polar_state(receiver_id, state, radius)
        return angle_jacobian(point, anchors) @ derivative

    fit = least_squares(
        residual,
        np.zeros(2),
        jac=jacobian,
        bounds=(
            [-0.8 * radius, -np.pi * radius / 3],
            [0.8 * radius, np.pi * radius / 3],
        ),
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=100,
    )
    point, _ = _polar_state(receiver_id, fit.x, radius)
    residual_norm = float(np.linalg.norm(fit.fun))
    # Local branch limits are computational guards, not a certified basin.
    success = bool(fit.success and residual_norm < 1e-8 and not np.any(fit.active_mask))
    return point, fit.x, residual_norm, success, int(fit.nfev), sigma


def decide_local_adjustment(
    receiver_id: int,
    transmitters: tuple[int, int, int],
    own_angles: np.ndarray,
    settings: LocalSettings = LocalSettings(),
) -> LocalDecision:
    """Choose among three reference pairs using six angles from ONE receiver.

    The unselected third transmitter supplies a local consistency check.
    Scheduler, other receivers' observations, and actual positions are absent.
    """
    if (
        receiver_id not in range(2, 10)
        or receiver_id in transmitters
        or len(transmitters) != 3
        or tuple(sorted(set(transmitters))) != transmitters
        or any(i not in range(1, 10) for i in transmitters)
    ):
        raise ValueError(
            "expected a receiver in 2..9 and three distinct sorted transmitters"
        )
    angles = np.asarray(own_angles, dtype=float)
    if angles.shape != (6,) or not np.isfinite(angles).all():
        raise ValueError("six finite receiver-local angles are required")
    if np.any((angles <= 0) | (angles >= np.pi)):
        raise ValueError("angles must be strictly between zero and pi")
    tx = (0, *transmitters)
    labels = tuple(combinations(tx, 2))
    nominal = np.array([fy_position(i, settings.radius_m) for i in tx])
    candidates = []
    for pair in combinations(transmitters, 2):
        indices = [labels.index(label) for label in combinations((0, *pair), 2)]
        try:
            point, state, residual, success, nfev, sigma = estimate_bias(
                receiver_id, pair, angles[indices], radius=settings.radius_m
            )
            # All six are local observations; three held-out angles check the
            # third reference while the fitted three check solver consistency.
            consistency = float(
                np.linalg.norm(pairwise_angles(point, nominal) - angles)
            )
            score = sigma / (1 + consistency / settings.selection_scale_rad)
            candidates.append(
                BiasEstimate(
                    pair=pair,
                    radial_bias_m=float(state[0]),
                    angular_bias_rad=float(state[1] / settings.radius_m),
                    fitted_residual_rad=residual,
                    consistency_rad=consistency,
                    geometry_sigma_min=sigma,
                    score=score,
                    success=success,
                    nfev=nfev,
                )
            )
        except GeometryError:
            continue
    eligible = [c for c in candidates if c.success]
    if not eligible:
        return LocalDecision(
            receiver_id, tuple(candidates), None, 0.0, 0.0, "fit_failed"
        )
    selected = min(eligible, key=lambda c: (-c.score, c.pair))
    radial = float(
        np.clip(
            -settings.gain * selected.radial_bias_m,
            -settings.max_radial_step_m,
            settings.max_radial_step_m,
        )
    )
    angular = float(
        np.clip(
            -settings.gain * selected.angular_bias_rad,
            -settings.max_angular_step_rad,
            settings.max_angular_step_rad,
        )
    )
    return LocalDecision(
        receiver_id, tuple(candidates), selected, radial, angular, "adjust"
    )


class ReceiverController:
    """Local settling memory; enter hold at half tolerance, resume at tolerance.

    Each unknown receiver receives 21 times per 28-slot public period. Requiring
    21 consecutive small estimates checks all its reference configurations.
    Every piece of persistent state belongs to this receiver alone.
    """

    def __init__(self, receiver_id, settings=LocalSettings()):
        self.receiver_id = receiver_id
        self.settings = settings
        self.small_count = 0
        self.holding = False

    def decide(self, transmitters, own_angles):
        decision = decide_local_adjustment(
            self.receiver_id, transmitters, own_angles, self.settings
        )
        candidate = decision.selected
        if candidate is None:
            self.holding = False
            self.small_count = 0
            return decision
        ratio = max(
            abs(candidate.radial_bias_m) / self.settings.radial_tolerance_m,
            abs(candidate.angular_bias_rad) / self.settings.angular_tolerance_rad,
            candidate.consistency_rad / self.settings.consistency_tolerance_rad,
        )
        if self.holding and ratio > 1:
            self.holding = False
            self.small_count = 0
        if not self.holding:
            self.small_count = self.small_count + 1 if ratio <= 0.5 else 0
            if self.small_count >= 21:
                self.holding = True
        if self.holding:
            return replace(
                decision,
                radial_step_m=0.0,
                angular_step_rad=0.0,
                status="within_tolerance",
            )
        return decision


def execute_relative_polar_step(position, radial_step_m, angular_step_rad):
    """Simulation actuator: execute relative radial travel and angular sweep.

    Actual coordinates belong to the simulator. The controller supplies only
    the two relative commands; this ideal actuator assumes exact execution
    around the sensed FY00 direction, with a consistent rotation convention.
    """
    r = np.linalg.norm(position) + radial_step_m
    if r <= 0:
        raise GeometryError("radial command crosses the center")
    theta = np.arctan2(position[1], position[0]) + angular_step_rad
    return r * np.array([np.cos(theta), np.sin(theta)])
