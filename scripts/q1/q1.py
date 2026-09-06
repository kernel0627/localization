from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass, replace
from functools import lru_cache
from itertools import combinations
from numbers import Integral
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import least_squares

# q1_1
FloatArray = NDArray[np.float64]

class GeometryError(ValueError):
    pass

@dataclass(frozen=True)
class LocalizationResult:
    position: FloatArray
    success: bool
    cost: float
    residual_norm: float
    jacobian_singular_values: FloatArray
    condition_number: float
    nfev: int
    message: str

def polar_to_cartesian(radius: float, angle_deg: float) -> FloatArray:
    angle = np.deg2rad(angle_deg)
    return np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=float)

def fy_position(drone_id: int, radius: float = 100.0) -> FloatArray:
    if drone_id == 0:
        return np.zeros(2, dtype=float)
    if not 1 <= drone_id <= 9:
        raise ValueError("drone_id must be between 0 and 9")
    return polar_to_cartesian(radius, 40.0 * (drone_id - 1))

def _as_anchor_array(anchors: ArrayLike) -> FloatArray:
    points = np.asarray(anchors, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("anchors must have shape (n, 2) with n >= 3")
    if not np.isfinite(points).all():
        raise ValueError("anchors must contain finite coordinates")
    return points

def _pairs(anchor_count: int) -> tuple[tuple[int, int], ...]:
    return tuple(combinations(range(anchor_count), 2))

def _receiver_vectors(
    receiver: ArrayLike, anchors: FloatArray
) -> tuple[FloatArray, FloatArray]:
    point = np.asarray(receiver, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("receiver must be one finite 2-D point")
    vectors = anchors - point
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-12):
        raise GeometryError(
            "receiver coincides with a transmitter, so an angle is undefined"
        )
    return (vectors, lengths)

def pairwise_angles(receiver: ArrayLike, anchors: ArrayLike) -> FloatArray:
    points = _as_anchor_array(anchors)
    (vectors, lengths) = _receiver_vectors(receiver, points)
    angles: list[float] = []
    for first, second in _pairs(points.shape[0]):
        cosine = np.dot(vectors[first], vectors[second]) / (
            lengths[first] * lengths[second]
        )
        angles.append(float(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return np.asarray(angles, dtype=float)

def _observed_cosines(observed_angles: ArrayLike, anchor_count: int) -> FloatArray:
    angles = np.asarray(observed_angles, dtype=float)
    expected = anchor_count * (anchor_count - 1) // 2
    if angles.shape != (expected,):
        raise ValueError(
            f"expected {expected} pairwise angles, got shape {angles.shape}"
        )
    if not np.isfinite(angles).all() or np.any(angles < 0.0) or np.any(angles > np.pi):
        raise ValueError("observed angles must be finite and lie in [0, pi]")
    return np.cos(angles)

def angle_residuals(
    receiver: ArrayLike, anchors: ArrayLike, observed_angles: ArrayLike
) -> FloatArray:
    points = _as_anchor_array(anchors)
    angles = np.asarray(observed_angles, dtype=float)
    _observed_cosines(angles, points.shape[0])
    return pairwise_angles(receiver, points) - angles

def cosine_jacobian(
    receiver: ArrayLike, anchors: ArrayLike, observed_angles: ArrayLike | None = None
) -> FloatArray:
    points = _as_anchor_array(anchors)
    if observed_angles is not None:
        _observed_cosines(observed_angles, points.shape[0])
    (vectors, lengths) = _receiver_vectors(receiver, points)
    rows: list[FloatArray] = []
    for first, second in _pairs(points.shape[0]):
        u = vectors[first]
        v = vectors[second]
        norm_u = lengths[first]
        norm_v = lengths[second]
        cosine = np.dot(u, v) / (norm_u * norm_v)
        derivative_u = v / (norm_u * norm_v) - cosine * u / norm_u**2
        derivative_v = u / (norm_u * norm_v) - cosine * v / norm_v**2
        rows.append(-(derivative_u + derivative_v))
    return np.asarray(rows, dtype=float)

def angle_jacobian(
    receiver: ArrayLike, anchors: ArrayLike, observed_angles: ArrayLike | None = None
) -> FloatArray:
    points = _as_anchor_array(anchors)
    if observed_angles is not None:
        _observed_cosines(observed_angles, points.shape[0])
    (vectors, lengths) = _receiver_vectors(receiver, points)
    cosines = np.asarray(
        [
            np.dot(vectors[first], vectors[second]) / (lengths[first] * lengths[second])
            for (first, second) in _pairs(points.shape[0])
        ],
        dtype=float,
    )
    clipped = np.clip(cosines, -1.0, 1.0)
    sine_magnitudes = np.sqrt(np.maximum(0.0, 1.0 - clipped**2))
    if np.any(sine_magnitudes <= 1e-12):
        raise GeometryError(
            "angle Jacobian is undefined for a pairwise angle of 0 or pi"
        )
    return -cosine_jacobian(receiver, points) / sine_magnitudes[:, None]

def localize_receiver(
    anchors: ArrayLike,
    observed_angles: ArrayLike,
    initial_position: ArrayLike,
    *,
    max_nfev: int = 500,
) -> LocalizationResult:
    points = _as_anchor_array(anchors)
    angles = np.asarray(observed_angles, dtype=float)
    _observed_cosines(angles, points.shape[0])
    initial = np.asarray(initial_position, dtype=float)
    _receiver_vectors(initial, points)
    result = least_squares(
        angle_residuals,
        initial,
        jac=angle_jacobian,
        args=(points, angles),
        method="trf",
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
        max_nfev=max_nfev,
    )
    singular_values = np.linalg.svd(result.jac, compute_uv=False)
    smallest = float(singular_values[-1])
    condition_number = (
        float(singular_values[0] / smallest) if smallest > 1e-15 else float("inf")
    )
    return LocalizationResult(
        position=np.asarray(result.x, dtype=float),
        success=bool(result.success),
        cost=float(result.cost),
        residual_norm=float(np.linalg.norm(result.fun)),
        jacobian_singular_values=np.asarray(singular_values, dtype=float),
        condition_number=condition_number,
        nfev=int(result.nfev),
        message=str(result.message),
    )

def solve_q1_1(
    receiver_id=3, transmitter_ids=(1, 5), observed_angles=None, radius=100.0
):
    a, b = transmitter_ids
    if (
        a == b
        or any(i not in range(1, 10) for i in (a, b, receiver_id))
        or receiver_id in (a, b)
        or not np.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError("invalid receiver, transmitter IDs or radius")
    anchors = np.array([fy_position(i, radius) for i in (0, a, b)])
    if observed_angles is None:
        observed_angles = pairwise_angles(
            table_positions()[receiver_id] * radius / 100, anchors
        )
    result = localize_receiver(
        anchors, observed_angles, fy_position(receiver_id, radius)
    )
    return {
        "receiver_id": receiver_id,
        "radius_m": radius,
        "transmitters": (0, a, b),
        "angles_deg": np.rad2deg(observed_angles),
        "result": result,
    }

# q1_2

@dataclass(frozen=True)
class PositionCandidate:
    emitter_id: int
    position: FloatArray
    residual_norm: float
    distance_to_ideal: float
    source: str

@dataclass(frozen=True)
class HypothesisResult:
    emitter_id: int
    candidates: tuple[PositionCandidate, ...]
    local_status: str
    local_residual_norm: float | None
    local_message: str
    continuum_distance_bounds: tuple[float, ...]

@dataclass(frozen=True)
class IdentificationResult:
    receiver_id: int
    status: str
    selected: PositionCandidate | None
    hypotheses: tuple[HypothesisResult, ...]
    distance_margin: float | None
    identity_margin: float | None
    continuum_distance_bound: float | None

def _circle_candidates(
    emitter_id: int,
    angles: FloatArray,
    ideal: FloatArray,
    radius: float,
    residual_tolerance: float,
) -> tuple[list[PositionCandidate], list[float]]:
    anchors = np.vstack([fy_position(i, radius) for i in (0, 1, emitter_id)])
    a = anchors[2] / radius
    perpendicular_a = np.array([-a[1], a[0]])
    candidates: list[PositionCandidate] = []
    continuum_bounds: list[float] = []
    for first_sign in (-1.0, 1.0):
        first_center = np.array([0.5, first_sign * 0.5 / np.tan(angles[0])])
        for second_sign in (-1.0, 1.0):
            second_center = (a + second_sign / np.tan(angles[1]) * perpendicular_a) / 2
            difference = first_center - second_center
            norm = float(np.linalg.norm(difference))
            scale = max(
                1.0, np.linalg.norm(first_center), np.linalg.norm(second_center)
            )
            if norm <= 1e-10 * scale:
                distance = radius * abs(
                    np.linalg.norm(ideal / radius - first_center)
                    - np.linalg.norm(first_center)
                )
                continuum_bounds.append(
                    max(0.0, float(distance) - radius * 1e-09 * scale)
                )
                continue
            direction = np.array([-difference[1], difference[0]]) / norm
            position = radius * 2 * np.dot(first_center, direction) * direction
            if np.min(np.linalg.norm(anchors - position, axis=1)) <= radius * 1e-10:
                continue
            residual = float(
                np.linalg.norm(pairwise_angles(position, anchors) - angles)
            )
            if residual > residual_tolerance:
                continue
            if any(
                (
                    np.linalg.norm(position - c.position) <= radius * 1e-07
                    for c in candidates
                )
            ):
                continue
            candidates.append(
                PositionCandidate(
                    emitter_id,
                    position,
                    residual,
                    float(np.linalg.norm(position - ideal)),
                    "circle_intersection",
                )
            )
    return (candidates, continuum_bounds)

def identify_anonymous_emitter(
    receiver_id: int,
    observed_angles: ArrayLike,
    *,
    radius: float = 100.0,
    residual_tolerance: float = 1e-08,
    distance_tie_tolerance: float | None = None,
    max_nfev: int = 500,
) -> IdentificationResult:
    if (
        isinstance(receiver_id, bool)
        or not isinstance(receiver_id, Integral)
        or (not 2 <= receiver_id <= 9)
    ):
        raise ValueError("receiver_id must be an integer from 2 through 9")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("residual_tolerance must be finite and positive")
    if (
        isinstance(max_nfev, bool)
        or not isinstance(max_nfev, Integral)
        or max_nfev <= 0
    ):
        raise ValueError("max_nfev must be a positive integer")
    tie_tolerance = (
        radius * 1e-07 if distance_tie_tolerance is None else distance_tie_tolerance
    )
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("distance_tie_tolerance must be finite and nonnegative")
    angles = np.asarray(observed_angles, dtype=float)
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError("expected three finite angles ordered (01, 0U, 1U)")
    if np.any(angles <= 0) or np.any(angles >= np.pi):
        raise ValueError(
            "exact-angle model requires each angle strictly between 0 and pi"
        )
    ideal = fy_position(receiver_id, radius)
    hypotheses: list[HypothesisResult] = []
    for emitter_id in range(2, 10):
        if emitter_id == receiver_id:
            continue
        anchors = np.vstack([fy_position(i, radius) for i in (0, 1, emitter_id)])
        local_residual = None
        local_candidate = None
        try:
            local = localize_receiver(anchors, angles, ideal, max_nfev=max_nfev)
            local_residual = local.residual_norm
            local_message = local.message
            if not local.success:
                local_status = "not_converged"
            elif local.residual_norm > residual_tolerance:
                local_status = "nonzero_residual"
            else:
                local_status = "consistent"
                local_candidate = PositionCandidate(
                    emitter_id,
                    local.position,
                    local.residual_norm,
                    float(np.linalg.norm(local.position - ideal)),
                    "q1_1_local",
                )
        except GeometryError as error:
            local_status = "geometry_error"
            local_message = str(error)
        (candidates, continuum_bounds) = _circle_candidates(
            emitter_id, angles, ideal, radius, residual_tolerance
        )
        if local_candidate is not None:
            match = next(
                (
                    i
                    for (i, c) in enumerate(candidates)
                    if np.linalg.norm(c.position - local_candidate.position)
                    <= radius * 1e-07
                ),
                None,
            )
            if match is None:
                candidates.append(local_candidate)
            else:
                candidates[match] = local_candidate
        candidates.sort(key=lambda c: c.distance_to_ideal)
        hypotheses.append(
            HypothesisResult(
                emitter_id,
                tuple(candidates),
                local_status,
                local_residual,
                local_message,
                tuple(continuum_bounds),
            )
        )
    all_candidates = sorted(
        (c for h in hypotheses for c in h.candidates), key=lambda c: c.distance_to_ideal
    )
    all_bounds = [bound for h in hypotheses for bound in h.continuum_distance_bounds]
    continuum_bound = min(all_bounds, default=None)
    if not all_candidates:
        return IdentificationResult(
            receiver_id,
            "degenerate" if all_bounds else "no_consistent_candidate",
            None,
            tuple(hypotheses),
            None,
            None,
            continuum_bound,
        )
    best = all_candidates[0]
    margin = (
        all_candidates[1].distance_to_ideal - best.distance_to_ideal
        if len(all_candidates) > 1
        else None
    )
    other_ids = [
        c.distance_to_ideal for c in all_candidates if c.emitter_id != best.emitter_id
    ]
    identity_margin = min(other_ids) - best.distance_to_ideal if other_ids else None
    if (
        continuum_bound is not None
        and continuum_bound <= best.distance_to_ideal + tie_tolerance
    ):
        (status, selected) = ("degenerate", None)
    elif margin is not None and margin <= tie_tolerance:
        (status, selected) = ("ambiguous", None)
    else:
        (status, selected) = ("selected", best)
    return IdentificationResult(
        receiver_id,
        status,
        selected,
        tuple(hypotheses),
        margin,
        identity_margin,
        continuum_bound,
    )

def solve_q1_2(receiver_id=3, observed_angles=None, radius=100.0):
    if observed_angles is None:
        if receiver_id == 5:
            raise ValueError(
                "the default example uses FY05; supply measured angles for receiver FY05"
            )
        observed_angles = pairwise_angles(
            table_positions()[receiver_id], nominal_positions()[[0, 1, 5]]
        )
    return identify_anonymous_emitter(receiver_id, observed_angles, radius=radius)

# q1_3_1

TABLE1_POLAR = (
    (0, 0),
    (100, 0),
    (98, 40.1),
    (112, 80.21),
    (105, 119.75),
    (98, 159.86),
    (112, 199.96),
    (105, 240.07),
    (98, 280.17),
    (112, 320.28),
)

FULL_SCHEDULE = tuple(((1, a, b) for (a, b) in combinations(range(2, 10), 2)))

TWO_SCHEDULE = ((1, 4, 5), (1, 7, 8))

def schedule_for(name):
    if name == "full":
        return FULL_SCHEDULE
    if name == "two":
        return TWO_SCHEDULE
    raise ValueError("schedule must be 'full' or 'two'")

def nominal_positions(radius=100.0):
    return np.array([fy_position(i, radius) for i in range(10)])

def table_positions():
    return np.array([polar_to_cartesian(r, angle) for (r, angle) in TABLE1_POLAR])

def random_positions(trial, seed=20260905):
    if not isinstance(trial, int) or trial < 0:
        raise ValueError("trial must be a nonnegative integer")
    rng = np.random.default_rng(np.random.SeedSequence([seed, trial, 7]))
    positions = nominal_positions()
    radii = 100 + rng.uniform(-12, 12, 8)
    angles = np.deg2rad(np.arange(1, 9) * 40 + rng.uniform(-0.3, 0.3, 8))
    positions[2:] = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    return positions

@dataclass(frozen=True)
class LocalSettings:
    radius_m: float = 100.0
    gain: float = 0.5
    radial_tolerance_m: float = 0.001
    angular_tolerance_rad: float = 1e-05
    consistency_tolerance_rad: float = 1e-05
    selection_scale_rad: float = 0.001
    max_radial_step_m: float = 5.0
    max_angular_step_rad: float = np.deg2rad(2.0)

    def __post_init__(self):
        values = vars(self)
        if any((not np.isfinite(value) or value <= 0 for value in values.values())):
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

def _polar_state(receiver_id, state, radius):
    r = radius + state[0]
    theta = np.deg2rad(40 * (receiver_id - 1)) + state[1] / radius
    radial = np.array([np.cos(theta), np.sin(theta)])
    tangent = np.array([-np.sin(theta), np.cos(theta)])
    return (r * radial, np.column_stack((radial, r / radius * tangent)))

@lru_cache(maxsize=None)
def _nominal_geometry(receiver_id, pair, radius):
    anchors = np.array([fy_position(i, radius) for i in (0, *pair)])
    sigma = float(
        np.linalg.svd(
            angle_jacobian(fy_position(receiver_id, radius), anchors), compute_uv=False
        )[-1]
    )
    return (anchors, sigma)

def estimate_bias(receiver_id, pair, observed_angles, *, radius=100.0):
    (anchors, sigma) = _nominal_geometry(receiver_id, tuple(pair), radius)

    def residual(state):
        (point, _) = _polar_state(receiver_id, state, radius)
        return angle_residuals(point, anchors, observed_angles)

    def jacobian(state):
        (point, derivative) = _polar_state(receiver_id, state, radius)
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
    (point, _) = _polar_state(receiver_id, fit.x, radius)
    residual_norm = float(np.linalg.norm(fit.fun))
    success = bool(
        fit.success and residual_norm < 1e-08 and (not np.any(fit.active_mask))
    )
    return (point, fit.x, residual_norm, success, int(fit.nfev), sigma)

def decide_local_adjustment(
    receiver_id: int,
    transmitters: tuple[int, int, int],
    own_angles: np.ndarray,
    settings: LocalSettings = LocalSettings(),
) -> LocalDecision:
    if (
        receiver_id not in range(2, 10)
        or receiver_id in transmitters
        or len(transmitters) != 3
        or (tuple(sorted(set(transmitters))) != transmitters)
        or any((i not in range(1, 10) for i in transmitters))
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
            (point, state, residual, success, nfev, sigma) = estimate_bias(
                receiver_id, pair, angles[indices], radius=settings.radius_m
            )
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
    r = np.linalg.norm(position) + radial_step_m
    if r <= 0:
        raise GeometryError("radial command crosses the center")
    theta = np.arctan2(position[1], position[0]) + angular_step_rad
    return r * np.array([np.cos(theta), np.sin(theta)])

@dataclass(frozen=True)
class SimulationNoise:
    bearing_std_deg: float = 0.0
    actuation_relative_std: float = 0.0
    link_bias_std_deg: float = 0.0
    seed: int = 20220905

    def __post_init__(self):
        scales = (
            self.bearing_std_deg,
            self.actuation_relative_std,
            self.link_bias_std_deg,
        )
        if any((not np.isfinite(s) or s < 0 for s in scales)):
            raise ValueError("noise standard deviations must be finite and nonnegative")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    def generator(self, slot, receiver_id, stream):
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, slot, receiver_id, stream])
        )

    def link_bias_rad(self, receiver_id, transmitter_id):
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, receiver_id, transmitter_id, 2])
        )
        return float(rng.normal(0, np.deg2rad(self.link_bias_std_deg)))

    def observe(self, position, transmitters, slot, receiver_id, transmitter_ids):
        if self.bearing_std_deg == self.link_bias_std_deg == 0:
            return pairwise_angles(position, transmitters)
        rays = np.asarray(transmitters, dtype=float) - np.asarray(position, dtype=float)
        if (
            rays.shape != (4, 2)
            or len(transmitter_ids) != 4
            or (not np.isfinite(rays).all())
        ):
            raise ValueError("expected four finite transmitter rays and their IDs")
        if np.any(np.linalg.norm(rays, axis=1) <= 1e-12):
            raise GeometryError("receiver coincides with a transmitter")
        directions = np.arctan2(rays[:, 1], rays[:, 0])
        if self.bearing_std_deg:
            directions += self.generator(slot, receiver_id, 0).normal(
                0, np.deg2rad(self.bearing_std_deg), 4
            )
        if self.link_bias_std_deg:
            directions += [
                self.link_bias_rad(receiver_id, tx) for tx in transmitter_ids
            ]
        return np.array(
            [
                abs(
                    np.arctan2(
                        np.sin(directions[a] - directions[b]),
                        np.cos(directions[a] - directions[b]),
                    )
                )
                for (a, b) in combinations(range(4), 2)
            ]
        )

    def execute(self, radial, angular, slot, receiver_id):
        if self.actuation_relative_std == 0:
            return (radial, angular)
        factors = 1 + self.generator(slot, receiver_id, 1).normal(
            0, self.actuation_relative_std, 2
        )
        return (radial * factors[0], angular * factors[1])

def serializable(value):
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, dict):
        return {str(k): serializable(v) for (k, v) in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serializable(value), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            fields = list(dict.fromkeys((key for row in rows for key in row)))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

def formation_metrics(positions):
    nominal = nominal_positions()
    radial = np.linalg.norm(positions[1:], axis=1) - 100
    theta = np.arctan2(positions[1:, 1], positions[1:, 0])
    angular = (theta - np.deg2rad(np.arange(9) * 40) + np.pi) % (2 * np.pi) - np.pi
    errors = np.linalg.norm(positions[1:] - nominal[1:], axis=1)
    return {
        "max_position_error_m": float(errors.max()),
        "rms_position_error_m": float(np.sqrt(np.mean(errors**2))),
        "max_radial_error_m": float(np.abs(radial).max()),
        "max_angular_error_deg": float(np.rad2deg(np.abs(angular).max())),
    }

def simulate_adjustment(
    initial_positions=None,
    *,
    schedule="full",
    settings=LocalSettings(),
    max_slots=560,
    noise=SimulationNoise(),
    retain_details=True,
):
    if settings.radius_m != 100:
        raise ValueError("Q1.3 uses the prescribed radius of 100 m")
    if not isinstance(max_slots, int) or isinstance(max_slots, bool) or max_slots < 1:
        raise ValueError("max_slots must be a positive integer")
    points = np.array(
        table_positions() if initial_positions is None else initial_positions,
        dtype=float,
        copy=True,
    )
    if points.shape != (10, 2) or not np.isfinite(points).all():
        raise ValueError("initial_positions must have finite shape (10, 2)")
    if not np.allclose(points[:2], nominal_positions()[:2], rtol=0, atol=1e-12):
        raise ValueError("FY00 and FY01 must equal their calibrated positions")
    phases = schedule_for(schedule)
    controllers = {i: ReceiverController(i, settings) for i in range(2, 10)}
    history = [
        {
            "slot": 0,
            "phase": 0,
            "cumulative_transmitter_uses": 0,
            "cumulative_endpoint_displacement_m": 0.0,
            **formation_metrics(points),
        }
    ]
    positions = [points.copy()]
    (steps, observations, candidates) = ([], [], [])
    (quiet_cycle, movement) = (True, 0.0)
    last_motion_slot = moves = active_slots = decisions_count = 0
    failure = None
    status = "budget_exhausted"
    for slot in range(1, max_slots + 1):
        phase = (slot - 1) % len(phases)
        if phase == 0:
            quiet_cycle = True
        circular_tx = phases[phase]
        tx = (0, *circular_tx)
        before = points.copy()
        (decisions, commands) = ({}, {})
        for i in range(2, 10):
            if i in tx:
                continue
            decisions_count += 1
            try:
                angles = noise.observe(before[i], before[list(tx)], slot, i, tx)
                decision = controllers[i].decide(circular_tx, angles)
            except (GeometryError, ValueError) as error:
                failure = {"slot": slot, "receiver_id": i, "message": str(error)}
                break
            decisions[i] = decision
            quiet_cycle &= decision.status == "within_tolerance"
            if retain_details:
                observations.append(
                    {
                        "slot": slot,
                        "receiver_id": i,
                        "transmitters": "-".join(map(str, tx)),
                        **{
                            f"angle_{a}_{b}_rad": float(value)
                            for ((a, b), value) in zip(
                                (
                                    (tx[a], tx[b])
                                    for a in range(4)
                                    for b in range(a + 1, 4)
                                ),
                                angles,
                            )
                        },
                    }
                )
                for candidate in decision.candidates:
                    row = asdict(candidate)
                    row["pair"] = "-".join(map(str, candidate.pair))
                    candidates.append({"slot": slot, "receiver_id": i, **row})
            if decision.selected is None:
                failure = {
                    "slot": slot,
                    "receiver_id": i,
                    "message": "no valid local reference fit",
                }
                break
            commands[i] = noise.execute(
                decision.radial_step_m, decision.angular_step_rad, slot, i
            )
        after = before.copy()
        if failure is None:
            for i, (radial, angular) in commands.items():
                if radial == angular == 0:
                    continue
                try:
                    after[i] = execute_relative_polar_step(before[i], radial, angular)
                except GeometryError as error:
                    failure = {"slot": slot, "receiver_id": i, "message": str(error)}
                    break
        if failure is not None:
            after = before.copy()
            status = "local_fit_failed" if len(commands) < 6 else "execution_failed"
        else:
            slot_moves = sum(
                (
                    d.radial_step_m != 0 or d.angular_step_rad != 0
                    for d in decisions.values()
                )
            )
            moves += slot_moves
            active_slots += int(slot_moves > 0)
            if slot_moves:
                last_motion_slot = slot
        if retain_details:
            for i, decision in decisions.items():
                chosen = decision.selected
                (radial, angular) = commands.get(i, (0.0, 0.0))
                steps.append(
                    {
                        "slot": slot,
                        "phase": phase + 1,
                        "receiver_id": i,
                        "transmitters": "-".join(map(str, tx)),
                        "status": decision.status,
                        "applied": failure is None,
                        "selected_pair": "-".join(map(str, chosen.pair))
                        if chosen
                        else "",
                        "estimated_radial_bias_m": chosen.radial_bias_m
                        if chosen
                        else None,
                        "estimated_angular_bias_rad": chosen.angular_bias_rad
                        if chosen
                        else None,
                        "consistency_rad": chosen.consistency_rad if chosen else None,
                        "radial_command_m": decision.radial_step_m,
                        "angular_command_rad": decision.angular_step_rad,
                        "executed_radial_m": radial if failure is None else 0.0,
                        "executed_angular_rad": angular if failure is None else 0.0,
                        "small_count": controllers[i].small_count,
                        "holding": controllers[i].holding,
                    }
                )
        points = after
        movement += float(np.linalg.norm(after - before, axis=1).sum())
        history.append(
            {
                "slot": slot,
                "phase": phase + 1,
                "cumulative_transmitter_uses": 4 * slot,
                "cumulative_endpoint_displacement_m": movement,
                **formation_metrics(points),
            }
        )
        positions.append(points.copy())
        if failure is not None:
            break
        if phase == len(phases) - 1 and quiet_cycle:
            status = "quiet_full_cycle"
            break
    thresholds = []
    for threshold in (0.02, 0.01, 0.001):
        first = next(
            (r["slot"] for r in history if r["max_position_error_m"] < threshold), None
        )
        thresholds.append(
            {
                "threshold_m": threshold,
                "first_slot": first,
                "transmitter_uses_at_first": None if first is None else 4 * first,
                "all_recorded_slots_after_first_below": first is not None
                and all(
                    (r["max_position_error_m"] < threshold for r in history[first:])
                ),
            }
        )
    summary = {
        "schedule": schedule,
        "settings": asdict(settings),
        "noise": asdict(noise),
        "max_slots": max_slots,
        "status": status,
        "failure": failure,
        "schedule_period_slots": len(phases),
        "measurement_slots": slot,
        "transmitter_uses": 4 * slot,
        "receiver_decisions": decisions_count,
        "nonzero_receiver_moves": moves,
        "slots_with_motion": active_slots,
        "last_motion_slot": last_motion_slot,
        "post_motion_confirmation_slots": slot - last_motion_slot,
        "total_endpoint_displacement_m": movement,
        "initial_metrics": history[0],
        "final_metrics": history[-1],
        "precision_thresholds": thresholds,
    }
    return {
        "summary": summary,
        "history": history,
        "steps": steps,
        "observations": observations,
        "candidates": candidates,
        "positions": np.array(positions),
        "final_positions": points,
    }

def save_run(output_dir, run):
    from pathlib import Path

    output_dir = Path(output_dir)
    write_json(output_dir / "summary.json", run["summary"])
    for name in ("history", "steps", "observations", "candidates"):
        write_csv(output_dir / f"{name}.csv", run[name])
    write_csv(
        output_dir / "final_positions.csv",
        [
            {"drone_id": i, "x_m": p[0], "y_m": p[1]}
            for (i, p) in enumerate(run["final_positions"])
        ],
    )
    np.save(output_dir / "positions.npy", run["positions"])
    write_csv(
        output_dir / "schedule.csv",
        [
            {
                "phase": phase,
                "transmitters": "-".join(map(str, (0, *tx))),
                "receivers": "-".join((str(i) for i in range(2, 10) if i not in tx)),
            }
            for (phase, tx) in enumerate(schedule_for(run["summary"]["schedule"]), 1)
        ],
    )

def solve_q1_3_1(
    initial_positions=None, gain=0.5, max_slots=560, noise=SimulationNoise()
):
    return simulate_adjustment(
        initial_positions,
        schedule="full",
        settings=LocalSettings(gain=gain),
        max_slots=max_slots,
        noise=noise,
    )

# q1_3_2

def solve_q1_3_2(
    initial_positions=None, gain=0.5, max_slots=560, noise=SimulationNoise()
):
    return simulate_adjustment(
        initial_positions,
        schedule="two",
        settings=LocalSettings(gain=gain),
        max_slots=max_slots,
        noise=noise,
    )

DIMENSION = 16

def linear_model(schedule="full", gain=0.5):
    if not np.isfinite(gain) or not 0 < gain <= 1:
        raise ValueError("gain must lie in (0, 1]")
    phases = schedule_for(schedule)
    q = nominal_positions()
    links = sorted(
        {(i, a) for tx in phases for i in range(2, 10) if i not in tx for a in (0, *tx)}
    )
    link_index = {link: column for (column, link) in enumerate(links)}
    (matrices, injections, choices, blocks) = ([], [], [], [])
    for phase, tx in enumerate(phases, 1):
        matrix = np.eye(DIMENSION)
        injection = np.zeros((DIMENSION, len(links)))
        for i in range(2, 10):
            if i in tx:
                continue
            candidates = []
            for pair in combinations(tx, 2):
                j = angle_jacobian(q[i], q[[0, *pair]]) @ polar_basis(i)
                sigma = float(np.linalg.svd(j, compute_uv=False)[-1])
                candidates.append((sigma, pair, j))
            candidates.sort(key=lambda c: (-c[0], c[1]))
            (sigma, pair, j) = candidates[0]
            inverse = np.linalg.pinv(j)
            propagation = inverse @ observation_derivative(i, (0, *pair))
            matrix[state_slice(i)] -= gain * propagation
            rays = q[[0, *pair]] - q[i]
            directions = np.arctan2(rays[:, 1], rays[:, 0])
            response = -gain * inverse @ ray_angle_derivative(directions)
            for column, a in enumerate((0, *pair)):
                injection[state_slice(i), link_index[i, a]] = response[:, column]
            choices.append(
                {
                    "phase": phase,
                    "receiver_id": i,
                    "pair": pair,
                    "sigma_min": sigma,
                    "relative_selection_gap": (sigma - candidates[1][0]) / sigma,
                }
            )
            for a in pair:
                if a >= 2:
                    blocks.append(
                        {
                            "phase": phase,
                            "receiver_id": i,
                            "reference_id": a,
                            "B": propagation[:, state_slice(a)],
                        }
                    )
        matrices.append(matrix)
        injections.append(injection)
    cycle = cycle_matrix(matrices)
    return {
        "schedule": schedule,
        "gain": gain,
        "links": links,
        "matrices": matrices,
        "injections": injections,
        "choices": choices,
        "reference_blocks": blocks,
        "cycle": cycle,
        "cycle_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(cycle)))),
        "cycle_operator_norm": float(np.linalg.norm(cycle, 2)),
        "two_cycle_operator_norm": float(np.linalg.norm(cycle @ cycle, 2)),
        "minimum_selection_gap": min((c["relative_selection_gap"] for c in choices)),
    }

def initial_second_moment():
    a = np.deg2rad(0.3)
    return np.kron(np.eye(8), np.diag([48.0, 100**2 * a**2 / 3]))

def exact_initial_rms():
    a = np.deg2rad(0.3)
    return float(np.sqrt(8 / 9 * (48 + 2 * 100**2 * (1 - np.sin(a) / a))))

def propagate_initial_moment(model, slots=560):
    if not isinstance(slots, int) or slots < 0:
        raise ValueError("slots must be a nonnegative integer")
    p = initial_second_moment()
    result = [p.copy()]
    matrices = model["matrices"]
    for slot in range(slots):
        m = matrices[slot % len(matrices)]
        p = m @ p @ m.T
        result.append(p.copy())
    return np.array(result)

def periodic_second_moment(model, bearing_std_deg=0.001, actuation_std=0.0):
    if any((not np.isfinite(s) or s < 0 for s in (bearing_std_deg, actuation_std))):
        raise ValueError("noise standard deviations must be finite and nonnegative")
    matrices = model["matrices"]
    innovations = [
        np.deg2rad(bearing_std_deg) ** 2 * g @ g.T for g in model["injections"]
    ]
    if actuation_std == 0:
        a = model["cycle"]
        rho = model["cycle_spectral_radius"] ** 2
        if rho >= 1:
            raise ValueError("local second moment is not stable")
        q_cycle = np.zeros((DIMENSION, DIMENSION))
        for m, q in zip(matrices, innovations):
            q_cycle = m @ q_cycle @ m.T + q
        endpoint = solve_discrete_lyapunov(a, q_cycle)
    else:
        operator = np.eye(DIMENSION**2)
        cycle_q = np.zeros(DIMENSION**2)
        for m, q in zip(matrices, innovations):
            step = covariance_operator(m, m - np.eye(DIMENSION), actuation_std)
            effective_q = q + actuation_std**2 * np.diag(np.diag(q))
            cycle_q = step @ cycle_q + effective_q.reshape(-1, order="F")
            operator = step @ operator
        rho = float(np.max(np.abs(np.linalg.eigvals(operator))))
        if rho >= 1:
            raise ValueError("local second moment is not stable")
        endpoint = np.linalg.solve(np.eye(DIMENSION**2) - operator, cycle_q).reshape(
            DIMENSION, DIMENSION, order="F"
        )
    endpoint = (endpoint + endpoint.T) / 2
    (phases, p) = ([], endpoint.copy())
    for m, q in zip(matrices, innovations):
        p = second_moment_step(p, m, m - np.eye(DIMENSION), q, actuation_std)
        phases.append(p)
    return {
        "phase_covariances": phases,
        "cycle_spectral_radius": float(rho),
        "phase_rms_m": [rms_from_covariance(p) for p in phases],
        "relative_residual": float(
            np.linalg.norm(p - endpoint) / max(np.linalg.norm(endpoint), 1e-300)
        ),
    }

def fixed_link_bias_response(model, bias_std_deg=0.001):
    if not np.isfinite(bias_std_deg) or bias_std_deg < 0:
        raise ValueError("bias standard deviation must be finite and nonnegative")
    if model["cycle_spectral_radius"] >= 1:
        raise ValueError("local cycle is not stable")
    loading = np.zeros_like(model["injections"][0])
    for m, g in zip(model["matrices"], model["injections"]):
        loading = m @ loading + g
    endpoint = np.linalg.solve(np.eye(DIMENSION) - model["cycle"], loading)
    (phases, current) = ([], endpoint.copy())
    for m, g in zip(model["matrices"], model["injections"]):
        current = m @ current + g
        phases.append(current)
    moments = [np.deg2rad(bias_std_deg) ** 2 * s @ s.T for s in phases]
    return {
        "phase_loadings": phases,
        "phase_covariances": moments,
        "phase_rms_m": [rms_from_covariance(p) for p in moments],
        "relative_residual": float(
            np.linalg.norm(current - endpoint) / max(np.linalg.norm(endpoint), 1e-300)
        ),
    }

def polar_basis(drone_id):
    theta = np.deg2rad(40 * (drone_id - 1))
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def state_slice(drone_id):
    return slice(2 * (drone_id - 2), 2 * (drone_id - 1))

def observation_derivative(receiver_id, transmitter_ids):
    q = nominal_positions()
    rows = []
    for a, b in combinations(transmitter_ids, 2):
        (u, v) = (q[a] - q[receiver_id], q[b] - q[receiver_id])
        (nu, nv) = (np.linalg.norm(u), np.linalg.norm(v))
        cosine = np.dot(u, v) / (nu * nv)
        sine = np.sqrt(1 - cosine**2)
        du = -(v / (nu * nv) - cosine * u / nu**2) / sine
        dv = -(u / (nu * nv) - cosine * v / nv**2) / sine
        row = np.zeros(DIMENSION)
        row[state_slice(receiver_id)] = -(du + dv) @ polar_basis(receiver_id)
        for drone_id, derivative in ((a, du), (b, dv)):
            if drone_id >= 2:
                row[state_slice(drone_id)] = derivative @ polar_basis(drone_id)
        rows.append(row)
    return np.array(rows)

def cycle_matrix(matrices):
    product = np.eye(DIMENSION)
    for matrix in matrices:
        product = matrix @ product
    return product

def ray_angle_derivative(directions):
    derivative = []
    for a, b in combinations(range(len(directions)), 2):
        if abs(np.sin(directions[a] - directions[b])) < 1e-12:
            raise ValueError("Unsigned angle derivative is undefined at 0 or pi")
        delta = np.arctan2(
            np.sin(directions[a] - directions[b]), np.cos(directions[a] - directions[b])
        )
        row = np.zeros(len(directions))
        (row[a], row[b]) = (np.sign(delta), -np.sign(delta))
        derivative.append(row)
    return np.array(derivative)

def covariance_operator(
    matrix: np.ndarray, command_matrix: np.ndarray, actuation_std: float
):
    n = matrix.shape[0]
    operator = np.kron(matrix, matrix)
    diagonal_rows = np.arange(n) * (n + 1)
    diagonal_map = (command_matrix[:, :, None] * command_matrix[:, None, :]).reshape(
        n, n * n, order="F"
    )
    operator[diagonal_rows, :] += actuation_std**2 * diagonal_map
    return operator

def second_moment_step(
    covariance: np.ndarray,
    matrix: np.ndarray,
    command_matrix: np.ndarray,
    innovation_covariance: np.ndarray,
    actuation_std: float,
) -> np.ndarray:
    deterministic = matrix @ covariance @ matrix.T
    multiplicative_state = actuation_std**2 * np.diag(
        np.diag(command_matrix @ covariance @ command_matrix.T)
    )
    innovation = innovation_covariance + actuation_std**2 * np.diag(
        np.diag(innovation_covariance)
    )
    return (
        deterministic
        + multiplicative_state
        + innovation
        + (deterministic + multiplicative_state + innovation).T
    ) / 2

def rms_from_covariance(covariance: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, np.trace(covariance) / 9)))

def wilson(successes: int, samples: int):
    z = 1.959963984540054
    probability = successes / samples
    denominator = 1 + z**2 / samples
    center = (probability + z**2 / (2 * samples)) / denominator
    half_width = (
        z
        * np.sqrt(probability * (1 - probability) / samples + z**2 / (4 * samples**2))
        / denominator
    )
    return (float(max(0, center - half_width)), float(min(1, center + half_width)))

def gaussian_root(covariance: np.ndarray) -> np.ndarray:
    symmetric = (covariance + covariance.T) / 2
    (eigenvalues, eigenvectors) = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if eigenvalues.min() < -1e-10 * scale:
        raise ValueError("saved covariance is not positive semidefinite")
    return eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))

def union_failure_bound(covariance: np.ndarray, epsilon_m: float) -> float:
    terms = []
    for drone in range(8):
        block = covariance[2 * drone : 2 * drone + 2, 2 * drone : 2 * drone + 2]
        maximum_variance = max(0.0, float(np.linalg.eigvalsh(block).max()))
        if maximum_variance == 0:
            terms.append(0.0)
        else:
            terms.append(float(np.exp(-(epsilon_m**2) / (2 * maximum_variance))))
    return float(min(1.0, sum(terms)))

def integrate_all_drone_probability(
    covariance: np.ndarray,
    thresholds_m: tuple[float, ...],
    samples: int,
    seed_sequence: list[int],
    batch_size: int = 20000,
):
    root = gaussian_root(covariance)
    rng = np.random.default_rng(np.random.SeedSequence(seed_sequence))
    successes = np.zeros(len(thresholds_m), dtype=np.int64)
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        state = rng.standard_normal((count, DIMENSION)) @ root.T
        maximum_error = np.sqrt((state.reshape(count, 8, 2) ** 2).sum(axis=2)).max(
            axis=1
        )
        successes += [(maximum_error < threshold).sum() for threshold in thresholds_m]
    return successes

CONDITIONS = {
    "exact": (0.0, 0.0, 0.0),
    "bearing_0.001deg": (0.001, 0.0, 0.0),
    "bearing_0.01deg": (0.01, 0.0, 0.0),
    "bearing_0.1deg": (0.1, 0.0, 0.0),
    "actuation_1pct": (0.0, 0.01, 0.0),
    "bearing_0.001deg_actuation_1pct": (0.001, 0.01, 0.0),
    "bias_0.001deg": (0.0, 0.0, 0.001),
}

def _run_trial(job):
    (output, schedule, gain, condition, trial, max_slots) = job
    (bearing, actuation, bias) = CONDITIONS[condition]
    noise = SimulationNoise(bearing, actuation, bias, seed=2026090500 + trial + 1)
    run = simulate_adjustment(
        random_positions(trial),
        schedule=schedule,
        settings=LocalSettings(gain=gain),
        max_slots=max_slots,
        noise=noise,
        retain_details=False,
    )
    destination = (
        Path(output) / f"{schedule}_gain{gain:g}" / condition / f"trial_{trial:03d}"
    )
    save_run(destination, run)
    summary = run["summary"]
    errors = np.array([h["max_position_error_m"] for h in run["history"]])
    bad = np.flatnonzero(errors >= 0.01)
    sustained = (
        0 if not len(bad) else int(bad[-1] + 1) if bad[-1] < len(errors) - 1 else None
    )
    failed = summary["failure"] is not None
    return {
        "schedule": schedule,
        "gain": gain,
        "condition": condition,
        "trial": trial,
        "status": summary["status"],
        "slots": summary["measurement_slots"],
        "first_cm_slot": summary["precision_thresholds"][1]["first_slot"],
        "sustained_cm_slot": sustained,
        "stopped": summary["status"] == "quiet_full_cycle",
        "failed": failed,
        "endpoint_cm": not failed and errors[-1] < 0.01,
        "endpoint_mm": not failed and errors[-1] < 0.001,
        "final_max_error_m": float(errors[-1]),
        "final_rms_m": summary["final_metrics"]["rms_position_error_m"],
    }

def run_random_experiments(
    output_dir, *, trials=100, workers=1, max_slots=560, conditions=None, methods=None
):
    if (
        not isinstance(trials, int)
        or trials < 1
        or (not isinstance(workers, int))
        or (workers < 1)
    ):
        raise ValueError("trials and workers must be positive integers")
    conditions = tuple(CONDITIONS) if conditions is None else tuple(conditions)
    if not conditions or any((c not in CONDITIONS for c in conditions)):
        raise ValueError("unknown or empty noise conditions")
    methods = (
        (("full", 0.5), ("two", 0.5), ("two", 1.0)) if methods is None else methods
    )
    jobs = [
        (str(output_dir), schedule, gain, condition, trial, max_slots)
        for (schedule, gain) in methods
        for condition in conditions
        if schedule == "two"
        or condition not in ("bias_0.001deg", "bearing_0.001deg_actuation_1pct")
        for trial in range(trials)
    ]
    if not jobs:
        raise ValueError("no methods match the requested conditions")
    rows = []

    def collect(iterator):
        for index, row in enumerate(iterator, 1):
            rows.append(row)
            print(
                f"{index}/{len(jobs)} {row['schedule']} gain={row['gain']} {row['condition']} trial={row['trial']} {row['status']}",
                flush=True,
            )

    if workers == 1:
        collect(map(_run_trial, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(_run_trial, jobs))
    groups = []
    for key in sorted({(r["schedule"], r["gain"], r["condition"]) for r in rows}):
        group = [r for r in rows if (r["schedule"], r["gain"], r["condition"]) == key]
        hits = [r["first_cm_slot"] for r in group if r["first_cm_slot"] is not None]
        stops = [r["slots"] for r in group if r["stopped"]]
        groups.append(
            {
                "schedule": key[0],
                "gain": key[1],
                "condition": key[2],
                "trials": len(group),
                "failed": sum((r["failed"] for r in group)),
                "stopped": len(stops),
                "first_cm_observed": len(hits),
                "endpoint_cm": sum((r["endpoint_cm"] for r in group)),
                "endpoint_mm": sum((r["endpoint_mm"] for r in group)),
                "median_first_cm_all_trials": float(np.median(hits))
                if len(hits) == len(group)
                else None,
                "median_stop_all_trials": float(np.median(stops))
                if len(stops) == len(group)
                else None,
                "median_final_max_error_m": float(
                    np.median([r["final_max_error_m"] for r in group])
                ),
            }
        )
    output = Path(output_dir)
    write_csv(output / "trials.csv", rows)
    write_json(output / "summary.json", groups)
    return groups

def analyze(output, samples=100000):
    rows = []
    for schedule, gain in (("full", 0.5), ("two", 0.5), ("two", 1.0)):
        model = linear_model(schedule, gain)
        path = output / f"{schedule}_gain{gain:g}"
        white = periodic_second_moment(model)
        result = {
            "model": model,
            "white_0.001deg": white,
            "exact_initial_root_expected_rms_squared_m": exact_initial_rms(),
            "scope": "Fixed nominal branches, active feedback, no clipping or hold. Gaussian precision probabilities apply to the local model only.",
        }
        if schedule == "two":
            result["white_0.001deg_actuation_1pct"] = periodic_second_moment(
                model, 0.001, 0.01
            )
            result["fixed_bias_0.001deg"] = fixed_link_bias_response(model)
            result["execution_stress"] = [
                {
                    "relative_std": std,
                    "cycle_spectral_radius": periodic_second_moment(model, 0, std)[
                        "cycle_spectral_radius"
                    ],
                }
                for std in (0.01, 0.05, 0.1)
            ]
        probabilities = []
        if samples > 0:
            for kind in ("white_0.001deg", "fixed_bias_0.001deg"):
                if kind not in result:
                    continue
                for phase, covariance in enumerate(
                    result[kind]["phase_covariances"], 1
                ):
                    hits = integrate_all_drone_probability(
                        covariance, (0.01, 0.001), samples, [20260906, phase]
                    )
                    for threshold, count in zip((0.01, 0.001), hits):
                        probabilities.append(
                            {
                                "kind": kind,
                                "phase": phase,
                                "threshold_m": threshold,
                                "samples": samples,
                                "probability": float(count / samples),
                                "mc_wilson_95": wilson(int(count), samples),
                                "failure_union_bound": union_failure_bound(
                                    covariance, threshold
                                ),
                            }
                        )
        result["local_gaussian_probabilities"] = probabilities
        write_json(path / "analysis.json", result)
        moments = propagate_initial_moment(model)
        np.savez_compressed(
            path / "matrices.npz",
            M=model["matrices"],
            G=model["injections"],
            cycle=model["cycle"],
            initial_moments=moments,
        )
        write_csv(
            path / "initial_moment_history.csv",
            [
                {"slot": slot, "root_expected_rms_squared_m": rms_from_covariance(p)}
                for (slot, p) in enumerate(moments)
            ],
        )
        rows.append(
            {
                "schedule": schedule,
                "gain": gain,
                "cycle_spectral_radius": model["cycle_spectral_radius"],
                "two_cycle_operator_norm": model["two_cycle_operator_norm"],
                "minimum_selection_gap": model["minimum_selection_gap"],
            }
        )
    write_json(output / "summary.json", rows)
    return rows

# 仿真与画图

def plot_results(output_dir, localization=None, identification=None, runs=()):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    colors = ("#2563A6", "#C75B39", "#36856C")
    q = nominal_positions()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.set_title("Q1.1  Position estimate", loc="left")
    if localization is not None:
        anchors = nominal_positions(localization["radius_m"])
        tx = list(localization["transmitters"])
        point = localization["result"].position
        ax.scatter(
            anchors[tx, 0],
            anchors[tx, 1],
            marker="^",
            s=55,
            color=colors[0],
            label="Transmitters",
        )
        ax.scatter(*point, marker="x", s=65, color=colors[1], label="Receiver estimate")
        for i in tx:
            ax.plot(
                [point[0], anchors[i, 0]],
                [point[1], anchors[i, 1]],
                color=colors[0],
                alpha=0.35,
            )
            ax.annotate(
                f"FY{i:02d}", anchors[i], xytext=(5, 5), textcoords="offset points"
            )
        ax.legend()
    ax.set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax = axes[0, 1]
    ax.set_title("Q1.2  Anonymous transmitter", loc="left")
    if identification is not None:
        hypotheses = sorted(
            (h for h in identification.hypotheses if h.candidates),
            key=lambda h: h.candidates[0].distance_to_ideal,
        )
        selected = identification.selected
        ax.barh(
            [f"FY{h.emitter_id:02d}" for h in hypotheses],
            [h.candidates[0].distance_to_ideal for h in hypotheses],
            color=[
                colors[1]
                if selected and h.emitter_id == selected.emitter_id
                else colors[0]
                for h in hypotheses
            ],
        )
        ax.invert_yaxis()
    ax.set_xlabel("Nearest candidate distance to ideal position (m)")
    ax = axes[1, 0]
    ax.set_title("Q1.3  Formation adjustment", loc="left")
    phi = np.linspace(0, 2 * np.pi, 361)
    ax.plot(100 * np.cos(phi), 100 * np.sin(phi), "--", color="0.65", linewidth=1)
    ax.scatter(q[:, 0], q[:, 1], marker="+", s=55, color="0.25", label="Ideal")
    if runs:
        states = runs[0][1]["positions"]
        ax.scatter(
            states[0, :, 0],
            states[0, :, 1],
            marker="o",
            facecolors="none",
            edgecolors=colors[1],
            label="Initial",
        )
        ax.scatter(
            states[-1, :, 0], states[-1, :, 1], s=15, color=colors[0], label="Final"
        )
        for i in range(2, 10):
            ax.plot(
                states[:, i, 0],
                states[:, i, 1],
                color=colors[0],
                alpha=0.55,
                linewidth=1,
            )
        ax.legend()
    ax.set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax = axes[1, 1]
    ax.set_title("Q1.3  Error by measurement slot", loc="left")
    for color, (label, run) in zip(colors, runs):
        history = run["history"]
        ax.semilogy(
            [r["slot"] for r in history],
            np.maximum([r["max_position_error_m"] for r in history], 1e-12),
            color=color,
            label=label,
        )
    ax.axhline(0.01, color="0.5", linestyle="--", linewidth=1, label="1 cm")
    ax.set(xlabel="Measurement slot", ylabel="Maximum position error (m)")
    ax.grid(alpha=0.2, which="both")
    ax.legend()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"q1.{extension}", dpi=300)
    plt.close(fig)

def plot_random_results(output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    with (output / "trials.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    methods = sorted({(r["schedule"], r["gain"]) for r in rows})
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for schedule, gain in methods:
        errors, stops = [], []
        for condition in conditions:
            group = [
                r
                for r in rows
                if (r["schedule"], r["gain"], r["condition"])
                == (schedule, gain, condition)
            ]
            errors.append(
                np.median([float(r["final_max_error_m"]) for r in group])
                if group
                else np.nan
            )
            stops.append(
                np.mean([r["stopped"] == "True" for r in group]) if group else np.nan
            )
        label = f"{schedule}, gain={gain}"
        axes[0].semilogy(
            range(len(conditions)), np.maximum(errors, 1e-12), "o-", label=label
        )
        axes[1].plot(range(len(conditions)), stops, "o-", label=label)
    for ax in axes:
        ax.set_xticks(range(len(conditions)), conditions, rotation=35, ha="right")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Median terminal maximum error (m)")
    axes[1].set(ylabel="Protocol stop fraction", ylim=(-0.05, 1.05))
    fig.savefig(output / "random_experiments.png", dpi=300)
    fig.savefig(output / "random_experiments.pdf")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        nargs="?",
        default="all",
        choices=("all", "q1_1", "q1_2", "q1_3_1", "q1_3_2", "analysis", "random"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    parser.add_argument("--receiver", type=int, default=3)
    parser.add_argument("--transmitters", nargs=2, type=int, default=(1, 5))
    parser.add_argument(
        "--angles-deg", nargs=3, type=float, help="01,0U,1U 或 0a,0b,ab"
    )
    parser.add_argument("--radius", type=float, default=100.0)
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--max-slots", type=int, default=560)
    parser.add_argument("--bearing-std-deg", type=float, default=0.0)
    parser.add_argument("--actuation-std", type=float, default=0.0)
    parser.add_argument("--bias-std-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20220905)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS))
    parser.add_argument("--probability-samples", type=int, default=100000)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    if args.task == "random":
        result = run_random_experiments(
            output,
            trials=args.trials,
            workers=args.workers,
            max_slots=args.max_slots,
            conditions=args.conditions,
        )
        if not args.no_plot:
            plot_random_results(output)
    elif args.task == "analysis":
        if args.probability_samples < 0:
            parser.error("probability-samples must be nonnegative")
        result = analyze(output, args.probability_samples)
    else:
        first = second = None
        runs = []
        result = {}
        angles = None if args.angles_deg is None else np.deg2rad(args.angles_deg)
        if args.task in ("all", "q1_1"):
            first = solve_q1_1(args.receiver, args.transmitters, angles, args.radius)
            result["q1_1"] = first
        if args.task in ("all", "q1_2"):
            second = solve_q1_2(args.receiver, angles, args.radius)
            result["q1_2"] = second
        methods = (
            [("full", 0.5), ("two", 0.5), ("two", 1.0)]
            if args.task == "all"
            else [("full", args.gain)]
            if args.task == "q1_3_1"
            else [("two", args.gain)]
            if args.task == "q1_3_2"
            else []
        )
        noise = SimulationNoise(
            args.bearing_std_deg, args.actuation_std, args.bias_std_deg, args.seed
        )
        for schedule, gain in methods:
            run = simulate_adjustment(
                schedule=schedule,
                settings=LocalSettings(gain=gain),
                max_slots=args.max_slots,
                noise=noise,
            )
            name = f"{schedule}_gain{gain:g}"
            save_run(output / name, run)
            result[name] = run["summary"]
            runs.append((f"{schedule}, gain={gain:g}", run))
        write_json(output / "q1.json", result)
        if not args.no_plot:
            plot_results(output, first, second, runs)
    for key, value in result.items() if isinstance(result, dict) else []:
        if isinstance(value, dict) and "measurement_slots" in value:
            hit = value["precision_thresholds"][1]["first_slot"]
            print(
                f"{key}: first cm={hit}, slots={value['measurement_slots']}, {value['status']}"
            )
    print(f"Results: {output.resolve()}")

if __name__ == "__main__":
    main()
