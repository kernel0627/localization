"""Finite identity enumeration using the Question 1(1) angle model.

Inputs are exact, labeled angles (01, 0U, 1U), in radians. For each identity,
the nominal-start local solve is supplemented with isoptic-circle intersections.
This prevents failed local optimization from being mistaken for impossibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from scripts.q1_1.localization import (
    FloatArray,
    GeometryError,
    fy_position,
    localize_receiver,
    pairwise_angles,
)


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
    # A coincident-circle branch may contain infinitely many positions. Its
    # distance to the full circle is a conservative lower bound for valid arcs.
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
    """Enumerate the four signed circle pairs; verify all three unsigned angles.

    Coordinates are normalized by R. Every circle passes through O, so subtracting
    their equations gives a line through O. Its second intersection is explicit.
    Coincident circles are retained as a distance bound, never discarded as empty.
    """

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
                # Allow for floating point circle-coincidence tolerance.
                continuum_bounds.append(
                    max(0.0, float(distance) - radius * 1e-9 * scale)
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
                np.linalg.norm(position - c.position) <= radius * 1e-7
                for c in candidates
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
    return candidates, continuum_bounds


def identify_anonymous_emitter(
    receiver_id: int,
    observed_angles: ArrayLike,
    *,
    radius: float = 100.0,
    residual_tolerance: float = 1e-8,
    distance_tie_tolerance: float | None = None,
    max_nfev: int = 500,
) -> IdentificationResult:
    """Enumerate the seven legal IDs and select the nearest consistent position.

    The receiver ID is known; the anonymous ID and true position are not inputs.
    The exact-angle geometric completion is not a noisy-data association method.
    Ties and possibly nearer continuous branches produce an explicit abstention.
    There is no invented physical displacement cutoff.
    """

    if (
        isinstance(receiver_id, bool)
        or not isinstance(receiver_id, Integral)
        or not 2 <= receiver_id <= 9
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
        radius * 1e-7 if distance_tie_tolerance is None else distance_tie_tolerance
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

        candidates, continuum_bounds = _circle_candidates(
            emitter_id,
            angles,
            ideal,
            radius,
            residual_tolerance,
        )
        if local_candidate is not None:
            match = next(
                (
                    i
                    for i, c in enumerate(candidates)
                    if np.linalg.norm(c.position - local_candidate.position)
                    <= radius * 1e-7
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
        (c for h in hypotheses for c in h.candidates),
        key=lambda c: c.distance_to_ideal,
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
        status, selected = "degenerate", None
    elif margin is not None and margin <= tie_tolerance:
        status, selected = "ambiguous", None
    else:
        status, selected = "selected", best
    return IdentificationResult(
        receiver_id,
        status,
        selected,
        tuple(hypotheses),
        margin,
        identity_margin,
        continuum_bound,
    )
