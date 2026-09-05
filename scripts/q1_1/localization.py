"""Question 1(1): localize one receiver from pairwise visual angles."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares


FloatArray = NDArray[np.float64]


class GeometryError(ValueError):
    """Raised when an angle is undefined for the supplied geometry."""


@dataclass(frozen=True)
class LocalizationResult:
    """Numerical estimate and diagnostics for one receiver."""

    position: FloatArray
    success: bool
    cost: float
    residual_norm: float
    jacobian_singular_values: FloatArray
    condition_number: float
    nfev: int
    message: str


def polar_to_cartesian(radius: float, angle_deg: float) -> FloatArray:
    """Convert the problem's polar coordinates to a Cartesian point."""

    angle = np.deg2rad(angle_deg)
    return np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=float)


def fy_position(drone_id: int, radius: float = 100.0) -> FloatArray:
    """Return an ideal circular-formation position for FY00--FY09."""

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


def _receiver_vectors(receiver: ArrayLike, anchors: FloatArray) -> tuple[FloatArray, FloatArray]:
    point = np.asarray(receiver, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("receiver must be one finite 2-D point")
    vectors = anchors - point
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-12):
        raise GeometryError("receiver coincides with a transmitter, so an angle is undefined")
    return vectors, lengths


def pairwise_angles(receiver: ArrayLike, anchors: ArrayLike) -> FloatArray:
    """Return all labeled pairwise transmitter angles in radians.

    The order is lexicographic: (0, 1), (0, 2), ..., (n-2, n-1).
    Each angle lies in [0, pi] and therefore does not require a global heading.
    """

    points = _as_anchor_array(anchors)
    vectors, lengths = _receiver_vectors(receiver, points)
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
        raise ValueError(f"expected {expected} pairwise angles, got shape {angles.shape}")
    if not np.isfinite(angles).all() or np.any(angles < 0.0) or np.any(angles > np.pi):
        raise ValueError("observed angles must be finite and lie in [0, pi]")
    return np.cos(angles)


def cosine_residuals(
    receiver: ArrayLike,
    anchors: ArrayLike,
    observed_angles: ArrayLike,
) -> FloatArray:
    """Evaluate dot-product cosine residuals for exact-constraint diagnostics."""

    points = _as_anchor_array(anchors)
    observed_cosines = _observed_cosines(observed_angles, points.shape[0])
    vectors, lengths = _receiver_vectors(receiver, points)
    predicted = [
        np.dot(vectors[first], vectors[second])
        / (lengths[first] * lengths[second])
        for first, second in _pairs(points.shape[0])
    ]
    return np.asarray(predicted, dtype=float) - observed_cosines


def angle_residuals(
    receiver: ArrayLike,
    anchors: ArrayLike,
    observed_angles: ArrayLike,
) -> FloatArray:
    """Return predicted-minus-observed pairwise angles in radians.

    The predicted angles are still obtained from the normalized dot product.
    Using angle-domain residuals keeps the least-squares objective consistent
    with an additive Gaussian angle-noise model.
    """

    points = _as_anchor_array(anchors)
    angles = np.asarray(observed_angles, dtype=float)
    _observed_cosines(angles, points.shape[0])
    return pairwise_angles(receiver, points) - angles


def cosine_jacobian(
    receiver: ArrayLike,
    anchors: ArrayLike,
    observed_angles: ArrayLike | None = None,
) -> FloatArray:
    """Analytic Jacobian of the cosine residuals with respect to (x, y)."""

    points = _as_anchor_array(anchors)
    if observed_angles is not None:
        _observed_cosines(observed_angles, points.shape[0])
    vectors, lengths = _receiver_vectors(receiver, points)
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
    receiver: ArrayLike,
    anchors: ArrayLike,
    observed_angles: ArrayLike | None = None,
) -> FloatArray:
    """Analytic Jacobian of angle residuals with respect to (x, y)."""

    points = _as_anchor_array(anchors)
    if observed_angles is not None:
        _observed_cosines(observed_angles, points.shape[0])
    vectors, lengths = _receiver_vectors(receiver, points)
    cosines = np.asarray(
        [
            np.dot(vectors[first], vectors[second])
            / (lengths[first] * lengths[second])
            for first, second in _pairs(points.shape[0])
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


def local_observability(receiver: ArrayLike, anchors: ArrayLike) -> FloatArray:
    """Return singular values of the angle Jacobian at a receiver point."""

    singular_values = np.linalg.svd(angle_jacobian(receiver, anchors), compute_uv=False)
    return np.asarray(singular_values, dtype=float)


def localize_receiver(
    anchors: ArrayLike,
    observed_angles: ArrayLike,
    initial_position: ArrayLike,
    *,
    max_nfev: int = 500,
) -> LocalizationResult:
    """Find the local receiver position that best matches all measured angles.

    For Question 1(1), the receiver identity is known and its position is only
    slightly perturbed. Its ideal formation position is therefore the intended
    initial value and selects the nearby physical solution.
    """

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


def find_local_candidates(
    anchors: ArrayLike,
    observed_angles: ArrayLike,
    initial_positions: ArrayLike,
    *,
    residual_tolerance: float = 1e-8,
    merge_distance: float = 1e-5,
) -> tuple[LocalizationResult, ...]:
    """Collect distinct zero-residual candidates reached from several starts.

    This is a diagnostic rather than the Question 1(1) selection rule. The
    receiver's known identity and small-deviation assumption select the
    candidate closest to its ideal formation position.
    """

    starts = np.asarray(initial_positions, dtype=float)
    if starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError("initial_positions must have shape (m, 2)")
    candidates: list[LocalizationResult] = []
    for initial in starts:
        try:
            candidate = localize_receiver(anchors, observed_angles, initial)
        except GeometryError:
            continue
        if not candidate.success or candidate.residual_norm > residual_tolerance:
            continue
        if any(
            np.linalg.norm(candidate.position - known.position) <= merge_distance
            for known in candidates
        ):
            continue
        candidates.append(candidate)
    return tuple(sorted(candidates, key=lambda item: item.residual_norm))
