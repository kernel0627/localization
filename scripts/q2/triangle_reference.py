"""Core geometry for the Q2 triangle-reference route.

The module fixes FY11=(0, 0), FY15=(4, 0), and the canonical upper-side
apex FY01=(2, 2*sqrt(3)).  It contains only angle observations, local
position estimation, and diagnostics needed by a batch runner.  The one
observation-generation helper accepts a supplied static state by design;
the estimation APIs do not accept true positions, movement commands, or
controller state.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares


FloatArray = NDArray[np.float64]

N_POINTS = 15
REFERENCE_IDS = (1, 11, 15)
RECEIVER_IDS = tuple(
    identifier for identifier in range(1, N_POINTS + 1) if identifier not in REFERENCE_IDS
)
ANCHOR_PAIRS = ((0, 1), (0, 2), (1, 2))
ANGLE_CROSS_TOL = 1e-10
ANGLE_RESIDUAL_TOLERANCE = 1e-8
MAX_NFEV = 500


class GeometryError(ValueError):
    """Raised when an angle or its ordinary Jacobian is undefined."""


def _as_position(point: ArrayLike, *, name: str = "position") -> FloatArray:
    value = np.asarray(point, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be one finite 2-D point")
    return value


def _as_positions(points: ArrayLike, *, name: str = "positions") -> FloatArray:
    value = np.asarray(points, dtype=float)
    if value.shape != (N_POINTS, 2) or not np.isfinite(value).all():
        raise ValueError(f"{name} must have shape (15, 2) and be finite")
    return value


def template() -> FloatArray:
    """Return the canonical five-layer template, numbered FY01 through FY15."""
    points: list[tuple[float, float]] = []
    for row in range(5):
        for column in range(row + 1):
            points.append(
                (
                    float(column - row / 2.0 + 2.0),
                    float(-np.sqrt(3.0) * row / 2.0 + 2.0 * np.sqrt(3.0)),
                )
            )
    result = np.asarray(points, dtype=float)
    # Keep the gauge contract explicit at the API boundary.
    assert np.allclose(result[0], (2.0, 2.0 * np.sqrt(3.0)))
    assert np.allclose(result[10], (0.0, 0.0))
    assert np.allclose(result[14], (4.0, 0.0))
    return result


TARGET_TEMPLATE = template()
TARGET_ANCHORS = TARGET_TEMPLATE[np.asarray(REFERENCE_IDS) - 1].copy()


def _validate_anchor_array(anchors: ArrayLike) -> FloatArray:
    value = np.asarray(anchors, dtype=float)
    if value.shape != (3, 2) or not np.isfinite(value).all():
        raise ValueError("anchors must have shape (3, 2) and be finite")
    return value


def _angle_and_gradient(
    receiver: FloatArray,
    first: FloatArray,
    second: FloatArray,
    *,
    require_regular: bool,
) -> tuple[float, float, FloatArray | None]:
    u = first - receiver
    v = second - receiver
    u2 = float(np.dot(u, u))
    v2 = float(np.dot(v, v))
    if u2 <= 1e-24 or v2 <= 1e-24:
        raise GeometryError("receiver coincides with a transmitter")
    cross = float(u[0] * v[1] - u[1] * v[0])
    dot = float(np.dot(u, v))
    normalized_cross = abs(cross) / np.sqrt(u2 * v2)
    angle = float(np.arctan2(abs(cross), dot))
    if normalized_cross <= ANGLE_CROSS_TOL:
        if require_regular:
            raise GeometryError("unsigned angle is undefined at 0 or pi")
        return angle, normalized_cross, None

    sign = np.sign(cross)
    rotation_u = np.array([-u[1], u[0]])
    rotation_v = np.array([-v[1], v[0]])
    d_first = -sign * rotation_u / u2
    d_second = sign * rotation_v / v2
    receiver_gradient = -d_first - d_second
    return angle, normalized_cross, receiver_gradient


def unsigned_angle(receiver: ArrayLike, first: ArrayLike, second: ArrayLike) -> float:
    """Return the labeled unsigned angle in radians, in ``[0, pi]``."""
    return _angle_and_gradient(
        _as_position(receiver, name="receiver"),
        _as_position(first, name="first transmitter"),
        _as_position(second, name="second transmitter"),
        require_regular=False,
    )[0]


def _angle_rows(receiver: FloatArray, anchors: FloatArray) -> tuple[FloatArray, FloatArray]:
    values: list[float] = []
    gradients: list[FloatArray] = []
    for first, second in ANCHOR_PAIRS:
        angle, _, gradient = _angle_and_gradient(
            receiver,
            anchors[first],
            anchors[second],
            require_regular=False,
        )
        values.append(angle)
        if gradient is None:
            gradients.append(np.array([np.nan, np.nan]))
        else:
            gradients.append(gradient)
    return np.asarray(values, dtype=float), np.asarray(gradients, dtype=float)


def receiver_angles(position: ArrayLike, anchors: ArrayLike) -> FloatArray:
    """Return angles for anchor pairs ``(0,1), (0,2), (1,2)``."""
    point = _as_position(position, name="position")
    points = _validate_anchor_array(anchors)
    return _angle_rows(point, points)[0]


def angle_jacobian(
    position: ArrayLike,
    anchors: ArrayLike,
    *,
    allow_degenerate: bool = False,
) -> FloatArray:
    """Return the three-row angle Jacobian with respect to receiver position.

    At an ideal collinear row, ``allow_degenerate=True`` returns NaNs for that
    row so callers can select a regular subset; the default raises explicitly.
    """
    point = _as_position(position, name="position")
    points = _validate_anchor_array(anchors)
    gradients = _angle_rows(point, points)[1]
    if not allow_degenerate and not np.isfinite(gradients).all():
        raise GeometryError("unsigned angle Jacobian is undefined at 0 or pi")
    return gradients


def bootstrap_angles(positions: ArrayLike) -> FloatArray:
    """Generate the two observed base angles from a 15-point static state.

    The state must use the canonical fixed FY11/FY15 positions.  The angles
    are still unsigned; :func:`estimate_apex` selects the canonical upper
    half-plane branch.
    """
    points = _as_positions(positions)
    if not np.allclose(points[10], TARGET_ANCHORS[1]) or not np.allclose(
        points[14], TARGET_ANCHORS[2]
    ):
        raise ValueError("bootstrap_angles expects canonical fixed FY11 and FY15")
    return np.asarray(
        [
            unsigned_angle(points[10], points[0], points[14]),
            unsigned_angle(points[14], points[0], points[10]),
        ],
        dtype=float,
    )


def _as_bootstrap_angles(observed: ArrayLike) -> tuple[float, float]:
    values = np.asarray(observed, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("observed bootstrap angles must have shape (2) and be finite")
    if np.any(values <= 0.0) or np.any(values >= np.pi):
        raise ValueError("bootstrap angles must lie strictly between 0 and pi")
    if float(values.sum()) >= np.pi:
        raise ValueError(
            "bootstrap angles must form a nondegenerate triangle; "
            "the returned apex uses the canonical upper branch"
        )
    return float(values[0]), float(values[1])


def bootstrap_from_angles(
    alpha: ArrayLike | float, beta: float | None = None
) -> FloatArray:
    """Recover the apex from two base angles on the canonical upper branch.

    ``alpha`` may be a length-two sequence when ``beta`` is omitted, or the
    two scalar angles may be passed separately.  The base is FY11--FY15 with
    length 4.  No true position is used.
    """
    if beta is None:
        alpha_value, beta_value = _as_bootstrap_angles(alpha)
    else:
        alpha_value, beta_value = _as_bootstrap_angles((alpha, beta))
    denominator = float(np.sin(alpha_value + beta_value))
    if denominator <= 1e-12:
        raise ValueError("bootstrap angles are too close to a degenerate branch")
    return np.asarray(
        [
            4.0
            * np.sin(beta_value)
            * np.cos(alpha_value)
            / denominator,
            4.0
            * np.sin(beta_value)
            * np.sin(alpha_value)
            / denominator,
        ],
        dtype=float,
    )


def estimate_apex(observed2: ArrayLike) -> FloatArray:
    """Alias for :func:`bootstrap_from_angles` using a length-two input."""
    return bootstrap_from_angles(observed2)


def _validate_receiver_id(receiver_id: int) -> int:
    if not isinstance(receiver_id, (int, np.integer)):
        raise ValueError("receiver_id must be an integer")
    identifier = int(receiver_id)
    if identifier < 1 or identifier > N_POINTS:
        raise ValueError("receiver_id must lie in 1..15")
    if identifier in REFERENCE_IDS:
        raise ValueError("reference transmitters are not receiver targets")
    return identifier


def _regular_rows_at_target(receiver_id: int) -> tuple[tuple[int, int], ...]:
    point = TARGET_TEMPLATE[receiver_id - 1]
    rows: list[tuple[int, FloatArray]] = []
    for index, (first, second) in enumerate(ANCHOR_PAIRS):
        _, _, gradient = _angle_and_gradient(
            point,
            TARGET_ANCHORS[first],
            TARGET_ANCHORS[second],
            require_regular=False,
        )
        if gradient is not None:
            rows.append((index, gradient))
    if len(rows) < 2:
        raise GeometryError(f"FY{receiver_id:02d} has fewer than two regular target angles")

    candidates: list[tuple[float, tuple[int, int], FloatArray]] = []
    for first, second in combinations(rows, 2):
        indices = (first[0], second[0])
        matrix = np.vstack((first[1], second[1]))
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        candidates.append((float(singular_values[-1]), indices, matrix))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return (candidates[0][1],)


def _selected_angle_indices(receiver_id: int) -> tuple[int, int]:
    return _regular_rows_at_target(_validate_receiver_id(receiver_id))[0]


SELECTED_ANGLE_INDICES = {
    identifier: _selected_angle_indices(identifier) for identifier in RECEIVER_IDS
}
SELECTED_ANGLE_PAIRS = {
    identifier: tuple(ANCHOR_PAIRS[index] for index in indices)
    for identifier, indices in SELECTED_ANGLE_INDICES.items()
}


def _as_observed_receiver_angles(observed3: ArrayLike) -> FloatArray:
    values = np.asarray(observed3, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("observed receiver angles must have shape (3,) and be finite")
    if np.any(values < 0.0) or np.any(values > np.pi):
        raise ValueError("observed receiver angles must lie in [0, pi]")
    return values


def _estimate_failure(
    initial: FloatArray,
    message: str,
    *,
    nfev: int = 0,
    residual: float = float("inf"),
) -> dict[str, Any]:
    return {
        "position": initial.copy(),
        "success": False,
        "max_angle_residual": float(residual),
        "nfev": int(nfev),
        "message": message,
    }


def estimate_receiver(
    receiver_id: int,
    observed3: ArrayLike,
    initial: ArrayLike | None = None,
) -> dict[str, Any]:
    """Estimate one receiver from its three labeled reference angles.

    The solver uses the two angle rows preselected at the canonical target
    for this receiver.  It then evaluates all three angles at the estimate;
    ``success`` requires the solver to converge and the complete-triangle
    maximum angle residual to be within ``ANGLE_RESIDUAL_TOLERANCE``.
    Geometry failures are returned as ``success=False`` records rather than
    replaced by a truth coordinate.
    """
    identifier = _validate_receiver_id(receiver_id)
    observed = _as_observed_receiver_angles(observed3)
    initial_point = (
        TARGET_TEMPLATE[identifier - 1].copy()
        if initial is None
        else _as_position(initial, name="initial")
    )
    indices = SELECTED_ANGLE_INDICES[identifier]
    try:
        # Validate the starting point before invoking SciPy, so failures are
        # explicit and no fallback coordinate is silently introduced.
        _angle_rows(initial_point, TARGET_ANCHORS)

        def residual_function(point: FloatArray) -> FloatArray:
            values, _ = _angle_rows(point, TARGET_ANCHORS)
            return values[list(indices)] - observed[list(indices)]

        def jacobian_function(point: FloatArray) -> FloatArray:
            _, gradients = _angle_rows(point, TARGET_ANCHORS)
            selected = gradients[list(indices)]
            if not np.isfinite(selected).all():
                raise GeometryError("selected angle Jacobian reached a 0 or pi row")
            return selected

        result = least_squares(
            residual_function,
            initial_point,
            jac=jacobian_function,
            method="trf",
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=MAX_NFEV,
        )
        position = np.asarray(result.x, dtype=float)
        predicted = receiver_angles(position, TARGET_ANCHORS)
        max_residual = float(np.max(np.abs(predicted - observed)))
        success = bool(result.success and max_residual <= ANGLE_RESIDUAL_TOLERANCE)
        message = str(result.message)
        if result.success and not success:
            message = (
                f"solver converged but complete-triangle residual "
                f"{max_residual:.3e} exceeds {ANGLE_RESIDUAL_TOLERANCE:.3e}"
            )
        return {
            "position": position,
            "success": success,
            "max_angle_residual": max_residual,
            "nfev": int(result.nfev),
            "message": message,
        }
    except (GeometryError, ValueError, FloatingPointError) as error:
        return _estimate_failure(initial_point, f"geometry/solver failure: {error}")


def selected_angle_pairs(receiver_id: int) -> tuple[tuple[int, int], ...]:
    """Return the canonical preselected transmitter pairs for one receiver."""
    return tuple(
        (REFERENCE_IDS[first], REFERENCE_IDS[second])
        for first, second in SELECTED_ANGLE_PAIRS[_validate_receiver_id(receiver_id)]
    )


def selected_angle_sigma_min(receiver_id: int) -> float:
    """Return the target local-block minimum singular value in rad/d."""
    identifier = _validate_receiver_id(receiver_id)
    point = TARGET_TEMPLATE[identifier - 1]
    gradients = []
    for angle_index in SELECTED_ANGLE_INDICES[identifier]:
        first, second = ANCHOR_PAIRS[angle_index]
        gradient = _angle_and_gradient(
            point,
            TARGET_ANCHORS[first],
            TARGET_ANCHORS[second],
            require_regular=True,
        )[2]
        assert gradient is not None
        gradients.append(gradient)
    return float(np.linalg.svd(np.vstack(gradients), compute_uv=False)[-1])
