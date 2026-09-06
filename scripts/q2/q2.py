"""第二问自包含代码：参考建立、并行调整与配套验证。

只依赖 Python 标准库、NumPy 和 SciPy；复制本文件即可独立运行。
以相邻间距 d=1 的二维规范坐标建模，估计器只读角度及参考信息。
仿真真值负责生成观测、施加理想动作和离线评分。

用法：
    conda run -n agent python q2.py triangle --output-dir results/triangle
    conda run -n agent python q2.py protocols --output-dir results/protocols
    conda run -n agent python q2.py noise --output-dir results/noise
    conda run -n agent python q2.py --help

文件顺序：几何核心、基础闭环、参考残差、几何边界、校准预算、
测角噪声与增益、协议对照、E0 几何探针、统一命令入口。
同名分析辅助函数和常量使用分区前缀，避免互相覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from itertools import product
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from pathlib import Path
from scipy.optimize import brentq
from scipy.optimize import least_squares
from scipy.stats import norm
from typing import Any
from typing import Iterable
import argparse
import csv
import itertools
import json
import numpy as np
import sys

# ============================================================================
# core: triangle_reference
# ============================================================================

FloatArray = NDArray[np.float64]

N_POINTS = 15
REFERENCE_IDS = (1, 11, 15)
RECEIVER_IDS = tuple(
    identifier
    for identifier in range(1, N_POINTS + 1)
    if identifier not in REFERENCE_IDS
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


def _angle_rows(
    receiver: FloatArray, anchors: FloatArray
) -> tuple[FloatArray, FloatArray]:
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
            4.0 * np.sin(beta_value) * np.cos(alpha_value) / denominator,
            4.0 * np.sin(beta_value) * np.sin(alpha_value) / denominator,
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
        raise GeometryError(
            f"FY{receiver_id:02d} has fewer than two regular target angles"
        )

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


# ============================================================================
# triangle: run_triangle_reference
# ============================================================================


triangle_DEFAULT_OUTPUT = Path("outputs/q2/triangle_reference")
triangle_PERTURBATION_LEVELS = (0.0, 0.01, 0.05, 0.10, 0.20)
triangle_SEEDS = (11, 23, 47)
triangle_GAINS = (1.0, 0.5)
MAX_ROUNDS = 30
ANGLE_TOLERANCE = float(ANGLE_RESIDUAL_TOLERANCE)
SHAPE_TOLERANCE = 1e-6
A_ID, B_ID, C_ID = 11, 15, 1
A_INDEX, B_INDEX, C_INDEX = A_ID - 1, B_ID - 1, C_ID - 1


@dataclass(frozen=True)
class InitialState:
    """One perturbed, AB-normalized true state and its construction metrics."""

    positions: np.ndarray
    rho: float
    seed: int | None
    actual_spacing: float
    raw_base_length: float
    raw_max_perturbation: float
    normalized_max_deviation: float
    normalized_rms_deviation: float
    mirrored_apex_side: bool


def triangle_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def triangle_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): triangle_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [triangle_json_value(item) for item in value]
    return value


def _normalize_by_ab(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Set A=(0,0), B=(4,0), retaining the orientation of the input."""

    value = np.asarray(points, dtype=float)
    base = value[B_INDEX] - value[A_INDEX]
    length = float(np.linalg.norm(base))
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError("perturbed A/B pair is degenerate")
    ex = base / length
    ey = np.array([-ex[1], ex[0]], dtype=float)
    vectors = value - value[A_INDEX]
    coordinates = np.column_stack((vectors @ ex, vectors @ ey))
    normalized = coordinates * (4.0 / length)
    normalized[A_INDEX] = (0.0, 0.0)
    normalized[B_INDEX] = (4.0, 0.0)
    return normalized, length / 4.0


def make_initial_state(rho: float, seed: int | None) -> InitialState:
    """Perturb every point, then let the simulator normalize by its actual AB."""

    if rho < 0 or not np.isfinite(rho):
        raise ValueError("rho must be finite and nonnegative")
    target = template()
    if rho == 0.0:
        raw = target.copy()
        seed_value = None
        perturbation = np.zeros_like(raw)
    else:
        if seed is None:
            raise ValueError("a positive rho requires a fixed seed")
        rng = np.random.default_rng(seed)
        perturbation = rng.normal(size=target.shape)
        row_norms = np.linalg.norm(perturbation, axis=1)
        perturbation /= float(np.max(row_norms))
        perturbation *= float(rho)
        raw = target + perturbation
        seed_value = int(seed)
    normalized, actual_spacing = _normalize_by_ab(raw)
    normalized_delta = normalized - target
    signed_side = float(
        np.cross(
            normalized[B_INDEX] - normalized[A_INDEX],
            normalized[C_INDEX] - normalized[A_INDEX],
        )
    )
    return InitialState(
        positions=normalized,
        rho=float(rho),
        seed=seed_value,
        actual_spacing=float(actual_spacing),
        raw_base_length=float(4.0 * actual_spacing),
        raw_max_perturbation=float(np.max(np.linalg.norm(perturbation, axis=1))),
        normalized_max_deviation=float(
            np.max(np.linalg.norm(normalized_delta, axis=1))
        ),
        normalized_rms_deviation=float(
            np.sqrt(np.mean(np.sum(normalized_delta**2, axis=1)))
        ),
        mirrored_apex_side=bool(signed_side < 0.0),
    )


def _bootstrap_target_angles() -> np.ndarray:
    return np.asarray(
        [
            # bootstrap_angles accepts a full state; this keeps the target
            # contract in one place and avoids a second angle implementation.
            *bootstrap_angles(TARGET_TEMPLATE),
        ],
        dtype=float,
    )


def _receiver_target_angles(receiver_id: int) -> np.ndarray:
    return receiver_angles(TARGET_TEMPLATE[receiver_id - 1], TARGET_ANCHORS)


def _bootstrap_record(
    *,
    case: InitialState,
    gain: float,
    max_rounds: int,
    position_tolerance: float | None = None,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run bootstrap, returning the post-bootstrap state and audit rows."""

    if position_tolerance is not None and (
        not np.isfinite(position_tolerance) or position_tolerance <= 0
    ):
        raise ValueError("position_tolerance must be finite and positive")
    state = case.positions.copy()
    target_angles = _bootstrap_target_angles()
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    action_count = 0
    relay_scalars = 0
    status = "budget_exhausted"
    stop_round: int | None = None
    failure: str | None = None
    for round_index in range(1, max_rounds + 1):
        observed = np.asarray(bootstrap_angles(state), dtype=float)
        estimate: np.ndarray | None = None
        estimate_error: str | None = None
        try:
            estimate = np.asarray(estimate_apex(observed), dtype=float)
        except (ValueError, FloatingPointError) as error:
            estimate_error = str(error)
        angle_error = float(np.max(np.abs(observed - target_angles)))
        relay_scalars += 2
        row = {
            "rho": case.rho,
            "seed": case.seed if case.seed is not None else "ideal",
            "gain": gain,
            "round": round_index,
            "alpha_rad": float(observed[0]),
            "beta_rad": float(observed[1]),
            "target_alpha_rad": float(target_angles[0]),
            "target_beta_rad": float(target_angles[1]),
            "max_angle_error_rad": angle_error,
            "relay_scalar_count": 2,
            "measurement_slot_count": 2,
            "measurement_tx_count": 4,
            "estimate_x": "",
            "estimate_y": "",
            "action_x": 0.0,
            "action_y": 0.0,
            "action_applied": False,
            "event": "observe",
            "failure": "",
        }
        if estimate is not None:
            row["estimate_x"] = float(estimate[0])
            row["estimate_y"] = float(estimate[1])
        if estimate_error is not None:
            row["failure"] = f"bootstrap_estimation_failure:{estimate_error}"
            rows.append(row)
            failure = row["failure"]
            status = "failure"
            break
        if position_tolerance is None:
            stop = angle_error <= ANGLE_TOLERANCE
        else:
            # This is an estimate from the observed base angles, not truth.
            row["estimated_position_error_d"] = float(
                np.linalg.norm(estimate - TARGET_TEMPLATE[C_INDEX])
            )
            stop = row["estimated_position_error_d"] <= position_tolerance
        if stop:
            row["event"] = (
                "stop_angle_threshold"
                if position_tolerance is None
                else "stop_estimated_position_threshold"
            )
            rows.append(row)
            # The online protocol has only the two unsigned angles.  Side or
            # mirror classification is recorded offline from simulator truth
            # below and is never used to select this stop.
            status = "converged"
            stop_round = round_index
            break
        delta = -float(gain) * (estimate - TARGET_TEMPLATE[C_INDEX])
        state[C_INDEX] += delta
        state[A_INDEX] = TARGET_ANCHORS[1]
        state[B_INDEX] = TARGET_ANCHORS[2]
        action_count += 1
        action_row = {
            "rho": case.rho,
            "seed": case.seed if case.seed is not None else "ideal",
            "gain": gain,
            "round": round_index,
            "receiver_id": C_ID,
            "delta_x": float(delta[0]),
            "delta_y": float(delta[1]),
            "true_position_after_x": float(state[C_INDEX, 0]),
            "true_position_after_y": float(state[C_INDEX, 1]),
            "action_number": action_count,
        }
        actions.append(action_row)
        row["action_x"] = float(delta[0])
        row["action_y"] = float(delta[1])
        row["action_applied"] = True
        row["event"] = "observe_and_move"
        rows.append(row)
    else:
        failure = "bootstrap_budget_exhausted"
        status = "failure"
    summary = {
        "status": status,
        "failure": failure,
        "rounds": len(rows),
        "stop_round": stop_round,
        "action_count": action_count,
        "relay_scalar_count": relay_scalars,
        "angle_slots": len(rows) * 2,
        # Each bootstrap observation has two physical slots (a and b), and
        # each slot uses C plus one fixed endpoint: 2 + 2 transmitter uses.
        "tx_uses": len(rows) * 4,
        "final_angle_error_rad": float(
            np.max(np.abs(bootstrap_angles(state) - target_angles))
        ),
        "final_apex_x": float(state[C_INDEX, 0]),
        "final_apex_y": float(state[C_INDEX, 1]),
    }
    if position_tolerance is not None:
        summary["position_tolerance_d"] = float(position_tolerance)
        summary["stop_rule"] = "estimated_apex_position"
    return state, summary, rows, actions


def _main_record(
    *,
    case: InitialState,
    gain: float,
    state: np.ndarray,
    max_rounds: int,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the simultaneous ABC broadcast loop for all 12 receivers."""

    current = state.copy()
    active = set(RECEIVER_IDS)
    records: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    per_receiver_rounds = {identifier: 0 for identifier in RECEIVER_IDS}
    per_receiver_actions = {identifier: 0 for identifier in RECEIVER_IDS}
    per_receiver_status = {identifier: "active" for identifier in RECEIVER_IDS}
    failures: dict[int, str] = {}
    broadcast_slots = 0
    for round_index in range(1, max_rounds + 1):
        if not active:
            break
        broadcast_slots += 1
        observed: dict[int, np.ndarray] = {}
        pending: list[int] = []
        for receiver_id in sorted(active):
            per_receiver_rounds[receiver_id] += 1
            position = current[receiver_id - 1]
            anchors = current[np.asarray(REFERENCE_IDS) - 1]
            angles = np.asarray(receiver_angles(position, anchors), dtype=float)
            target_angles = _receiver_target_angles(receiver_id)
            angle_error = float(np.max(np.abs(angles - target_angles)))
            observed[receiver_id] = angles
            row = {
                "rho": case.rho,
                "seed": case.seed if case.seed is not None else "ideal",
                "gain": gain,
                "round": round_index,
                "receiver_id": receiver_id,
                "angle_01_rad": float(angles[0]),
                "angle_02_rad": float(angles[1]),
                "angle_12_rad": float(angles[2]),
                "target_angle_01_rad": float(target_angles[0]),
                "target_angle_02_rad": float(target_angles[1]),
                "target_angle_12_rad": float(target_angles[2]),
                "max_angle_error_rad": angle_error,
                "measurement_slot": broadcast_slots,
                "measurement_tx_count": 3,
                "estimated_x": "",
                "estimated_y": "",
                "estimator_residual_rad": "",
                "estimator_success": "",
                "nfev": "",
                "delta_x": 0.0,
                "delta_y": 0.0,
                "action_applied": False,
                "event": "observe",
                "failure": "",
            }
            if angle_error <= ANGLE_TOLERANCE:
                row["event"] = "stop_angle_threshold"
                per_receiver_status[receiver_id] = "converged"
                active.remove(receiver_id)
            else:
                pending.append(receiver_id)
            records.append(row)
        for receiver_id in pending:
            if receiver_id not in active:
                continue
            observed3 = observed[receiver_id]
            estimate: dict[str, Any]
            try:
                # This API boundary receives only the measured angles and the
                # template-selected initial point, never current truth.
                estimate = estimate_receiver(
                    receiver_id,
                    observed3,
                    initial=TARGET_TEMPLATE[receiver_id - 1],
                )
            except (ValueError, FloatingPointError) as error:
                estimate = {
                    "position": TARGET_TEMPLATE[receiver_id - 1].copy(),
                    "success": False,
                    "max_angle_residual": float("inf"),
                    "nfev": 0,
                    "message": str(error),
                }
            row = next(
                item
                for item in reversed(records)
                if item["round"] == round_index and item["receiver_id"] == receiver_id
            )
            estimate_position = np.asarray(estimate["position"], dtype=float)
            row["estimated_x"] = float(estimate_position[0])
            row["estimated_y"] = float(estimate_position[1])
            row["estimator_residual_rad"] = float(estimate["max_angle_residual"])
            row["estimator_success"] = bool(estimate["success"])
            row["nfev"] = int(estimate["nfev"])
            if not bool(estimate["success"]):
                message = str(estimate.get("message", "estimator failed"))
                failure = f"receiver_estimation_failure:{message}"
                row["event"] = "failure"
                row["failure"] = failure
                failures[receiver_id] = failure
                per_receiver_status[receiver_id] = "failure"
                active.remove(receiver_id)
                continue
            delta = -float(gain) * (
                estimate_position - TARGET_TEMPLATE[receiver_id - 1]
            )
            current[receiver_id - 1] += delta
            per_receiver_actions[receiver_id] += 1
            row["delta_x"] = float(delta[0])
            row["delta_y"] = float(delta[1])
            row["action_applied"] = True
            row["event"] = "observe_and_move"
            actions.append(
                {
                    "rho": case.rho,
                    "seed": case.seed if case.seed is not None else "ideal",
                    "gain": gain,
                    "round": round_index,
                    "receiver_id": receiver_id,
                    "delta_x": float(delta[0]),
                    "delta_y": float(delta[1]),
                    "true_position_after_x": float(current[receiver_id - 1, 0]),
                    "true_position_after_y": float(current[receiver_id - 1, 1]),
                    "action_number": per_receiver_actions[receiver_id],
                }
            )
        # All updates happen after the same broadcast has been consumed.  The
        # next loop iteration is the only online verification of the update.
    for receiver_id in sorted(active):
        per_receiver_status[receiver_id] = "budget_exhausted"
        failures[receiver_id] = "receiver_budget_exhausted"
    status = "converged" if not failures else "failure"
    final_angle_errors = {
        receiver_id: float(
            np.max(
                np.abs(
                    receiver_angles(
                        current[receiver_id - 1],
                        current[np.asarray(REFERENCE_IDS) - 1],
                    )
                    - _receiver_target_angles(receiver_id)
                )
            )
        )
        for receiver_id in RECEIVER_IDS
    }
    summary = {
        "status": status,
        "failure_count": len(failures),
        "failures": {str(key): value for key, value in failures.items()},
        "broadcast_slots": broadcast_slots,
        "measurement_slots": broadcast_slots,
        "angle_rows": len(records) * 3,
        "tx_uses": broadcast_slots * 3,
        "action_count": len(actions),
        "receiver_rounds": per_receiver_rounds,
        "receiver_actions": per_receiver_actions,
        "receiver_status": per_receiver_status,
        "final_angle_error_rad": final_angle_errors,
        "final_max_angle_error_rad": max(final_angle_errors.values()),
    }
    return current, summary, records, actions


def run_case(
    rho: float,
    seed: int | None,
    gain: float,
    max_rounds: int,
    *,
    bootstrap_position_tolerance: float | None = None,
) -> dict[str, Any]:
    """Run one case; optional finite calibration is for bounded audits.

    The default retains the original angle stop. The main-stage rule and
    historical 1e-6 d offline acceptance remain the same for both modes.
    """
    case = make_initial_state(rho, seed)
    bootstrap_state, bootstrap_summary, bootstrap_rows, bootstrap_actions = (
        _bootstrap_record(
            case=case,
            gain=gain,
            max_rounds=max_rounds,
            position_tolerance=bootstrap_position_tolerance,
        )
    )
    if bootstrap_summary["status"] == "converged":
        main_state, main_summary, main_rows, main_actions = _main_record(
            case=case,
            gain=gain,
            state=bootstrap_state,
            max_rounds=max_rounds,
        )
    else:
        # Do not start ABC estimation with an uncalibrated reference triangle.
        # This keeps a bootstrap failure visible instead of masking it with a
        # downstream run that violates the staged protocol.
        main_state = bootstrap_state.copy()
        main_rows = []
        main_actions = []
        main_summary = {
            "status": "skipped_bootstrap_failure",
            "failure_count": 0,
            "failures": {},
            "broadcast_slots": 0,
            "measurement_slots": 0,
            "angle_rows": 0,
            "tx_uses": 0,
            "action_count": 0,
            "receiver_rounds": {},
            "receiver_actions": {},
            "receiver_status": {
                str(identifier): "skipped" for identifier in RECEIVER_IDS
            },
            "final_angle_error_rad": {},
            "final_max_angle_error_rad": float("inf"),
        }
    initial_error = np.linalg.norm(case.positions - TARGET_TEMPLATE, axis=1)
    final_error = np.linalg.norm(main_state - TARGET_TEMPLATE, axis=1)
    shape_error = float(np.max(final_error))
    shape_rms = float(np.sqrt(np.mean(final_error**2)))
    bootstrap_distance = float(
        sum(
            np.linalg.norm([row["delta_x"], row["delta_y"]])
            for row in bootstrap_actions
        )
    )
    main_distance = float(
        sum(np.linalg.norm([row["delta_x"], row["delta_y"]]) for row in main_actions)
    )
    online_status = (
        "success"
        if bootstrap_summary["status"] == "converged"
        and main_summary["status"] == "converged"
        and main_summary["final_max_angle_error_rad"] <= ANGLE_TOLERANCE
        else "failure"
    )
    failure_types: list[str] = []
    if bootstrap_summary.get("failure"):
        failure_types.append(str(bootstrap_summary["failure"]))
    if main_summary.get("failures"):
        failure_types.extend(str(value) for value in main_summary["failures"].values())
    if case.mirrored_apex_side:
        failure_types.append("mirror_branch")
    if shape_error > SHAPE_TOLERANCE:
        failure_types.append("shape_error_above_1e-6d")
    tx_uses = int(bootstrap_summary["tx_uses"] + main_summary["tx_uses"])
    measurement_slots = int(
        bootstrap_summary["angle_slots"] + main_summary["measurement_slots"]
    )
    return {
        "rho": case.rho,
        "seed": case.seed if case.seed is not None else "ideal",
        "gain": float(gain),
        "max_rounds": int(max_rounds),
        "online_status": online_status,
        "status": "success" if not failure_types else "failure",
        "failure_types": sorted(set(failure_types)),
        "actual_spacing_d": case.actual_spacing,
        "d_actual": case.actual_spacing,
        "d_actual_ratio": case.actual_spacing,
        "actual_spacing_over_template_d": case.actual_spacing,
        "raw_base_length": case.raw_base_length,
        "raw_max_perturbation_d": case.raw_max_perturbation,
        "initial_max_deviation_d": case.normalized_max_deviation,
        "initial_rms_deviation_d": case.normalized_rms_deviation,
        "initial_max_error_d": float(np.max(initial_error)),
        "initial_rms_error_d": float(np.sqrt(np.mean(initial_error**2))),
        "mirrored_apex_side": case.mirrored_apex_side,
        "bootstrap": bootstrap_summary,
        "main": main_summary,
        "measurement_slots": measurement_slots,
        "tx_uses": tx_uses,
        "bootstrap_relay_scalars": int(bootstrap_summary["relay_scalar_count"]),
        "action_count": int(
            bootstrap_summary["action_count"] + main_summary["action_count"]
        ),
        "bootstrap_cumulative_move_d": bootstrap_distance,
        "main_cumulative_move_d": main_distance,
        "total_cumulative_move_d": bootstrap_distance + main_distance,
        "bootstrap_cumulative_move_raw_d": bootstrap_distance * case.actual_spacing,
        "main_cumulative_move_raw_d": main_distance * case.actual_spacing,
        "total_cumulative_move_raw_d": (bootstrap_distance + main_distance)
        * case.actual_spacing,
        "final_max_position_error_d": shape_error,
        "final_rms_position_error_d": shape_rms,
        "shape_error_pass": bool(shape_error <= SHAPE_TOLERANCE),
        "final_positions": main_state,
        "bootstrap_rows": bootstrap_rows,
        "bootstrap_actions": bootstrap_actions,
        "main_rows": main_rows,
        "main_actions": main_actions,
    }


def _case_seed(rho: float, seed: int) -> int | None:
    return None if rho == 0.0 else seed


def run_batch(
    output_dir: Path,
    *,
    gains: tuple[float, ...] = triangle_GAINS,
    max_rounds: int = MAX_ROUNDS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [(0.0, None)] + [
        (rho, seed)
        for rho in triangle_PERTURBATION_LEVELS[1:]
        for seed in triangle_SEEDS
    ]
    summaries: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_actions: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    main_actions: list[dict[str, Any]] = []
    for gain in gains:
        for rho, seed in cases:
            result = run_case(
                rho,
                _case_seed(rho, seed or 0),
                float(gain),
                max_rounds,
            )
            summary = {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "final_positions",
                    "bootstrap_rows",
                    "bootstrap_actions",
                    "main_rows",
                    "main_actions",
                }
            }
            summaries.append(summary)
            identity = {
                "rho": rho,
                "seed": result["seed"],
                "gain": gain,
            }
            for row in result["bootstrap_rows"]:
                bootstrap_rows.append({**identity, **row})
            for row in result["bootstrap_actions"]:
                bootstrap_actions.append({**identity, **row})
            for row in result["main_rows"]:
                main_rows.append({**identity, **row})
            for row in result["main_actions"]:
                main_actions.append({**identity, **row})
            case_dir = (
                output_dir
                / f"gain_{gain:g}"
                / f"rho_{rho:g}"
                / f"seed_{result['seed']}"
            )
            case_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                case_dir / "final_positions.csv",
                result["final_positions"],
                delimiter=",",
            )
            case_summary = {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "final_positions",
                    "bootstrap_rows",
                    "bootstrap_actions",
                    "main_rows",
                    "main_actions",
                }
            }
            (case_dir / "summary.json").write_text(
                json.dumps(
                    triangle_json_value(case_summary), ensure_ascii=False, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
    triangle_write_csv(output_dir / "bootstrap_observations.csv", bootstrap_rows)
    triangle_write_csv(output_dir / "bootstrap_actions.csv", bootstrap_actions)
    triangle_write_csv(output_dir / "receiver_observations.csv", main_rows)
    triangle_write_csv(output_dir / "receiver_actions.csv", main_actions)
    flat_summaries: list[dict[str, Any]] = []
    for row in summaries:
        flat = {
            key: value
            for key, value in row.items()
            if key not in {"bootstrap", "main", "failure_types"}
        }
        flat["bootstrap_status"] = row["bootstrap"]["status"]
        flat["bootstrap_rounds"] = row["bootstrap"]["rounds"]
        flat["bootstrap_actions"] = row["bootstrap"]["action_count"]
        flat["main_status"] = row["main"]["status"]
        flat["main_broadcast_slots"] = row["main"]["broadcast_slots"]
        flat["main_actions"] = row["main"]["action_count"]
        flat["failure_types_json"] = json.dumps(
            row["failure_types"], ensure_ascii=False, separators=(",", ":")
        )
        flat["bootstrap_summary_json"] = json.dumps(
            triangle_json_value(row["bootstrap"]),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        flat["main_summary_json"] = json.dumps(
            triangle_json_value(row["main"]), ensure_ascii=False, separators=(",", ":")
        )
        flat_summaries.append(flat)
    triangle_write_csv(output_dir / "runs.csv", flat_summaries)
    summary = {
        "protocol": "triangle_reference_bootstrap_then_ABC_parallel_closed_loop",
        "estimator_truth_boundary": "estimator receives observations and canonical template only; simulator owns truth and applies ideal normalized-coordinate moves",
        "reference_ids": list(REFERENCE_IDS),
        "receiver_ids": list(RECEIVER_IDS),
        "target_anchor_coordinates": TARGET_ANCHORS,
        "target_template": TARGET_TEMPLATE,
        "gains": list(gains),
        "perturbation_levels": list(triangle_PERTURBATION_LEVELS),
        "seeds": list(triangle_SEEDS),
        "case_count_per_gain": len(cases),
        "run_count": len(summaries),
        "max_rounds_per_stage": max_rounds,
        "angle_tolerance_rad": ANGLE_TOLERANCE,
        "shape_tolerance_d": SHAPE_TOLERANCE,
        "measurement_accounting": {
            "bootstrap_angle_slots_per_observation": 2,
            "bootstrap_tx_uses_per_observation": 4,
            "bootstrap_tx_uses_per_angle_slot": 2,
            "main_tx_uses_per_broadcast": 3,
            "bootstrap_relay_scalars_per_observation": 2,
            "main_broadcast_is_shared_by_12_receivers": True,
        },
        "runs": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(triangle_json_value(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def triangle_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} triangle",
        description="Run the bounded noiseless Q2 triangle-reference closed loop.",
    )
    parser.add_argument("--output-dir", type=Path, default=triangle_DEFAULT_OUTPUT)
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=list(triangle_GAINS),
        help="gains to run, default: 1.0 0.5",
    )
    args = parser.parse_args(argv)
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    gains = tuple(float(value) for value in args.gains)
    if any(not np.isfinite(value) or value <= 0 or value > 1 for value in gains):
        parser.error("all gains must lie in (0, 1]")
    summary = run_batch(args.output_dir, gains=gains, max_rounds=args.max_rounds)
    success = sum(row["status"] == "success" for row in summary["runs"])
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "run_count": summary["run_count"],
                "success_count": int(success),
                "failure_count": int(summary["run_count"] - success),
            },
            ensure_ascii=False,
        )
    )


# ============================================================================
# residual: analyze_reference_residual
# ============================================================================


def apex_angle_jacobian(position: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Derivative of the three unsigned angles with respect to anchor 0.

    Nonsmooth rows return NaNs, even when the apex is absent from that row.
    Only the production estimator's smooth selected rows enter the analysis.
    """
    result = np.zeros((3, 2))
    for row, (first, second) in enumerate(ANCHOR_PAIRS):
        u, v = anchors[first] - position, anchors[second] - position
        u2, v2 = u @ u, v @ v
        if min(u2, v2) <= 1e-24:
            raise ValueError("receiver coincides with anchor")
        cross = u[0] * v[1] - u[1] * v[0]
        if abs(cross) / np.sqrt(u2 * v2) <= 1e-10:
            result[row] = np.nan
        elif first == 0:
            result[row] = -np.sign(cross) * np.array([-u[1], u[0]]) / u2
        elif second == 0:
            result[row] = np.sign(cross) * np.array([-v[1], v[0]]) / v2
    return result


def bootstrap_jacobian(apex: np.ndarray) -> np.ndarray:
    """Two base-angle derivatives on the canonical upper half-plane."""
    x, y = apex
    if y <= 0:
        raise ValueError("bootstrap derivative requires upper branch")
    return np.array(
        [
            [-y, x] / np.array(x * x + y * y),
            [y, 4 - x] / np.array((4 - x) ** 2 + y * y),
        ]
    )


def propagation_blocks(receiver_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = TARGET_TEMPLATE[receiver_id - 1]
    selected = list(SELECTED_ANGLE_INDICES[receiver_id])
    j = angle_jacobian(point, TARGET_ANCHORS, allow_degenerate=True)[selected]
    b = apex_angle_jacobian(point, TARGET_ANCHORS)[selected]
    return j, b, np.linalg.solve(j, b)


def central_difference(function, point: np.ndarray, step: float) -> np.ndarray:
    return np.column_stack(
        [
            (function(point + offset) - function(point - offset)) / (2 * step)
            for offset in np.eye(2) * step
        ]
    )


def linear_audit() -> tuple[list[dict], dict]:
    jc = bootstrap_jacobian(TARGET_ANCHORS[0])
    inv_jc = np.linalg.inv(jc)
    corners = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    rows = []
    max_j_error = max_b_error = max_g_error = 0.0
    for identifier in RECEIVER_IDS:
        point = TARGET_TEMPLATE[identifier - 1]
        selected = list(SELECTED_ANGLE_INDICES[identifier])
        j, b, g = propagation_blocks(identifier)

        def angles_at_apex(apex):
            anchors = TARGET_ANCHORS.copy()
            anchors[0] = apex
            return receiver_angles(point, anchors)

        def estimated_at_apex(apex):
            # Retain the returned two-row candidate even when the complete
            # triangle check rejects it; acceptance is audited separately.
            return estimate_receiver(identifier, angles_at_apex(apex))["position"]

        for step in (1e-4, 1e-5):
            nj = central_difference(
                lambda p: receiver_angles(p, TARGET_ANCHORS)[selected], point, step
            )
            nb = central_difference(
                lambda c: angles_at_apex(c)[selected], TARGET_ANCHORS[0], step
            )
            ng = central_difference(estimated_at_apex, TARGET_ANCHORS[0], step)
            max_j_error = max(max_j_error, float(np.max(np.abs(j - nj))))
            max_b_error = max(max_b_error, float(np.max(np.abs(b - nb))))
            max_g_error = max(max_g_error, float(np.max(np.abs(g - ng))))
        rows.append(
            {
                "receiver_id": identifier,
                "selected_indices": ";".join(map(str, selected)),
                "sigma_min_rad_per_d": np.linalg.svd(j, compute_uv=False)[-1],
                "g_00": g[0, 0],
                "g_01": g[0, 1],
                "g_10": g[1, 0],
                "g_11": g[1, 1],
                "g_spectral_norm": np.linalg.norm(g, 2),
                "angle_box_to_position_d_per_rad": max(
                    np.linalg.norm(g @ inv_jc @ s) for s in corners
                ),
            }
        )

    def base_angles(apex):
        points = TARGET_TEMPLATE.copy()
        points[0] = apex
        return bootstrap_angles(points)

    jc_error = max(
        float(
            np.max(np.abs(jc - central_difference(base_angles, TARGET_ANCHORS[0], h)))
        )
        for h in (1e-4, 1e-5)
    )
    return rows, {
        "bootstrap_jacobian": jc,
        "bootstrap_sigma_min": np.linalg.svd(jc, compute_uv=False)[-1],
        "bootstrap_angle_box_factor": max(np.linalg.norm(inv_jc @ s) for s in corners),
        "g_max": max(row["g_spectral_norm"] for row in rows),
        "receiver_angle_box_factor": max(
            row["angle_box_to_position_d_per_rad"] for row in rows
        ),
        "max_bootstrap_derivative_error": jc_error,
        "max_receiver_derivative_error": max_j_error,
        "max_apex_derivative_error": max_b_error,
        "max_solver_candidate_derivative_error": max_g_error,
    }


def residual_probes() -> tuple[list[dict], list[dict], list[dict]]:
    """Eight directions, five magnitudes, all receivers; 480 estimates."""
    rows, loops, observations = [], [], []
    for amplitude in (1e-5, 1e-3, 0.005, 0.01, 0.02):
        for direction in range(8):
            theta = direction * np.pi / 4
            delta = amplitude * np.array([np.cos(theta), np.sin(theta)])
            anchors = TARGET_ANCHORS.copy()
            anchors[0] += delta
            for identifier in RECEIVER_IDS:
                point = TARGET_TEMPLATE[identifier - 1]
                observed = receiver_angles(point, anchors)
                result = estimate_receiver(identifier, observed)
                bias = result["position"] - point
                prediction = propagation_blocks(identifier)[2] @ delta
                rows.append(
                    {
                        "amplitude_d": amplitude,
                        "direction_deg": direction * 45,
                        "delta_c_x": delta[0],
                        "delta_c_y": delta[1],
                        "receiver_id": identifier,
                        "candidate_bias_x": bias[0],
                        "candidate_bias_y": bias[1],
                        "candidate_bias_norm_d": np.linalg.norm(bias),
                        "linear_prediction_x": prediction[0],
                        "linear_prediction_y": prediction[1],
                        "linear_error_d": np.linalg.norm(bias - prediction),
                        "accepted": result["success"],
                        "full_residual_rad": result["max_angle_residual"],
                        "message": result["message"],
                    }
                )
            # A controlled injection at the main-stage entry, not a complete
            # bootstrap protocol. Preserve every production failure/stop rule.
            if amplitude in (0.005, 0.02):
                for gain in (1.0, 0.5):
                    state = TARGET_TEMPLATE.copy()
                    state[0] += delta
                    final, summary, records, actions = _main_record(
                        case=make_initial_state(0.0, None),
                        gain=gain,
                        state=state,
                        max_rounds=30,
                    )
                    label = {
                        "amplitude_d": amplitude,
                        "direction_deg": direction * 45,
                        "gain": gain,
                    }
                    receiver_error = np.linalg.norm(
                        (final - TARGET_TEMPLATE)[np.array(RECEIVER_IDS) - 1], axis=1
                    )
                    loops.append(
                        {
                            **label,
                            "status": summary["status"],
                            "failure_count": summary["failure_count"],
                            "failures": json.dumps(
                                summary["failures"], ensure_ascii=False
                            ),
                            "main_slots": summary["broadcast_slots"],
                            "main_tx_uses": summary["tx_uses"],
                            "main_actions": len(actions),
                            "main_displacement_d": sum(
                                np.hypot(a["delta_x"], a["delta_y"]) for a in actions
                            ),
                            "receiver_max_error_d": max(receiver_error),
                            "full_team_max_error_d": max(
                                amplitude, max(receiver_error)
                            ),
                            "final_max_angle_error_rad": summary[
                                "final_max_angle_error_rad"
                            ],
                        }
                    )
                    observations.extend({**label, **record} for record in records)
    return rows, loops, observations


def geometry_probes() -> tuple[list[dict], list[dict]]:
    base = []
    for height in (2 * np.sqrt(3), 1.0, 0.1, 0.01, 0.001):
        j = bootstrap_jacobian(np.array([2.0, height]))
        singular = np.linalg.svd(j, compute_uv=False)
        base.append(
            {
                "height_d": height,
                "sigma_min": singular[-1],
                "condition": singular[0] / singular[-1],
            }
        )
    branches = []
    for identifier in RECEIVER_IDS:
        actual = TARGET_TEMPLATE[identifier - 1] + np.array([0.03, -0.02])
        observed = receiver_angles(actual, TARGET_ANCHORS)
        target = TARGET_TEMPLATE[identifier - 1]
        starts = [
            target,
            target + [0.2, 0.2],
            target + [-0.2, -0.2],
            target * np.array([1.0, -1.0]),
            np.array([2.0, 5.0]),
        ]
        for index, initial in enumerate(starts):
            result = estimate_receiver(identifier, observed, initial=initial)
            branches.append(
                {
                    "receiver_id": identifier,
                    "start_id": index,
                    "initial_x": initial[0],
                    "initial_y": initial[1],
                    "actual_x": actual[0],
                    "actual_y": actual[1],
                    "candidate_x": result["position"][0],
                    "candidate_y": result["position"][1],
                    "accepted": result["success"],
                    "position_error_d": np.linalg.norm(result["position"] - actual),
                    "full_residual_rad": result["max_angle_residual"],
                    "message": result["message"],
                }
            )
    return base, branches


def threshold_comparison() -> tuple[list[dict], list[dict]]:
    """27 complete protocols: three seeds, three gains, three stop rules."""
    rows, cases = [], []
    for seed in (11, 23, 47):
        for gain in (1.0, 0.8, 0.5):
            for tolerance in (None, 0.005, 0.01):
                result = run_case(
                    0.1, seed, gain, 30, bootstrap_position_tolerance=tolerance
                )
                label = (
                    "angle_1e-8" if tolerance is None else f"position_{tolerance:g}d"
                )
                cases.append({"calibration_rule": label, **result})
                # Reconstruct the state and costs independently from action
                # logs, including the finite residual of the frozen apex.
                reconstructed = make_initial_state(0.1, seed).positions.copy()
                actions = result["bootstrap_actions"] + result["main_actions"]
                for action in actions:
                    reconstructed[action["receiver_id"] - 1] += [
                        action["delta_x"],
                        action["delta_y"],
                    ]
                np.testing.assert_allclose(
                    reconstructed, result["final_positions"], atol=1e-13, rtol=0
                )
                displacement = sum(
                    np.hypot(a["delta_x"], a["delta_y"]) for a in actions
                )
                np.testing.assert_allclose(
                    displacement, result["total_cumulative_move_d"], atol=1e-12
                )
                b, m = result["bootstrap"], result["main"]
                assert (
                    result["measurement_slots"]
                    == 2 * b["rounds"] + m["broadcast_slots"]
                )
                assert result["tx_uses"] == 4 * b["rounds"] + 3 * m["broadcast_slots"]
                rows.append(
                    {
                        "seed": seed,
                        "gain": gain,
                        "calibration_rule": label,
                        "online_status": result["online_status"],
                        "historical_1e_minus6_status": result["status"],
                        "failure_types": json.dumps(result["failure_types"]),
                        "illustrative_0_01d_pass": result["online_status"] == "success"
                        and result["final_max_position_error_d"] <= 0.01,
                        "final_apex_error_d": np.linalg.norm(
                            result["final_positions"][0] - TARGET_TEMPLATE[0]
                        ),
                        "final_max_position_error_d": result[
                            "final_max_position_error_d"
                        ],
                        "bootstrap_slots": b["angle_slots"],
                        "main_slots": m["broadcast_slots"],
                        "complete_slots": result["measurement_slots"],
                        "complete_tx_uses": result["tx_uses"],
                        "bootstrap_relay_scalars": result["bootstrap_relay_scalars"],
                        "complete_displacement_d": result["total_cumulative_move_d"],
                    }
                )
    return rows, cases


def residual_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} residual",
        description="Bounded, noiseless audit of a frozen FY01 calibration residual.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/q2/reference_residual")
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blocks, linear = linear_audit()
    probes, loops, observations = residual_probes()
    base, branches = geometry_probes()
    thresholds, threshold_cases = threshold_comparison()
    (args.output_dir / "threshold_cases.json").write_text(
        json.dumps(triangle_json_value(threshold_cases), indent=2, allow_nan=False)
        + "\n"
    )
    angle_box = []
    target_angles = bootstrap_angles(TARGET_TEMPLATE)
    for epsilon_deg in (0.01, 0.05, 0.1):
        epsilon = np.deg2rad(epsilon_deg)
        for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            c = bootstrap_from_angles(target_angles + epsilon * np.array([sx, sy]))
            angle_box.append(
                {
                    "epsilon_deg": epsilon_deg,
                    "alpha_sign": sx,
                    "beta_sign": sy,
                    "apex_error_d": np.linalg.norm(c - TARGET_ANCHORS[0]),
                    "linear_box_bound_d": linear["bootstrap_angle_box_factor"]
                    * epsilon,
                }
            )
    tables = {
        "propagation_blocks": blocks,
        "residual_probes": probes,
        "main_stage_runs": loops,
        "main_stage_observations": observations,
        "bootstrap_conditioning": base,
        "multistart_probes": branches,
        "bootstrap_angle_box": angle_box,
        "threshold_comparison": thresholds,
    }
    for name, rows in tables.items():
        triangle_write_csv(args.output_dir / f"{name}.csv", rows)
    summary = {
        "contract": "Noiseless frozen FY01 offset; canonical upper local branch; production estimator unchanged. Injected main-stage costs exclude bootstrap; threshold comparison includes the complete two-stage protocol.",
        "linear": linear,
        "residual_probes": {
            "count": len(probes),
            "rejected": sum(not row["accepted"] for row in probes),
            "by_amplitude": [
                {
                    "amplitude_d": amplitude,
                    "max_candidate_gain": max(
                        row["candidate_bias_norm_d"] / amplitude
                        for row in probes
                        if row["amplitude_d"] == amplitude
                    ),
                    "max_linear_error_d": max(
                        row["linear_error_d"]
                        for row in probes
                        if row["amplitude_d"] == amplitude
                    ),
                    "rejected": sum(
                        not row["accepted"]
                        for row in probes
                        if row["amplitude_d"] == amplitude
                    ),
                }
                for amplitude in (1e-5, 1e-3, 0.005, 0.01, 0.02)
            ],
        },
        "main_stage_runs": {
            "count": len(loops),
            "converged": sum(row["status"] == "converged" for row in loops),
        },
        "multistart": {
            "count": len(branches),
            "accepted": sum(row["accepted"] for row in branches),
            "accepted_wrong_position": sum(
                row["accepted"] and row["position_error_d"] > 1e-6 for row in branches
            ),
        },
        "threshold_comparison": {
            "count": len(thresholds),
            "online_success": sum(
                row["online_status"] == "success" for row in thresholds
            ),
            "illustrative_0_01d_pass": sum(
                row["illustrative_0_01d_pass"] for row in thresholds
            ),
            "historical_1e_minus6_success": sum(
                row["historical_1e_minus6_status"] == "success" for row in thresholds
            ),
            "action_reconstruction_and_cost_checks": "passed for every complete case",
        },
        "limits": [
            "Linear coefficients are at the target only.",
            "Rejected two-row candidates are not successful localizations.",
            "Finite thresholds tested on three fixed initial states, not an entire neighborhood.",
            "No noisy stopping rule or global uniqueness is validated.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(triangle_json_value(summary), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(triangle_json_value(summary), indent=2, allow_nan=False))


# ============================================================================
# geometry: analyze_geometry_boundaries
# ============================================================================

geometry_DEFAULT_OUTPUT = Path("outputs/q2/geometry_boundaries")
ANCHOR_NAMES = ("FY01", "FY11", "FY15")
BOOTSTRAP_BASE_H = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8)
G_AMPLITUDES = (0.002, 0.0048, 0.005, 0.01, 0.02)
RECEIVER_OFFSET_SCENARIOS = {
    "Q": np.array([0.0, 0.0]),
    "plus_0.05_x": np.array([0.05, 0.0]),
    "plus_0.05_y": np.array([0.0, 0.05]),
    "plus_0.10_x": np.array([0.10, 0.0]),
    "plus_0.10_y": np.array([0.0, 0.10]),
}
RECEIVER_PATH_IDS = tuple(RECEIVER_IDS)


def geometry_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): geometry_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [geometry_json_value(item) for item in value]
    return value


def geometry_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            geometry_json_value(value), ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _finite_svd_metrics(matrix: np.ndarray) -> dict[str, Any]:
    finite_rows = np.isfinite(matrix).all(axis=1)
    regular = matrix[finite_rows]
    if len(regular) == 0:
        return {
            "regular_row_count": 0,
            "rank": 0,
            "sigma_min": float("nan"),
            "sigma_max": float("nan"),
            "inverse_amplification": float("nan"),
        }
    singular = np.linalg.svd(regular, compute_uv=False)
    rank = int(np.linalg.matrix_rank(regular, tol=1e-10))
    sigma_min = float(singular[-1]) if len(singular) >= 2 else float("nan")
    return {
        "regular_row_count": int(len(regular)),
        "rank": rank,
        "sigma_min": sigma_min,
        "sigma_max": float(singular[0]),
        "inverse_amplification": (
            float(1.0 / sigma_min)
            if np.isfinite(sigma_min) and sigma_min > 0.0
            else float("inf")
            if sigma_min == 0.0
            else float("nan")
        ),
    }


def _triangle_state(apex: np.ndarray) -> np.ndarray:
    state = TARGET_TEMPLATE.copy()
    state[0] = apex
    return state


def _bootstrap_row(path: str, path_index: int, apex: np.ndarray) -> dict[str, Any]:
    observed = bootstrap_angles(_triangle_state(apex))
    interior = np.array([observed[0], observed[1], np.pi - observed.sum()])
    jacobian = _bootstrap_jacobian(apex)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    corners = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    estimate = bootstrap_from_angles(observed)
    return {
        "path": path,
        "path_index": path_index,
        "x_d": float(apex[0]),
        "h_d": float(apex[1]),
        "alpha_rad": float(observed[0]),
        "beta_rad": float(observed[1]),
        "angle_sum_rad": float(observed.sum()),
        "min_triangle_interior_angle_rad": float(np.min(interior)),
        "apex_interior_angle_rad": float(interior[2]),
        "sigma_min_rad_per_d": float(singular[-1]),
        "sigma_max_rad_per_d": float(singular[0]),
        "condition_number": float(singular[0] / singular[-1]),
        "angle_box_inverse_amplification_d_per_rad": float(
            max(np.linalg.norm(np.linalg.solve(jacobian, corner)) for corner in corners)
        ),
        "inverse_reconstruction_error_d": float(np.linalg.norm(estimate - apex)),
    }


def _bootstrap_jacobian(apex: np.ndarray) -> np.ndarray:
    x, h = apex
    if h <= 0.0:
        raise ValueError("bootstrap boundary audit uses the upper branch h > 0")
    return np.array(
        [
            [-h, x] / (x * x + h * h),
            [h, 4.0 - x] / ((4.0 - x) ** 2 + h * h),
        ]
    )


def bootstrap_boundary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path_index = 0
    for x, name in ((2.0, "base_interior_x2"), (-1.0, "base_exterior_x_minus1")):
        for h in BOOTSTRAP_BASE_H:
            rows.append(_bootstrap_row(name, path_index, np.array([x, h])))
            path_index += 1
    for h in (4.0, 10.0, 100.0, 1000.0):
        rows.append(_bootstrap_row("high_vertical_x2", path_index, np.array([2.0, h])))
        path_index += 1
    for t in (10.0, 100.0, 1000.0):
        rows.append(
            _bootstrap_row("far_diagonal_x2t_h_t", path_index, np.array([2.0 * t, t]))
        )
        path_index += 1
    for x in (10.0, 100.0, 1000.0):
        rows.append(_bootstrap_row("far_horizontal_h1", path_index, np.array([x, 1.0])))
        path_index += 1
    return rows


def _receiver_row(
    path: str,
    path_index: int,
    receiver_id: int,
    point: np.ndarray,
    *,
    t: float,
    anchor_label: str,
    circle_theta: float | None = None,
) -> dict[str, Any]:
    observed = receiver_angles(point, TARGET_ANCHORS)
    jacobian = angle_jacobian(point, TARGET_ANCHORS, allow_degenerate=True)
    selected = jacobian[list(SELECTED_ANGLE_INDICES[receiver_id])]
    full_metrics = _finite_svd_metrics(jacobian)
    selected_metrics = _finite_svd_metrics(selected)
    result = estimate_receiver(
        receiver_id,
        observed,
        initial=TARGET_TEMPLATE[receiver_id - 1],
    )
    candidate = np.asarray(result["position"], dtype=float)
    endpoint_singular = [
        int(index)
        for index, value in enumerate(observed)
        if min(float(value), float(np.pi - value)) <= 1e-10
    ]
    singular_rows = [
        int(index) for index, row in enumerate(jacobian) if not np.isfinite(row).all()
    ]
    return {
        "path": path,
        "path_index": path_index,
        "receiver_id": receiver_id,
        "anchor": anchor_label,
        "t": float(t),
        "circle_theta_rad": "" if circle_theta is None else float(circle_theta),
        "point_x": float(point[0]),
        "point_y": float(point[1]),
        "angle_01_rad": float(observed[0]),
        "angle_02_rad": float(observed[1]),
        "angle_12_rad": float(observed[2]),
        "min_observed_angle_rad": float(np.min(observed)),
        "endpoint_zero_or_pi_rows": ";".join(map(str, endpoint_singular)),
        "nonsmooth_jacobian_rows": ";".join(map(str, singular_rows)),
        "full_regular_row_count": full_metrics["regular_row_count"],
        "full_rank": full_metrics["rank"],
        "full_sigma_min": full_metrics["sigma_min"],
        "selected_indices": ";".join(map(str, SELECTED_ANGLE_INDICES[receiver_id])),
        "selected_regular_row_count": selected_metrics["regular_row_count"],
        "selected_rank": selected_metrics["rank"],
        "selected_sigma_min": selected_metrics["sigma_min"],
        "selected_inverse_amplification": selected_metrics["inverse_amplification"],
        "estimator_success": bool(result["success"]),
        "candidate_x": float(candidate[0]),
        "candidate_y": float(candidate[1]),
        "candidate_error_to_path_d": float(np.linalg.norm(candidate - point)),
        "full_residual_rad": float(result["max_angle_residual"]),
        "estimator_message": str(result["message"]),
    }


def receiver_boundary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path_index = 0
    ray_t = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        start, end = TARGET_ANCHORS[first], TARGET_ANCHORS[second]
        label = f"{ANCHOR_NAMES[first]}_toward_{ANCHOR_NAMES[second]}"
        for t in ray_t:
            point = start + t * (end - start)
            for receiver_id in RECEIVER_PATH_IDS:
                rows.append(
                    _receiver_row(
                        "reference_ray",
                        path_index,
                        receiver_id,
                        point,
                        t=t,
                        anchor_label=label,
                    )
                )
            path_index += 1
    point_t = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
    for anchor_index in range(3):
        direction_angle = 0.37 + 0.51 * anchor_index
        direction = np.array([np.cos(direction_angle), np.sin(direction_angle)])
        anchor = TARGET_ANCHORS[anchor_index]
        label = f"{ANCHOR_NAMES[anchor_index]}_generic_ray"
        for t in point_t:
            point = anchor + t * direction
            for receiver_id in RECEIVER_PATH_IDS:
                rows.append(
                    _receiver_row(
                        "reference_point",
                        path_index,
                        receiver_id,
                        point,
                        t=t,
                        anchor_label=label,
                    )
                )
            path_index += 1

    center = np.array([2.0, 2.0 * np.sqrt(3.0) / 3.0])
    radius = 4.0 / np.sqrt(3.0)
    circle_offsets = (0.1, 0.01, 0.001, 0.0001, 0.0)
    for theta in (0.0, 0.3):
        unit = np.array([np.cos(theta), np.sin(theta)])
        for offset in circle_offsets:
            point = center + (radius + offset) * unit
            for receiver_id in RECEIVER_PATH_IDS:
                rows.append(
                    _receiver_row(
                        "reference_circumcircle",
                        path_index,
                        receiver_id,
                        point,
                        t=offset,
                        anchor_label="reference_circumcircle",
                        circle_theta=theta,
                    )
                )
            path_index += 1
    return rows


def _right_worst_direction(matrix: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(matrix)
    vector = vh[0]
    first = np.flatnonzero(np.abs(vector) > 1e-12)
    if len(first) and vector[first[0]] < 0:
        vector = -vector
    return vector


def g_worst_direction_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        receiver_id: propagation_blocks(receiver_id) for receiver_id in RECEIVER_IDS
    }
    global_receiver = max(
        RECEIVER_IDS,
        key=lambda identifier: (np.linalg.norm(blocks[identifier][2], 2), -identifier),
    )
    global_g = blocks[global_receiver][2]
    for receiver_id in RECEIVER_IDS:
        g = blocks[receiver_id][2]
        direction = _right_worst_direction(g)
        for sign in (1, -1):
            for amplitude in G_AMPLITUDES:
                delta = float(sign * amplitude) * direction
                anchors = TARGET_ANCHORS.copy()
                anchors[0] += delta
                for offset_label, offset in RECEIVER_OFFSET_SCENARIOS.items():
                    actual = TARGET_TEMPLATE[receiver_id - 1] + offset
                    try:
                        observed = receiver_angles(actual, anchors)
                        result = estimate_receiver(
                            receiver_id,
                            observed,
                            initial=TARGET_TEMPLATE[receiver_id - 1],
                        )
                        candidate = np.asarray(result["position"], dtype=float)
                        error = candidate - actual
                        candidate_x = float(candidate[0])
                        candidate_y = float(candidate[1])
                        residual = float(result["max_angle_residual"])
                        accepted = bool(result["success"])
                        message = str(result["message"])
                    except (ValueError, FloatingPointError) as exc:
                        error = np.array([np.nan, np.nan])
                        candidate_x = float("nan")
                        candidate_y = float("nan")
                        residual = float("inf")
                        accepted = False
                        message = str(exc)
                    prediction = g @ delta
                    rows.append(
                        {
                            "receiver_id": receiver_id,
                            "offset_label": offset_label,
                            "offset_x_d": float(offset[0]),
                            "offset_y_d": float(offset[1]),
                            "sign": sign,
                            "amplitude_d": amplitude,
                            "delta_c_x_d": float(delta[0]),
                            "delta_c_y_d": float(delta[1]),
                            "right_singular_vector_x": float(direction[0]),
                            "right_singular_vector_y": float(direction[1]),
                            "g_spectral_norm": float(np.linalg.norm(g, 2)),
                            "predicted_bias_x_d": float(prediction[0]),
                            "predicted_bias_y_d": float(prediction[1]),
                            "candidate_bias_x_d": float(error[0]),
                            "candidate_bias_y_d": float(error[1]),
                            "candidate_linear_error_d": float(
                                np.linalg.norm(error - prediction)
                            ),
                            "accepted": accepted,
                            "candidate_x": candidate_x,
                            "candidate_y": candidate_y,
                            "full_residual_rad": residual,
                            "message": message,
                        }
                    )
    return rows, {
        "global_worst_receiver_id": int(global_receiver),
        "global_worst_g_spectral_norm": float(np.linalg.norm(global_g, 2)),
        "global_worst_right_singular_vector": _right_worst_direction(global_g),
        "amplitudes_d": list(G_AMPLITUDES),
        "offset_scenarios": {
            key: value for key, value in RECEIVER_OFFSET_SCENARIOS.items()
        },
        "row_count": len(rows),
        "accepted_count": int(sum(row["accepted"] for row in rows)),
        "rejected_count": int(sum(not row["accepted"] for row in rows)),
    }


def main_stage_rows(global_info: dict[str, Any]) -> list[dict[str, Any]]:
    receiver_id = int(global_info["global_worst_receiver_id"])
    direction = np.asarray(
        global_info["global_worst_right_singular_vector"], dtype=float
    )
    blocks = {
        identifier: propagation_blocks(identifier)[2] for identifier in RECEIVER_IDS
    }
    rows: list[dict[str, Any]] = []
    for offset_label, receiver_offset in RECEIVER_OFFSET_SCENARIOS.items():
        for sign in (1, -1):
            for amplitude in (0.0048, 0.005, 0.01):
                delta = float(sign * amplitude) * direction
                state = TARGET_TEMPLATE.copy()
                state[0] += delta
                state[np.asarray(RECEIVER_IDS) - 1] += receiver_offset
                final, summary, records, actions = _main_record(
                    case=make_initial_state(0.0, None),
                    gain=1.0,
                    state=state,
                    max_rounds=30,
                )
                receiver_errors = {
                    str(identifier): float(
                        np.linalg.norm(
                            final[identifier - 1] - TARGET_TEMPLATE[identifier - 1]
                        )
                    )
                    for identifier in RECEIVER_IDS
                }
                predicted_receiver_errors = {
                    str(identifier): float(np.linalg.norm(blocks[identifier] @ delta))
                    for identifier in RECEIVER_IDS
                }
                failures = summary["failures"]
                rows.append(
                    {
                        "global_worst_receiver_id": receiver_id,
                        "receiver_offset_label": offset_label,
                        "receiver_offset_x_d": float(receiver_offset[0]),
                        "receiver_offset_y_d": float(receiver_offset[1]),
                        "sign": sign,
                        "amplitude_d": amplitude,
                        "delta_c_x_d": float(delta[0]),
                        "delta_c_y_d": float(delta[1]),
                        "status": summary["status"],
                        "failure_count": int(summary["failure_count"]),
                        "failures": json.dumps(
                            failures, ensure_ascii=False, sort_keys=True
                        ),
                        "broadcast_slots": int(summary["broadcast_slots"]),
                        "tx_uses": int(summary["tx_uses"]),
                        "action_count": int(len(actions)),
                        "records": int(len(records)),
                        "final_apex_error_d": float(
                            np.linalg.norm(final[0] - TARGET_TEMPLATE[0])
                        ),
                        "final_max_receiver_error_d": max(receiver_errors.values()),
                        "final_team_max_error_d": float(
                            max(np.linalg.norm(final - TARGET_TEMPLATE, axis=1))
                        ),
                        # This is the target-point G prediction for delta C only;
                        # the receiver offset is deliberately kept in the actual run.
                        "predicted_max_receiver_error_d": max(
                            predicted_receiver_errors.values()
                        ),
                        "receiver_errors_json": json.dumps(
                            receiver_errors, sort_keys=True
                        ),
                        "predicted_receiver_errors_json": json.dumps(
                            predicted_receiver_errors, sort_keys=True
                        ),
                        "final_max_angle_error_rad": float(
                            summary["final_max_angle_error_rad"]
                        ),
                    }
                )
    return rows


def counterexample_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    upper = TARGET_TEMPLATE[0].copy()
    lower = np.array([upper[0], -upper[1]])
    upper_observed = bootstrap_angles(_triangle_state(upper))
    lower_observed = bootstrap_angles(_triangle_state(lower))
    mirror_estimate = bootstrap_from_angles(lower_observed)
    rows.append(
        {
            "family": "bootstrap_mirror",
            "case": "lower_apex_same_unsigned_base_angles",
            "receiver_id": "",
            "initial_id": "",
            "point_x": float(lower[0]),
            "point_y": float(lower[1]),
            "candidate_x": float(mirror_estimate[0]),
            "candidate_y": float(mirror_estimate[1]),
            "observation_difference": float(
                np.linalg.norm(lower_observed - upper_observed)
            ),
            "position_error_d": float(np.linalg.norm(mirror_estimate - lower)),
            "accepted": True,
            "message": "unsigned bootstrap angles choose the canonical upper branch",
        }
    )

    center = np.array([2.0, 2.0 * np.sqrt(3.0) / 3.0])
    radius = 4.0 / np.sqrt(3.0)
    circle_points = [
        center + radius * np.array([np.cos(theta), np.sin(theta)])
        for theta in (0.0, 0.3)
    ]
    circle_observations = [
        receiver_angles(point, TARGET_ANCHORS) for point in circle_points
    ]
    for index, (point, observed) in enumerate(zip(circle_points, circle_observations)):
        default_result = estimate_receiver(2, observed, initial=TARGET_TEMPLATE[1])
        on_circle_result = estimate_receiver(2, observed, initial=point)
        rows.extend(
            [
                {
                    "family": "circumcircle_default_start",
                    "case": f"theta_{index}",
                    "receiver_id": 2,
                    "initial_id": "template",
                    "point_x": float(point[0]),
                    "point_y": float(point[1]),
                    "candidate_x": float(default_result["position"][0]),
                    "candidate_y": float(default_result["position"][1]),
                    "observation_difference": float(
                        np.linalg.norm(observed - circle_observations[0])
                    ),
                    "position_error_d": float(
                        np.linalg.norm(default_result["position"] - point)
                    ),
                    "accepted": bool(default_result["success"]),
                    "message": str(default_result["message"]),
                },
                {
                    "family": "circumcircle_on_branch_start",
                    "case": f"theta_{index}",
                    "receiver_id": 2,
                    "initial_id": "point_itself",
                    "point_x": float(point[0]),
                    "point_y": float(point[1]),
                    "candidate_x": float(on_circle_result["position"][0]),
                    "candidate_y": float(on_circle_result["position"][1]),
                    "observation_difference": float(
                        np.linalg.norm(observed - circle_observations[0])
                    ),
                    "position_error_d": float(
                        np.linalg.norm(on_circle_result["position"] - point)
                    ),
                    "accepted": bool(on_circle_result["success"]),
                    "message": str(on_circle_result["message"]),
                },
            ]
        )

    receiver_id = 2
    actual = TARGET_TEMPLATE[receiver_id - 1] + np.array([0.03, -0.02])
    starts = {
        "template": TARGET_TEMPLATE[receiver_id - 1],
        "nearby": TARGET_TEMPLATE[receiver_id - 1] + np.array([0.2, 0.2]),
        "reflected": TARGET_TEMPLATE[receiver_id - 1] * np.array([1.0, -1.0]),
        "far": np.array([2.0, 5.0]),
    }
    observed = receiver_angles(actual, TARGET_ANCHORS)
    for initial_id, initial in starts.items():
        result = estimate_receiver(receiver_id, observed, initial=initial)
        rows.append(
            {
                "family": "receiver_multistart",
                "case": "actual_Q_plus_0.03_minus_0.02",
                "receiver_id": receiver_id,
                "initial_id": initial_id,
                "point_x": float(actual[0]),
                "point_y": float(actual[1]),
                "candidate_x": float(result["position"][0]),
                "candidate_y": float(result["position"][1]),
                "observation_difference": 0.0,
                "position_error_d": float(np.linalg.norm(result["position"] - actual)),
                "accepted": bool(result["success"]),
                "message": str(result["message"]),
            }
        )
    return rows


def geometry_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} geometry",
        description="Finite geometry and branch-boundary audit for the FY01 triangle route.",
    )
    parser.add_argument("--output-dir", type=Path, default=geometry_DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_rows = bootstrap_boundary_rows()
    receiver_rows = receiver_boundary_rows()
    g_rows, g_info = g_worst_direction_rows()
    main_rows = main_stage_rows(g_info)
    counter_rows = counterexample_rows()

    triangle_write_csv(args.output_dir / "bootstrap_paths.csv", bootstrap_rows)
    triangle_write_csv(args.output_dir / "receiver_paths.csv", receiver_rows)
    triangle_write_csv(args.output_dir / "g_worst_direction.csv", g_rows)
    triangle_write_csv(args.output_dir / "main_stage_worst_direction.csv", main_rows)
    triangle_write_csv(args.output_dir / "counterexamples.csv", counter_rows)

    main_stage_by_amplitude: dict[str, dict[str, Any]] = {}
    for amplitude in sorted({float(row["amplitude_d"]) for row in main_rows}):
        selected = [row for row in main_rows if row["amplitude_d"] == amplitude]
        final_team_errors = [float(row["final_team_max_error_d"]) for row in selected]
        main_stage_by_amplitude[str(amplitude)] = {
            "row_count": len(selected),
            "converged_count": sum(row["status"] == "converged" for row in selected),
            "failure_count": sum(row["status"] != "converged" for row in selected),
            "final_team_max_error_range_d": [
                min(final_team_errors),
                max(final_team_errors),
            ],
            "broadcast_slots_range": [
                min(int(row["broadcast_slots"]) for row in selected),
                max(int(row["broadcast_slots"]) for row in selected),
            ],
            "tx_uses_range": [
                min(int(row["tx_uses"]) for row in selected),
                max(int(row["tx_uses"]) for row in selected),
            ],
        }

    circle_rows = [
        row
        for row in receiver_rows
        if row["path"] == "reference_circumcircle" and row["t"] == 0.0
    ]
    summary = {
        "contract": (
            "Finite noiseless geometry and branch-boundary audit; production estimator "
            "unchanged; rows are evidence only for listed paths, amplitudes, offsets, "
            "and starts."
        ),
        "bootstrap": {
            "row_count": len(bootstrap_rows),
            "paths": sorted({row["path"] for row in bootstrap_rows}),
            "interior_small_h_last_angle_sum": next(
                row["angle_sum_rad"]
                for row in reversed(bootstrap_rows)
                if row["path"] == "base_interior_x2"
            ),
            "exterior_small_h_last_angle_sum": next(
                row["angle_sum_rad"]
                for row in reversed(bootstrap_rows)
                if row["path"] == "base_exterior_x_minus1"
            ),
            "high_far_included": True,
        },
        "receiver_boundary": {
            "row_count": len(receiver_rows),
            "circumcircle_exact_row_count": len(circle_rows),
            "reference_ray_rows_with_one_nonsmooth_angle": sum(
                bool(row["nonsmooth_jacobian_rows"])
                and row["full_regular_row_count"] >= 2
                for row in receiver_rows
                if row["path"] == "reference_ray"
            ),
            "circumcircle_exact_selected_rank_deficient": sum(
                row["selected_rank"] < 2 for row in circle_rows
            ),
        },
        "g_worst_direction": g_info,
        "main_stage": {
            "row_count": len(main_rows),
            "converged_count": sum(row["status"] == "converged" for row in main_rows),
            "failure_count": sum(row["status"] != "converged" for row in main_rows),
            "receiver_offset_scenarios": list(RECEIVER_OFFSET_SCENARIOS),
            "by_amplitude": main_stage_by_amplitude,
        },
        "counterexamples": {
            "row_count": len(counter_rows),
            "rejected_count": sum(not row["accepted"] for row in counter_rows),
            "circumcircle_observation_pair_difference": float(
                np.linalg.norm(
                    receiver_angles(
                        np.array(
                            [circle_rows[0]["point_x"], circle_rows[0]["point_y"]]
                        ),
                        TARGET_ANCHORS,
                    )
                    - receiver_angles(
                        np.array(
                            [circle_rows[12]["point_x"], circle_rows[12]["point_y"]]
                        ),
                        TARGET_ANCHORS,
                    )
                )
            ),
        },
        "limits": [
            "No noisy observations, execution errors, or global uniqueness claim.",
            "A finite path approaching a boundary does not establish behavior elsewhere.",
            "Circumcircle rank loss is local and exact for the listed reference triangle.",
            "Main-stage runs retain the true FY01 residual and report online stopping separately from position error.",
        ],
    }
    geometry_write_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            geometry_json_value(summary), ensure_ascii=False, indent=2, allow_nan=False
        )
    )


# ============================================================================
# budget: calibration_budget
# ============================================================================


def apex_angle_box(
    observed: ArrayLike,
    half_width: ArrayLike | float,
    target: ArrayLike = TARGET_ANCHORS[0],
) -> dict:
    """Enclose every feasible apex under deterministic angle bounds.

    ``half_width`` bounds errors in radians. A standard deviation alone is
    not a deterministic bound. The known upper-side branch is required.
    """
    center = np.asarray(observed, dtype=float)
    widths = np.broadcast_to(np.asarray(half_width, dtype=float), (2,)).copy()
    goal = np.asarray(target, dtype=float)
    if center.shape != (2,) or goal.shape != (2,):
        raise ValueError("observed and target must each have shape (2,)")
    if (
        not np.isfinite(center).all()
        or not np.isfinite(widths).all()
        or not np.isfinite(goal).all()
    ):
        raise ValueError("angle box and target must be finite")
    if np.any(widths < 0):
        raise ValueError("angle half widths must be nonnegative")
    lower, upper = center - widths, center + widths
    if np.any(lower <= 0) or float(upper.sum()) >= np.pi:
        raise ValueError(
            "angle box must lie strictly in the bounded upper-triangle domain"
        )
    vertices = np.array(
        [
            bootstrap_from_angles([alpha, beta])
            for alpha, beta in product((lower[0], upper[0]), (lower[1], upper[1]))
        ]
    )
    distances = np.linalg.norm(vertices - goal, axis=1)
    return {
        "lower_angles_rad": lower.tolist(),
        "upper_angles_rad": upper.tolist(),
        "vertices": vertices.tolist(),
        "maximum_position_error_d": float(distances.max()),
        "maximizing_vertex": int(distances.argmax()),
    }


def equal_angle_budget(position_budget: float) -> float:
    """Largest equal base-angle target residual fitting an apex budget."""
    if not np.isfinite(position_budget) or position_budget <= 0:
        raise ValueError("position budget must be finite and positive")
    target_angles = np.full(2, np.pi / 3)
    return float(
        brentq(
            lambda epsilon: (
                apex_angle_box(target_angles, epsilon)["maximum_position_error_d"]
                - position_budget
            ),
            0.0,
            np.pi / 6 - 1e-6,
            xtol=1e-15,
        )
    )


def gaussian_angle_half_width(
    sigma_rad: float,
    samples_per_angle: int,
    *,
    family_error: float = 0.01,
    planned_checks: int = 1,
) -> float:
    """Simultaneous two-angle mean interval with a declared error budget.

    Bonferroni splits the family error across two angles and a predeclared
    finite number of checks. Known Gaussian marginal variances and independent
    repeated samples within each angle are assumed. Across-angle/check
    independence is not needed for the union bound.
    """
    if not np.isfinite(sigma_rad) or sigma_rad <= 0:
        raise ValueError("sigma_rad must be finite and positive")
    if not isinstance(samples_per_angle, (int, np.integer)) or samples_per_angle <= 0:
        raise ValueError("samples_per_angle must be a positive integer")
    if not isinstance(planned_checks, (int, np.integer)) or planned_checks <= 0:
        raise ValueError("planned_checks must be a positive integer")
    if not 0 < family_error < 1:
        raise ValueError("family_error must lie in (0,1)")
    z = float(norm.ppf(1 - family_error / (4 * planned_checks)))
    return z * sigma_rad / np.sqrt(samples_per_angle)


def gaussian_minimum_samples(
    position_budget: float,
    sigma_rad: float,
    *,
    planned_checks: int = 1,
    family_error: float = 0.01,
) -> int:
    """Best-centered K: interval is centered exactly at target base angles.

    Actual measured means away from target need the full apex_angle_box test
    and may require more samples or another adjustment.
    """
    epsilon = equal_angle_budget(position_budget)
    one_sample_width = gaussian_angle_half_width(
        sigma_rad,
        1,
        family_error=family_error,
        planned_checks=planned_checks,
    )
    return max(1, int(np.ceil((one_sample_width / epsilon) ** 2)))


def budget_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} budget",
        description="Exact upper-branch apex bounds from two bounded base-angle intervals.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/q2/calibration_budget")
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for degree in (0.01, 0.05, 0.1):
        epsilon = float(np.deg2rad(degree))
        result = apex_angle_box(np.full(2, np.pi / 3), epsilon)
        rows.append(
            {
                "each_angle_bound_deg": degree,
                "linear_apex_bound_d": 8 * epsilon,
                "exact_apex_bound_d": result["maximum_position_error_d"],
            }
        )
    budget_rows = []
    for budget in (0.002, 0.0048, 0.005, 0.01):
        epsilon = equal_angle_budget(budget)
        budget_rows.append(
            {
                "apex_position_budget_d": budget,
                "equal_angle_bound_rad": epsilon,
                "equal_angle_bound_deg": float(np.rad2deg(epsilon)),
                "verified_exact_bound_d": apex_angle_box(
                    np.full(2, np.pi / 3), epsilon
                )["maximum_position_error_d"],
            }
        )
    triangle_write_csv(args.output_dir / "angle_to_position.csv", rows)
    triangle_write_csv(args.output_dir / "position_to_angle.csv", budget_rows)
    statistical_rows = []
    for sigma_deg in (0.01, 0.05, 0.1):
        for budget in (0.0048, 0.005):
            for checks in (1, 30):
                count = gaussian_minimum_samples(
                    budget, np.deg2rad(sigma_deg), planned_checks=checks
                )
                half_width = gaussian_angle_half_width(
                    np.deg2rad(sigma_deg), count, planned_checks=checks
                )
                bound = apex_angle_box(np.full(2, np.pi / 3), half_width)[
                    "maximum_position_error_d"
                ]
                statistical_rows.append(
                    {
                        "sigma_each_angle_deg": sigma_deg,
                        "apex_budget_d": budget,
                        "planned_checks": checks,
                        "family_error_probability": 0.01,
                        "samples_per_angle_best_centered": count,
                        "one_check_measurement_slots": 2 * count,
                        "one_check_tx_uses": 4 * count,
                        "angle_mean_interval_half_width_deg": np.rad2deg(half_width),
                        "best_centered_exact_apex_radius_d": bound,
                    }
                )
    triangle_write_csv(
        args.output_dir / "gaussian_confidence_budget.csv", statistical_rows
    )
    summary = {
        "deterministic_contract": "Bounded intervals for two base angles; canonical upper branch; alpha>0, beta>0, alpha+beta<pi. Geometry alone assumes no noise distribution.",
        "gaussian_contract": "Known Gaussian single-sample variance; fixed independent samples per angle within each stationary batch; predeclared maximum checks; Bonferroni family error control. Sample requirements assume means centered exactly at target, not guaranteed acceptance.",
        "angle_to_position": rows,
        "position_to_angle": budget_rows,
        "gaussian_confidence_budget": statistical_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


# ============================================================================
# noise: analyze_noise_gain
# ============================================================================

OUTPUT_DIR = Path("outputs/q2/noise_gain")
SCENARIO_SIGMA_DEG = (0.01, 0.05, 0.10)
noise_GAINS = (0.25, 0.50, 0.75, 1.00)
NONLINEAR_GAIN_PAIRS = tuple((eta, eta) for eta in noise_GAINS) + ((0.50, 0.25),)
N_BOOTSTRAP = 8
N_MAIN = 18
N_LINEAR_TRIALS = 8_000
NONLINEAR_CASES = (
    (11, 0.01),
    (23, 0.01),
    (47, 0.05),
    (71, 0.05),
    (101, 0.10),
    (131, 0.10),
)
PI = float(np.pi)
FOLD_TOL = 1e-10


def noise_json_value(value: Any) -> Any:
    """Convert numpy values to strict JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): noise_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [noise_json_value(item) for item in value]
    return value


def noise_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wrap_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle to ``[-pi, pi)`` without changing array shape."""

    return (np.asarray(angle) + PI) % (2.0 * PI) - PI


def fold_unsigned_difference(first: float, second: float) -> float:
    """Return the unsigned angle in ``[0, pi]`` with the 0/pi fold."""

    return float(abs(float(wrap_pi(first - second))))


def azimuth_angles(receiver: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Return the three labeled ray azimuths from one receiver."""

    vectors = np.asarray(anchors, dtype=float) - np.asarray(receiver, dtype=float)
    return np.arctan2(vectors[:, 1], vectors[:, 0])


def pairwise_from_azimuth(azimuth: np.ndarray) -> np.ndarray:
    """Generate the three unsigned angles from three noisy azimuths.

    ``azimuth`` may have shape ``(3,)`` or any leading batch shape ending in
    ``(3,)``.  The pair ordering follows ``ANCHOR_PAIRS`` in the production
    estimator.  The explicit fold is important near 0 and pi.
    """

    values = np.asarray(azimuth, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError("azimuth must end in a length-three axis")
    result = np.empty(values.shape[:-1] + (3,), dtype=float)
    for row, (first, second) in enumerate(ANCHOR_PAIRS):
        result[..., row] = np.abs(wrap_pi(values[..., first] - values[..., second]))
    return result


def _azimuth_difference_sign(first: float, second: float) -> float | None:
    """Derivative sign for one smooth folded pair, or ``None`` at a fold."""

    difference = float(wrap_pi(first - second))
    if abs(difference) <= FOLD_TOL or abs(abs(difference) - PI) <= FOLD_TOL:
        return None
    return float(np.sign(difference))


def azimuth_difference_matrix(
    receiver_id: int,
    *,
    selected_only: bool = True,
) -> tuple[tuple[int, int], np.ndarray]:
    """Return the local ``D`` matrix mapping azimuth errors to angle errors.

    A selected row is smooth by construction.  Asking for all rows retains
    ``nan`` at a 0/pi fold, making the undefined linearization visible rather
    than silently assigning a sign.
    """

    receiver = TARGET_TEMPLATE[receiver_id - 1]
    azimuth = azimuth_angles(receiver, TARGET_ANCHORS)
    if selected_only:
        indices = tuple(int(value) for value in SELECTED_ANGLE_INDICES[receiver_id])
    else:
        indices = tuple(range(3))
    # Selected rows are regular and need a genuine linear map with zeros in
    # the unused azimuth columns.  The full three-row diagnostic retains NaN
    # only where the 0/pi fold makes a derivative undefined.
    matrix = np.zeros((len(indices), 3), dtype=float)
    for output_row, angle_index in enumerate(indices):
        first, second = ANCHOR_PAIRS[angle_index]
        sign = _azimuth_difference_sign(azimuth[first], azimuth[second])
        if sign is not None:
            matrix[output_row, first] = sign
            matrix[output_row, second] = -sign
        elif not selected_only:
            matrix[output_row, :] = np.nan
    return indices, matrix


def bootstrap_covariance(sigma_deg: float) -> dict[str, np.ndarray | float]:
    """Return base-angle and FY01 position noise covariance for one scenario."""

    sigma = np.deg2rad(float(sigma_deg))
    sigma_theta = sigma
    jacobian = bootstrap_jacobian(TARGET_ANCHORS[0])
    inverse = np.linalg.inv(jacobian)
    sigma_angles = sigma_theta**2 * np.eye(2)
    sigma_position = inverse @ sigma_angles @ inverse.T
    return {
        "sigma_theta_rad": sigma_theta,
        "sigma_theta_deg": float(sigma_deg),
        "sigma_angles": sigma_angles,
        "J_C": jacobian,
        "J_C_inverse": inverse,
        "Sigma_C_measurement": sigma_position,
    }


def receiver_noise_contract(sigma_deg: float) -> list[dict[str, Any]]:
    """Build selected-row D and angle covariance tables for all receivers."""

    sigma_theta = np.deg2rad(float(sigma_deg))
    sigma_azimuth = sigma_theta / np.sqrt(2.0)
    rows: list[dict[str, Any]] = []
    for receiver_id in RECEIVER_IDS:
        selected, d_selected = azimuth_difference_matrix(
            receiver_id, selected_only=True
        )
        _, d_full = azimuth_difference_matrix(receiver_id, selected_only=False)
        sigma_selected = sigma_azimuth**2 * (d_selected @ d_selected.T)
        j, b, g = propagation_blocks(receiver_id)
        j_inverse = np.linalg.inv(j)
        sigma_position = j_inverse @ sigma_selected @ j_inverse.T
        rows.append(
            {
                "sigma_theta_deg": float(sigma_deg),
                "sigma_azimuth_deg": float(np.rad2deg(sigma_azimuth)),
                "receiver_id": receiver_id,
                "selected_indices": ";".join(str(index) for index in selected),
                "selected_pairs": ";".join(
                    f"{first + 1}-{second + 1}"
                    for first, second in (ANCHOR_PAIRS[index] for index in selected)
                ),
                "D_selected": json.dumps(noise_json_value(d_selected)),
                "D_full": json.dumps(noise_json_value(d_full)),
                "Sigma_theta_selected": json.dumps(noise_json_value(sigma_selected)),
                "Sigma_receiver_measurement": json.dumps(
                    noise_json_value(sigma_position)
                ),
                "sigma_receiver_x": float(np.sqrt(sigma_position[0, 0])),
                "sigma_receiver_y": float(np.sqrt(sigma_position[1, 1])),
                "g_spectral_norm": float(np.linalg.norm(g, 2)),
                "selected_rows_are_smooth": bool(np.isfinite(d_selected).all()),
            }
        )
    return rows


def row_selection_comparison(sigma_deg: float = 0.05) -> list[dict[str, Any]]:
    """Compare geometric row selection with covariance-aware alternatives.

    The production selector maximizes the unweighted ``sigma_min(J)``.  With
    shared azimuth noise the angle covariance is ``sigma_phi^2 D D.T``; this
    table computes the resulting position covariance for every regular pair.
    It is diagnostic only and does not replace the production choice.
    """

    sigma_phi = np.deg2rad(float(sigma_deg)) / np.sqrt(2.0)
    rows: list[dict[str, Any]] = []
    for receiver_id in RECEIVER_IDS:
        point = TARGET_TEMPLATE[receiver_id - 1]
        jacobian_full = angle_jacobian(point, TARGET_ANCHORS, allow_degenerate=True)
        _, d_full = azimuth_difference_matrix(receiver_id, selected_only=False)
        regular = [
            index
            for index in range(3)
            if np.isfinite(jacobian_full[index]).all()
            and np.isfinite(d_full[index]).all()
        ]
        production = tuple(int(value) for value in SELECTED_ANGLE_INDICES[receiver_id])
        candidates: list[dict[str, Any]] = []
        for pair in combinations(regular, 2):
            j_pair = jacobian_full[list(pair)]
            d_pair = d_full[list(pair)]
            sigma_theta = sigma_phi**2 * (d_pair @ d_pair.T)
            inverse = np.linalg.inv(j_pair)
            sigma_position = inverse @ sigma_theta @ inverse.T
            singular = np.linalg.svd(j_pair, compute_uv=False)
            candidate = {
                "receiver_id": receiver_id,
                "pair_indices": ";".join(map(str, pair)),
                "pair_labels": ";".join(
                    f"{first + 1}-{second + 1}"
                    for first, second in (ANCHOR_PAIRS[index] for index in pair)
                ),
                "production_selected": bool(tuple(pair) == production),
                "sigma_min_J": float(singular[-1]),
                "trace_position_cov_d2": float(np.trace(sigma_position)),
                "lambda_max_position_cov_d2": float(
                    np.linalg.eigvalsh(sigma_position)[-1]
                ),
                "position_covariance": json.dumps(noise_json_value(sigma_position)),
            }
            candidates.append(candidate)
        best_trace_value = min(row["trace_position_cov_d2"] for row in candidates)
        best_lambda_value = min(row["lambda_max_position_cov_d2"] for row in candidates)
        trace_tolerance = 1e-10 * max(1.0, abs(best_trace_value))
        lambda_tolerance = 1e-10 * max(1.0, abs(best_lambda_value))
        for candidate in candidates:
            candidate["covariance_best_trace"] = (
                abs(candidate["trace_position_cov_d2"] - best_trace_value)
                <= trace_tolerance
            )
            candidate["covariance_best_lambda_max"] = (
                abs(candidate["lambda_max_position_cov_d2"] - best_lambda_value)
                <= lambda_tolerance
            )
            candidate["geometric_selector_matches_trace"] = bool(
                candidate["production_selected"] and candidate["covariance_best_trace"]
            )
            candidate["geometric_selector_matches_lambda_max"] = bool(
                candidate["production_selected"]
                and candidate["covariance_best_lambda_max"]
            )
            rows.append(candidate)
    return rows


def _geometric_sum(eta: float, count: int) -> float:
    """Return ``eta^2 sum_{j=0}^{count-1}(1-eta)^(2j)``."""

    if count <= 0:
        return 0.0
    a = 1.0 - float(eta)
    if abs(1.0 - a * a) < 1e-14:
        return float(count * eta * eta)
    return float(eta * eta * (1.0 - a ** (2 * count)) / (1.0 - a * a))


def _locked_reference_theory(
    eta_c: float,
    *,
    n_bootstrap: int,
    e_c0: np.ndarray,
    sigma_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of the frozen FY01 error after bootstrap."""

    covariance = bootstrap_covariance(sigma_deg)
    a = 1.0 - float(eta_c)
    mean = a**n_bootstrap * np.asarray(e_c0, dtype=float)
    covariance_n = (
        _geometric_sum(eta_c, n_bootstrap) * covariance["Sigma_C_measurement"]
    )
    return mean, np.asarray(covariance_n, dtype=float)


def linear_theory(
    sigma_deg: float,
    eta_c: float,
    eta_r: float,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    n_main: int = N_MAIN,
    e_c0: np.ndarray | None = None,
    e_receiver0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Finite-horizon bias/covariance formula for all 12 receivers.

    ``e_receiver0`` is a deterministic ``(12,2)`` array by default.  The
    reference covariance appears in every cross-machine block because the
    reference is measured, locked, and then held fixed.
    """

    if not (0.0 < eta_c < 2.0 and 0.0 < eta_r < 2.0):
        raise ValueError("both gains must satisfy 0 < eta < 2")
    if e_c0 is None:
        e_c0 = np.array([0.08, -0.06], dtype=float)
    if e_receiver0 is None:
        e_receiver0 = np.zeros((len(RECEIVER_IDS), 2), dtype=float)
    e_c0 = np.asarray(e_c0, dtype=float)
    e_receiver0 = np.asarray(e_receiver0, dtype=float)
    if e_c0.shape != (2,) or e_receiver0.shape != (len(RECEIVER_IDS), 2):
        raise ValueError("initial error shapes are invalid")

    mean_c, covariance_c = _locked_reference_theory(
        eta_c, n_bootstrap=n_bootstrap, e_c0=e_c0, sigma_deg=sigma_deg
    )
    a_r = 1.0 - float(eta_r)
    attenuation = a_r**n_main
    main_sum = _geometric_sum(eta_r, n_main)
    sigma_theta = np.deg2rad(float(sigma_deg))
    sigma_azimuth = sigma_theta / np.sqrt(2.0)
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    g_matrices: list[np.ndarray] = []
    receiver_covariances: list[np.ndarray] = []
    for index, receiver_id in enumerate(RECEIVER_IDS):
        selected, d_selected = azimuth_difference_matrix(
            receiver_id, selected_only=True
        )
        sigma_theta_selected = sigma_azimuth**2 * (d_selected @ d_selected.T)
        j, _, g = propagation_blocks(receiver_id)
        v = np.linalg.inv(j)
        sigma_v = v @ sigma_theta_selected @ v.T
        mean_i = attenuation * e_receiver0[index] - (1.0 - attenuation) * g @ mean_c
        covariance_i = (
            1.0 - attenuation
        ) ** 2 * g @ covariance_c @ g.T + main_sum * sigma_v
        means.append(mean_i)
        covariances.append(covariance_i)
        g_matrices.append(g)
        receiver_covariances.append(sigma_v)
    means_array = np.asarray(means)
    covariance_array = np.asarray(covariances)
    cross_covariance = np.empty((len(RECEIVER_IDS), len(RECEIVER_IDS), 2, 2))
    for i, g_i in enumerate(g_matrices):
        for j, g_j in enumerate(g_matrices):
            cross_covariance[i, j] = (
                (1.0 - attenuation) ** 2 * g_i @ covariance_c @ g_j.T
            )
            if i == j:
                cross_covariance[i, j] += main_sum * receiver_covariances[i]
    per_receiver_mse = np.sum(means_array**2, axis=1) + np.trace(
        covariance_array, axis1=1, axis2=2
    )
    return {
        "sigma_theta_deg": float(sigma_deg),
        "eta_c": float(eta_c),
        "eta_r": float(eta_r),
        "n_bootstrap": int(n_bootstrap),
        "n_main": int(n_main),
        "mean_reference_error": mean_c,
        "covariance_reference_error": covariance_c,
        "mean_receiver_error": means_array,
        "covariance_receiver_error": covariance_array,
        "cross_covariance_receiver_error": cross_covariance,
        "g_matrices": np.asarray(g_matrices),
        "receiver_measurement_covariances": np.asarray(receiver_covariances),
        "position_rms_d": float(np.sqrt(np.mean(per_receiver_mse))),
        "reference_rms_d": float(np.sqrt(np.sum(mean_c**2) + np.trace(covariance_c))),
        "white_noise_steady_multiplier": float(eta_r / (2.0 - eta_r)),
        "reference_white_noise_steady_multiplier": float(eta_c / (2.0 - eta_c)),
    }


def _apply_linear_bootstrap_noise(
    base_noise: np.ndarray,
    eta_c: float,
    e_c0: np.ndarray,
    sigma_deg: float,
) -> np.ndarray:
    covariance = bootstrap_covariance(sigma_deg)
    inverse = np.asarray(covariance["J_C_inverse"])
    measurement = np.einsum("ab,nkb->nka", inverse, base_noise)
    current = np.broadcast_to(e_c0, (base_noise.shape[0], 2)).copy()
    for round_index in range(base_noise.shape[1]):
        current = (1.0 - eta_c) * current - eta_c * measurement[:, round_index]
    return current


def linear_monte_carlo(
    sigma_deg: float,
    eta_c: float,
    eta_r: float,
    *,
    base_noise: np.ndarray,
    azimuth_noise: np.ndarray,
    e_c0: np.ndarray,
    e_receiver0: np.ndarray,
) -> dict[str, Any]:
    """Vectorized linear recurrence using shared noise arrays."""

    reference_error = _apply_linear_bootstrap_noise(base_noise, eta_c, e_c0, sigma_deg)
    current = np.broadcast_to(
        e_receiver0, (base_noise.shape[0], len(RECEIVER_IDS), 2)
    ).copy()
    for round_index in range(azimuth_noise.shape[1]):
        update = np.empty_like(current)
        for receiver_index, receiver_id in enumerate(RECEIVER_IDS):
            selected, d_selected = azimuth_difference_matrix(
                receiver_id, selected_only=True
            )
            epsilon = np.einsum(
                "ab,nb->na",
                d_selected,
                azimuth_noise[:, round_index, receiver_index, :],
            )
            epsilon *= 1.0  # D already maps azimuth errors; scale is in the input.
            j, _, _ = propagation_blocks(receiver_id)
            candidate_noise = np.einsum("ab,nb->na", np.linalg.inv(j), epsilon)
            _, _, g = propagation_blocks(receiver_id)
            update[:, receiver_index, :] = (1.0 - eta_r) * current[
                :, receiver_index, :
            ] - eta_r * (np.einsum("ab,nb->na", g, reference_error) + candidate_noise)
        current = update
    final = np.concatenate((reference_error[:, None, :], current), axis=1)
    position_rms = np.sqrt(np.mean(np.sum(current**2, axis=2)))
    max_error = np.max(np.linalg.norm(final, axis=2), axis=1)
    return {
        "final_reference_error": reference_error,
        "final_receiver_error": current,
        "position_rms_d": float(position_rms),
        "position_max_error_sample_d": float(np.max(max_error)),
        "position_p99_error_sample_d": float(np.quantile(max_error, 0.99)),
    }


def _initial_team(
    seed: int, *, reference_error: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build one deterministic local initial state and its receiver errors."""

    rng = np.random.default_rng(seed + 900_000)
    perturbations = rng.normal(size=(len(RECEIVER_IDS), 2))
    norms = np.linalg.norm(perturbations, axis=1, keepdims=True)
    perturbations = perturbations / np.maximum(norms, 1e-12) * 0.04
    team = TARGET_TEMPLATE.copy()
    team[0] += reference_error
    for index, receiver_id in enumerate(RECEIVER_IDS):
        team[receiver_id - 1] += perturbations[index]
    return team, perturbations


def nonlinear_fixed_budget(
    *,
    seed: int,
    sigma_deg: float,
    eta_c: float,
    eta_r: float,
    base_noise: np.ndarray,
    azimuth_noise: np.ndarray,
    initial_reference_error: np.ndarray,
    initial_receiver_errors: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one fixed-budget nonlinear trajectory and preserve failures."""

    team = TARGET_TEMPLATE.copy()
    team[0] += initial_reference_error
    for index, receiver_id in enumerate(RECEIVER_IDS):
        team[receiver_id - 1] += initial_receiver_errors[index]
    target_apex = TARGET_TEMPLATE[0]
    failures: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    bootstrap_displacement = 0.0
    main_displacement = 0.0
    bootstrap_failures = 0
    receiver_failures = 0
    production_acceptance_hits = 0
    original_angle_stop_hits = 0
    target_bootstrap_angles = bootstrap_angles(TARGET_TEMPLATE)

    # The two base-angle noises are fixed per case and replayed across gains.
    for round_index in range(N_BOOTSTRAP):
        true_angles = bootstrap_angles(team)
        observed = true_angles + base_noise[round_index]
        try:
            estimate = np.asarray(bootstrap_from_angles(observed), dtype=float)
            if not np.isfinite(estimate).all():
                raise ValueError("non-finite bootstrap estimate")
            error = estimate - target_apex
            action = -eta_c * error
            team[0] += action
            bootstrap_displacement += float(np.linalg.norm(action))
            trace_rows.append(
                {
                    "stage": "bootstrap",
                    "round": round_index + 1,
                    "receiver_id": "FY01",
                    "observed": observed.copy(),
                    "estimate": estimate.copy(),
                    "action": action.copy(),
                    "success": True,
                    "max_angle_residual_rad": None,
                }
            )
            if float(np.max(np.abs(observed - target_bootstrap_angles))) <= 1e-8:
                original_angle_stop_hits += 1
        except (ValueError, FloatingPointError) as error:
            bootstrap_failures += 1
            failures.append(
                {
                    "seed": seed,
                    "sigma_theta_deg": sigma_deg,
                    "eta_c": eta_c,
                    "eta_r": eta_r,
                    "stage": "bootstrap",
                    "round": round_index + 1,
                    "receiver_id": "FY01",
                    "failure": str(error),
                }
            )
            trace_rows.append(
                {
                    "stage": "bootstrap",
                    "round": round_index + 1,
                    "receiver_id": "FY01",
                    "observed": observed.copy(),
                    "estimate": None,
                    "action": np.zeros(2),
                    "success": False,
                    "max_angle_residual_rad": None,
                }
            )

    anchors = np.vstack((team[0], team[10], team[14]))
    for round_index in range(N_MAIN):
        for receiver_index, receiver_id in enumerate(RECEIVER_IDS):
            point = team[receiver_id - 1]
            true_azimuth = azimuth_angles(point, anchors)
            observed = pairwise_from_azimuth(
                true_azimuth + azimuth_noise[round_index, receiver_index]
            )
            result = estimate_receiver(receiver_id, observed)
            position = np.asarray(result["position"], dtype=float)
            if bool(result["success"]):
                production_acceptance_hits += 1
            if not bool(result["success"]):
                receiver_failures += 1
                failures.append(
                    {
                        "seed": seed,
                        "sigma_theta_deg": sigma_deg,
                        "eta_c": eta_c,
                        "eta_r": eta_r,
                        "stage": "main",
                        "round": round_index + 1,
                        "receiver_id": f"FY{receiver_id:02d}",
                        "failure": result["message"],
                        "max_angle_residual_rad": result["max_angle_residual"],
                        "nfev": result["nfev"],
                    }
                )
            # A rejected production estimate is preserved as a failure and
            # does not move the simulator.  This keeps fixed-budget scoring
            # faithful if a future noise/initial-state case fails.
            action = np.zeros(2, dtype=float)
            if bool(result["success"]) and np.isfinite(position).all():
                action = -eta_r * (position - TARGET_TEMPLATE[receiver_id - 1])
                team[receiver_id - 1] += action
                main_displacement += float(np.linalg.norm(action))
            trace_rows.append(
                {
                    "stage": "main",
                    "round": round_index + 1,
                    "receiver_id": f"FY{receiver_id:02d}",
                    "observed": observed.copy(),
                    "estimate": position.copy(),
                    "action": action.copy(),
                    "success": bool(result["success"]),
                    "max_angle_residual_rad": float(result["max_angle_residual"]),
                }
            )
            target_angles = receiver_angles(
                TARGET_TEMPLATE[receiver_id - 1], TARGET_ANCHORS
            )
            if float(np.max(np.abs(observed - target_angles))) <= 1e-8:
                original_angle_stop_hits += 1
        anchors = np.vstack((team[0], team[10], team[14]))

    errors = team - TARGET_TEMPLATE
    receiver_errors = errors[np.asarray(RECEIVER_IDS) - 1]
    all_norms = np.linalg.norm(errors, axis=1)
    summary = {
        "seed": int(seed),
        "sigma_theta_deg": float(sigma_deg),
        "eta_c": float(eta_c),
        "eta_r": float(eta_r),
        "n_bootstrap": N_BOOTSTRAP,
        "n_main": N_MAIN,
        "online_status": "not_evaluated_fixed_budget",
        "original_1e-8_stop_applicable": False,
        "production_acceptance_hits": int(production_acceptance_hits),
        "original_angle_stop_hits": int(original_angle_stop_hits),
        "bootstrap_failure_count": int(bootstrap_failures),
        "receiver_failure_count": int(receiver_failures),
        "failure_count": int(len(failures)),
        "final_reference_error_norm_d": float(np.linalg.norm(errors[0])),
        "final_receiver_rms_d": float(
            np.sqrt(np.mean(np.sum(receiver_errors**2, axis=1)))
        ),
        "final_receiver_max_d": float(np.max(np.linalg.norm(receiver_errors, axis=1))),
        "final_team_rms_d": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        "final_team_max_d": float(np.max(all_norms)),
        "bootstrap_action_displacement_d": float(bootstrap_displacement),
        "main_action_displacement_d": float(main_displacement),
        "total_action_displacement_d": float(
            bootstrap_displacement + main_displacement
        ),
        "measurement_slots": int(2 * N_BOOTSTRAP + N_MAIN),
        "transmitter_uses": int(4 * N_BOOTSTRAP + 3 * N_MAIN),
        "final_positions": team,
    }
    return summary, failures, trace_rows


def _noise_inputs_for_case(
    seed: int, sigma_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sigma_theta = np.deg2rad(float(sigma_deg))
    sigma_azimuth = sigma_theta / np.sqrt(2.0)
    base = rng.normal(0.0, sigma_theta, size=(N_BOOTSTRAP, 2))
    azimuth = rng.normal(
        0.0,
        sigma_azimuth,
        size=(N_MAIN, len(RECEIVER_IDS), 3),
    )
    return base, azimuth


def _linear_noise_tables() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    e_c0 = np.array([0.08, -0.06], dtype=float)
    linear_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    noise_archive: dict[str, Any] = {}
    rng = np.random.default_rng(20260906)
    for sigma_deg in SCENARIO_SIGMA_DEG:
        contract_rows.extend(receiver_noise_contract(sigma_deg))
        base = rng.normal(
            0.0, np.deg2rad(sigma_deg), size=(N_LINEAR_TRIALS, N_BOOTSTRAP, 2)
        )
        azimuth = rng.normal(
            0.0,
            np.deg2rad(sigma_deg) / np.sqrt(2.0),
            size=(N_LINEAR_TRIALS, N_MAIN, len(RECEIVER_IDS), 3),
        )
        for eta in noise_GAINS:
            theory = linear_theory(sigma_deg, eta, eta, e_c0=e_c0)
            simulation = linear_monte_carlo(
                sigma_deg,
                eta,
                eta,
                base_noise=base,
                azimuth_noise=azimuth,
                e_c0=e_c0,
                e_receiver0=np.zeros((len(RECEIVER_IDS), 2)),
            )
            linear_rows.append(
                {
                    "sigma_theta_deg": sigma_deg,
                    "sigma_azimuth_deg": sigma_deg / np.sqrt(2.0),
                    "eta_c": eta,
                    "eta_r": eta,
                    "n_bootstrap": N_BOOTSTRAP,
                    "n_main": N_MAIN,
                    "trials": N_LINEAR_TRIALS,
                    "theory_reference_rms_d": theory["reference_rms_d"],
                    "theory_receiver_rms_d": theory["position_rms_d"],
                    "mc_reference_rms_d": float(
                        np.sqrt(
                            np.mean(
                                np.sum(simulation["final_reference_error"] ** 2, axis=1)
                            )
                        )
                    ),
                    "mc_receiver_rms_d": simulation["position_rms_d"],
                    "mc_team_max_error_p99_d": simulation[
                        "position_p99_error_sample_d"
                    ],
                    "mc_team_max_error_sample_d": simulation[
                        "position_max_error_sample_d"
                    ],
                    "white_noise_steady_multiplier_eta_r": theory[
                        "white_noise_steady_multiplier"
                    ],
                    "white_noise_steady_multiplier_eta_c": theory[
                        "reference_white_noise_steady_multiplier"
                    ],
                    "shared_reference_covariance_trace_d2": float(
                        np.trace(theory["covariance_reference_error"])
                    ),
                }
            )
        # Keep one compact seed-independent archive for reproducibility.  Full
        # linear arrays are regenerated from this fixed master seed in tests.
        noise_archive[f"sigma_{sigma_deg:g}_base_seed"] = 20260906
    return linear_rows, contract_rows, noise_archive


def independent_gain_grid() -> list[dict[str, Any]]:
    """Enumerate all 4x4 ``(eta_C, eta_R)`` theory combinations."""

    rows: list[dict[str, Any]] = []
    initial_reference = np.array([0.08, -0.06], dtype=float)
    # Use the same nonzero receiver perturbation construction as the
    # nonlinear fixed-budget cases.  A zero receiver initial error would make
    # the grid answer only the steady-noise question and would understate the
    # transient cost of eta_R.
    _, initial_receivers = _initial_team(11, reference_error=initial_reference)
    for sigma_deg in SCENARIO_SIGMA_DEG:
        for eta_c in noise_GAINS:
            for eta_r in noise_GAINS:
                result = linear_theory(
                    sigma_deg,
                    eta_c,
                    eta_r,
                    e_c0=initial_reference,
                    e_receiver0=initial_receivers,
                )
                rows.append(
                    {
                        "sigma_theta_deg": sigma_deg,
                        "eta_c": eta_c,
                        "eta_r": eta_r,
                        "n_bootstrap": N_BOOTSTRAP,
                        "n_main": N_MAIN,
                        "initial_receiver_error_rms_d": float(
                            np.sqrt(np.mean(np.sum(initial_receivers**2, axis=1)))
                        ),
                        "theory_reference_rms_d": result["reference_rms_d"],
                        "theory_receiver_rms_d": result["position_rms_d"],
                        "shared_reference_covariance_trace_d2": float(
                            np.trace(result["covariance_reference_error"])
                        ),
                        "white_noise_steady_multiplier_eta_c": result[
                            "reference_white_noise_steady_multiplier"
                        ],
                        "white_noise_steady_multiplier_eta_r": result[
                            "white_noise_steady_multiplier"
                        ],
                    }
                )
    return rows


def run_analysis(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Run all cheap and bounded nonlinear analyses and write artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    linear_rows, contract_rows, noise_archive = _linear_noise_tables()
    gain_grid_rows = independent_gain_grid()
    selection_rows = row_selection_comparison(0.05)
    nonlinear_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    nonlinear_traces: list[dict[str, Any]] = []
    input_archive: dict[str, Any] = {}
    initial_reference_error = np.array([0.08, -0.06], dtype=float)
    for seed, sigma_deg in NONLINEAR_CASES:
        base_noise, azimuth_noise = _noise_inputs_for_case(seed, sigma_deg)
        _, receiver_errors = _initial_team(
            seed, reference_error=initial_reference_error
        )
        input_archive[f"case_{seed}_{sigma_deg:g}_base"] = base_noise
        input_archive[f"case_{seed}_{sigma_deg:g}_azimuth"] = azimuth_noise
        input_archive[f"case_{seed}_{sigma_deg:g}_receiver_initial"] = receiver_errors
        for eta_c, eta_r in NONLINEAR_GAIN_PAIRS:
            result, failures, trace = nonlinear_fixed_budget(
                seed=seed,
                sigma_deg=sigma_deg,
                eta_c=eta_c,
                eta_r=eta_r,
                base_noise=base_noise,
                azimuth_noise=azimuth_noise,
                initial_reference_error=initial_reference_error,
                initial_receiver_errors=receiver_errors,
            )
            row = dict(result)
            row.pop("final_positions", None)
            nonlinear_rows.append(row)
            failure_rows.extend(failures)
            nonlinear_traces.append(
                {
                    "seed": seed,
                    "sigma_theta_deg": sigma_deg,
                    "eta_c": eta_c,
                    "eta_r": eta_r,
                    "summary": row,
                    "final_positions": result["final_positions"],
                    "trace": trace,
                }
            )
    np.savez_compressed(output_dir / "nonlinear_noise_inputs.npz", **input_archive)
    noise_write_csv(output_dir / "noise_contract.csv", contract_rows)
    noise_write_csv(output_dir / "row_selection_comparison.csv", selection_rows)
    noise_write_csv(output_dir / "linear_monte_carlo.csv", linear_rows)
    noise_write_csv(output_dir / "independent_gain_grid.csv", gain_grid_rows)
    noise_write_csv(output_dir / "nonlinear_trajectories.csv", nonlinear_rows)
    noise_write_csv(output_dir / "nonlinear_failures.csv", failure_rows)
    (output_dir / "nonlinear_traces.json").write_text(
        json.dumps(
            noise_json_value(nonlinear_traces),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    theory_example = linear_theory(0.05, 0.5, 0.5, e_c0=initial_reference_error)
    bootstrap = bootstrap_covariance(0.05)
    summary = {
        "contract": {
            "bootstrap": "每轮两底角独立 Gaussian；sigma_theta 为每条角行边际标准差",
            "main": "三条方位角先独立加 Gaussian，再按 0/pi fold 生成三条无向夹角；sigma_azimuth=sigma_theta/sqrt(2)",
            "selected_rows": "各接收机仅在线性协方差和局部控制更新中使用预选的两条光滑行；退化行保留 fold 观测和生产一致性失败",
            "execution": "坐标动作精确执行；无额外执行误差",
            "locking": "FY01 校准结束后固定；其随机误差是所有接收机的共同偏置源",
            "fixed_budget": f"n_bootstrap={N_BOOTSTRAP}, n_main={N_MAIN}; 不运行在线停止保证",
        },
        "gain_condition": {
            "linear_stability": "0 < eta_C < 2 且 0 < eta_R < 2",
            "local_scope": "目标附近、选定光滑行、连续调整、同一分支",
            "steady_white_noise_multiplier": "eta/(2-eta)",
            "interpretation": "固定参考残差不按白噪声稳态公式压缩；它进入共同固定偏置项",
        },
        "bootstrap_example_sigma_0.05deg": bootstrap,
        "example_theory_eta_0.5_sigma_0.05deg": theory_example,
        "linear_rows": len(linear_rows),
        "independent_gain_grid_rows": len(gain_grid_rows),
        "row_selection_rows": len(selection_rows),
        "row_selection_production_matches": {
            "trace": sum(
                bool(row["geometric_selector_matches_trace"])
                for row in selection_rows
                if row["production_selected"]
            ),
            "lambda_max": sum(
                bool(row["geometric_selector_matches_lambda_max"])
                for row in selection_rows
                if row["production_selected"]
            ),
        },
        "nonlinear_trajectories": len(nonlinear_rows),
        "nonlinear_failure_records": len(failure_rows),
        "nonlinear_failures_by_gain_pair": {
            f"{eta_c:g},{eta_r:g}": sum(
                row["eta_c"] == eta_c and row["eta_r"] == eta_r for row in failure_rows
            )
            for eta_c, eta_r in NONLINEAR_GAIN_PAIRS
        },
        "nonlinear_success_wording": "这些行都是固定预算轨迹；online_status 明确为 not_evaluated_fixed_budget，failure_count 保留生产 strict success 失败",
        "reproducibility": {
            "linear_master_seed": 20260906,
            "nonlinear_noise_file": "nonlinear_noise_inputs.npz",
            "nonlinear_cases": NONLINEAR_CASES,
            "nonlinear_gain_pairs": NONLINEAR_GAIN_PAIRS,
            "gains": noise_GAINS,
        },
        "limits": [
            "线性协方差只在 canonical 局部支、固定参考、连续调整下成立",
            "样本最大误差是所运行 Monte Carlo 的样本统计，不是硬上界",
            "非线性轨迹仅为 30 个固定预算案例，不能替代全初态域或在线停止成功率",
            "原有 1e-8 全三角残差阈值不适合作为这些含噪观测的停止保证",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(
            noise_json_value(summary), indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    # The archive metadata is small and human-readable; the actual arrays are
    # in the compressed NPZ next to the CSV tables.
    (output_dir / "noise_archive.json").write_text(
        json.dumps(noise_json_value(noise_archive), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return summary


def noise_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} noise",
        description="Q2 angle-noise, reference-residual, and gain analysis.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    summary = run_analysis(args.output_dir)
    print(json.dumps(noise_json_value(summary), indent=2, ensure_ascii=False))


# ============================================================================
# protocols: compare_reference_protocols
# ============================================================================

protocols_DEFAULT_OUTPUT = Path("outputs/q2/protocol_comparison")
RHO = 0.1
protocols_SEEDS = (11, 23, 47)
TAU_C_VALUES = (0.0048, 0.005)
GAIN_PAIRS = ((1.0, 1.0), (1.0, 0.75), (0.75, 0.75), (0.5, 0.5), (0.5, 1.0))
FINAL_POSITION_BUDGET = 0.01
FAIR_TAU_C = 0.005
FAIR_RECEIVER_ESTIMATE_BUDGET = 0.0045
STRICT_ANGLE_TOLERANCE = float(ANGLE_RESIDUAL_TOLERANCE)
protocols_MAX_NFEV = 500
# The local first-order FY01-to-receiver multiplier audited in
# docs/q2/参考残差传播与校准阈值.md.  It is reported as a budget diagnostic,
# not promoted to a nonlinear guarantee.
G_MAX_FIRST_ORDER = 1.040833


def _norm(value: Any) -> Any:
    """Convert numpy values recursively for JSON and CSV-friendly records."""

    return triangle_json_value(value)


def protocols_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_norm(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _as_anchors(anchors: np.ndarray) -> np.ndarray:
    value = np.asarray(anchors, dtype=float)
    if value.shape != (3, 2) or not np.isfinite(value).all():
        raise ValueError("known anchors must have shape (3, 2) and be finite")
    return value


def _choose_local_rows(receiver_id: int, anchors: np.ndarray) -> tuple[int, int]:
    """Select two smooth rows at the template initial point for known anchors.

    This is intentionally a local branch: the initial point is the canonical
    template location for the labelled receiver, while the anchor geometry is
    the explicit current estimate supplied by the protocol.  Every candidate
    pair is scored by its minimum singular value, and the complete triangle is
    checked after solving.
    """

    point = TARGET_TEMPLATE[int(receiver_id) - 1]
    gradients = angle_jacobian(point, _as_anchors(anchors), allow_degenerate=True)
    candidates: list[tuple[float, tuple[int, int]]] = []
    for first in range(len(ANCHOR_PAIRS)):
        for second in range(first + 1, len(ANCHOR_PAIRS)):
            matrix = gradients[[first, second]]
            if not np.isfinite(matrix).all():
                continue
            sigma_min = float(np.linalg.svd(matrix, compute_uv=False)[-1])
            if sigma_min > 0.0:
                candidates.append((sigma_min, (first, second)))
    if not candidates:
        raise ValueError(f"FY{receiver_id:02d} has no smooth local two-row branch")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def estimate_receiver_with_anchors(
    receiver_id: int,
    observed3: np.ndarray,
    known_anchors: np.ndarray,
    *,
    initial: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate one receiver using explicit current anchors.

    The estimator boundary contains only labelled angles, three known anchor
    coordinates, and an optional local template initial point.  It does not
    access the true state.  Two rows chosen at that initial point are solved;
    all three rows are then evaluated and the same $10^{-8}$ rad acceptance
    contract as the production solver is applied.
    """

    identifier = int(receiver_id)
    observed = np.asarray(observed3, dtype=float)
    if identifier not in RECEIVER_IDS:
        raise ValueError("receiver_id must be a non-reference identifier")
    if observed.shape != (3,) or not np.isfinite(observed).all():
        raise ValueError("observed angles must have shape (3,) and be finite")
    if np.any(observed < 0.0) or np.any(observed > np.pi):
        raise ValueError("observed angles must lie in [0, pi]")
    anchors = _as_anchors(known_anchors)
    start = (
        TARGET_TEMPLATE[identifier - 1].copy()
        if initial is None
        else np.asarray(initial, dtype=float).copy()
    )
    if start.shape != (2,) or not np.isfinite(start).all():
        raise ValueError("initial must be one finite 2-D point")
    try:
        selected = _choose_local_rows(identifier, anchors)

        def residual(point: np.ndarray) -> np.ndarray:
            return (
                receiver_angles(point, anchors)[list(selected)]
                - observed[list(selected)]
            )

        def jacobian(point: np.ndarray) -> np.ndarray:
            rows = angle_jacobian(point, anchors, allow_degenerate=True)[list(selected)]
            if not np.isfinite(rows).all():
                raise ValueError("selected local row reached an angle singularity")
            return rows

        result = least_squares(
            residual,
            start,
            jac=jacobian,
            method="trf",
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=protocols_MAX_NFEV,
        )
        position = np.asarray(result.x, dtype=float)
        predicted = receiver_angles(position, anchors)
        max_residual = float(np.max(np.abs(predicted - observed)))
        success = bool(result.success and max_residual <= STRICT_ANGLE_TOLERANCE)
        message = str(result.message)
        if result.success and not success:
            message = (
                f"solver converged but complete-triangle residual {max_residual:.3e} "
                f"exceeds {STRICT_ANGLE_TOLERANCE:.3e}"
            )
        return {
            "position": position,
            "success": success,
            "max_angle_residual": max_residual,
            "nfev": int(result.nfev),
            "selected_indices": list(selected),
            "message": message,
        }
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return {
            "position": start,
            "success": False,
            "max_angle_residual": float("inf"),
            "nfev": 0,
            "selected_indices": [],
            "message": f"geometry/solver failure: {error}",
        }


def _initial_result(case: Any) -> dict[str, Any]:
    return {
        "rho": float(case.rho),
        "seed": case.seed if case.seed is not None else "ideal",
        "initial_max_error_d": float(case.normalized_max_deviation),
        "initial_rms_error_d": float(case.normalized_rms_deviation),
        "actual_spacing_d": float(case.actual_spacing),
    }


def _action_distance(actions: Iterable[dict[str, Any]]) -> float:
    return float(
        sum(np.hypot(float(row["delta_x"]), float(row["delta_y"])) for row in actions)
    )


def _score_result(
    case: Any,
    state: np.ndarray,
    *,
    protocol: str,
    eta_c: float,
    eta_r: float,
    tau_c: float,
    stop_rule: str,
    measurement_slots: int,
    tx_uses: int,
    relay_scalars: int,
    rows: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    status: str,
    failures: list[str],
    bootstrap_summary: dict[str, Any] | None = None,
    main_summary: dict[str, Any] | None = None,
    receiver_estimate_budget: float | None = None,
    budget_group: str = "strict_angle",
    public_reference_scalars: int = 0,
) -> dict[str, Any]:
    errors = np.linalg.norm(np.asarray(state) - TARGET_TEMPLATE, axis=1)
    final_max_error = float(np.max(errors))
    budget_excess = max(0.0, final_max_error - FINAL_POSITION_BUDGET)
    budget_plus_reference = (
        None
        if receiver_estimate_budget is None
        else float(receiver_estimate_budget + G_MAX_FIRST_ORDER * tau_c)
    )
    return {
        **_initial_result(case),
        "protocol": protocol,
        "eta_c": float(eta_c),
        "eta_r": float(eta_r),
        "tau_c_d": float(tau_c),
        "final_position_budget_d": float(FINAL_POSITION_BUDGET),
        "receiver_estimate_budget_d": (
            None
            if receiver_estimate_budget is None
            else float(receiver_estimate_budget)
        ),
        "budget_group": budget_group,
        "stop_rule": stop_rule,
        "status": status,
        "online_status": "success" if status == "success" else "failure",
        "failures": sorted(set(failures)),
        "measurement_slots": int(measurement_slots),
        "tx_uses": int(tx_uses),
        "relay_scalar_count": int(relay_scalars),
        "public_reference_scalars": int(public_reference_scalars),
        "final_max_position_error_d": final_max_error,
        "final_position_budget_pass": bool(final_max_error <= FINAL_POSITION_BUDGET),
        "final_position_budget_excess_d": budget_excess,
        "first_order_reference_propagation_bound_d": float(G_MAX_FIRST_ORDER * tau_c),
        "online_budget_plus_reference_bound_d": budget_plus_reference,
        "final_rms_position_error_d": float(np.sqrt(np.mean(errors**2))),
        "cumulative_displacement_d": _action_distance(actions),
        "action_count": int(len(actions)),
        "bootstrap": bootstrap_summary or {},
        "main": main_summary or {},
        "rows": rows,
        "actions": actions,
        "final_positions": np.asarray(state),
    }


def run_staged_case(
    rho: float,
    seed: int,
    eta_c: float,
    eta_r: float,
    tau_c: float | None,
    *,
    max_rounds: int = MAX_ROUNDS,
) -> dict[str, Any]:
    """Run finite calibration followed by the production strict-angle stage."""

    case = make_initial_state(rho, seed)
    state, bootstrap, bootstrap_rows, bootstrap_actions = _bootstrap_record(
        case=case, gain=eta_c, max_rounds=max_rounds, position_tolerance=tau_c
    )
    bootstrap = {**bootstrap, "gain": float(eta_c)}
    failures: list[str] = []
    rows = [{"stage": "bootstrap", **row} for row in bootstrap_rows]
    actions = [{"stage": "bootstrap", **row} for row in bootstrap_actions]
    if bootstrap["status"] != "converged":
        failures.append(str(bootstrap.get("failure", "bootstrap_failure")))
        return _score_result(
            case,
            state,
            protocol="staged_fixed_reference",
            eta_c=eta_c,
            eta_r=eta_r,
            tau_c=0.0 if tau_c is None else tau_c,
            stop_rule="strict_main_angle_after_finite_calibration",
            measurement_slots=2 * int(bootstrap["rounds"]),
            tx_uses=int(bootstrap["tx_uses"]),
            relay_scalars=int(bootstrap["relay_scalar_count"]),
            rows=rows,
            actions=actions,
            status="failure",
            failures=failures,
            bootstrap_summary=bootstrap,
            receiver_estimate_budget=None,
            budget_group="strict_angle",
            public_reference_scalars=int(bootstrap["relay_scalar_count"]),
        )
    state, main, main_rows, main_actions = _main_record(
        case=case, gain=eta_r, state=state, max_rounds=max_rounds
    )
    main = {**main, "gain": float(eta_r)}
    rows.extend({"stage": "main", **row} for row in main_rows)
    actions.extend({"stage": "main", **row} for row in main_actions)
    if main["status"] != "converged":
        failures.extend(str(value) for value in main.get("failures", {}).values())
    return _score_result(
        case,
        state,
        protocol="staged_fixed_reference",
        eta_c=eta_c,
        eta_r=eta_r,
        tau_c=0.0 if tau_c is None else tau_c,
        stop_rule="strict_main_angle_after_finite_calibration",
        measurement_slots=2 * int(bootstrap["rounds"]) + int(main["broadcast_slots"]),
        tx_uses=int(bootstrap["tx_uses"] + main["tx_uses"]),
        relay_scalars=int(bootstrap["relay_scalar_count"]),
        rows=rows,
        actions=actions,
        status="success" if not failures else "failure",
        failures=failures,
        bootstrap_summary=bootstrap,
        main_summary=main,
        receiver_estimate_budget=None,
        budget_group="strict_angle",
        public_reference_scalars=int(bootstrap["relay_scalar_count"]),
    )


def _run_fixed_position_main(
    case: Any,
    state: np.ndarray,
    eta_r: float,
    max_rounds: int,
    position_budget: float,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Static-reference main stage with an observable estimated-position stop."""

    current = np.asarray(state, dtype=float).copy()
    active = set(RECEIVER_IDS)
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    failures: dict[int, str] = {}
    rounds = 0
    for round_i in range(1, max_rounds + 1):
        if not active:
            break
        rounds += 1
        observed: dict[int, np.ndarray] = {}
        pending: list[int] = []
        for receiver_id in sorted(active):
            angles = np.asarray(
                receiver_angles(
                    current[receiver_id - 1], current[np.asarray(REFERENCE_IDS) - 1]
                ),
                dtype=float,
            )
            observed[receiver_id] = angles
            rows.append(
                {
                    "stage": "main_budget",
                    "round": round_i,
                    "receiver_id": receiver_id,
                    "measurement_slot": round_i,
                    "measurement_tx_count": 3,
                    "angle_error_to_target_rad": float(
                        np.max(
                            np.abs(
                                angles
                                - receiver_angles(
                                    TARGET_TEMPLATE[receiver_id - 1], TARGET_ANCHORS
                                )
                            )
                        )
                    ),
                    "estimated_x": "",
                    "estimated_y": "",
                    "estimated_position_error_d": "",
                    "estimator_success": "",
                    "estimator_residual_rad": "",
                    "delta_x": 0.0,
                    "delta_y": 0.0,
                    "action_applied": False,
                    "event": "observe",
                    "failure": "",
                }
            )
            pending.append(receiver_id)
        for receiver_id in pending:
            estimate = estimate_receiver(
                receiver_id,
                observed[receiver_id],
                initial=TARGET_TEMPLATE[receiver_id - 1],
            )
            row = next(
                item
                for item in reversed(rows)
                if item["stage"] == "main_budget"
                and item["round"] == round_i
                and item["receiver_id"] == receiver_id
            )
            position = np.asarray(estimate["position"], dtype=float)
            row["estimated_x"] = float(position[0])
            row["estimated_y"] = float(position[1])
            row["estimated_position_error_d"] = float(
                np.linalg.norm(position - TARGET_TEMPLATE[receiver_id - 1])
            )
            row["estimator_success"] = bool(estimate["success"])
            row["estimator_residual_rad"] = float(estimate["max_angle_residual"])
            if not bool(estimate["success"]):
                failure = f"receiver_estimation_failure:{estimate['message']}"
                row["event"] = "failure"
                row["failure"] = failure
                failures[receiver_id] = failure
                active.remove(receiver_id)
                continue
            if row["estimated_position_error_d"] <= position_budget:
                row["event"] = "stop_estimated_position_budget"
                active.remove(receiver_id)
                continue
            delta = -float(eta_r) * (position - TARGET_TEMPLATE[receiver_id - 1])
            row["delta_x"] = float(delta[0])
            row["delta_y"] = float(delta[1])
            row["action_applied"] = True
            row["event"] = "observe_and_move"
            current[receiver_id - 1] += delta
            actions.append(
                {
                    "stage": "main_budget",
                    "round": round_i,
                    "receiver_id": receiver_id,
                    "delta_x": float(delta[0]),
                    "delta_y": float(delta[1]),
                    "true_position_after_x": float(current[receiver_id - 1, 0]),
                    "true_position_after_y": float(current[receiver_id - 1, 1]),
                }
            )
    for receiver_id in sorted(active):
        failures[receiver_id] = "receiver_budget_exhausted"
    final_errors = {
        receiver_id: float(
            np.max(
                np.abs(
                    receiver_angles(
                        current[receiver_id - 1], current[np.asarray(REFERENCE_IDS) - 1]
                    )
                    - receiver_angles(TARGET_TEMPLATE[receiver_id - 1], TARGET_ANCHORS)
                )
            )
        )
        for receiver_id in RECEIVER_IDS
    }
    summary = {
        "status": "converged" if not failures else "failure",
        "failure_count": len(failures),
        "failures": {str(key): value for key, value in failures.items()},
        "broadcast_slots": rounds,
        "tx_uses": rounds * 3,
        "stop_rule": "estimated_receiver_position",
        "position_budget_d": float(position_budget),
        "final_max_angle_error_rad": max(final_errors.values()),
        "receiver_status": {
            str(identifier): ("failure" if identifier in failures else "converged")
            for identifier in RECEIVER_IDS
        },
    }
    return current, summary, rows, actions


def run_staged_budget_case(
    rho: float,
    seed: int,
    eta_c: float,
    eta_r: float,
    tau_c: float,
    *,
    max_rounds: int = MAX_ROUNDS,
    position_budget: float = FINAL_POSITION_BUDGET,
    budget_group: str = "unallocated_reference_budget",
) -> dict[str, Any]:
    """Run staged finite calibration with the common observable position stop."""

    case = make_initial_state(rho, seed)
    state, bootstrap, bootstrap_rows, bootstrap_actions = _bootstrap_record(
        case=case, gain=eta_c, max_rounds=max_rounds, position_tolerance=tau_c
    )
    bootstrap = {**bootstrap, "gain": float(eta_c)}
    rows = [{"stage": "bootstrap", **row} for row in bootstrap_rows]
    actions = [{"stage": "bootstrap", **row} for row in bootstrap_actions]
    failures: list[str] = []
    main: dict[str, Any] = {}
    if bootstrap["status"] == "converged":
        state, main, main_rows, main_actions = _run_fixed_position_main(
            case, state, eta_r, max_rounds, position_budget
        )
        main = {**main, "gain": float(eta_r)}
        rows.extend(main_rows)
        actions.extend(main_actions)
        failures.extend(str(value) for value in main.get("failures", {}).values())
    else:
        failures.append(str(bootstrap.get("failure", "bootstrap_failure")))
    return _score_result(
        case,
        state,
        protocol="staged_fixed_reference",
        eta_c=eta_c,
        eta_r=eta_r,
        tau_c=0.0 if tau_c is None else tau_c,
        stop_rule="estimated_position_budget_after_finite_calibration",
        measurement_slots=2 * int(bootstrap["rounds"])
        + int(main.get("broadcast_slots", 0)),
        tx_uses=int(bootstrap["tx_uses"] + main.get("tx_uses", 0)),
        relay_scalars=int(bootstrap["relay_scalar_count"]),
        rows=rows,
        actions=actions,
        status="success" if not failures else "failure",
        failures=failures,
        bootstrap_summary=bootstrap,
        main_summary=main,
        receiver_estimate_budget=position_budget,
        budget_group=budget_group,
        public_reference_scalars=int(bootstrap["relay_scalar_count"]),
    )


def run_dynamic_case(
    rho: float,
    seed: int,
    eta_c: float,
    eta_r: float,
    tau_c: float,
    *,
    max_rounds: int = MAX_ROUNDS,
    position_budget: float = FINAL_POSITION_BUDGET,
    budget_group: str = "unallocated_reference_budget",
) -> dict[str, Any]:
    """Run the synchronised current-reference protocol.

    At each round every node is stationary during two bottom-angle slots and
    one (1,11,15) broadcast slot.  The two measured angles yield ``C_hat``;
    the three-anchor estimator receives ``[C_hat, FY11, FY15]``.  Only after
    all estimates have been computed are FY01 and all active receivers moved
    together.  The stop uses observable ``C_hat`` and receiver estimates.
    """

    case = make_initial_state(rho, seed)
    current = case.positions.copy()
    active_receivers = set(RECEIVER_IDS)
    calibration_active = True
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    failures: list[str] = []
    c_hat_error = float("inf")
    receiver_estimate_errors: dict[int, float] = {}
    round_summaries: list[dict[str, Any]] = []

    for round_i in range(1, max_rounds + 1):
        anchors_true = current[np.asarray(REFERENCE_IDS) - 1].copy()
        observed_bootstrap = np.asarray(bootstrap_angles(current), dtype=float)
        try:
            c_hat = np.asarray(bootstrap_from_angles(observed_bootstrap), dtype=float)
        except (ValueError, FloatingPointError) as error:
            failures.append(f"bootstrap_estimation_failure:{error}")
            break
        c_hat_error = float(np.linalg.norm(c_hat - TARGET_ANCHORS[0]))
        rows.append(
            {
                "stage": "dynamic_calibration",
                "round": round_i,
                "measurement_slot_count": 2,
                "measurement_tx_count": 4,
                "relay_scalar_count": 2,
                "alpha_rad": float(observed_bootstrap[0]),
                "beta_rad": float(observed_bootstrap[1]),
                "c_hat_x": float(c_hat[0]),
                "c_hat_y": float(c_hat[1]),
                "c_hat_error_to_target_d": c_hat_error,
                "event": "measure_current_reference",
                "failure": "",
            }
        )
        # FY11/FY15 are fixed by the coordinate contract and therefore use
        # their known canonical coordinates; only FY01 is inferred this round.
        known_anchors = np.vstack((c_hat, TARGET_ANCHORS[1], TARGET_ANCHORS[2]))
        pending: list[tuple[int, dict[str, Any]]] = []
        current_receiver_estimates: dict[int, np.ndarray] = {}
        receiver_failures: dict[int, str] = {}
        # All receiver observations are generated before any action is applied.
        for receiver_id in sorted(active_receivers):
            observed3 = np.asarray(
                receiver_angles(current[receiver_id - 1], anchors_true), dtype=float
            )
            result = estimate_receiver_with_anchors(
                receiver_id,
                observed3,
                known_anchors,
                initial=TARGET_TEMPLATE[receiver_id - 1],
            )
            estimate_position = np.asarray(result["position"], dtype=float)
            estimate_error = float(
                np.linalg.norm(estimate_position - TARGET_TEMPLATE[receiver_id - 1])
            )
            receiver_estimate_errors[receiver_id] = estimate_error
            row = {
                "stage": "dynamic_main",
                "round": round_i,
                "receiver_id": receiver_id,
                "measurement_slot_count": 1,
                "measurement_tx_count": 3,
                "c_hat_x_used": float(c_hat[0]),
                "c_hat_y_used": float(c_hat[1]),
                "angle_error_to_target_rad": float(
                    np.max(
                        np.abs(
                            observed3
                            - receiver_angles(
                                TARGET_TEMPLATE[receiver_id - 1], TARGET_ANCHORS
                            )
                        )
                    )
                ),
                "estimated_x": float(estimate_position[0]),
                "estimated_y": float(estimate_position[1]),
                "estimated_position_error_d": estimate_error,
                "estimator_success": bool(result["success"]),
                "estimator_residual_rad": float(result["max_angle_residual"]),
                "selected_indices": ";".join(
                    map(str, result.get("selected_indices", []))
                ),
                "delta_x": 0.0,
                "delta_y": 0.0,
                "action_applied": False,
                "event": "observe",
                "failure": "",
            }
            rows.append(row)
            pending.append((receiver_id, result))
            current_receiver_estimates[receiver_id] = estimate_position
            if not result["success"]:
                failure = f"receiver_estimation_failure:{result['message']}"
                row["event"] = "failure"
                row["failure"] = failure
                receiver_failures[receiver_id] = failure

        failures.extend(receiver_failures.values())
        # Observable stop is checked before the synchronized move.  A receiver
        # that already meets the budget is held; all remaining actions are still
        # applied together after the same three-slot round.
        for receiver_id in receiver_failures:
            active_receivers.discard(receiver_id)
        receiver_budget_met = {
            receiver_id
            for receiver_id, position in current_receiver_estimates.items()
            if receiver_id in active_receivers
            and np.linalg.norm(position - TARGET_TEMPLATE[receiver_id - 1])
            <= position_budget
        }
        for receiver_id in receiver_budget_met:
            active_receivers.discard(receiver_id)
            row = next(
                item
                for item in reversed(rows)
                if item.get("stage") == "dynamic_main"
                and item.get("round") == round_i
                and item.get("receiver_id") == receiver_id
            )
            row["event"] = "stop_estimated_position_budget"

        calibration_active = c_hat_error > tau_c
        if not calibration_active and not active_receivers:
            round_summaries.append(
                {
                    "round": round_i,
                    "c_hat_error_to_target_d": c_hat_error,
                    "receiver_budget_met_count": len(receiver_budget_met),
                    "calibration_action": False,
                    "receiver_action_count": 0,
                    "event": "stop_after_observable_verification",
                }
            )
            break

        # Apply all synchronized actions only after the complete round is
        # observed.  FY01 uses the current C_hat; no true C is read by a
        # controller and no previous C_hat survives into the next round.
        round_actions = 0
        if calibration_active:
            delta_c = -float(eta_c) * (c_hat - TARGET_ANCHORS[0])
            current[C_INDEX] += delta_c
            round_actions += 1
            actions.append(
                {
                    "stage": "dynamic_calibration",
                    "round": round_i,
                    "receiver_id": 1,
                    "delta_x": float(delta_c[0]),
                    "delta_y": float(delta_c[1]),
                    "true_position_after_x": float(current[C_INDEX, 0]),
                    "true_position_after_y": float(current[C_INDEX, 1]),
                    "source": "current_C_hat",
                }
            )
            row = next(
                item
                for item in reversed(rows)
                if item.get("stage") == "dynamic_calibration"
                and item.get("round") == round_i
            )
            row["calibration_delta_x"] = float(delta_c[0])
            row["calibration_delta_y"] = float(delta_c[1])
            row["calibration_action_applied"] = True
        else:
            row = next(
                item
                for item in reversed(rows)
                if item.get("stage") == "dynamic_calibration"
                and item.get("round") == round_i
            )
            row["calibration_delta_x"] = 0.0
            row["calibration_delta_y"] = 0.0
            row["calibration_action_applied"] = False
        for receiver_id, result in pending:
            if receiver_id not in active_receivers:
                continue
            if not result["success"]:
                continue
            estimate_position = current_receiver_estimates[receiver_id]
            delta = -float(eta_r) * (
                estimate_position - TARGET_TEMPLATE[receiver_id - 1]
            )
            current[receiver_id - 1] += delta
            round_actions += 1
            actions.append(
                {
                    "stage": "dynamic_main",
                    "round": round_i,
                    "receiver_id": receiver_id,
                    "delta_x": float(delta[0]),
                    "delta_y": float(delta[1]),
                    "true_position_after_x": float(current[receiver_id - 1, 0]),
                    "true_position_after_y": float(current[receiver_id - 1, 1]),
                    "source": "known_current_anchors",
                }
            )
            row = next(
                item
                for item in reversed(rows)
                if item.get("stage") == "dynamic_main"
                and item.get("round") == round_i
                and item.get("receiver_id") == receiver_id
            )
            row["delta_x"] = float(delta[0])
            row["delta_y"] = float(delta[1])
            row["action_applied"] = True
            row["event"] = "observe_and_move"
        round_summaries.append(
            {
                "round": round_i,
                "c_hat_error_to_target_d": c_hat_error,
                "receiver_budget_met_count": len(receiver_budget_met),
                "calibration_action": calibration_active,
                "receiver_action_count": round_actions - int(calibration_active),
                "event": "observe_and_synchronize_move",
            }
        )
    else:
        failures.append("global_budget_exhausted")

    if not failures and not calibration_active and not active_receivers:
        status = "success"
    else:
        status = "failure"
    summary = {
        "status": status,
        "rounds": len(round_summaries),
        "stop_rule": "c_hat_position_and_receiver_estimated_position_budget",
        "position_budget_d": float(position_budget),
        "tau_c_d": float(tau_c),
        "broadcast_slots": len(round_summaries),
        "measurement_slots": len(round_summaries) * 3,
        "tx_uses": len(round_summaries) * 7,
        "relay_scalar_count": len(round_summaries) * 2,
        "public_reference_scalars": len(round_summaries) * 2,
        "failures": failures,
        "rounds_detail": round_summaries,
        "final_c_hat_error_d": c_hat_error,
    }
    return _score_result(
        case,
        current,
        protocol="dynamic_current_reference",
        eta_c=eta_c,
        eta_r=eta_r,
        tau_c=tau_c,
        stop_rule="c_hat_position_and_receiver_estimated_position_budget",
        measurement_slots=int(summary["measurement_slots"]),
        tx_uses=int(summary["tx_uses"]),
        relay_scalars=int(summary["relay_scalar_count"]),
        rows=rows,
        actions=actions,
        status=status,
        failures=failures,
        main_summary=summary,
        receiver_estimate_budget=position_budget,
        budget_group=budget_group,
        public_reference_scalars=int(summary["relay_scalar_count"]),
    )


def run_comparison(
    output_dir: Path = protocols_DEFAULT_OUTPUT,
    *,
    rho: float = RHO,
    seeds: tuple[int, ...] = protocols_SEEDS,
    tau_values: tuple[float, ...] = TAU_C_VALUES,
    gain_pairs: tuple[tuple[float, float], ...] = GAIN_PAIRS,
    max_rounds: int = MAX_ROUNDS,
) -> dict[str, Any]:
    """Run bounded strict, fair-budget staged, and dynamic comparisons."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    case_index = 0

    def add(result: dict[str, Any], *, record_kind: str) -> None:
        nonlocal case_index
        case_index += 1
        seed_value = result["seed"]
        reconstructed = make_initial_state(
            rho,
            None if seed_value == "ideal" else int(seed_value),
        ).positions.copy()
        for action in result["actions"]:
            reconstructed[int(action["receiver_id"]) - 1] += np.asarray(
                [float(action["delta_x"]), float(action["delta_y"])]
            )
        np.testing.assert_allclose(
            reconstructed,
            result["final_positions"],
            atol=1e-12,
            rtol=0.0,
            err_msg=f"action reconstruction failed for case {case_index}",
        )
        case_dir = output_dir / "cases" / f"case_{case_index:03d}_{record_kind}"
        case_dir.mkdir(parents=True, exist_ok=True)
        protocols_write_json(
            case_dir / "summary.json",
            {
                k: v
                for k, v in result.items()
                if k not in {"rows", "actions", "final_positions"}
            },
        )
        np.savetxt(
            case_dir / "final_positions.csv", result["final_positions"], delimiter=","
        )
        triangle_write_csv(case_dir / "observations.csv", result["rows"])
        triangle_write_csv(case_dir / "actions.csv", result["actions"])
        row = {
            k: v
            for k, v in result.items()
            if k
            not in {
                "bootstrap",
                "main",
                "rows",
                "actions",
                "final_positions",
                "failures",
            }
        }
        row["record_kind"] = record_kind
        row["failures_json"] = json.dumps(
            result["failures"], ensure_ascii=False, separators=(",", ":")
        )
        row["bootstrap_summary_json"] = json.dumps(
            _norm(result.get("bootstrap", {})),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        row["main_summary_json"] = json.dumps(
            _norm(result.get("main", {})), ensure_ascii=False, separators=(",", ":")
        )
        row["case_dir"] = str(case_dir)
        rows.append(row)
        details.append(
            {
                "case_dir": str(case_dir),
                "protocol": result["protocol"],
                "record_kind": record_kind,
            }
        )

    # Original strict main-angle contract, one unit-gain reference case per seed.
    for seed in seeds:
        add(
            run_staged_case(rho, seed, 1.0, 1.0, None, max_rounds=max_rounds),
            record_kind="strict_angle",
        )
    for tau_c in tau_values:
        for eta_c, eta_r in gain_pairs:
            for seed in seeds:
                add(
                    run_staged_budget_case(
                        rho,
                        seed,
                        eta_c,
                        eta_r,
                        tau_c,
                        max_rounds=max_rounds,
                    ),
                    record_kind="unallocated_reference_budget",
                )
                add(
                    run_dynamic_case(
                        rho,
                        seed,
                        eta_c,
                        eta_r,
                        tau_c,
                        max_rounds=max_rounds,
                    ),
                    record_kind="unallocated_reference_budget",
                )
    # Formal common-budget group: the receiver estimate gets only the
    # remaining .0045d after tau_C=.005d and a small nonlinear margin.  Both
    # protocols use this same observable receiver threshold; final true Emax
    # is still reported separately and is the only .01d acceptance claim.
    for eta_c, eta_r in gain_pairs:
        for seed in seeds:
            add(
                run_staged_budget_case(
                    rho,
                    seed,
                    eta_c,
                    eta_r,
                    FAIR_TAU_C,
                    max_rounds=max_rounds,
                    position_budget=FAIR_RECEIVER_ESTIMATE_BUDGET,
                    budget_group="fair_shared_position_budget",
                ),
                record_kind="fair_shared_position_budget",
            )
            add(
                run_dynamic_case(
                    rho,
                    seed,
                    eta_c,
                    eta_r,
                    FAIR_TAU_C,
                    max_rounds=max_rounds,
                    position_budget=FAIR_RECEIVER_ESTIMATE_BUDGET,
                    budget_group="fair_shared_position_budget",
                ),
                record_kind="fair_shared_position_budget",
            )
    triangle_write_csv(output_dir / "protocol_comparison.csv", rows)
    summary = {
        "contract": {
            "rho": rho,
            "seeds": list(seeds),
            "tau_c_values_d": list(tau_values),
            "gain_pairs_eta_c_eta_r": [list(pair) for pair in gain_pairs],
            "final_position_budget_d": FINAL_POSITION_BUDGET,
            "fair_shared_budget": {
                "tau_c_d": FAIR_TAU_C,
                "receiver_estimate_budget_d": FAIR_RECEIVER_ESTIMATE_BUDGET,
                "first_order_reference_bound_d": G_MAX_FIRST_ORDER * FAIR_TAU_C,
                "nonlinear_margin_d": 0.0002,
                "budget_formula": "0.01 - 1.040833*0.005 - 0.0002 is conservatively rounded to 0.0045",
            },
            "strict_angle_tolerance_rad": STRICT_ANGLE_TOLERANCE,
            "max_rounds_per_stage": max_rounds,
            "units": "normalised d; no metre scale inferred from angles",
        },
        "accounting": {
            "bootstrap_round": {
                "measurement_slots": 2,
                "transmitter_uses": 4,
                "relay_scalars": 2,
            },
            "fixed_main_broadcast": {"measurement_slots": 1, "transmitter_uses": 3},
            "dynamic_round": {
                "measurement_slots": 3,
                "transmitter_uses": 7,
                "relay_scalars": 2,
                "explanation": "two bottom-angle slots (FY01+one bottom corner each) plus one (1,11,15) broadcast",
            },
            "static_full_protocol": "2*n_C+n_R slots, 4*n_C+3*n_R transmitter uses",
            "dynamic_full_protocol": "3*n_dynamic slots, 7*n_dynamic transmitter uses, 2*n_dynamic public scalars",
            "dynamic_payload_visibility": "the 2 current C_hat coordinates are counted separately as public_reference_scalars; no extra physical slot is invented",
        },
        "run_count": len(rows),
        "success_count": sum(row["status"] == "success" for row in rows),
        "failure_count": sum(row["status"] != "success" for row in rows),
        "success_count_meaning": "online protocol stop only; actual final position budget is counted separately",
        "online_success_count": sum(row["online_status"] == "success" for row in rows),
        "final_position_budget_pass_count": sum(
            row["final_position_budget_pass"] for row in rows
        ),
        "final_position_budget_fail_count": sum(
            not row["final_position_budget_pass"] for row in rows
        ),
        "actual_budget_by_group": {
            group: {
                "runs": sum(row["record_kind"] == group for row in rows),
                "passed": sum(
                    row["record_kind"] == group and row["final_position_budget_pass"]
                    for row in rows
                ),
                "failed": sum(
                    row["record_kind"] == group
                    and not row["final_position_budget_pass"]
                    for row in rows
                ),
            }
            for group in (
                "strict_angle",
                "unallocated_reference_budget",
                "fair_shared_position_budget",
            )
        },
        "strict_angle_records": sum(
            row["record_kind"] == "strict_angle" for row in rows
        ),
        "record_groups": {
            "strict_angle": sum(row["record_kind"] == "strict_angle" for row in rows),
            "unallocated_reference_budget": sum(
                row["record_kind"] == "unallocated_reference_budget" for row in rows
            ),
            "fair_shared_position_budget": sum(
                row["record_kind"] == "fair_shared_position_budget" for row in rows
            ),
        },
        "details": details,
        "decision": {
            "mainline": "staged_fixed_reference_with_finite_tau_C_then_12_parallel_receivers",
            "dynamic_role": "bounded comparison; current-anchor remeasurement pays 7 transmitter uses and 2 public scalars per round",
            "interpretation": "A dynamic synchronized round can match the six-slot strict unit-gain static route in the exact model, but it has no cost advantage there; its value is conditional on avoiding a long finite calibration or on reference motion requiring fresh anchors.",
        },
    }
    protocols_write_json(output_dir / "summary.json", summary)
    return summary


def protocols_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} protocols",
        description="Bounded comparison of staged and synchronised Q2 reference protocols.",
    )
    parser.add_argument("--output-dir", type=Path, default=protocols_DEFAULT_OUTPUT)
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    args = parser.parse_args(argv)
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    summary = run_comparison(args.output_dir, max_rounds=args.max_rounds)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "run_count": summary["run_count"],
                "online_success_count": summary["online_success_count"],
                "online_failure_count": summary["failure_count"],
                "final_position_budget_pass_count": summary[
                    "final_position_budget_pass_count"
                ],
                "final_position_budget_fail_count": summary[
                    "final_position_budget_fail_count"
                ],
            },
            ensure_ascii=False,
        )
    )


# ============================================================================
# e0: e0_shape_observability
# ============================================================================

e0_N_POINTS = 15
DIMENSION = 2
e0_ANGLE_CROSS_TOL = 1e-10
RANK_RELATIVE_THRESHOLDS = (1e-8, 1e-9, 1e-10)
DOUBLE_SCAN_LIMIT = 200
REQUESTED_CONFIGURATIONS = ((1, 11, 15), (2, 7, 10))
e0_PERTURBATION_LEVELS = (1e-3, 1e-2)
PERTURBATION_SEEDS = (11, 23, 47)
FIXED_GENERAL_PERTURBATION_LEVEL = 0.013
FIXED_GENERAL_PERTURBATION_SEED = 20260906


def triangular_template(spacing: float = 1.0) -> np.ndarray:
    """Return the numbered five-layer triangular template, IDs 1 through 15."""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive")
    points: list[tuple[float, float]] = []
    for row in range(5):
        for column in range(row + 1):
            points.append(
                (
                    spacing * (column - row / 2.0),
                    -spacing * np.sqrt(3.0) * row / 2.0,
                )
            )
    return np.asarray(points, dtype=float)


def validate_configuration(configuration: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(configuration) != 3 or len(set(configuration)) != 3:
        raise ValueError("a configuration must contain three distinct IDs")
    if any(not isinstance(value, (int, np.integer)) for value in configuration):
        raise ValueError("configuration IDs must be integers")
    if any(value < 1 or value > e0_N_POINTS for value in configuration):
        raise ValueError("configuration IDs must be in 1..15")
    return tuple(sorted(int(value) for value in configuration))


def observation_layout(configuration: tuple[int, int, int]) -> np.ndarray:
    """Return rows (receiver ID, transmitter ID, transmitter ID), one per pair."""
    tx = validate_configuration(configuration)
    rows: list[tuple[int, int, int]] = []
    for receiver in range(1, e0_N_POINTS + 1):
        if receiver not in tx:
            rows.extend(
                (receiver, first, second)
                for first, second in itertools.combinations(tx, 2)
            )
    return np.asarray(rows, dtype=int)


def _vectors(positions: np.ndarray, layout: np.ndarray):
    points = np.asarray(positions, dtype=float)
    if points.shape != (e0_N_POINTS, DIMENSION) or not np.isfinite(points).all():
        raise ValueError("positions must have shape (15, 2) and contain finite values")
    if layout.ndim != 2 or layout.shape[1] != 3:
        raise ValueError("layout must have shape (rows, 3)")
    receiver, first, second = (layout[:, index] - 1 for index in range(3))
    u = points[first] - points[receiver]
    v = points[second] - points[receiver]
    u2 = np.einsum("ij,ij->i", u, u)
    v2 = np.einsum("ij,ij->i", v, v)
    if np.any((u2 <= 1e-24) | (v2 <= 1e-24)):
        raise ValueError("a receiver coincides with a transmitter")
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dot = np.einsum("ij,ij->i", u, v)
    normalized_cross = np.abs(cross) / np.sqrt(u2 * v2)
    return receiver, first, second, u, v, u2, v2, cross, dot, normalized_cross


def predict_angles(positions: np.ndarray, layout: np.ndarray) -> np.ndarray:
    """Return unsigned angles in [0, pi] using atan2(|cross|, dot)."""
    *_, cross, dot, _ = _vectors(positions, layout)
    return np.arctan2(np.abs(cross), dot)


def regular_row_mask(positions: np.ndarray, layout: np.ndarray) -> np.ndarray:
    """Rows where the original unsigned-angle map has an ordinary derivative."""
    return _vectors(positions, layout)[-1] > e0_ANGLE_CROSS_TOL


def e0_angle_jacobian(positions: np.ndarray, layout: np.ndarray) -> np.ndarray:
    """Analytic Jacobian of angle rows with respect to all 30 coordinates."""
    receiver, first, second, u, v, u2, v2, cross, _, normalized_cross = _vectors(
        positions, layout
    )
    if np.any(normalized_cross <= e0_ANGLE_CROSS_TOL):
        raise ValueError("unsigned-angle Jacobian is undefined at 0 or pi")
    sign = np.sign(cross)[:, None]
    rotation_u = np.column_stack((-u[:, 1], u[:, 0]))
    rotation_v = np.column_stack((-v[:, 1], v[:, 0]))
    d_first = -sign * rotation_u / u2[:, None]
    d_second = sign * rotation_v / v2[:, None]
    d_receiver = -d_first - d_second
    jacobian = np.zeros((len(layout), e0_N_POINTS * DIMENSION), dtype=float)
    for ids, derivative in (
        (receiver, d_receiver),
        (first, d_first),
        (second, d_second),
    ):
        rows = np.arange(len(layout))
        columns = DIMENSION * ids
        jacobian[rows, columns] += derivative[:, 0]
        jacobian[rows, columns + 1] += derivative[:, 1]
    return jacobian


def finite_difference_jacobian(
    positions: np.ndarray, layout: np.ndarray, step: float
) -> np.ndarray:
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    base = np.asarray(positions, dtype=float)
    result = np.empty((len(layout), e0_N_POINTS * DIMENSION), dtype=float)
    for column in range(e0_N_POINTS * DIMENSION):
        plus = base.copy().reshape(-1)
        minus = base.copy().reshape(-1)
        plus[column] += step
        minus[column] -= step
        result[:, column] = (
            predict_angles(plus.reshape(e0_N_POINTS, DIMENSION), layout)
            - predict_angles(minus.reshape(e0_N_POINTS, DIMENSION), layout)
        ) / (2.0 * step)
    return result


def singular_values_30(jacobian: np.ndarray) -> np.ndarray:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return np.pad(
        singular_values,
        (0, e0_N_POINTS * DIMENSION - len(singular_values)),
        mode="constant",
    )


def ranks_for_thresholds(singular_values: np.ndarray) -> dict[str, int]:
    sigma_max = float(singular_values[0]) if len(singular_values) else 0.0
    return {
        f"rank_{relative:.0e}": int(
            np.count_nonzero(singular_values > sigma_max * relative)
        )
        for relative in RANK_RELATIVE_THRESHOLDS
    }


def gauge_basis(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return four similarity tangent vectors and a 26-column orthogonal complement."""
    points = np.asarray(positions, dtype=float)
    center = points.mean(axis=0)
    centered = points - center
    basis = np.zeros((e0_N_POINTS * DIMENSION, 4), dtype=float)
    basis[0::2, 0] = 1.0  # x translation
    basis[1::2, 1] = 1.0  # y translation
    basis[0::2, 2] = -centered[:, 1]
    basis[1::2, 2] = centered[:, 0]
    basis[0::2, 3] = centered[:, 0]
    basis[1::2, 3] = centered[:, 1]
    q, _ = np.linalg.qr(basis, mode="complete")
    return q[:, :4], q[:, 4:]


def gauge_diagnostics(jacobian: np.ndarray, positions: np.ndarray) -> dict:
    gauge, complement = gauge_basis(positions)
    restricted = jacobian @ complement
    restricted_singular = np.linalg.svd(restricted, compute_uv=False)
    restricted_singular = np.pad(
        restricted_singular,
        (0, 26 - len(restricted_singular)),
        mode="constant",
    )
    return {
        "gauge_residual": float(
            np.linalg.norm(jacobian @ gauge, ord="fro")
            / max(1.0, np.linalg.norm(jacobian, ord="fro"))
        ),
        "shape_singular_values_26": [float(value) for value in restricted_singular],
        "shape_ranks": ranks_for_thresholds(restricted_singular),
        "shape_sigma_min": float(restricted_singular[-1]),
    }


def jacobian_record(jacobian: np.ndarray, positions: np.ndarray) -> dict:
    singular = singular_values_30(jacobian)
    return {
        **ranks_for_thresholds(singular),
        "singular_values_30": [float(value) for value in singular],
        "sigma_26": float(singular[25]),
        "sigma_27": float(singular[26]),
        **gauge_diagnostics(jacobian, positions),
    }


def perturbation(spacing_template: np.ndarray, level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    delta = rng.normal(size=spacing_template.shape)
    norms = np.linalg.norm(delta, axis=1)
    delta /= np.max(norms)
    return spacing_template + level * delta


def derivative_checks(template: np.ndarray) -> dict:
    configuration = REQUESTED_CONFIGURATIONS[0]
    layout = observation_layout(configuration)
    q_mask = regular_row_mask(template, layout)
    q_layout = layout[q_mask]
    q_jacobian = e0_angle_jacobian(template, q_layout)
    generic = perturbation(
        template,
        FIXED_GENERAL_PERTURBATION_LEVEL,
        FIXED_GENERAL_PERTURBATION_SEED,
    )
    generic_mask = regular_row_mask(generic, q_layout)
    # The selected ideal regular rows remain regular at this small fixed perturbation.
    if not generic_mask.all():
        raise RuntimeError("fixed regular rows became degenerate in derivative check")
    records = {}
    for name, state, jacobian in (
        ("ideal", template, q_jacobian),
        ("perturbed", generic, e0_angle_jacobian(generic, q_layout)),
    ):
        step_records = {}
        for step in (1e-5, 1e-6, 1e-7):
            numerical = finite_difference_jacobian(state, q_layout, step)
            difference = jacobian - numerical
            step_records[f"{step:.0e}"] = {
                "relative_frobenius_error": float(
                    np.linalg.norm(difference, ord="fro")
                    / max(1.0, np.linalg.norm(jacobian, ord="fro"))
                ),
                "max_absolute_error": float(np.max(np.abs(difference))),
            }
        records[name] = {
            "configuration": list(configuration),
            "regular_rows": int(len(q_layout)),
            "steps": step_records,
        }
    return records


def invariance_checks(template: np.ndarray, layout: np.ndarray) -> dict:
    angle = predict_angles(template, layout)
    theta = 0.37
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    transformed = 1.7 * template @ rotation.T + np.array([0.37, -0.23])
    reflected = template.copy()
    reflected[:, 0] *= -1.0
    reflected += np.array([-0.19, 0.41])
    return {
        "similarity_max_abs_angle_error": float(
            np.max(np.abs(angle - predict_angles(transformed, layout)))
        ),
        "reflection_max_abs_angle_error": float(
            np.max(np.abs(angle - predict_angles(reflected, layout)))
        ),
    }


def configuration_row_data(
    template: np.ndarray, configuration: tuple[int, int, int]
) -> dict:
    layout = observation_layout(configuration)
    mask = regular_row_mask(template, layout)
    jacobian = e0_angle_jacobian(template, layout[mask])
    return {
        "configuration": list(configuration),
        "full_rows": int(len(layout)),
        "degenerate_rows": int(np.count_nonzero(~mask)),
        "regular_rows": int(np.count_nonzero(mask)),
        "degenerate_row_indices": [int(index) for index in np.flatnonzero(~mask)],
        "regular_jacobian": jacobian,
        "regular_layout": layout[mask],
        "diagnostics": jacobian_record(jacobian, template),
    }


def configuration_summary_row(data: dict) -> dict:
    diagnostics = data["diagnostics"]
    configuration = data["configuration"]
    return {
        "configuration": "-".join(map(str, configuration)),
        "full_rows": data["full_rows"],
        "degenerate_rows": data["degenerate_rows"],
        "regular_rows": data["regular_rows"],
        "rank_1e-08": diagnostics["rank_1e-08"],
        "rank_1e-09": diagnostics["rank_1e-09"],
        "rank_1e-10": diagnostics["rank_1e-10"],
        "sigma_26": diagnostics["sigma_26"],
        "sigma_27": diagnostics["sigma_27"],
        "shape_rank_1e-08": diagnostics["shape_ranks"]["rank_1e-08"],
        "shape_rank_1e-09": diagnostics["shape_ranks"]["rank_1e-09"],
        "shape_rank_1e-10": diagnostics["shape_ranks"]["rank_1e-10"],
        "gauge_residual": diagnostics["gauge_residual"],
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: Path, double_limit: int = DOUBLE_SCAN_LIMIT) -> dict:
    if not isinstance(double_limit, int) or not 1 <= double_limit <= DOUBLE_SCAN_LIMIT:
        raise ValueError(f"double_limit must be an integer in 1..{DOUBLE_SCAN_LIMIT}")
    output_dir.mkdir(parents=True, exist_ok=True)
    template = triangular_template()
    all_layout = np.concatenate(
        [
            observation_layout(configuration)
            for configuration in itertools.combinations(range(1, 16), 3)
        ]
    )
    single_rows: list[dict] = []
    single_data: dict[tuple[int, int, int], dict] = {}
    for configuration in itertools.combinations(range(1, 16), 3):
        data = configuration_row_data(template, configuration)
        single_data[configuration] = data
        single_rows.append(configuration_summary_row(data))
    write_csv(
        output_dir / "single_configurations.csv",
        single_rows,
        list(single_rows[0].keys()),
    )

    fixed_general_state = perturbation(
        template,
        FIXED_GENERAL_PERTURBATION_LEVEL,
        FIXED_GENERAL_PERTURBATION_SEED,
    )
    single_perturbed_rows: list[dict] = []
    for configuration in itertools.combinations(range(1, 16), 3):
        data = configuration_row_data(fixed_general_state, configuration)
        row = configuration_summary_row(data)
        row["perturbation_level"] = FIXED_GENERAL_PERTURBATION_LEVEL
        row["perturbation_seed"] = FIXED_GENERAL_PERTURBATION_SEED
        single_perturbed_rows.append(row)
    write_csv(
        output_dir / "single_configurations_perturbed.csv",
        single_perturbed_rows,
        list(single_perturbed_rows[0].keys()),
    )

    triple_list = list(itertools.combinations(range(1, 16), 3))
    requested_pair = tuple(sorted(REQUESTED_CONFIGURATIONS))
    candidates = [requested_pair]
    candidates.extend(
        pair
        for pair in itertools.combinations(triple_list, 2)
        if pair != requested_pair
    )
    double_rows: list[dict] = []
    double_details: list[dict] = []
    witnesses: list[dict] = []
    for index, (first, second) in enumerate(candidates[:double_limit], start=1):
        jacobian = np.vstack(
            (
                single_data[first]["regular_jacobian"],
                single_data[second]["regular_jacobian"],
            )
        )
        diagnostics = jacobian_record(jacobian, template)
        stable_rank = all(
            diagnostics[f"rank_{relative:.0e}"] == 26
            for relative in RANK_RELATIVE_THRESHOLDS
        )
        row = {
            "scan_index": index,
            "configuration_a": "-".join(map(str, first)),
            "configuration_b": "-".join(map(str, second)),
            "full_rows_a": single_data[first]["full_rows"],
            "full_rows_b": single_data[second]["full_rows"],
            "degenerate_rows_a": single_data[first]["degenerate_rows"],
            "degenerate_rows_b": single_data[second]["degenerate_rows"],
            "regular_rows_a": single_data[first]["regular_rows"],
            "regular_rows_b": single_data[second]["regular_rows"],
            "full_rows_total": single_data[first]["full_rows"]
            + single_data[second]["full_rows"],
            "degenerate_rows_total": single_data[first]["degenerate_rows"]
            + single_data[second]["degenerate_rows"],
            "regular_rows_total": single_data[first]["regular_rows"]
            + single_data[second]["regular_rows"],
            "rank_1e-08": diagnostics["rank_1e-08"],
            "rank_1e-09": diagnostics["rank_1e-09"],
            "rank_1e-10": diagnostics["rank_1e-10"],
            "shape_rank_1e-08": diagnostics["shape_ranks"]["rank_1e-08"],
            "shape_rank_1e-09": diagnostics["shape_ranks"]["rank_1e-09"],
            "shape_rank_1e-10": diagnostics["shape_ranks"]["rank_1e-10"],
            "shape_sigma_min": diagnostics["shape_sigma_min"],
            "gauge_residual": diagnostics["gauge_residual"],
            "stable_rank_26": stable_rank,
        }
        double_rows.append(row)
        detail = {
            **row,
            "singular_values_30": diagnostics["singular_values_30"],
            "shape_singular_values_26": diagnostics["shape_singular_values_26"],
            "degenerate_row_indices_a": single_data[first]["degenerate_row_indices"],
            "degenerate_row_indices_b": single_data[second]["degenerate_row_indices"],
        }
        double_details.append(detail)
        if stable_rank:
            witnesses.append(detail)
            if len(witnesses) >= 3:
                break
    write_csv(
        output_dir / "double_configurations_checked.csv",
        double_rows,
        list(double_rows[0].keys()),
    )

    witness_continuation: list[dict] = []
    for witness in witnesses:
        first = tuple(int(value) for value in witness["configuration_a"].split("-"))
        second = tuple(int(value) for value in witness["configuration_b"].split("-"))
        for level in e0_PERTURBATION_LEVELS:
            for seed in PERTURBATION_SEEDS:
                state = perturbation(template, level, seed)
                jacobian_a = e0_angle_jacobian(
                    state, single_data[first]["regular_layout"]
                )
                jacobian_b = e0_angle_jacobian(
                    state, single_data[second]["regular_layout"]
                )
                diagnostics = jacobian_record(
                    np.vstack((jacobian_a, jacobian_b)), state
                )
                witness_continuation.append(
                    {
                        "configuration_a": witness["configuration_a"],
                        "configuration_b": witness["configuration_b"],
                        "level": level,
                        "seed": seed,
                        "rank_1e-08": diagnostics["rank_1e-08"],
                        "rank_1e-09": diagnostics["rank_1e-09"],
                        "rank_1e-10": diagnostics["rank_1e-10"],
                        "shape_rank_1e-08": diagnostics["shape_ranks"]["rank_1e-08"],
                        "shape_rank_1e-09": diagnostics["shape_ranks"]["rank_1e-09"],
                        "shape_rank_1e-10": diagnostics["shape_ranks"]["rank_1e-10"],
                        "shape_sigma_min": diagnostics["shape_sigma_min"],
                        "gauge_residual": diagnostics["gauge_residual"],
                    }
                )
    write_csv(
        output_dir / "witness_perturbation_continuation.csv",
        witness_continuation,
        list(witness_continuation[0].keys())
        if witness_continuation
        else ["configuration_a"],
    )

    derivative = derivative_checks(template)
    invariance = invariance_checks(template, all_layout)
    template_center = template.mean(axis=0)
    nearest_distances = [
        np.linalg.norm(template[first] - template[second])
        for first, second in itertools.combinations(range(e0_N_POINTS), 2)
    ]
    nearest = [distance for distance in nearest_distances if distance < 1.01]
    summary = {
        "schema_version": 1,
        "scope": "Q2 E0 geometry only: static labeled angle batches, three transmitters per configuration, centralized angle collection",
        "not_done": [
            "angle inversion",
            "feedback control",
            "noise",
            "Monte Carlo recovery",
            "global uniqueness",
        ],
        "template": {
            "point_count": e0_N_POINTS,
            "layers": [1, 2, 3, 4, 5],
            "spacing": 1.0,
            "numbering": "FY01..FY15, row-major by layers",
            "center": [float(value) for value in template_center],
            "min_pair_distance": float(min(nearest_distances)),
            "nearest_neighbor_pair_count": len(nearest),
            "coordinates": [[float(value) for value in row] for row in template],
        },
        "observation_contract": {
            "angle": "atan2(abs(cross(Pj-Pi, Pk-Pi)), dot(Pj-Pi, Pk-Pi))",
            "value_range": "[0, pi]",
            "rows_per_configuration": 36,
            "static_batch": True,
            "labeled_ids": True,
            "angle_collection": "centralized for E0",
            "degeneracy_rule": f"normalized absolute cross <= {e0_ANGLE_CROSS_TOL:.0e} is excluded from ordinary Jacobian rows",
        },
        "free_dimension": {
            "coordinate_columns": 30,
            "similarity_gauge_dimension": 4,
            "shape_upper_bound": 26,
            "reflection": "discrete ambiguity; not subtracted from differential rank",
        },
        "single_configuration_summary": {
            "count": len(single_rows),
            "max_rank_1e-08": max(row["rank_1e-08"] for row in single_rows),
            "max_rank_1e-09": max(row["rank_1e-09"] for row in single_rows),
            "max_rank_1e-10": max(row["rank_1e-10"] for row in single_rows),
            "all_rank_at_most_24": all(row["rank_1e-09"] <= 24 for row in single_rows),
            "regular_row_count_distribution": sorted(
                {row["regular_rows"] for row in single_rows}
            ),
        },
        "single_configuration_perturbed_summary": {
            "count": len(single_perturbed_rows),
            "perturbation_level": FIXED_GENERAL_PERTURBATION_LEVEL,
            "perturbation_seed": FIXED_GENERAL_PERTURBATION_SEED,
            "max_rank_1e-08": max(row["rank_1e-08"] for row in single_perturbed_rows),
            "max_rank_1e-09": max(row["rank_1e-09"] for row in single_perturbed_rows),
            "max_rank_1e-10": max(row["rank_1e-10"] for row in single_perturbed_rows),
            "all_rank_at_most_24": all(
                row["rank_1e-09"] <= 24 for row in single_perturbed_rows
            ),
            "regular_row_count_distribution": sorted(
                {row["regular_rows"] for row in single_perturbed_rows}
            ),
        },
        "double_scan": {
            "requested_pair_first": [
                list(configuration) for configuration in requested_pair
            ],
            "requested_pair_row_counts": {
                "full_rows": sum(
                    single_data[configuration]["full_rows"]
                    for configuration in requested_pair
                ),
                "degenerate_rows": sum(
                    single_data[configuration]["degenerate_rows"]
                    for configuration in requested_pair
                ),
                "regular_rows": sum(
                    single_data[configuration]["regular_rows"]
                    for configuration in requested_pair
                ),
            },
            "limit": double_limit,
            "checked": len(double_rows),
            "stopped_after_witnesses": len(witnesses) >= 3,
            "witness_count": len(witnesses),
            "witnesses": [
                {
                    "configuration_a": witness["configuration_a"],
                    "configuration_b": witness["configuration_b"],
                    "rank_1e-08": witness["rank_1e-08"],
                    "rank_1e-09": witness["rank_1e-09"],
                    "rank_1e-10": witness["rank_1e-10"],
                    "shape_sigma_min": witness["shape_sigma_min"],
                    "gauge_residual": witness["gauge_residual"],
                }
                for witness in witnesses
            ],
            "checked_details": double_details,
        },
        "derivative_checks": derivative,
        "invariance_checks": invariance,
        "witness_perturbation_continuation": {
            "levels": list(e0_PERTURBATION_LEVELS),
            "seeds": list(PERTURBATION_SEEDS),
            "rows": witness_continuation,
        },
        "interpretation": {
            "single_configuration": "The regular-row probe has at most 24 differential dimensions at the ideal template; this is a design-class cross-check, not a theorem about every non-smooth full-angle configuration.",
            "double_configuration": "A stable rank-26 witness is a local shape-observability sufficient condition after removing translation, rotation, and scale; it does not establish inversion success, global uniqueness, control convergence, or physical-scale recovery.",
            "scale": "d=1 is a coordinate convention; angle data alone do not identify absolute meters.",
            "reflection": "Mirror images preserve unsigned angles and must be handled as a discrete branch in later inversion.",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def e0_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} e0",
        description="Bounded Q2 geometry E0: local shape observability for labeled angle data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/q2/e0_shape"),
        help="directory for the bounded E0 outputs",
    )
    parser.add_argument(
        "--double-limit",
        type=int,
        default=DOUBLE_SCAN_LIMIT,
        help=f"maximum double configurations to check (1..{DOUBLE_SCAN_LIMIT})",
    )
    args = parser.parse_args(argv)
    summary = run(args.output_dir, args.double_limit)
    print(
        json.dumps(
            {
                "single_configurations": summary["single_configuration_summary"],
                "single_configurations_perturbed": summary[
                    "single_configuration_perturbed_summary"
                ],
                "double_scan": summary["double_scan"],
                "derivative_checks": summary["derivative_checks"],
                "invariance_checks": summary["invariance_checks"],
            },
            ensure_ascii=False,
        )
    )


# 统一命令入口。各分析的参数解析函数接受 argv，不改写进程参数。
COMMANDS = {
    "triangle": triangle_main,
    "residual": residual_main,
    "geometry": geometry_main,
    "budget": budget_main,
    "noise": noise_main,
    "protocols": protocols_main,
    "e0": e0_main,
}


def main(argv: list[str] | None = None) -> None:
    """Dispatch a self-contained experiment; use COMMAND --help for options."""
    parser = argparse.ArgumentParser(
        description="第二问：三角编队参考建立、并行调整及完整分析。",
        epilog=(
            "triangle: 基础闭环；residual: 参考残差；geometry: 几何边界；"
            "budget: 校准位置预算；noise: 噪声和增益；"
            "protocols: 两阶段/同步对照；e0: 局部形状可辨识性。"
            "运行 COMMAND --help 查看该分析的参数。"
        ),
    )
    parser.add_argument("command", choices=tuple(COMMANDS))
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return
    command = parser.parse_args(arguments[:1]).command
    COMMANDS[command](arguments[1:])


if __name__ == "__main__":
    main()
