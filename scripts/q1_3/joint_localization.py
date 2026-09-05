"""Estimate eight unknown UAV positions from static, labeled angle slots."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from scripts.q1_1.localization import FloatArray, GeometryError, fy_position

AuxiliaryPair = tuple[int, int]


def ideal_formation(radius: float = 100.0) -> FloatArray:
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    return np.array([fy_position(i, radius) for i in range(10)])


def validate_pair(pair: AuxiliaryPair) -> AuxiliaryPair:
    if (
        len(pair) != 2
        or any(not isinstance(i, (int, np.integer)) or not 2 <= i <= 9 for i in pair)
        or pair[0] == pair[1]
    ):
        raise ValueError("an auxiliary pair must contain two distinct IDs in 2..9")
    return tuple(sorted(pair))


def observation_layout(configurations: tuple[AuxiliaryPair, ...]) -> np.ndarray:
    """Rows (receiver, first transmitter, second transmitter), slot by slot."""
    if not configurations:
        raise ValueError("at least one measurement configuration is required")
    rows = []
    for pair in configurations:
        transmitters = (0, 1, *validate_pair(pair))
        for receiver in range(2, 10):
            if receiver not in transmitters:
                rows.extend((receiver, j, k) for j, k in combinations(transmitters, 2))
    return np.asarray(rows, dtype=int)


def _vectors(positions: ArrayLike, layout: np.ndarray):
    points = np.asarray(positions, dtype=float)
    if points.shape != (10, 2) or not np.isfinite(points).all():
        raise ValueError("positions must have shape (10, 2) and be finite")
    i, j, k = layout.T
    u, v = points[j] - points[i], points[k] - points[i]
    u2, v2 = np.sum(u * u, axis=1), np.sum(v * v, axis=1)
    if np.any(np.minimum(u2, v2) <= 1e-24):
        raise GeometryError("receiver coincides with a transmitter")
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return u, v, u2, v2, cross


def predict_angles(positions: ArrayLike, layout: np.ndarray) -> FloatArray:
    """Unsigned [0, pi] angles; atan2 is equivalent to the Q1(1) acos model."""
    u, v, _, _, cross = _vectors(positions, layout)
    return np.arctan2(np.abs(cross), np.sum(u * v, axis=1))


def joint_jacobian(positions: ArrayLike, layout: np.ndarray) -> FloatArray:
    """Angle derivatives w.r.t. (x2,y2,...,x9,y9), in rad/m."""
    u, v, u2, v2, cross = _vectors(positions, layout)
    if np.any(np.abs(cross) / np.sqrt(u2 * v2) <= 1e-12):
        raise GeometryError("unsigned-angle Jacobian is undefined at 0 or pi")
    sign = np.sign(cross)[:, None]
    du = -sign * np.column_stack((-u[:, 1], u[:, 0])) / u2[:, None]
    dv = sign * np.column_stack((-v[:, 1], v[:, 0])) / v2[:, None]
    jac = np.zeros((len(layout), 16))
    for ids, derivative in zip(layout.T, (-du - dv, du, dv)):
        rows = np.flatnonzero(ids >= 2)
        columns = 2 * (ids[rows] - 2)
        jac[rows, columns] += derivative[rows, 0]
        jac[rows, columns + 1] += derivative[rows, 1]
    return jac


def jacobian_diagnostics(jac: FloatArray) -> dict:
    singular = np.linalg.svd(jac, compute_uv=False)
    tolerance = float(singular[0] * 1e-9)
    rank = int(np.count_nonzero(singular > tolerance))
    return {
        "rank": rank,
        "rank_tolerance": tolerance,
        "sigma_min_rad_per_m": float(singular[-1]),
        "sigma_max_rad_per_m": float(singular[0]),
        "condition_number": float(singular[0] / singular[-1]) if rank == 16 else None,
    }


@dataclass(frozen=True)
class JointResult:
    positions: FloatArray
    success: bool
    residual_norm_rad: float
    diagnostics: dict
    nfev: int
    message: str


def localize_formation(
    configurations: tuple[AuxiliaryPair, ...],
    observed_angles: ArrayLike,
    *,
    radius: float = 100.0,
    max_nfev: int = 500,
    residual_tolerance: float = 1e-8,
) -> JointResult:
    """Exact-angle local fit from the nominal formation, with fixed FY00/FY01.

    Inputs contain no simulated true coordinates or true displacement. Full
    rank and small residual certify only a locally consistent numerical fit.
    """
    nominal = ideal_formation(radius)
    layout = observation_layout(configurations)
    observed = np.asarray(observed_angles, dtype=float)
    if observed.shape != (len(layout),) or not np.isfinite(observed).all():
        raise ValueError("one finite observed angle is required per layout row")
    if np.any((observed <= 0) | (observed >= np.pi)):
        raise ValueError("observed angles must lie strictly between 0 and pi")
    if not isinstance(max_nfev, int) or max_nfev <= 0:
        raise ValueError("max_nfev must be a positive integer")
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("residual_tolerance must be finite and positive")

    def unpack(state):
        return np.vstack((nominal[:2], state.reshape(8, 2)))

    fit = least_squares(
        lambda state: predict_angles(unpack(state), layout) - observed,
        nominal[2:].ravel(),
        jac=lambda state: joint_jacobian(unpack(state), layout),
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
        max_nfev=max_nfev,
    )
    diagnostics = jacobian_diagnostics(fit.jac)
    residual = float(np.linalg.norm(fit.fun))
    return JointResult(
        positions=unpack(fit.x),
        success=bool(
            fit.success and diagnostics["rank"] == 16 and residual <= residual_tolerance
        ),
        residual_norm_rad=residual,
        diagnostics=diagnostics,
        nfev=int(fit.nfev),
        message=str(fit.message),
    )


def movement_commands(
    estimate: ArrayLike,
    held_pair: AuxiliaryPair,
    *,
    radius: float = 100.0,
    gain: float = 1.0,
) -> FloatArray:
    """Displacements computed from estimates; transmitters hold position."""
    held = validate_pair(held_pair)
    points = np.asarray(estimate, dtype=float)
    if points.shape != (10, 2) or not np.isfinite(points).all():
        raise ValueError("estimate must be finite with shape (10, 2)")
    if not np.isfinite(gain) or not 0 < gain <= 1:
        raise ValueError("gain must lie in (0, 1]")
    commands = gain * (ideal_formation(radius) - points)
    commands[[0, 1, *held]] = 0
    return commands
