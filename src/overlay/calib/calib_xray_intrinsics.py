# -*- coding: utf-8 -*-
"""
calib_xray_intrinsics.py

X-ray intrinsic calibration from multiple planar homographies
using Zhang's method (2000).

Assumes:
    x ~ H X
where
    x = (u, v, 1)^T   image pixel coordinates
    X = (X, Y, 1)^T   planar PCB coordinates (e.g. mm)

No distortion model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict

import numpy as np


# ============================================================
# Data containers
# ============================================================

@dataclass
class XrayIntrinsicsResult:
    K: np.ndarray
    num_views: int


@dataclass
class HomographyPose:
    R: np.ndarray
    t: np.ndarray


# ============================================================
# Zhang helpers
# ============================================================

def _normalize_H(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be (3,3), got {H.shape}")
    if abs(H[2, 2]) > 1e-12:
        return H / H[2, 2]
    return H / (np.linalg.norm(H) + 1e-12)


def _v_ij(H: np.ndarray, i: int, j: int) -> np.ndarray:
    h = H
    return np.array([
        h[0, i] * h[0, j],
        h[0, i] * h[1, j] + h[1, i] * h[0, j],
        h[1, i] * h[1, j],
        h[2, i] * h[0, j] + h[0, i] * h[2, j],
        h[2, i] * h[1, j] + h[1, i] * h[2, j],
        h[2, i] * h[2, j],
    ], dtype=np.float64)


# ============================================================
# Intrinsics estimation (Zhang)
# ============================================================

def estimate_intrinsics_from_homographies(
    H_list: Sequence[np.ndarray],
    *,
    enforce_zero_skew: bool = False,
) -> XrayIntrinsicsResult:
    """
    Estimate intrinsic matrix Kx from multiple homographies (Zhang 2000).

    Requires >= 3 sufficiently distinct views.

    Notes
    -----
    - The SVD solution for b is defined only up to scale/sign. We MUST fix the
      sign ambiguity, otherwise b11 can be negative and sqrt(lam/b11) becomes NaN.
    - We enforce a consistent sign by making b11 > 0 (common convention).
    """

    if len(H_list) < 3:
        raise ValueError("At least 3 homographies required.")

    V_rows: List[np.ndarray] = []

    for H in H_list:
        Hn = _normalize_H(H)

        v12 = _v_ij(Hn, 0, 1)
        v11 = _v_ij(Hn, 0, 0)
        v22 = _v_ij(Hn, 1, 1)

        V_rows.append(v12)
        V_rows.append(v11 - v22)

    V = np.stack(V_rows, axis=0)

    # Solve V b = 0 via SVD (b is last right-singular vector)
    _, _, VT = np.linalg.svd(V)
    b = VT[-1, :].astype(np.float64)

    # ---- Fix SVD sign ambiguity (critical) ----
    # Ensure b11 > 0 to avoid negative under sqrt in alpha = sqrt(lam / b11).
    if b[0] < 0:
        b = -b

    b11, b12, b22, b13, b23, b33 = b.tolist()

    denom = b11 * b22 - b12 * b12
    if abs(denom) < 1e-18:
        raise RuntimeError("Degenerate configuration (denominator ~ 0).")

    v0 = (b12 * b13 - b11 * b23) / denom
    lam = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11

    # In rare cases numeric noise can still give lam < 0. Since b is up to sign,
    # try flipping once. If still negative, fall back to abs(lam) as last resort.
    if lam < 0:
        b = -b
        b11, b12, b22, b13, b23, b33 = b.tolist()

        denom = b11 * b22 - b12 * b12
        if abs(denom) < 1e-18:
            raise RuntimeError("Degenerate configuration (denominator ~ 0) after sign flip.")

        v0 = (b12 * b13 - b11 * b23) / denom
        lam = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11

        if lam < 0:
            lam = abs(lam)

    alpha = np.sqrt(lam / b11)
    beta = np.sqrt(lam * b11 / denom)
    gamma = -b12 * alpha * alpha * beta / lam
    u0 = gamma * v0 / beta - b13 * alpha * alpha / lam

    if enforce_zero_skew:
        gamma = 0.0
        u0 = -b13 * alpha * alpha / lam

    K = np.array([
        [alpha, gamma, u0],
        [0.0,   beta,  v0],
        [0.0,   0.0,   1.0],
    ], dtype=np.float64)

    return XrayIntrinsicsResult(
        K=K,
        num_views=len(H_list),
    )


# ============================================================
# Homography decomposition (pose recovery)
# ============================================================

def decompose_homography(
    K: np.ndarray,
    H: np.ndarray,
) -> HomographyPose:
    """
    Recover (R, t) from K and homography H.
    """

    Hn = _normalize_H(H)
    Kinv = np.linalg.inv(K)

    B = Kinv @ Hn

    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]

    scale = 1.0 / (np.linalg.norm(b1) + 1e-12)

    r1 = scale * b1
    r2 = scale * b2
    t = scale * b3

    r3 = np.cross(r1, r2)
    R_approx = np.stack([r1, r2, r3], axis=1)

    U, _, VT = np.linalg.svd(R_approx)
    R = U @ VT

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ VT

    return HomographyPose(R=R, t=t)


# ============================================================
# Utility
# ============================================================

def rotation_matrix_to_euler_xyz_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to XYZ Euler angles (degrees).

    Convention:
        R = Rz * Ry * Rx  (intrinsic XYZ)
    Returns:
        (x_deg, y_deg, z_deg)
    """

    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-9

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0

    # Convert to degrees
    x_deg = np.degrees(x)
    y_deg = np.degrees(y)
    z_deg = np.degrees(z)

    return float(x_deg), float(y_deg), float(z_deg)


def relative_board_angles_deg(
    R_ref: np.ndarray,
    R: np.ndarray,
    *,
    xray_axes: bool = False,
) -> Tuple[float, float, float]:
    """
    Relative rotation angles (deg) of pose R w.r.t. reference pose R_ref,
    expressed in the BOARD coordinate system of the reference.

    Returns (tilt_xg_deg, tilt_yg_deg, inplane_zg_deg) using XYZ Euler angles.

    If xray_axes=True, we re-interpret the in-plane board axes to match a
    "X-ray image view" convention where:
        x_g points UP in the X-ray image      => x' = -y
        y_g points RIGHT in the X-ray image   => y' =  x
        z_g unchanged                         => z' =  z

    Additionally, in xray_axes mode we flip the signs of tilt_xg and tilt_yg
    to match the right-hand-rule interpretation you apply when judging the
    rotation directly from the X-ray view / setup.
    """
    R_ref = np.asarray(R_ref, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    # Relative rotation expressed in reference board frame
    R_rel_board = R_ref.T @ R

    if xray_axes:
        # Basis change in the board plane: (x',y',z') = (-y, x, z)
        # Columns are x', y', z' expressed in the old (x,y,z) basis.
        S = np.array([
            [0.0,  1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64)

        # Express the same rotation in the new basis
        R_rel_board = S.T @ R_rel_board @ S

    tilt_xg_deg, tilt_yg_deg, inplane_zg_deg = rotation_matrix_to_euler_xyz_deg(R_rel_board)

    if xray_axes:
        tilt_xg_deg = -tilt_xg_deg
        tilt_yg_deg = -tilt_yg_deg

    return float(tilt_xg_deg), float(tilt_yg_deg), float(inplane_zg_deg)


def relative_shift_board_mm(
    R_ref: np.ndarray,
    t_ref: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    *,
    xray_axes: bool = False,
) -> np.ndarray:
    """
    Relative translation vector (mm) w.r.t reference, expressed in board frame.
    If xray_axes=True, applies the same axis convention + sign convention as
    relative_board_angles_deg(..., xray_axes=True).
    """
    dt_cam = np.asarray(t, dtype=np.float64) - np.asarray(t_ref, dtype=np.float64)
    dt_board = np.asarray(R_ref, dtype=np.float64).T @ dt_cam  # (3,)

    if xray_axes:
        S = np.array([
            [0.0,  1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64)

        # same basis change as for angles
        dt_board = S.T @ dt_board

        # same sign convention you decided for angles:
        # tilt_xg = -tilt_xg, tilt_yg = -tilt_yg  => flip x and y components
        dt_board[0] = -dt_board[0]
        dt_board[1] = -dt_board[1]

    return dt_board
