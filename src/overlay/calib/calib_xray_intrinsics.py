# -*- coding: utf-8 -*-
"""
calib_xray_intrinsics.py

X-ray intrinsic calibration from multiple planar homographies
using Zhang's method (2000), with optional global nonlinear refinement
via OpenCV calibrateCamera.

Assumes:
    x ~ H X
where
    x = (u, v, 1)^T   image pixel coordinates
    X = (X, Y, 1)^T   planar PCB coordinates (e.g. mm)

Default refinement model:
    - zero skew
    - no tangential distortion
    - no radial distortion refinement
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional

import numpy as np
import cv2


# ============================================================
# Data containers
# ============================================================

@dataclass
class XrayIntrinsicsResult:
    K: np.ndarray
    num_views: int
    rms_reproj_error: Optional[float] = None
    dist_coeffs: Optional[np.ndarray] = None
    refined: bool = False


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
    enforce_zero_skew: bool = True,
    global_optimization: bool = False,
    image_size: Optional[tuple[int, int]] = None,
    object_points_per_view: Optional[Sequence[np.ndarray]] = None,
    image_points_per_view: Optional[Sequence[np.ndarray]] = None,
    radial_model: str = "none",
    fix_principal_point: bool = False,
    principal_point_mode: str = "init",
) -> XrayIntrinsicsResult:
    """
    Estimate intrinsic matrix Kx from multiple homographies using Zhang (2000),
    with optional global nonlinear refinement using OpenCV calibrateCamera.

    Parameters
    ----------
    H_list : sequence of (3,3) homographies
        Homographies mapping planar board coordinates to image pixels:
            x ~ H X

    enforce_zero_skew : bool
        If True, enforce gamma = 0 directly in the linear Zhang solve.

    global_optimization : bool
        If True, run global nonlinear refinement in OpenCV using Zhang K as
        initialization and per-view poses derived from the homographies.

    image_size : (width, height), optional
        Required if global_optimization=True.

    object_points_per_view : sequence of (N,3) arrays, optional
        Required if global_optimization=True.
        Known board points for each view.

    image_points_per_view : sequence of (N,2) arrays, optional
        Required if global_optimization=True.
        Measured X-ray image points for each view.

    radial_model : {"none", "k1", "k1k2"}
        Which radial distortion terms are allowed to vary during refinement.

    fix_principal_point : bool
        If True, principal point is kept fixed during nonlinear refinement.

    principal_point_mode : {"init", "image_center"}
        If fix_principal_point=True:
            - "init" keeps the Zhang-estimated principal point fixed
            - "image_center" fixes the principal point to the image center

    Returns
    -------
    XrayIntrinsicsResult
    """

    allowed_models = {"none", "k1", "k1k2"}
    if radial_model not in allowed_models:
        raise ValueError(
            f"radial_model must be one of {sorted(allowed_models)}, got {radial_model!r}"
        )

    allowed_pp_modes = {"init", "image_center"}
    if principal_point_mode not in allowed_pp_modes:
        raise ValueError(
            f"principal_point_mode must be one of {sorted(allowed_pp_modes)}, "
            f"got {principal_point_mode!r}"
        )

    min_views = 2 if enforce_zero_skew else 3
    if len(H_list) < min_views:
        raise ValueError(
            f"At least {min_views} homographies required "
            f"(got {len(H_list)}; enforce_zero_skew={enforce_zero_skew})."
        )

    V_rows: List[np.ndarray] = []

    for H in H_list:
        Hn = _normalize_H(H)

        v12 = _v_ij(Hn, 0, 1)
        v11 = _v_ij(Hn, 0, 0)
        v22 = _v_ij(Hn, 1, 1)

        V_rows.append(v12)
        V_rows.append(v11 - v22)

    if enforce_zero_skew:
        V_rows.append(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))

    V = np.stack(V_rows, axis=0)

    _, _, VT = np.linalg.svd(V)
    b = VT[-1, :].astype(np.float64)

    if b[0] < 0:
        b = -b

    b11, b12, b22, b13, b23, b33 = b.tolist()

    if enforce_zero_skew:
        b12 = 0.0

    denom = b11 * b22 - b12 * b12
    if abs(denom) < 1e-18:
        raise RuntimeError("Degenerate configuration: b11*b22 - b12^2 is ~0.")

    if abs(b11) < 1e-18:
        raise RuntimeError("Degenerate configuration: b11 is ~0.")

    v0 = (b12 * b13 - b11 * b23) / denom
    lam = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11

    if lam <= 0:
        b = -b
        b11, b12, b22, b13, b23, b33 = b.tolist()

        if enforce_zero_skew:
            b12 = 0.0

        denom = b11 * b22 - b12 * b12
        if abs(denom) < 1e-18:
            raise RuntimeError("Degenerate configuration after sign flip: denominator ~0.")
        if abs(b11) < 1e-18:
            raise RuntimeError("Degenerate configuration after sign flip: b11 ~0.")

        v0 = (b12 * b13 - b11 * b23) / denom
        lam = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11

    if lam <= 0:
        raise RuntimeError(
            f"Invalid solution: lambda must be > 0, got {lam:.6e}. "
            "Likely insufficient pose diversity or noisy/degenerate homographies."
        )

    alpha_sq = lam / b11
    beta_sq = lam * b11 / denom

    if alpha_sq <= 0 or beta_sq <= 0:
        raise RuntimeError(
            "Invalid intrinsic recovery: alpha^2 or beta^2 <= 0. "
            "Likely degenerate geometry or unstable homographies."
        )

    alpha = np.sqrt(alpha_sq)
    beta = np.sqrt(beta_sq)

    gamma = 0.0 if enforce_zero_skew else (-b12 * alpha * alpha * beta / lam)
    u0 = gamma * v0 / beta - b13 * alpha * alpha / lam

    K_init = np.array([
        [alpha, gamma, u0],
        [0.0,   beta,  v0],
        [0.0,   0.0,   1.0],
    ], dtype=np.float64)

    if not global_optimization:
        return XrayIntrinsicsResult(
            K=K_init,
            num_views=len(H_list),
            rms_reproj_error=None,
            dist_coeffs=None,
            refined=False,
        )

    if not enforce_zero_skew:
        raise ValueError(
            "For the current OpenCV refinement branch, enforce_zero_skew=True is required "
            "to stay consistent with the chosen zero-skew camera model."
        )

    if image_size is None:
        raise ValueError("image_size is required when global_optimization=True.")
    if object_points_per_view is None or image_points_per_view is None:
        raise ValueError(
            "object_points_per_view and image_points_per_view are required "
            "when global_optimization=True."
        )
    if len(object_points_per_view) != len(H_list) or len(image_points_per_view) != len(H_list):
        raise ValueError("H_list, object_points_per_view, and image_points_per_view must have the same length.")

    K_refined, dist_coeffs, rms = _refine_intrinsics_opencv(
        K_init=K_init,
        H_list=H_list,
        image_size=image_size,
        object_points_per_view=object_points_per_view,
        image_points_per_view=image_points_per_view,
        radial_model=radial_model,
        fix_principal_point=fix_principal_point,
        principal_point_mode=principal_point_mode,
    )

    return XrayIntrinsicsResult(
        K=K_refined,
        num_views=len(H_list),
        rms_reproj_error=float(rms),
        dist_coeffs=dist_coeffs,
        refined=True,
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
# OpenCV global refinement
# ============================================================

def _as_opencv_object_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"object points must have shape (N,3), got {pts.shape}")
    return pts.astype(np.float32)


def _as_opencv_image_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"image points must have shape (N,2), got {pts.shape}")
    return pts.astype(np.float32)


def _pose_to_rvec_tvec(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)

    rvec, _ = cv2.Rodrigues(R)
    tvec = t.astype(np.float64)

    return rvec, tvec


def _refine_intrinsics_opencv(
    *,
    K_init: np.ndarray,
    H_list: Sequence[np.ndarray],
    image_size: tuple[int, int],
    object_points_per_view: Sequence[np.ndarray],
    image_points_per_view: Sequence[np.ndarray],
    radial_model: str = "none",
    fix_principal_point: bool = False,
    principal_point_mode: str = "init",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Global nonlinear refinement of K using OpenCV calibrateCamera.

    radial_model:
        - "none" : no radial terms are refined
        - "k1"   : refine only k1
        - "k1k2" : refine k1 and k2

    principal_point_mode:
        - "init"         : keep Zhang principal point
        - "image_center" : set principal point to image center and fix it
    """

    allowed_models = {"none", "k1", "k1k2"}
    if radial_model not in allowed_models:
        raise ValueError(f"radial_model must be one of {sorted(allowed_models)}, got {radial_model!r}")

    allowed_pp_modes = {"init", "image_center"}
    if principal_point_mode not in allowed_pp_modes:
        raise ValueError(
            f"principal_point_mode must be one of {sorted(allowed_pp_modes)}, "
            f"got {principal_point_mode!r}"
        )

    object_points_cv: list[np.ndarray] = []
    image_points_cv: list[np.ndarray] = []
    rvecs_init: list[np.ndarray] = []
    tvecs_init: list[np.ndarray] = []

    for H, obj_pts, img_pts in zip(H_list, object_points_per_view, image_points_per_view):
        obj_pts_cv = _as_opencv_object_points(obj_pts)
        img_pts_cv = _as_opencv_image_points(img_pts)

        if len(obj_pts_cv) != len(img_pts_cv):
            raise ValueError("Each view must have same number of object and image points.")

        pose = decompose_homography(K_init, H)
        rvec, tvec = _pose_to_rvec_tvec(pose.R, pose.t)

        object_points_cv.append(obj_pts_cv)
        image_points_cv.append(img_pts_cv)
        rvecs_init.append(rvec)
        tvecs_init.append(tvec)

    camera_matrix = np.asarray(K_init, dtype=np.float64).copy()
    dist_coeffs = np.zeros((8, 1), dtype=np.float64)

    if fix_principal_point:
        if principal_point_mode == "init":
            pass
        elif principal_point_mode == "image_center":
            camera_matrix[0, 2] = image_size[0] / 2.0
            camera_matrix[1, 2] = image_size[1] / 2.0
        else:
            raise ValueError(f"Unsupported principal_point_mode: {principal_point_mode!r}")

    flags = 0
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_USE_EXTRINSIC_GUESS
    flags |= cv2.CALIB_ZERO_TANGENT_DIST

    if fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

    if radial_model == "none":
        flags |= cv2.CALIB_FIX_K1
        flags |= cv2.CALIB_FIX_K2
        flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_FIX_K4
        flags |= cv2.CALIB_FIX_K5
        flags |= cv2.CALIB_FIX_K6

    elif radial_model == "k1":
        flags |= cv2.CALIB_FIX_K2
        flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_FIX_K4
        flags |= cv2.CALIB_FIX_K5
        flags |= cv2.CALIB_FIX_K6

    elif radial_model == "k1k2":
        flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_FIX_K4
        flags |= cv2.CALIB_FIX_K5
        flags |= cv2.CALIB_FIX_K6

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
        200,
        1e-12,
    )

    rms, K_refined, dist_refined, _, _ = cv2.calibrateCamera(
        objectPoints=object_points_cv,
        imagePoints=image_points_cv,
        imageSize=image_size,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        rvecs=rvecs_init,
        tvecs=tvecs_init,
        flags=flags,
        criteria=criteria,
    )

    return K_refined, dist_refined, float(rms)


# ============================================================
# Utility
# ============================================================

def relative_board_tilt_from_normal_deg(
    R_ref: np.ndarray,
    R: np.ndarray,
    *,
    xray_axes: bool = False,
) -> tuple[float, float, float]:
    R_ref = np.asarray(R_ref, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    R_rel = R_ref.T @ R

    if xray_axes:
        S = np.array([
            [0.0,  1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64)
        R_rel = S.T @ R_rel @ S

    n = R_rel @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    nx, ny, nz = n.tolist()

    if nz < 0:
        nx, ny, nz = -nx, -ny, -nz

    tilt_x = np.arctan2(ny, nz)
    tilt_y = -np.arctan2(nx, nz)
    tilt_mag = np.arctan2(np.sqrt(nx * nx + ny * ny), nz)

    return float(np.degrees(tilt_x)), float(np.degrees(tilt_y)), float(np.degrees(tilt_mag))


def relative_shift_board_mm(
    R_ref: np.ndarray,
    t_ref: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    *,
    xray_axes: bool = False,
) -> np.ndarray:
    dt_cam = np.asarray(t, dtype=np.float64) - np.asarray(t_ref, dtype=np.float64)
    dt_board = np.asarray(R_ref, dtype=np.float64).T @ dt_cam

    if xray_axes:
        S = np.array([
            [0.0,  1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64)

        dt_board = S.T @ dt_board
        dt_board[0] = -dt_board[0]
        dt_board[1] = -dt_board[1]

    return dt_board