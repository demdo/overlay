# -*- coding: utf-8 -*-
"""
calib_camera_to_xray.py

Camera-to-X-ray calibration from 3-D/2-D correspondences or planar board
image points, depending on the chosen pose method.

Two workflows are supported, selected via ``pose_method``:

PnP workflow  (``"iterative"``, ``"iterative_ransac"``, ``"ippe"``)
--------------------------------------------------------------------
Requires pre-computed 3-D marker positions in the camera frame and their
2-D counterparts in the X-ray image.  The camera->xray transform is solved
directly as a single PnP problem.

    T_cx  ←  solve_pose(points_xyz_camera, points_uv_xray, K_xray)

Homography workflow  (``"homography"``)
---------------------------------------
Requires 2-D board detections in both the RGB and X-ray images plus both
intrinsic matrices.  Two board poses are estimated via DLT homography
decomposition and then composed:

    T_bc  ←  decompose(H_bc, K_rgb)   # board -> camera
    T_bx  ←  decompose(H_bx, K_xray)  # board -> xray
    T_cx  =  T_bx @ inv(T_bc)         # camera -> xray

Conventions
-----------
T_ab  means: transform *from* frame a *to* frame b

    T_bc : board  -> camera
    T_bx : board  -> xray
    T_cx : camera -> xray   (primary output)
    T_xc : xray   -> camera (convenience inverse)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from overlay.tools.homography import (
    build_planar_correspondences,
    estimate_homography_dlt,
    decompose_homography_to_pose,
)
from overlay.tracking.pose_solvers import solve_pose, PoseSolveResult
from overlay.tracking.transforms import invert_transform, rvec_tvec_to_transform


# ============================================================
# Result container
# ============================================================

@dataclass(frozen=True)
class CameraToXrayCalibrationResult:
    """
    Output of ``calibrate_camera_to_xray``.

    Attributes
    ----------
    rvec : (3,1) float64
        Rodrigues rotation vector of the camera->xray transform.
    tvec : (3,1) float64
        Translation vector of the camera->xray transform.
    rotation : (3,3) float64
        Rotation matrix derived from ``rvec``.
    translation : (3,1) float64
        Alias for ``tvec`` kept for API symmetry.
    T_cx : (4,4) float64
        Homogeneous camera->xray transform.
    T_xc : (4,4) float64
        Homogeneous xray->camera transform (inverse of T_cx).
    inliers : (M,1) int32 or None
        OpenCV-style inlier indices; only set when RANSAC is used.
    inlier_idx : (M,) or (N,) int64
        Flat inlier index array; equals arange(N) when no RANSAC is used.
    uv_proj : (N,2) float64
        X-ray image points reprojected from the solved pose.
    reproj_errors_px : (N,) float64
        Per-point reprojection error in pixels.
    reproj_mean_px : float
    reproj_median_px : float
    reproj_max_px : float
    pose_result : PoseSolveResult
        Full solver output for downstream inspection.
    """

    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    T_cx: np.ndarray
    T_xc: np.ndarray

    inliers: np.ndarray | None
    inlier_idx: np.ndarray

    uv_proj: np.ndarray
    reproj_errors_px: np.ndarray
    reproj_mean_px: float
    reproj_median_px: float
    reproj_max_px: float

    pose_result: PoseSolveResult


# ============================================================
# Internal helpers
# ============================================================

def _as_uv(arr: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N,2), got {pts.shape}")
    return pts


def _as_K(arr: np.ndarray, name: str) -> np.ndarray:
    K = np.asarray(arr, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3,3), got {K.shape}")
    return K


def _result_from_pose(
    pose_result: PoseSolveResult,
    T_cx: np.ndarray,
) -> CameraToXrayCalibrationResult:
    """
    Assemble a ``CameraToXrayCalibrationResult`` from a ``PoseSolveResult``
    and a pre-computed homogeneous transform.

    Used by both the PnP and homography workflows so that result construction
    is not duplicated.
    """
    rotation, _ = cv2.Rodrigues(pose_result.rvec)
    translation = np.asarray(pose_result.tvec, dtype=np.float64).reshape(3, 1)
    T_cx = np.asarray(T_cx, dtype=np.float64)
    T_xc = invert_transform(T_cx)

    return CameraToXrayCalibrationResult(
        rvec=pose_result.rvec,
        tvec=pose_result.tvec,
        rotation=rotation,
        translation=translation,
        T_cx=T_cx,
        T_xc=T_xc,
        inliers=pose_result.inliers,
        inlier_idx=pose_result.inlier_idx,
        uv_proj=pose_result.uv_proj,
        reproj_errors_px=pose_result.reproj_errors_px,
        reproj_mean_px=pose_result.reproj_mean_px,
        reproj_median_px=pose_result.reproj_median_px,
        reproj_max_px=pose_result.reproj_max_px,
        pose_result=pose_result,
    )


# ============================================================
# Public API
# ============================================================

def calibrate_camera_to_xray(
    K_xray: np.ndarray,
    *,
    # ── PnP inputs (required for all non-homography methods) ──
    points_xyz_camera: np.ndarray | None = None,
    points_uv_xray: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    # ── Homography inputs (required when pose_method="homography") ──
    rgb_points_uv: np.ndarray | None = None,
    xray_points_uv: np.ndarray | None = None,
    K_rgb: np.ndarray | None = None,
    pitch_mm: float = 2.54,
    nrows: int = 11,
    ncols: int = 11,
    # ── Shared solver options ──
    pose_method: str = "iterative_ransac",
    refine_with_iterative: bool = False,
    ransac_reprojection_error_px: float = 3.0,
    ransac_confidence: float = 0.99,
    ransac_iterations_count: int = 5000,
) -> CameraToXrayCalibrationResult:
    """
    Estimate the rigid camera->xray transform.

    The function supports two workflows selected by ``pose_method``.

    PnP workflow  (``pose_method`` ∈ ``{"iterative", "iterative_ransac", "ippe"}``)
    -------------------------------------------------------------------------------
    Solves the camera->xray transform directly from 3-D marker positions in
    the camera frame and their 2-D counterparts in the X-ray image.

    Required parameters
    ~~~~~~~~~~~~~~~~~~~
    points_xyz_camera : (N,3) ndarray
        3-D marker positions in the camera frame.
    points_uv_xray : (N,2) ndarray
        Corresponding 2-D detections in the X-ray image.

    Homography workflow  (``pose_method="homography"``)
    ---------------------------------------------------
    Estimates the camera->xray transform by composing two board poses
    (board->rgb and board->xray) recovered from planar DLT homographies:

        T_cx = T_bx @ inv(T_bc)

    The ``uv_proj`` and reprojection statistics in the returned result
    describe the board->xray pose only (not the composed T_cx), consistent
    with what ``pose_result`` also reports.

    Required parameters
    ~~~~~~~~~~~~~~~~~~~
    rgb_points_uv : (N,2) ndarray
        Board corner detections in the RGB image, row-major grid order.
    xray_points_uv : (N,2) ndarray
        Corresponding board corner detections in the X-ray image.
    K_rgb : (3,3) ndarray
        RGB intrinsic matrix.

    Optional board geometry parameters
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    pitch_mm : float
        Physical board pitch in mm.  Default 2.54.
    nrows, ncols : int
        Board grid dimensions.  Default 11×11.

    Shared parameters
    -----------------
    K_xray : (3,3) ndarray
        X-ray intrinsic matrix.
    dist_coeffs : array-like or None
        X-ray distortion coefficients.  Ignored by the homography method
        during estimation; used only if ``refine_with_iterative=True``.
    pose_method : str
        One of ``"iterative"``, ``"iterative_ransac"``, ``"ippe"``,
        ``"homography"``.  Default ``"iterative_ransac"``.
    refine_with_iterative : bool
        Follow the primary solve with an iterative LM refinement pass.
        Default False.
    ransac_reprojection_error_px : float
        Inlier threshold in pixels (``"iterative_ransac"`` only).
    ransac_confidence : float
        RANSAC confidence (``"iterative_ransac"`` only).
    ransac_iterations_count : int
        Maximum RANSAC iterations (``"iterative_ransac"`` only).

    Returns
    -------
    CameraToXrayCalibrationResult
    """
    K_xray = _as_K(K_xray, "K_xray")
    method = str(pose_method).lower().strip()

    # ── Homography workflow ───────────────────────────────────────────────────
    if method == "homography":
        if rgb_points_uv is None or xray_points_uv is None or K_rgb is None:
            raise ValueError(
                "pose_method='homography' requires rgb_points_uv, "
                "xray_points_uv, and K_rgb."
            )

        rgb_uv_raw = _as_uv(rgb_points_uv, "rgb_points_uv")
        xray_uv = _as_uv(xray_points_uv, "xray_points_uv")
        K_rgb = _as_K(K_rgb, "K_rgb")

        if len(rgb_uv_raw) != len(xray_uv):
            raise ValueError(
                "rgb_points_uv and xray_points_uv must have the same length, "
                f"got {len(rgb_uv_raw)} and {len(xray_uv)}."
            )

        dbg = {"nu": int(ncols - 1), "nv": int(nrows - 1)}
        board_xy, rgb_uv, _ = build_planar_correspondences(
            rgb_uv_raw,
            dbg,
            pitch_mm=float(pitch_mm),
        )

        if len(board_xy) != len(xray_uv):
            raise ValueError(
                "Generated board point count does not match xray point count: "
                f"{len(board_xy)} vs {len(xray_uv)}."
            )

        H_bc = estimate_homography_dlt(rgb_uv, board_xy)
        H_bx = estimate_homography_dlt(xray_uv, board_xy)

        _, _, T_bc = decompose_homography_to_pose(H_bc, K_rgb)
        _, _, T_bx = decompose_homography_to_pose(H_bx, K_xray)

        T_cx = T_bx @ invert_transform(T_bc)

        # Express the xray board pose as a PoseSolveResult via the homography
        # solver so that uv_proj and reproj stats describe the xray side.
        board_xyz = np.hstack([
            board_xy,
            np.zeros((len(board_xy), 1), dtype=np.float64),
        ])

        pose_result = solve_pose(
            object_points_xyz=board_xyz,
            image_points_uv=xray_uv,
            K=K_xray,
            dist_coeffs=dist_coeffs,
            pose_method="homography",
            refine_with_iterative=refine_with_iterative,
        )

        return _result_from_pose(pose_result, T_cx)

    # ── PnP workflow ─────────────────────────────────────────────────────────
    if points_xyz_camera is None or points_uv_xray is None:
        raise ValueError(
            f"pose_method='{pose_method}' requires points_xyz_camera and "
            "points_uv_xray."
        )

    xyz = np.asarray(points_xyz_camera, dtype=np.float64)
    uv = _as_uv(points_uv_xray, "points_uv_xray")

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points_xyz_camera must have shape (N, 3).")
    if xyz.shape[0] != uv.shape[0]:
        raise ValueError("3-D and 2-D point counts must match.")

    pose_result = solve_pose(
        object_points_xyz=xyz,
        image_points_uv=uv,
        K=K_xray,
        dist_coeffs=dist_coeffs,
        pose_method=pose_method,
        refine_with_iterative=refine_with_iterative,
        ransac_reprojection_error_px=ransac_reprojection_error_px,
        ransac_confidence=ransac_confidence,
        ransac_iterations_count=ransac_iterations_count,
    )

    T_cx = rvec_tvec_to_transform(pose_result.rvec, pose_result.tvec)
    return _result_from_pose(pose_result, T_cx)