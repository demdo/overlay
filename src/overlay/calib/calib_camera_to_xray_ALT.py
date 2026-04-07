# -*- coding: utf-8 -*-
"""
camera_to_xray.py

Camera-to-X-ray calibration implementation.

Implements the workflow sketched in the notes:
1) Fit a calibration plane from the RGB-D point cloud (RANSAC).
2) Intersect 3 checkerboard extreme corners with that plane to recover their 3D positions.
3) Interpolate the 3D marker grid in the camera frame.
4) Solve a PnP problem between 3D camera-frame markers and 2D X-ray detections.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import cv2
import numpy as np

from overlay.tracking.pose_solvers import solve_pose
from overlay.tracking.transforms import invert_transform, rvec_tvec_to_transform


@dataclass(frozen=True)
class PnPResult:
    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray          # (3,3)
    translation: np.ndarray       # (3,1)

    T_4x4: np.ndarray             # (4,4) homogeneous transform

    inliers: np.ndarray | None    # (M,1) OpenCV-style OR None
    inlier_idx: np.ndarray        # (M,) or (N,) fallback if no inliers

    uv_proj: np.ndarray           # (N,2) projected points in X-ray
    reproj_errors_px: np.ndarray  # (N,)
    reproj_mean_px: float
    reproj_median_px: float
    reproj_max_px: float


def calibrate_camera_to_xray(
    points_xyz_camera: np.ndarray,
    points_uv_xray: np.ndarray,
    xray_intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    use_ransac: bool = True,
    *,
    ransac_reproj_error_px: float = 3.0,
    ransac_confidence: float = 0.99,
    ransac_iterations: int = 5000,
) -> PnPResult:
    """
    Estimate the rigid pose (R, t) of the X-ray system relative to the camera
    from 3D–2D correspondences using a perspective projection model.

    If use_ransac=True, a RANSAC-based PnP formulation is applied in order
    to robustly reject outlier correspondences prior to computing the final
    reprojection statistics.

    Parameters
    ----------
    ransac_reproj_error_px : float
        Inlier threshold in pixels used by solvePnPRansac.

    ransac_confidence : float
        Desired probability that at least one randomly sampled minimal
        subset is free of outliers.

    ransac_iterations : int
        Maximum number of RANSAC iterations.

    Returns
    -------
    PnPResult
        Contains rotation, translation, homogeneous transform,
        inlier indices, projected points, and reprojection error statistics.
    """
    points_xyz_camera = np.asarray(points_xyz_camera, dtype=np.float64)
    points_uv_xray = np.asarray(points_uv_xray, dtype=np.float64)

    if points_xyz_camera.ndim != 2 or points_xyz_camera.shape[1] != 3:
        raise ValueError("points_xyz_camera must have shape (N, 3).")
    if points_uv_xray.ndim != 2 or points_uv_xray.shape[1] != 2:
        raise ValueError("points_uv_xray must have shape (N, 2).")
    if points_xyz_camera.shape[0] != points_uv_xray.shape[0]:
        raise ValueError("3D and 2D point counts must match.")
    if points_xyz_camera.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required for PnP.")

    Kx = np.asarray(xray_intrinsics, dtype=np.float64)
    if Kx.shape != (3, 3):
        raise ValueError("xray_intrinsics must be a 3x3 matrix.")

    pose_result = solve_pose(
        object_points_xyz=points_xyz_camera,
        image_points_uv=points_uv_xray,
        K=Kx,
        dist_coeffs=dist_coeffs,
        pose_method="iterative_ransac" if use_ransac else "iterative",
        refine_with_iterative=False,
        ransac_reprojection_error_px=ransac_reproj_error_px,
        ransac_confidence=ransac_confidence,
        ransac_iterations_count=ransac_iterations,
    )

    rotation, _ = cv2.Rodrigues(pose_result.rvec)
    translation = np.asarray(pose_result.tvec, dtype=np.float64).reshape(3, 1)
    T_4x4 = rvec_tvec_to_transform(pose_result.rvec, pose_result.tvec)

    return PnPResult(
        rvec=pose_result.rvec,
        tvec=pose_result.tvec,
        rotation=rotation,
        translation=translation,
        T_4x4=T_4x4,
        inliers=pose_result.inliers,
        inlier_idx=pose_result.inlier_idx,
        uv_proj=pose_result.uv_proj,
        reproj_errors_px=pose_result.reproj_errors_px,
        reproj_mean_px=pose_result.reproj_mean_px,
        reproj_median_px=pose_result.reproj_median_px,
        reproj_max_px=pose_result.reproj_max_px,
    )