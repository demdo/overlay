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

#from overlay.tools import checkerboard_corner_detection as cbd
from overlay.tools import ransac_plane_fitting as rpf


@dataclass(frozen=True)
class PnPResult:
    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    inliers: np.ndarray | None
    
    
def _intersect_corners_with_plane(
    corners_uv: np.ndarray,
    rgb_intrinsics: np.ndarray,
    plane_model: np.ndarray,
) -> np.ndarray:
    """
    Intersect pixel rays with the calibration plane to recover 3D corner positions.
    
    corners_uv: (3, 2) array [u, v] for top-left, top-right, bottom-left.
    """
    corners_uv = np.asarray(corners_uv, dtype=np.float64)
    if corners_uv.shape != (3, 2):
        raise ValueError("corners_uv must have shape (3, 2).")

    K = np.asarray(rgb_intrinsics, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("rgb_intrinsics must be a 3x3 matrix.")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    a, b, c, d = [float(x) for x in plane_model]
    
    xyz = np.zeros((3, 3), dtype=np.float64)
    for i, (u, v) in enumerate(corners_uv):
        x = (u - cx) / fx
        y = (v - cy) / fy
        denom = a * x + b * y + c
        if abs(denom) < 1e-12:
            raise ValueError("Corner ray is parallel to the fitted plane.")
        z = -d / denom
        xyz[i] = np.array([x * z, y * z, z], dtype=np.float64)
    return xyz


def _interpolate_marker_grid(
    corner_xyz: np.ndarray,
    steps_per_edge: int,
) -> np.ndarray:
    """
    Interpolate the marker grid from 3 corner points.
    
    corner_xyz: (3, 3) array [P_tl, P_tr, P_bl]
    steps_per_edge: s, with alpha/beta in {0, 1, ..., s}
    Returns points in row-major order with shape ((s+1)*(s+1), 3).
    """
    if steps_per_edge <= 0:
        raise ValueError("steps_per_edge must be > 0.")

    corner_xyz = np.asarray(corner_xyz, dtype=np.float64)
    if corner_xyz.shape != (3, 3):
        raise ValueError("corner_xyz must have shape (3, 3).")

    p_tl, p_tr, p_bl = corner_xyz
    step_x = (p_tr - p_tl) / float(steps_per_edge)
    step_y = (p_bl - p_tl) / float(steps_per_edge)

    points = []
    for beta in range(steps_per_edge + 1):
        for alpha in range(steps_per_edge + 1):
            points.append(p_tl + alpha * step_x + beta * step_y)
    return np.asarray(points, dtype=np.float64)


def _solve_xray_pnp(
    points_xyz_camera: np.ndarray,
    points_uv_xray: np.ndarray,
    xray_intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    use_ransac: bool = True,
) -> PnPResult:
    """
    Solve the PnP alignment between camera-frame 3D points and X-ray 2D detections.
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
    dist = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else dist_coeffs

    if use_ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_xyz_camera,
            points_uv_xray,
            Kx,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        success, rvec, tvec = cv2.solvePnP(
            points_xyz_camera,
            points_uv_xray,
            Kx,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inliers = None

    if not success:
        raise RuntimeError("PnP failed to converge.")

    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.reshape(3, 1)
    return PnPResult(
        rvec=rvec,
        tvec=tvec,
        rotation=rotation,
        translation=translation,
        inliers=None if inliers is None else np.asarray(inliers, dtype=np.int64),
    )
    

def calibrate_camera_to_xray(
    points_xyz: np.ndarray,
    corners_uv: np.ndarray,
    rgb_intrinsics: np.ndarray,
    xray_points_uv: np.ndarray,
    xray_intrinsics: np.ndarray,
    *,
    plane_model: np.ndarray | None = None,
    steps_per_edge: int | None = None,
    dist_coeffs: np.ndarray | None = None,
    use_ransac_pnp: bool = True,
) -> tuple[PnPResult, np.ndarray, np.ndarray]:
    """
    End-to-end camera-to-X-ray calibration.
    
    Returns:
        (pnp_result, corner_xyz, marker_xyz_camera)
    """
    if plane_model is None:
        plane_model, _ = rpf.fit_plane_from_points(
            points_xyz,
            distance_threshold=0.005,
            ransac_n=3,
            num_iterations=1000,
        )
    
    corner_xyz = _intersect_corners_with_plane(corners_uv, rgb_intrinsics, plane_model)

    if steps_per_edge is None:
        raise ValueError("steps_per_edge is required to build the marker grid.")
    marker_xyz = _interpolate_marker_grid(corner_xyz, steps_per_edge=steps_per_edge)
    
    pnp = _solve_xray_pnp(
        marker_xyz,
        xray_points_uv,
        xray_intrinsics,
        dist_coeffs=dist_coeffs,
        use_ransac=use_ransac_pnp,
    )
    return pnp, corner_xyz, marker_xyz

    
