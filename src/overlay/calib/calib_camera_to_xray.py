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
    

def _solve_xray_pnp(
    points_xyz_camera: np.ndarray,
    points_uv_xray: np.ndarray,
    xray_intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    use_ransac: bool = True,
    *,
    ransac_reproj_error_px: float = 3.0,   # threshold
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
        A correspondence is classified as an inlier if its reprojection error
        (Euclidean distance between measured and projected image point)
        is below this value.
    
        The threshold is defined in pixel units on the detector.
    
        For the ARCADIS Orbic system (23 cm detector field-of-view over
        1024 px resolution), the detector scale is approximately:
    
            230 mm / 1024 px ≈ 0.225 mm/px   (at SOD = SID = 980 mm)
    
        However, the effective mm/px in the object plane depends on
        geometric magnification:
    
            mm/px ≈ (230 mm / 1024 px) * (SOD / SID)
    
        Assuming a typical board position of
            SOD ≈ 450 mm and SID ≈ 980 mm,
    
        this yields:
    
            mm/px ≈ 0.10 mm/px
    
        Consequently, a RANSAC threshold of 3 px corresponds to
        approximately 0.3 mm in the board plane.
    
        Typical values in practice: 2–4 px.
    
    ransac_confidence : float
        Desired probability that at least one randomly sampled minimal
        subset is free of outliers. This parameter controls the statistical
        robustness of RANSAC but does not directly influence pose accuracy.
        Commonly set to 0.99.
    
    ransac_iterations : int
        Maximum number of RANSAC iterations. Higher values increase the
        probability of finding a valid inlier-only model when the outlier
        ratio is high, at the cost of increased computation time.
    
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

    dist = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)

    if use_ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_xyz_camera,
            points_uv_xray,
            Kx,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=ransac_reproj_error_px,
            confidence=ransac_confidence,
            iterationsCount=ransac_iterations,
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
    
    T_4x4 = np.eye(4, dtype=np.float64)
    T_4x4[:3, :3] = rotation
    T_4x4[:3, 3:4] = translation

    # Reproject points
    proj, _ = cv2.projectPoints(points_xyz_camera, rvec, tvec, Kx, dist)
    uv_proj = proj.reshape(-1, 2)

    uv_meas = points_uv_xray.reshape(-1, 2)
    reproj_errors = np.linalg.norm(uv_meas - uv_proj, axis=1)

    if inliers is None or len(inliers) == 0:
        inlier_idx = np.arange(len(uv_meas), dtype=np.int64)
    else:
        inlier_idx = inliers.reshape(-1).astype(np.int64)

    inlier_errs = reproj_errors[inlier_idx]
    reproj_mean = float(np.mean(inlier_errs))
    reproj_median = float(np.median(inlier_errs))
    reproj_max = float(np.max(inlier_errs))

    return PnPResult(
        rvec=rvec,
        tvec=tvec,
        rotation=rotation,
        translation=translation,
        T_4x4=T_4x4,
        inliers=None if inliers is None else np.asarray(inliers, dtype=np.int64),
        inlier_idx=inlier_idx,
        uv_proj=uv_proj,
        reproj_errors_px=reproj_errors,
        reproj_mean_px=reproj_mean,
        reproj_median_px=reproj_median,
        reproj_max_px=reproj_max,
    )


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3:4]
    Tinv = np.eye(4, dtype=T.dtype)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3:4] = -R.T @ t
    return Tinv
