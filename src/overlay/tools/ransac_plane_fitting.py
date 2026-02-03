# -*- coding: utf-8 -*-
"""
RANSAC_plane_fitting.py

Helper utilities for RANSAC plane fitting and visualization math.
"""

import numpy as np
import open3d as o3d


# =========================
# Open3D RANSAC plane fit
# =========================
def ransac_plane_open3d(points_xyz, distance_threshold: float, ransac_n: int, num_iterations: int):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    return np.asarray(plane_model, dtype=np.float64), np.asarray(inliers, dtype=np.int64)


def fit_plane_from_points(
    points_xyz: np.ndarray,
    distance_threshold: float = 0.005,
    ransac_n: int = 3,
    num_iterations: int = 1000,
):
    """
    Fit a plane model to an arbitrary point cloud.

    Returns plane coefficients (a, b, c, d) and inlier indices.
    """
    if points_xyz is None or len(points_xyz) == 0:
        raise ValueError("points_xyz must contain at least one point.")
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3).")
    return ransac_plane_open3d(points_xyz, distance_threshold, ransac_n, num_iterations)


