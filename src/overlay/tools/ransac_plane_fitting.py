# -*- coding: utf-8 -*-
"""
RANSAC_plane_fitting.py

Helper utilities for RANSAC plane fitting and visualization math.
"""

import numpy as np
import open3d as o3d
import pyrealsense2 as rs


def rect_from_pts(pts_uv: np.ndarray, w: int, h: int, pad: int) -> tuple[int, int, int, int]:
    umin = int(np.floor(np.min(pts_uv[:, 0]) - pad))
    umax = int(np.ceil(np.max(pts_uv[:, 0]) + pad))
    vmin = int(np.floor(np.min(pts_uv[:, 1]) - pad))
    vmax = int(np.ceil(np.max(pts_uv[:, 1]) + pad))

    umin = max(0, min(w - 1, umin))
    umax = max(0, min(w - 1, umax))
    vmin = max(0, min(h - 1, vmin))
    vmax = max(0, min(h - 1, vmax))

    if umax < umin:
        umin, umax = umax, umin
    if vmax < vmin:
        vmin, vmax = vmax, vmin

    return (umin, vmin, umax, vmax)


def sample_pts3d(
    depth_frame,
    rect: tuple[int, int, int, int],
    max_points: int,
    z_min: float,
    z_max: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    intr = depth_frame.profile.as_video_stream_profile().intrinsics

    umin, vmin, umax, vmax = rect
    roi_w = umax - umin + 1
    roi_h = vmax - vmin + 1
    if roi_w <= 2 or roi_h <= 2:
        return np.empty((0, 3), dtype=np.float64)

    n_pix = roi_w * roi_h
    n = min(max_points, n_pix)
    idx = rng.choice(n_pix, size=n, replace=False)

    us = (idx % roi_w).astype(np.int32) + umin
    vs = (idx // roi_w).astype(np.int32) + vmin

    pts = []
    for u, v in zip(us, vs):
        z = float(depth_frame.get_distance(int(u), int(v)))
        if z <= 0 or z < z_min or z > z_max:
            continue
        xyz = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], z)
        pts.append(xyz)

    if not pts:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def deviations(pts: np.ndarray, plane: np.ndarray) -> np.ndarray:
    n = plane[:3]
    denom = float(np.linalg.norm(n))
    if denom <= 1e-12:
        return np.zeros((pts.shape[0],), dtype=np.float64)
    return np.abs(pts @ n + plane[3]) / denom


def intersect_corners_with_plane(
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


def interpolate_marker_grid(
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


def fit_plane_stable(
    points_xyz: np.ndarray,
    distance_threshold: float = 0.0015,   # 1.5 mm
    ransac_n: int = 8,
    num_iterations: int = 3000,
    n_runs: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run RANSAC plane fitting n_runs times and return the spherically averaged
    plane normal + the inlier set of the best run.

    All normals are flipped to a consistent hemisphere before averaging
    (convention: n_z < 0, i.e. normal points toward the camera).

    Parameters
    ----------
    points_xyz        : (N, 3) point cloud
    distance_threshold: RANSAC inlier threshold in metres
    ransac_n          : minimum sample size per RANSAC iteration
    num_iterations    : RANSAC iterations per run
    n_runs            : number of independent RANSAC runs to average

    Returns
    -------
    plane  : (4,) normalised [a, b, c, d]
    inliers: (M,) int64 inlier indices from the best run
    """
    normals: list[np.ndarray] = []
    offsets: list[float] = []
    best_inliers = np.array([], dtype=np.int64)
    reference_normal: np.ndarray | None = None

    for _ in range(n_runs):
        plane_raw, inliers = ransac_plane_open3d(
            points_xyz, distance_threshold, ransac_n, num_iterations
        )
        n_vec = plane_raw[:3]
        norm = float(np.linalg.norm(n_vec))
        if norm < 1e-9:
            continue

        n_vec = n_vec / norm
        d_val = float(plane_raw[3]) / norm

        # Establish sign convention on first valid run,
        # then keep all subsequent runs consistent with it.
        if reference_normal is None:
            if n_vec[2] > 0.0:          # normal points away from camera → flip
                n_vec, d_val = -n_vec, -d_val
            reference_normal = n_vec.copy()
        else:
            if np.dot(n_vec, reference_normal) < 0.0:
                n_vec, d_val = -n_vec, -d_val

        normals.append(n_vec)
        offsets.append(d_val)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if not normals:
        raise RuntimeError("fit_plane_stable: all RANSAC runs failed.")

    # Spherical mean on S²: sum → normalise (Fréchet mean approximation)
    mean_n = np.mean(normals, axis=0)
    mean_n /= np.linalg.norm(mean_n)
    mean_d = float(np.mean(offsets))

    plane_out = np.array([mean_n[0], mean_n[1], mean_n[2], mean_d], dtype=np.float64)
    return plane_out, best_inliers


