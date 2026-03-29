from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_OVERLAY_DIR = _THIS_FILE.parents[1]   # .../src/overlay

if str(_OVERLAY_DIR) not in sys.path:
    sys.path.insert(0, str(_OVERLAY_DIR))

import numpy as np
import cv2

from tracking.pose_solvers import (
    solve_pose,
    normalize_dist_coeffs,
    compute_reprojection_error_px,
)


# ============================================================
# Config
# ============================================================

NPZ_PATH = Path(__file__).resolve().with_name("T_cx_debug_new.npz")
DIST_COEFFS = None
PRINT_ROTATION_MATRIX = True


# ============================================================
# Basic helpers
# ============================================================

def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ t)
    return T_inv


def transform_point(T_ab: np.ndarray, p_a: np.ndarray) -> np.ndarray:
    p_a = np.asarray(p_a, dtype=np.float64).reshape(3)
    return T_ab[:3, :3] @ p_a + T_ab[:3, 3]


def transform_direction(T_ab: np.ndarray, v_a: np.ndarray) -> np.ndarray:
    v_a = np.asarray(v_a, dtype=np.float64).reshape(3)
    return T_ab[:3, :3] @ v_a


def normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


# ============================================================
# I/O
# ============================================================

def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    with np.load(str(path), allow_pickle=False) as npz:
        required = ["K_xray", "points_xyz_camera", "points_uv_xray"]
        missing = [k for k in required if k not in npz.files]
        if missing:
            raise KeyError(
                f"{path.name} is missing keys: {missing}. "
                f"Found keys: {list(npz.files)}"
            )

        K = np.asarray(npz["K_xray"], dtype=np.float64)
        xyz = np.asarray(npz["points_xyz_camera"], dtype=np.float64)
        uv = np.asarray(npz["points_uv_xray"], dtype=np.float64)

    if K.shape != (3, 3):
        raise ValueError(f"K_xray has shape {K.shape}, expected (3, 3)")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"points_xyz_camera has shape {xyz.shape}, expected (N, 3)")
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"points_uv_xray has shape {uv.shape}, expected (N, 2)")
    if xyz.shape[0] != uv.shape[0]:
        raise ValueError(
            f"Point count mismatch: xyz has {xyz.shape[0]}, uv has {uv.shape[0]}"
        )

    return K, xyz, uv


# ============================================================
# Plane analysis
# ============================================================

def fit_plane_svd(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit plane to 3D points.

    Returns
    -------
    center : (3,)
        Centroid of the points.
    normal : (3,)
        Unit normal from SVD (sign still ambiguous).
    """
    P = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if P.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane.")

    center = P.mean(axis=0)
    X = P - center

    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    normal = Vt[-1, :]
    normal = normalize_vec(normal)

    return center, normal


def orient_normal_towards_camera_origin(center_c: np.ndarray, normal_c: np.ndarray) -> np.ndarray:
    """
    Flip the plane normal in camera frame so that it points towards
    the camera origin (0,0,0) of the RGB camera frame.

    This gives a deterministic sign choice in the camera frame.
    """
    center_c = np.asarray(center_c, dtype=np.float64).reshape(3)
    normal_c = normalize_vec(normal_c)

    vec_plane_to_cam = -center_c
    if np.dot(normal_c, vec_plane_to_cam) < 0.0:
        normal_c = -normal_c
    return normal_c


def plane_metrics_in_xray(
    center_c: np.ndarray,
    normal_c: np.ndarray,
    T_cx: np.ndarray,
) -> dict:
    """
    Transform plane center + normal from camera frame to xray frame
    and compute several distance metrics relative to the xray source.

    Convention:
    - T_cx maps camera -> xray
    - Xray source is assumed at origin of xray frame
    - Plane equation in xray frame: n_x^T X = d_signed

    Returns
    -------
    dict with:
        center_x
        normal_x_raw
        d_signed_raw
        normal_x_source_facing
        d_x_source_facing
    """
    center_x = transform_point(T_cx, center_c)
    normal_x_raw = normalize_vec(transform_direction(T_cx, normal_c))

    # raw signed plane distance for current normal sign
    d_signed_raw = float(np.dot(normal_x_raw, center_x))

    # choose normal so that it points from plane towards source at origin
    # then d_x becomes positive for the usual source-to-plane distance
    normal_x_source_facing = normal_x_raw.copy()
    if np.dot(normal_x_source_facing, -center_x) < 0.0:
        normal_x_source_facing = -normal_x_source_facing

    d_x_source_facing = float(np.dot(normal_x_source_facing, center_x))
    # This should be >= 0 by construction, up to numerical precision.

    return {
        "center_x": center_x,
        "normal_x_raw": normal_x_raw,
        "d_signed_raw": d_signed_raw,
        "normal_x_source_facing": normal_x_source_facing,
        "d_x_source_facing": d_x_source_facing,
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate_pose(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    uv_proj, mean_px, median_px, max_px, _ = compute_reprojection_error_px(
        object_points_xyz=object_points_xyz,
        image_points_uv=image_points_uv,
        rvec=rvec,
        tvec=tvec,
        K=K,
        dist=dist,
    )
    return mean_px, median_px, max_px, uv_proj


def print_pose_result(
    title: str,
    rvec: np.ndarray,
    tvec: np.ndarray,
    mean_px: float,
    median_px: float,
    max_px: float,
    xyz_c: np.ndarray,
) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    R_cx = rodrigues_to_R(rvec)
    T_cx = make_transform(R_cx, tvec)
    T_xc = invert_transform(T_cx)

    print("rvec =")
    print(rvec)

    print("\ntvec =")
    print(tvec)

    if PRINT_ROTATION_MATRIX:
        print("\nR_cx =")
        print(R_cx)

    print("\nT_cx =")
    print(T_cx)

    print("\nT_xc =")
    print(T_xc)

    print(f"\nreproj mean   = {mean_px:.6f} px")
    print(f"reproj median = {median_px:.6f} px")
    print(f"reproj max    = {max_px:.6f} px")

    # --------------------------------------------------------
    # Plane analysis
    # --------------------------------------------------------
    center_c, normal_c_raw = fit_plane_svd(xyz_c)
    normal_c = orient_normal_towards_camera_origin(center_c, normal_c_raw)

    metrics = plane_metrics_in_xray(
        center_c=center_c,
        normal_c=normal_c,
        T_cx=T_cx,
    )

    print("\n--- Plane / board geometry ---")
    print("board center in camera frame =")
    print(center_c.reshape(3,))

    print("\nboard normal in camera frame (towards camera origin) =")
    print(normal_c.reshape(3,))

    print("\nboard center in xray frame =")
    print(metrics["center_x"].reshape(3,))

    print("\nboard normal in xray frame (raw transformed) =")
    print(metrics["normal_x_raw"].reshape(3,))

    print(f"\nraw signed plane distance in xray frame = {metrics['d_signed_raw']:.6f}")

    print("\nboard normal in xray frame (source-facing) =")
    print(metrics["normal_x_source_facing"].reshape(3,))

    print(f"\nd_x (source-facing, should be >= 0) = {metrics['d_x_source_facing']:.6f}")

    # extra diagnostics
    center_x = metrics["center_x"]
    print(f"\n||board center in xray frame|| = {np.linalg.norm(center_x):.6f}")
    print(f"center_x.z = {center_x[2]:.6f}")


# ============================================================
# Solver runners
# ============================================================

def run_reference_solvers(K: np.ndarray, xyz: np.ndarray, uv: np.ndarray) -> None:
    print("\n" + "#" * 70)
    print("REFERENCE SOLVERS")
    print("#" * 70)

    res_iter = solve_pose(
        object_points_xyz=xyz,
        image_points_uv=uv,
        K=K,
        dist_coeffs=DIST_COEFFS,
        pose_method="iterative",
    )

    print_pose_result(
        title="ITERATIVE",
        rvec=res_iter.rvec,
        tvec=res_iter.tvec,
        mean_px=res_iter.reproj_mean_px,
        median_px=res_iter.reproj_median_px,
        max_px=res_iter.reproj_max_px,
        xyz_c=xyz,
    )

    res_ransac = solve_pose(
        object_points_xyz=xyz,
        image_points_uv=uv,
        K=K,
        dist_coeffs=DIST_COEFFS,
        pose_method="iterative_ransac",
        ransac_reprojection_error_px=1.5,
        ransac_confidence=0.99,
        ransac_iterations_count=5000,
    )

    print_pose_result(
        title="ITERATIVE_RANSAC",
        rvec=res_ransac.rvec,
        tvec=res_ransac.tvec,
        mean_px=res_ransac.reproj_mean_px,
        median_px=res_ransac.reproj_median_px,
        max_px=res_ransac.reproj_max_px,
        xyz_c=xyz,
    )


def run_ippe_all_candidates(K: np.ndarray, xyz: np.ndarray, uv: np.ndarray) -> None:
    print("\n" + "#" * 70)
    print("IPPE ALL CANDIDATES")
    print("#" * 70)

    dist = normalize_dist_coeffs(DIST_COEFFS)

    success, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
        objectPoints=np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        imagePoints=np.asarray(uv, dtype=np.float64).reshape(-1, 2),
        cameraMatrix=np.asarray(K, dtype=np.float64).reshape(3, 3),
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not success or rvecs is None or tvecs is None or len(rvecs) == 0:
        raise RuntimeError("cv2.solvePnPGeneric(..., SOLVEPNP_IPPE) failed.")

    print(f"Number of IPPE candidates: {len(rvecs)}")

    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

        mean_px, median_px, max_px, _ = evaluate_pose(
            object_points_xyz=xyz,
            image_points_uv=uv,
            K=K,
            dist=dist,
            rvec=rvec,
            tvec=tvec,
        )

        print_pose_result(
            title=f"IPPE CANDIDATE {i}",
            rvec=rvec,
            tvec=tvec,
            mean_px=mean_px,
            median_px=median_px,
            max_px=max_px,
            xyz_c=xyz,
        )

        if reproj_errs is not None and len(reproj_errs) > i:
            try:
                print(f"OpenCV reprojErr[{i}] = {float(reproj_errs[i]):.6f}")
            except Exception:
                pass


# ============================================================
# Main
# ============================================================

def main() -> None:
    np.set_printoptions(precision=8, suppress=True)

    K, xyz, uv = load_data(NPZ_PATH)

    print("=" * 70)
    print("Loaded data")
    print("=" * 70)
    print(f"NPZ path   : {NPZ_PATH}")
    print(f"K shape    : {K.shape}")
    print(f"xyz shape  : {xyz.shape}")
    print(f"uv shape   : {uv.shape}")

    run_reference_solvers(K, xyz, uv)
    run_ippe_all_candidates(K, xyz, uv)


if __name__ == "__main__":
    main()