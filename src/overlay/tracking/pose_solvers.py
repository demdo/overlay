# overlay/pose/pose_solvers.py

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


# ============================================================
# Helpers
# ============================================================

def normalize_dist_coeffs(dist_coeffs: np.ndarray | None) -> np.ndarray:
    """
    Normalize distortion coefficients to a float64 column vector.

    If None is given, a zero-distortion model is assumed.
    """
    if dist_coeffs is None:
        return np.zeros((5, 1), dtype=np.float64)

    dist = np.asarray(dist_coeffs, dtype=np.float64)
    return dist.reshape(-1, 1)


def compute_reprojection_error_px(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, float, float, float, np.ndarray]:
    """
    Project 3D object points into the image and compute reprojection errors.

    Returns
    -------
    uv_proj : (N,2)
    mean_px : float
    median_px : float
    max_px : float
    per_point_px : (N,)
    """
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    uv_proj, _ = cv2.projectPoints(
        object_points_xyz,
        rvec,
        tvec,
        K,
        dist,
    )
    uv_proj = uv_proj.reshape(-1, 2)

    per_point_px = np.linalg.norm(image_points_uv - uv_proj, axis=1)
    mean_px = float(np.mean(per_point_px))
    median_px = float(np.median(per_point_px))
    max_px = float(np.max(per_point_px))

    return uv_proj, mean_px, median_px, max_px, per_point_px


def _compute_inlier_idx(
    inliers: np.ndarray | None,
    num_points: int,
) -> np.ndarray:
    """
    Convert OpenCV-style inlier output to a flat int64 index array.

    If no inliers are provided, all points are treated as inliers.
    """
    if inliers is None or len(inliers) == 0:
        return np.arange(num_points, dtype=np.int64)

    return np.asarray(inliers, dtype=np.int64).reshape(-1)


def _compute_reprojection_stats_from_subset(
    reproj_errors_px: np.ndarray,
    inlier_idx: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute mean / median / max reprojection error on the selected subset.
    """
    reproj_errors_px = np.asarray(reproj_errors_px, dtype=np.float64).reshape(-1)
    inlier_idx = np.asarray(inlier_idx, dtype=np.int64).reshape(-1)

    if reproj_errors_px.ndim != 1:
        raise ValueError("reproj_errors_px must be 1D.")
    if inlier_idx.ndim != 1:
        raise ValueError("inlier_idx must be 1D.")
    if len(reproj_errors_px) == 0:
        raise ValueError("At least one reprojection error is required.")

    if len(inlier_idx) == 0:
        inlier_idx = np.arange(len(reproj_errors_px), dtype=np.int64)

    inlier_errs = reproj_errors_px[inlier_idx]
    mean_px = float(np.mean(inlier_errs))
    median_px = float(np.median(inlier_errs))
    max_px = float(np.max(inlier_errs))

    return mean_px, median_px, max_px


def _refine_pose_iterative(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec_init: np.ndarray,
    tvec_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine an existing pose estimate using SOLVEPNP_ITERATIVE
    with useExtrinsicGuess=True.
    """
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    rvec_init = np.asarray(rvec_init, dtype=np.float64).reshape(3, 1)
    tvec_init = np.asarray(tvec_init, dtype=np.float64).reshape(3, 1)

    success, rvec_ref, tvec_ref = cv2.solvePnP(
        object_points_xyz,
        image_points_uv,
        K,
        dist,
        rvec=rvec_init,
        tvec=tvec_init,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        raise RuntimeError("Iterative pose refinement failed.")

    rvec_ref = np.asarray(rvec_ref, dtype=np.float64).reshape(3, 1)
    tvec_ref = np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1)
    return rvec_ref, tvec_ref


# ============================================================
# Result type
# ============================================================

@dataclass(frozen=True)
class PoseSolveResult:
    rvec: np.ndarray
    tvec: np.ndarray
    method: str
    raw_pnp_flag: int
    used_extrinsic_guess: bool
    candidate_index: int | None
    refined_with_iterative: bool
    refinement_pnp_flag: int | None

    inliers: np.ndarray | None
    inlier_idx: np.ndarray

    uv_proj: np.ndarray
    reproj_errors_px: np.ndarray
    reproj_mean_px: float
    reproj_median_px: float
    reproj_max_px: float


# ============================================================
# Core solvers
# ============================================================

def _solve_pose_iterative(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    *,
    rvec_init: np.ndarray | None = None,
    tvec_init: np.ndarray | None = None,
    use_extrinsic_guess: bool = False,
    refine_with_iterative: bool = True,
) -> PoseSolveResult:
    """
    Internal helper for classical solvePnP with SOLVEPNP_ITERATIVE.

    This function is single-frame only.
    No temporal prior or Kalman filtering is used here.
    """
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = normalize_dist_coeffs(dist_coeffs)

    if object_points_xyz.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required.")

    used_guess = bool(
        use_extrinsic_guess and
        rvec_init is not None and
        tvec_init is not None
    )

    if used_guess:
        rvec_init = np.asarray(rvec_init, dtype=np.float64).reshape(3, 1)
        tvec_init = np.asarray(tvec_init, dtype=np.float64).reshape(3, 1)

        success, rvec, tvec = cv2.solvePnP(
            object_points_xyz,
            image_points_uv,
            K,
            dist,
            rvec=rvec_init,
            tvec=tvec_init,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        success, rvec, tvec = cv2.solvePnP(
            object_points_xyz,
            image_points_uv,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not success:
        raise RuntimeError("SOLVEPNP_ITERATIVE failed.")

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    if refine_with_iterative:
        rvec, tvec = _refine_pose_iterative(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            K=K,
            dist=dist,
            rvec_init=rvec,
            tvec_init=tvec,
        )

    uv_proj, _, _, _, per_point_px = compute_reprojection_error_px(
        object_points_xyz=object_points_xyz,
        image_points_uv=image_points_uv,
        rvec=rvec,
        tvec=tvec,
        K=K,
        dist=dist,
    )

    inliers = None
    inlier_idx = _compute_inlier_idx(inliers, num_points=len(image_points_uv))
    mean_px, median_px, max_px = _compute_reprojection_stats_from_subset(
        per_point_px,
        inlier_idx,
    )

    return PoseSolveResult(
        rvec=rvec,
        tvec=tvec,
        method="iterative",
        raw_pnp_flag=int(cv2.SOLVEPNP_ITERATIVE),
        used_extrinsic_guess=used_guess,
        candidate_index=None,
        refined_with_iterative=bool(refine_with_iterative),
        refinement_pnp_flag=int(cv2.SOLVEPNP_ITERATIVE) if refine_with_iterative else None,
        inliers=inliers,
        inlier_idx=inlier_idx,
        uv_proj=uv_proj,
        reproj_errors_px=per_point_px,
        reproj_mean_px=mean_px,
        reproj_median_px=median_px,
        reproj_max_px=max_px,
    )


def _solve_pose_iterative_ransac(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    *,
    refine_with_iterative: bool = True,
    ransac_reprojection_error_px: float = 8.0,
    ransac_confidence: float = 0.99,
    ransac_iterations_count: int = 100,
) -> PoseSolveResult:
    """
    Internal helper for solvePnPRansac with SOLVEPNP_ITERATIVE.

    This function is single-frame only.
    No temporal prior or Kalman filtering is used here.

    Notes
    -----
    RANSAC is used for robust initial pose estimation in the presence of
    outlier correspondences. The resulting pose may optionally be refined
    afterwards with SOLVEPNP_ITERATIVE on the full correspondence set.
    """
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = normalize_dist_coeffs(dist_coeffs)

    if object_points_xyz.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required.")

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points_xyz,
        imagePoints=image_points_uv,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=float(ransac_reprojection_error_px),
        confidence=float(ransac_confidence),
        iterationsCount=int(ransac_iterations_count),
    )

    if not success:
        raise RuntimeError("SOLVEPNP_ITERATIVE_RANSAC failed.")

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    inliers = None if inliers is None else np.asarray(inliers, dtype=np.int64)

    if refine_with_iterative:
        rvec, tvec = _refine_pose_iterative(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            K=K,
            dist=dist,
            rvec_init=rvec,
            tvec_init=tvec,
        )

    uv_proj, _, _, _, per_point_px = compute_reprojection_error_px(
        object_points_xyz=object_points_xyz,
        image_points_uv=image_points_uv,
        rvec=rvec,
        tvec=tvec,
        K=K,
        dist=dist,
    )

    inlier_idx = _compute_inlier_idx(inliers, num_points=len(image_points_uv))
    mean_px, median_px, max_px = _compute_reprojection_stats_from_subset(
        per_point_px,
        inlier_idx,
    )

    return PoseSolveResult(
        rvec=rvec,
        tvec=tvec,
        method="iterative_ransac",
        raw_pnp_flag=int(cv2.SOLVEPNP_ITERATIVE),
        used_extrinsic_guess=False,
        candidate_index=None,
        refined_with_iterative=bool(refine_with_iterative),
        refinement_pnp_flag=int(cv2.SOLVEPNP_ITERATIVE) if refine_with_iterative else None,
        inliers=inliers,
        inlier_idx=inlier_idx,
        uv_proj=uv_proj,
        reproj_errors_px=per_point_px,
        reproj_mean_px=mean_px,
        reproj_median_px=median_px,
        reproj_max_px=max_px,
    )


def _solve_pose_ippe(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    *,
    refine_with_iterative: bool = True,
) -> PoseSolveResult:
    """
    Internal helper for IPPE-based pose estimation with reprojection-based
    candidate selection and optional iterative refinement.

    Procedure
    ---------
    1. Compute IPPE pose candidates with cv2.solvePnPGeneric(..., SOLVEPNP_IPPE)
    2. Evaluate all returned candidates by reprojection error
    3. Select the best candidate
    4. Optionally refine the selected candidate with SOLVEPNP_ITERATIVE

    Notes
    -----
    This function is single-frame only.
    No temporal prior or Kalman filtering is used here.
    """
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = normalize_dist_coeffs(dist_coeffs)

    if object_points_xyz.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required.")

    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        objectPoints=object_points_xyz,
        imagePoints=image_points_uv,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not success or rvecs is None or tvecs is None or len(rvecs) == 0:
        raise RuntimeError("SOLVEPNP_IPPE failed.")

    if len(rvecs) != len(tvecs):
        raise RuntimeError("IPPE returned inconsistent candidate counts.")

    best_idx: int | None = None
    best_mean_px: float | None = None
    best_rvec: np.ndarray | None = None
    best_tvec: np.ndarray | None = None

    for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
        rv = np.asarray(rv, dtype=np.float64).reshape(3, 1)
        tv = np.asarray(tv, dtype=np.float64).reshape(3, 1)

        _, mean_px, _, _, _ = compute_reprojection_error_px(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            rvec=rv,
            tvec=tv,
            K=K,
            dist=dist,
        )

        if best_mean_px is None or mean_px < best_mean_px:
            best_mean_px = mean_px
            best_idx = i
            best_rvec = rv
            best_tvec = tv

    if best_rvec is None or best_tvec is None or best_idx is None:
        raise RuntimeError("IPPE candidate selection failed.")

    rvec = best_rvec
    tvec = best_tvec

    if refine_with_iterative:
        rvec, tvec = _refine_pose_iterative(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            K=K,
            dist=dist,
            rvec_init=rvec,
            tvec_init=tvec,
        )

    uv_proj, _, _, _, per_point_px = compute_reprojection_error_px(
        object_points_xyz=object_points_xyz,
        image_points_uv=image_points_uv,
        rvec=rvec,
        tvec=tvec,
        K=K,
        dist=dist,
    )

    inliers = None
    inlier_idx = _compute_inlier_idx(inliers, num_points=len(image_points_uv))
    mean_px, median_px, max_px = _compute_reprojection_stats_from_subset(
        per_point_px,
        inlier_idx,
    )

    return PoseSolveResult(
        rvec=rvec,
        tvec=tvec,
        method="ippe",
        raw_pnp_flag=int(cv2.SOLVEPNP_IPPE),
        used_extrinsic_guess=False,
        candidate_index=int(best_idx),
        refined_with_iterative=bool(refine_with_iterative),
        refinement_pnp_flag=int(cv2.SOLVEPNP_ITERATIVE) if refine_with_iterative else None,
        inliers=inliers,
        inlier_idx=inlier_idx,
        uv_proj=uv_proj,
        reproj_errors_px=per_point_px,
        reproj_mean_px=mean_px,
        reproj_median_px=median_px,
        reproj_max_px=max_px,
    )


def solve_pose(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    *,
    pose_method: str = "iterative",
    rvec_init: np.ndarray | None = None,
    tvec_init: np.ndarray | None = None,
    use_extrinsic_guess: bool = False,
    refine_with_iterative: bool = True,
    ransac_reprojection_error_px: float = 8.0,
    ransac_confidence: float = 0.99,
    ransac_iterations_count: int = 100,
) -> PoseSolveResult:
    """
    Dispatch pose estimation by method name.

    Supported methods
    -----------------
    - "iterative": classical PnP
    - "iterative_ransac": classical PnP with RANSAC initialization
    - "ippe": planar PnP with IPPE candidate selection

    Notes
    -----
    - All methods operate on a single frame.
    - Temporal filtering (e.g. Kalman) must be applied externally.
    - All methods may optionally be followed by iterative refinement.
    """
    method = str(pose_method).lower().strip()

    if method == "iterative":
        return _solve_pose_iterative(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            K=K,
            dist_coeffs=dist_coeffs,
            rvec_init=rvec_init,
            tvec_init=tvec_init,
            use_extrinsic_guess=use_extrinsic_guess,
            refine_with_iterative=refine_with_iterative,
        )

    if method == "iterative_ransac":
        return _solve_pose_iterative_ransac(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            K=K,
            dist_coeffs=dist_coeffs,
            refine_with_iterative=refine_with_iterative,
            ransac_reprojection_error_px=ransac_reprojection_error_px,
            ransac_confidence=ransac_confidence,
            ransac_iterations_count=ransac_iterations_count,
        )

    if method == "ippe":
        return _solve_pose_ippe(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            K=K,
            dist_coeffs=dist_coeffs,
            refine_with_iterative=refine_with_iterative,
        )

    raise ValueError(
        f"Unknown pose_method '{pose_method}'. "
        f"Expected 'iterative', 'iterative_ransac', or 'ippe'."
    )