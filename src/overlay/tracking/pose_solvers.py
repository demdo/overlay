from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from overlay.tools.homography import (
    estimate_homography_dlt,
    decompose_homography_to_pose,
    project_homography,
)


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
# Result types
# ============================================================

@dataclass(frozen=True)
class IppeCandidateResult:
    candidate_index: int
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    uv_proj: np.ndarray
    reproj_errors_px: np.ndarray
    reproj_mean_px: float
    reproj_median_px: float
    reproj_max_px: float


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

    all_candidates: list[IppeCandidateResult] | None = None


# ============================================================
# IPPE candidate selection
# ============================================================

@dataclass(frozen=True)
class _IppeCandidate:
    """
    Intermediate representation of a single IPPE solution before selection.

    Attributes
    ----------
    rvec : (3,1) float64
    tvec : (3,1) float64
    R : (3,3) float64 – rotation matrix derived from rvec
    reproj_mean_px : float – mean reprojection error over all correspondences
    """

    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    reproj_mean_px: float


def _build_ippe_candidates(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> list[_IppeCandidate]:
    """
    Run cv2.solvePnPGeneric with SOLVEPNP_IPPE and collect all returned
    solutions as ``_IppeCandidate`` objects.

    Raises
    ------
    RuntimeError
        If OpenCV reports failure or returns no candidates.
    """
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

    candidates: list[_IppeCandidate] = []

    for rv_raw, tv_raw in zip(rvecs, tvecs):
        rv = np.asarray(rv_raw, dtype=np.float64).reshape(3, 1)
        tv = np.asarray(tv_raw, dtype=np.float64).reshape(3, 1)
        R, _ = cv2.Rodrigues(rv)

        _, mean_px, _, _, _ = compute_reprojection_error_px(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            rvec=rv,
            tvec=tv,
            K=K,
            dist=dist,
        )

        candidates.append(_IppeCandidate(rvec=rv, tvec=tv, R=R, reproj_mean_px=mean_px))

    return candidates


def _export_ippe_candidate_result(
    candidate: _IppeCandidate,
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    candidate_index: int,
) -> IppeCandidateResult:
    """
    Convert one internal _IppeCandidate to a debug-friendly result object.
    """
    uv_proj, mean_px, median_px, max_px, per_point_px = compute_reprojection_error_px(
        object_points_xyz=object_points_xyz,
        image_points_uv=image_points_uv,
        rvec=candidate.rvec,
        tvec=candidate.tvec,
        K=K,
        dist=dist,
    )

    return IppeCandidateResult(
        candidate_index=int(candidate_index),
        rvec=np.asarray(candidate.rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(candidate.tvec, dtype=np.float64).reshape(3, 1),
        R=np.asarray(candidate.R, dtype=np.float64).reshape(3, 3),
        uv_proj=uv_proj,
        reproj_errors_px=per_point_px,
        reproj_mean_px=float(mean_px),
        reproj_median_px=float(median_px),
        reproj_max_px=float(max_px),
    )


def _select_ippe_candidate(candidates: list[_IppeCandidate]) -> int:
    """
    Select the physically correct IPPE solution via an early-exit decision
    tree that encodes the empirically verified coordinate-system geometry of
    the X-ray / camera setup.

    Coordinate system (empirically verified)
    -----------------------------------------
    X-Ray frame  :  x_x → table down,  y_x → table left,  z_x → intensifier up
    Camera frame :  y_c → right,        z_c → table down

    Criteria (evaluated in order, first discriminating criterion wins)
    -------------------------------------------------------------------
    1. tz > 0      X-Ray origin is in front of the camera.
    2. R[2,2] < 0  The z-axes of the X-Ray and camera frames point in
                   opposite directions.
    3. ty < 0      The camera sits to the right of the intensifier.

    Fallback
    --------
    If all three criteria are tied, the candidate with the smaller mean
    reprojection error is chosen.
    """
    if len(candidates) != 2:
        raise ValueError(
            f"Expected exactly 2 IPPE candidates, got {len(candidates)}."
        )

    s0, s1 = candidates[0], candidates[1]

    # Criterion 1: tz > 0
    c1 = [s.tvec[2, 0] > 0 for s in candidates]
    if c1[0] != c1[1]:
        return 0 if c1[0] else 1
    if not c1[0]:
        return 0 if s0.reproj_mean_px <= s1.reproj_mean_px else 1

    # Criterion 2: R[2,2] < 0
    c2 = [s.R[2, 2] < 0 for s in candidates]
    if c2[0] != c2[1]:
        return 0 if c2[0] else 1

    # Criterion 3: ty < 0
    c3 = [s.tvec[1, 0] < 0 for s in candidates]
    if c3[0] != c3[1]:
        return 0 if c3[0] else 1

    # Fallback: smaller reprojection error
    return 0 if s0.reproj_mean_px <= s1.reproj_mean_px else 1


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
        all_candidates=None,
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
        all_candidates=None,
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
    Internal helper for IPPE-based pose estimation with geometry-aware
    candidate selection and optional iterative refinement.
    """
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = normalize_dist_coeffs(dist_coeffs)

    if object_points_xyz.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required.")

    candidates = _build_ippe_candidates(
        object_points_xyz=object_points_xyz,
        image_points_uv=image_points_uv,
        K=K,
        dist=dist,
    )

    all_candidates = [
        _export_ippe_candidate_result(
            candidate=c,
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            K=K,
            dist=dist,
            candidate_index=i,
        )
        for i, c in enumerate(candidates)
    ]

    best_idx = _select_ippe_candidate(candidates)
    best = candidates[best_idx]

    rvec = best.rvec
    tvec = best.tvec

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
        all_candidates=all_candidates,
    )


def _solve_pose_homography(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    *,
    refine_with_iterative: bool = False,
) -> PoseSolveResult:
    """
    Internal helper for planar pose estimation via DLT homography decomposition.
    """
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = normalize_dist_coeffs(dist_coeffs)

    if object_points_xyz.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required.")

    board_xy = object_points_xyz[:, :2]

    H = estimate_homography_dlt(image_points_uv, board_xy)
    R, t, _ = decompose_homography_to_pose(H, K)

    rvec, _ = cv2.Rodrigues(R)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(t, dtype=np.float64).reshape(3, 1)

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
        method="homography",
        raw_pnp_flag=-1,
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
        all_candidates=None,
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

    if method == "homography":
        return _solve_pose_homography(
            object_points_xyz=object_points_xyz,
            image_points_uv=image_points_uv,
            K=K,
            dist_coeffs=dist_coeffs,
            refine_with_iterative=refine_with_iterative,
        )

    raise ValueError(
        f"Unknown pose_method '{pose_method}'. "
        f"Expected 'iterative', 'iterative_ransac', 'ippe', or 'homography'."
    )