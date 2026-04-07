# overlay/tools/homography.py

from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np
import cv2

Cell = Tuple[int, int]


def build_board_xyz_canonical(
    *,
    nu: int,
    nv: int,
    pitch_mm: float = 2.54,
) -> np.ndarray:
    """
    Build canonical board coordinates (3D, z=0) in mm.

    Canonical board frame (DEFINED ONCE FOR ENTIRE PROJECT)
    ------------------------------------------------------
    - origin  : bottom-left board corner (point 1)
    - +x_b    : upward on the board
    - +y_b    : to the right on the board
    - +z_b    : into plane (right-handed)

    Grid layout
    -----------
    - ncols = nu + 1
    - nrows = nv + 1
    - row-major ordering:
        first row front->back, then next row

    Returns
    -------
    XYZ : (N,3) float64
        Board coordinates in mm
    """

    ncols = int(nu) + 1
    nrows = int(nv) + 1

    jj, ii = np.meshgrid(
        np.arange(ncols, dtype=np.float64),
        np.arange(nrows, dtype=np.float64),
        indexing="xy",
    )

    X = jj * float(pitch_mm)   # +x_b → right
    Y = ii * float(pitch_mm)   # +y_b → down
    Z = np.zeros_like(X)

    XYZ = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float64)
    return XYZ


# ============================================================
# Internal helper
# ============================================================

def _largest_contiguous_run(vals_sorted_unique: List[int]) -> Tuple[int, int]:
    """
    Return (start, end) of the longest contiguous run
    in a sorted unique integer list.
    """
    if not vals_sorted_unique:
        raise ValueError("_largest_contiguous_run: empty input")

    best_s = best_e = vals_sorted_unique[0]
    best_len = 1

    cur_s = cur_e = vals_sorted_unique[0]
    cur_len = 1

    for v in vals_sorted_unique[1:]:
        if v == cur_e + 1:
            cur_e = v
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_s, best_e = cur_s, cur_e
            cur_s = cur_e = v
            cur_len = 1

    if cur_len > best_len:
        best_s, best_e = cur_s, cur_e

    return best_s, best_e


# ============================================================
# Build correspondences
# ============================================================

def build_planar_correspondences(
    roi_uv: np.ndarray,
    dbg: dict,
    *,
    pitch_mm: float = 2.54,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build (XY_mm, uv_px, meta) with GUARANTEED consistent ordering.

    Assumptions
    ----------
    - roi_uv is already ordered in grid row-major order:
        row 0: left->right, then row 1, ...
    - dbg contains 'nu' and 'nv' (grid steps along u and v), so that
        N = (nv+1)*(nu+1) matches roi_uv length.

    Returns
    -------
    XY_mm : np.ndarray
        (N,2) float64 grid coordinates in mm, row-major order.
    uv_px : np.ndarray
        (N,2) float64 image coordinates in px (copy of roi_uv).
    meta : dict
        Contains inferred (nrows, ncols, nu, nv).
    """
    uv = np.asarray(roi_uv, dtype=np.float64).reshape(-1, 2)
    if uv.size == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            dict(nrows=0, ncols=0, nu=None, nv=None),
        )

    nu = int(dbg.get("nu", -1))
    nv = int(dbg.get("nv", -1))
    if nu < 0 or nv < 0:
        raise ValueError("dbg must contain integer 'nu' and 'nv'.")

    ncols = nu + 1
    nrows = nv + 1

    expected = nrows * ncols
    got = int(uv.shape[0])
    if got != expected:
        raise ValueError(
            f"ROI size mismatch: expected (nv+1)*(nu+1)={expected} points, got {got} "
            f"(nv={nv}, nu={nu})."
        )

    # Row-major grid to match uv row-major
    jj, ii = np.meshgrid(
        np.arange(ncols, dtype=np.float64),
        np.arange(nrows, dtype=np.float64),
        indexing="xy",
    )
    X = jj * float(pitch_mm)
    Y = ii * float(pitch_mm)
    XY = np.stack([X, Y], axis=-1).reshape(-1, 2).astype(np.float64)

    meta = dict(nrows=int(nrows), ncols=int(ncols), nu=int(nu), nv=int(nv))
    return XY, uv, meta


# ============================================================
# Homography estimation
# ============================================================


def estimate_homography_dlt(
    uv_img: np.ndarray,
    XY_grid: np.ndarray,
) -> np.ndarray:
    """
    Estimate planar homography H mapping grid -> image.

    Relation:
        [u, v, 1]^T ~ H [X, Y, 1]^T

    Parameters
    ----------
    uv_img : (N,2)
        Image points (u,v) in pixels.
    XY_grid : (N,2)
        Corresponding planar grid coordinates (X,Y).

    Returns
    -------
    H : (3,3) ndarray
        Homography matrix mapping grid -> image.
    """

    uv_img = np.asarray(uv_img, dtype=np.float64)
    XY_grid = np.asarray(XY_grid, dtype=np.float64)

    if uv_img.shape != XY_grid.shape or uv_img.shape[1] != 2:
        raise ValueError("Inputs must both have shape (N,2) and match in size.")
    if uv_img.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required.")

    # grid -> image (pure DLT, no RANSAC)
    H, _ = cv2.findHomography(
        XY_grid,
        uv_img,
        method=0
    )

    if H is None:
        raise RuntimeError("Homography estimation failed.")

    # normalize scale (H[2,2] = 1)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H


# ============================================================
# Plane-induced homography (X-ray -> Camera)
# ============================================================

def estimate_plane_induced_homography(
    K_c: np.ndarray,
    R_xc: np.ndarray,
    t_xc: np.ndarray,
    K_x: np.ndarray,
    d_x: float,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute plane-induced homography from X-ray image to camera image.
    
    This homography maps pixel coordinates from the X-ray image to
    pixel coordinates in the RGB camera image, assuming all points lie
    on a single plane in the X-ray frame defined by
    
        n_x^T X_x = d_x
    
    with
    
        n_x = [0, 0, 1]^T
    
    and d_x being the positive plane depth in the X-ray frame
    (e.g. the pointer tip z-coordinate expressed in the X-ray frame).
    
    Mathematical formulation
    ------------------------
        H_xc = K_c ( R_xc + (t_xc n_x^T) / d_x ) K_x^{-1}
    
    such that:
        u_c ~ H_xc * u_x
    
    Coordinate conventions
    ----------------------
    - R_xc, t_xc transform 3D points from X-ray frame to camera frame:
    
          X_c = R_xc X_x + t_xc
    
    - d_x is NOT the offset term of a plane written as n^T X + d = 0.
      Instead, it is the positive depth parameter in the plane equation
    
          n_x^T X_x = d_x

    Parameters
    ----------
    K_c : (3,3) float64
        Intrinsic matrix of the RGB camera.

    R_xc : (3,3) float64
        Rotation from X-ray frame to camera frame.

    t_xc : (3,) or (3,1) float64
        Translation from X-ray origin to camera origin (in camera frame).

    K_x : (3,3) float64
        Intrinsic matrix of the X-ray imaging system.

    d_x : float
        Plane depth in X-ray frame.

    normalize : bool, optional
        If True, normalize H such that H[2,2] = 1.

    Returns
    -------
    H_xc : (3,3) float64
        Homography mapping X-ray pixels -> camera pixels.
    """

    K_c = np.asarray(K_c, dtype=np.float64)
    R_xc = np.asarray(R_xc, dtype=np.float64)
    K_x = np.asarray(K_x, dtype=np.float64)
    t_xc = np.asarray(t_xc, dtype=np.float64).reshape(3, 1)

    if K_c.shape != (3, 3) or K_x.shape != (3, 3):
        raise ValueError("K_c and K_x must be (3,3).")
    if R_xc.shape != (3, 3):
        raise ValueError("R_xc must be (3,3).")
    if t_xc.shape != (3, 1):
        raise ValueError("t_xc must be shape (3,) or (3,1).")
    if abs(d_x) < 1e-9:
        raise ValueError("d_x must be non-zero.")

    n_x = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
    d_x = float(d_x) * 1e-3   # mm -> m

    plane_term = (t_xc @ n_x.T) / float(d_x)
    A = R_xc + plane_term

    H_xc = K_c @ A @ np.linalg.inv(K_x)

    if normalize and abs(H_xc[2, 2]) > 1e-12:
        H_xc = H_xc / H_xc[2, 2]

    return H_xc


# ============================================================
# Homography evaluation utils
# ============================================================

def project_homography(H: np.ndarray, XY: np.ndarray) -> np.ndarray:
    """
    Apply homography to planar points.

    Relation:
        [u, v, 1]^T ~ H [X, Y, 1]^T

    Parameters
    ----------
    H : (3,3)
        Homography mapping grid -> image.
    XY : (N,2)
        Planar points.

    Returns
    -------
    uv_proj : (N,2)
        Projected image points.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError("H must have shape (3,3).")

    XY = np.asarray(XY, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((XY.shape[0], 1), dtype=np.float64)
    Xh = np.hstack([XY, ones])              # (N,3)
    Uh = (H @ Xh.T).T                       # (N,3)

    w = Uh[:, 2]
    uv = np.empty((XY.shape[0], 2), dtype=np.float64)

    # avoid divide-by-zero
    good = np.isfinite(w) & (np.abs(w) > 1e-12) & np.isfinite(Uh[:, 0]) & np.isfinite(Uh[:, 1])
    uv[:] = np.nan
    uv[good, 0] = Uh[good, 0] / w[good]
    uv[good, 1] = Uh[good, 1] / w[good]

    return uv


def homography_reproj_stats(
    H: np.ndarray,
    XY: np.ndarray,
    uv: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute reprojection error stats for a homography.

    Returns
    -------
    mean_px, median_px, rmse_px : float
    """
    XY = np.asarray(XY, dtype=np.float64).reshape(-1, 2)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if XY.shape != uv.shape:
        raise ValueError("XY and uv must have the same shape (N,2).")

    uvp = project_homography(H, XY)

    finite = (
        np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) &
        np.isfinite(uvp[:, 0]) & np.isfinite(uvp[:, 1])
    )
    if not np.any(finite):
        return float("nan"), float("nan"), float("nan")

    e = np.linalg.norm(uvp[finite] - uv[finite], axis=1)
    mean_e = float(np.mean(e))
    med_e = float(np.median(e))
    rmse_e = float(np.sqrt(np.mean(e * e)))
    return mean_e, med_e, rmse_e



def decompose_homography_to_pose(
    H: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a planar homography into pose (R, t).

    Assumes that H maps planar board coordinates (X, Y, 0)
    to image coordinates:

        [u, v, 1]^T ~ H [X, Y, 1]^T

    and that the camera model is:

        H = K [r1 r2 t]

    where:
        R = [r1 r2 r3] is a rotation matrix
        t is translation

    Parameters
    ----------
    H : (3,3)
        Homography mapping board -> image.
    K : (3,3)
        Intrinsic matrix.

    Returns
    -------
    R : (3,3)
        Rotation matrix (board -> camera).
    t : (3,)
        Translation vector.
    T : (4,4)
        Homogeneous transform.
    """

    H = np.asarray(H, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    if H.shape != (3, 3):
        raise ValueError(f"H must be (3,3), got {H.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")

    # remove intrinsics
    K_inv = np.linalg.inv(K)
    B = K_inv @ H

    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]

    # scale (robust)
    scale = 2.0 / (np.linalg.norm(b1) + np.linalg.norm(b2))

    r1 = scale * b1
    r2 = scale * b2
    r3 = np.cross(r1, r2)

    t = scale * b3

    # enforce orthonormal rotation via SVD
    R_approx = np.column_stack((r1, r2, r3))
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt

    # fix improper rotation
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
        t *= -1.0

    # build transform
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return R, t, T


