# overlay/tools/homography.py

from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np
import cv2

Cell = Tuple[int, int]


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
    circles_grid: np.ndarray,
    roi_cells: Iterable[Cell],
    pitch_mm: float = 2.54,
    row_tol_px: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, list[Cell]]:
    """
    Build (XY_mm, uv_px, cells_used) in GEOMETRIC row-wise ordering in IMAGE space.

    Key idea:
    - roi_cells gives us which (i,j) to use
    - but ordering is derived from their (u,v) coordinates (image geometry), not from (i,j)
    - rows are formed by clustering v with tolerance row_tol_px
    - columns by sorting u within each row
    - XY becomes a clean grid: X = col*pitch, Y = row*pitch
    """

    if circles_grid is None or circles_grid.ndim != 3 or circles_grid.shape[2] < 2:
        raise ValueError("circles_grid must have shape (nrows, ncols, 3).")

    H, W = circles_grid.shape[:2]

    # collect finite uv for all roi cells
    pts: list[tuple[float, float, int, int]] = []  # (u, v, i, j)
    for (i, j) in set(roi_cells):
        if not (0 <= i < H and 0 <= j < W):
            continue
        u, v = circles_grid[i, j, 0], circles_grid[i, j, 1]
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        pts.append((float(u), float(v), int(i), int(j)))

    if not pts:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            [],
        )

    # sort by v then u (geometric)
    pts.sort(key=lambda t: (t[1], t[0]))

    # cluster into rows by v
    rows: list[list[tuple[float, float, int, int]]] = []
    cur: list[tuple[float, float, int, int]] = []
    v_ref: float | None = None

    for p in pts:
        u, v, i, j = p
        if v_ref is None:
            cur = [p]
            v_ref = v
            continue

        if abs(v - v_ref) <= float(row_tol_px):
            cur.append(p)
            # keep row reference stable-ish (running mean)
            v_ref = (v_ref * (len(cur) - 1) + v) / len(cur)
        else:
            rows.append(cur)
            cur = [p]
            v_ref = v

    if cur:
        rows.append(cur)

    XY_list: list[list[float]] = []
    uv_list: list[list[float]] = []
    cells_used: list[Cell] = []

    for r, row in enumerate(rows):
        # left->right
        row.sort(key=lambda t: t[0])
        Y = float(r) * float(pitch_mm)

        for c, (u, v, i, j) in enumerate(row):
            X = float(c) * float(pitch_mm)
            XY_list.append([X, Y])
            uv_list.append([u, v])
            cells_used.append((i, j))

    return (
        np.asarray(XY_list, dtype=np.float64),
        np.asarray(uv_list, dtype=np.float64),
        cells_used,
    )


"""
def build_planar_correspondences(
    circles_grid: np.ndarray,
    roi_cells: Iterable[Cell],
    pitch_mm: float = 2.54,
) -> tuple[np.ndarray, np.ndarray, list[Cell]]:
    
    #Build (XY_mm, uv_px, cells_used) in guaranteed row-major ordering.

    #- Determine dominant contiguous i-block (rows).
    #- Within that block, determine dominant contiguous j-block (cols).
    #- Build dense rectangle (i0..i1, j0..j1).
    #- Only return cells with finite uv.
    #- XY starts at (0,0) for (i0,j0).

    #Returns
    #-------
    #XY : (N,2) float64
    #uv : (N,2) float64
    #cells_used : list[(i,j)] in row-major order
    
    if circles_grid is None or circles_grid.ndim != 3 or circles_grid.shape[2] < 2:
        raise ValueError("circles_grid must have shape (nrows, ncols, 3).")

    roi = list(roi_cells)
    if not roi:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            [],
        )

    # dominant i-run
    i_vals = sorted({i for (i, _j) in roi})
    i0, i1 = _largest_contiguous_run(i_vals)

    roi_i = [(i, j) for (i, j) in roi if i0 <= i <= i1]
    if not roi_i:
        roi_i = roi
        i0 = min(i for (i, _j) in roi_i)
        i1 = max(i for (i, _j) in roi_i)

    # dominant j-run inside i-run
    j_vals = sorted({j for (_i, j) in roi_i})
    j0, j1 = _largest_contiguous_run(j_vals)

    H, W = circles_grid.shape[:2]

    XY_list: list[list[float]] = []
    uv_list: list[list[float]] = []
    cells_used: list[Cell] = []

    for i in range(i0, i1 + 1):
        Y = (i - i0) * float(pitch_mm)
        for j in range(j0, j1 + 1):
            if not (0 <= i < H and 0 <= j < W):
                continue

            x, y = circles_grid[i, j, 0], circles_grid[i, j, 1]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            X = (j - j0) * float(pitch_mm)

            XY_list.append([X, Y])
            uv_list.append([float(x), float(y)])
            cells_used.append((i, j))

    XY = np.asarray(XY_list, dtype=np.float64)
    uv = np.asarray(uv_list, dtype=np.float64)

    return XY, uv, cells_used
"""

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



