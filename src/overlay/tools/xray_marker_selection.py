# overlay/tools/xray_marker_selection.py
# -*- coding: utf-8 -*-
"""
xray_marker_selection.py

Marker selection helpers without UI dependencies.

Public API
----------
- run_xray_marker_detection(...)
- compute_roi_from_grid(...)

Design intent (per your project)
--------------------------------
- Detection returns a ROW-WISE sorted list of circles (no fixed grid_shape here),
  because the full PCB layout / detected rows/cols may vary per image.
- The GUI (or any caller) may build its own circles_grid (with padding etc.).
- ROI is computed purely from a provided circles_grid and selected anchor cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, List

import cv2
import numpy as np

from overlay.tools.blob_detection import detect_blobs_hough, estimate_pitch_nn

Cell = Tuple[int, int]


## debug
from pathlib import Path

from pathlib import Path

def _debug_dump_detection(
    *,
    circles: np.ndarray,
    circles_sorted: np.ndarray,
    circles_grid: np.ndarray | None,
    row_tol_px: float,
) -> None:

    base = Path(r"C:\Users\domin\Documents\Studium\Master\Masterarbeit\Projekt\Data")

    # Raw + sorted circles
    np.savetxt(base / "Xray_Calib_View4__circles_raw.txt",
               circles, fmt="%.3f", header="x y r")

    np.savetxt(base / "Xray_Calib_View4__circles_sorted.txt",
               circles_sorted, fmt="%.3f", header="x y r")

    if circles_grid is None:
        (base / "Xray_Calib_View4__grid_stats.txt").write_text("circles_grid = None\n")
        return

    g = circles_grid
    finite = np.isfinite(g[..., 0]) & np.isfinite(g[..., 1])
    per_row = finite.sum(axis=1)

    first_j = np.full((g.shape[0],), -1, dtype=int)
    last_j  = np.full((g.shape[0],), -1, dtype=int)

    for i in range(g.shape[0]):
        js = np.where(finite[i])[0]
        if js.size > 0:
            first_j[i] = int(js.min())
            last_j[i]  = int(js.max())

    lines = []
    lines.append(f"row_tol_px: {float(row_tol_px):.3f}\n")
    lines.append(f"grid_shape: {g.shape}\n")
    lines.append(f"finite_total: {int(finite.sum())}\n")
    lines.append(
        "finite_per_row_min/med/max: "
        f"{int(per_row.min())} / {float(np.median(per_row)):.1f} / {int(per_row.max())}\n"
    )
    lines.append("row_first_last_j (i: first..last, count):\n")

    for i in range(g.shape[0]):
        lines.append(
            f"{i:03d}: {first_j[i]:4d} .. {last_j[i]:4d}   (n={int(per_row[i])})\n"
        )

    (base / "Xray_Calib_View4__grid_stats.txt").write_text("".join(lines))

    np.savez_compressed(
        base / "Xray_Calib_View4__grid_dump.npz",
        circles=circles,
        circles_sorted=circles_sorted,
        circles_grid=circles_grid,
        row_tol_px=float(row_tol_px),
    )
##

# ============================================================
# Result container (public)
# ============================================================

@dataclass(frozen=True)
class XrayMarkerDetectionResult:
    """
    Result bundle for the marker DETECTION pipeline (no ROI computation).

    Attributes
    ----------
    img_proc : np.ndarray
        Preprocessed image (e.g., CLAHE applied), uint8 (H,W).
    mask : np.ndarray
        Detector mask, uint8 (H,W), values in {0,255}.
    img_masked : np.ndarray
        Masked image fed into blob detection, uint8 (H,W).
    circles : Optional[np.ndarray]
        Raw/deduped circle detections, (N,3) [x,y,r] or None.
    circles_sorted : Optional[np.ndarray]
        Circles sorted approximately row-major (rows by y, within row by x),
        (N,3) [x,y,r] or None.
    circles_grid : Optional[np.ndarray]
        Padded circles grid built from circles_sorted, shape (nrows,ncols,3) [x,y,r],
        with NaNs for missing entries. None if detection failed.
    """
    img_proc: np.ndarray
    mask: np.ndarray
    img_masked: np.ndarray
    circles: Optional[np.ndarray]
    circles_sorted: Optional[np.ndarray]
    circles_grid: Optional[np.ndarray]


# ============================================================
# Private helpers (module-internal only)
# ============================================================

def _fit_circle_kasa(points_xy: np.ndarray) -> Tuple[float, float, float] | None:
    """
    Fit a circle to 2D points using Kasa's least-squares method.

    Parameters
    ----------
    points_xy : np.ndarray
        (N,2) array of 2D points.

    Returns
    -------
    (xc, yc, r) : tuple[float,float,float] | None
        Circle center and radius, or None if insufficient points / degenerate fit.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] < 30:
        return None

    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2

    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    xc, yc, c = sol
    r2 = c + xc**2 + yc**2
    if r2 <= 0:
        return None

    r = np.sqrt(r2)
    return float(xc), float(yc), float(r)


def _detector_mask(
    img_gray: np.ndarray,
    blur_ks: int = 11,
    thr_mode: str = "otsu",
    adaptive_block: int = 51,
    adaptive_C: int = -5,
    close_ks: int = 41,
    close_iter: int = 1,
) -> np.ndarray:
    """
    Create a broad support mask for the detector region.

    Returns
    -------
    mask : np.ndarray
        uint8 (H,W), 255 inside ROI, 0 outside.
    """
    if img_gray.ndim != 2:
        raise ValueError("img_gray must be grayscale (H,W).")
    h, w = img_gray.shape

    # 1) Smooth
    k = blur_ks if blur_ks % 2 == 1 else blur_ks + 1
    blur = cv2.GaussianBlur(img_gray, (k, k), 0)

    # 2) Binarize
    if thr_mode == "adaptive":
        blk = adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1
        bw = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blk, adaptive_C
        )
    else:
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if background went white
    if float(np.mean(bw)) > 220.0:
        bw = 255 - bw

    # 3) Contours
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.full((h, w), 255, dtype=np.uint8)

    def touches_border(c: np.ndarray) -> bool:
        xs = c[:, 0, 0]
        ys = c[:, 0, 1]
        return (xs.min() <= 1) or (ys.min() <= 1) or (xs.max() >= w - 2) or (ys.max() >= h - 2)

    # 4) Candidates
    nb = [c for c in cnts if not touches_border(c)]
    candidates = nb if len(nb) > 0 else cnts

    largest = max(candidates, key=cv2.contourArea)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

    # 5) Closing (fill small holes)
    if close_ks > 1:
        kc = close_ks if close_ks % 2 == 1 else close_ks + 1
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((kc, kc), np.uint8),
            iterations=int(close_iter),
        )

    return mask


def _estimate_grid_axes_from_knn(xy: np.ndarray, k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate two (approximately) orthonormal grid axes (a,b) from local neighbor
    displacement vectors.

    Output:
        a, b : unit vectors in image XY (float64), roughly along grid directions.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("xy must have shape (N,2) (or at least 2 columns).")
    xy = xy[:, :2]

    N = xy.shape[0]
    if N < 10:
        return np.array([1.0, 0.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)

    kk = int(max(2, min(k, N - 1)))

    # brute-force kNN (OK for N~1500)
    d2 = (xy[:, None, 0] - xy[None, :, 0]) ** 2 + (xy[:, None, 1] - xy[None, :, 1]) ** 2
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argpartition(d2, kth=kk, axis=1)[:, :kk]

    vecs = xy[nn_idx] - xy[:, None, :]     # (N,k,2)
    lens = np.linalg.norm(vecs, axis=2)    # (N,k)

    l_flat = lens.ravel()
    l_flat = l_flat[np.isfinite(l_flat)]
    if l_flat.size < 50:
        return np.array([1.0, 0.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)

    # keep "short" neighbor vectors around pitch-ish (robust quantiles)
    l0 = np.quantile(l_flat, 0.20)
    l1 = np.quantile(l_flat, 0.60)
    mask = (lens >= max(1e-6, 0.5 * l0)) & (lens <= 1.8 * l1)

    v = vecs[mask]  # (M,2)
    if v.shape[0] < 200:
        return np.array([1.0, 0.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)

    # normalize for angle histogram
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    ang = np.arctan2(v[:, 1], v[:, 0])
    ang = np.mod(ang, np.pi)  # [0, pi)

    nbins = 180
    hist, edges = np.histogram(ang, bins=nbins, range=(0.0, np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])

    i1 = int(np.argmax(hist))
    theta1 = float(centers[i1])

    def circ_dist(a_: np.ndarray, b_: float) -> np.ndarray:
        d = np.abs(a_ - b_)
        return np.minimum(d, np.pi - d)

    d_to_perp = circ_dist(centers, (theta1 + 0.5 * np.pi) % np.pi)
    score = hist * np.exp(-(d_to_perp ** 2) / (2 * (0.20 ** 2)))  # ~11°
    i2 = int(np.argmax(score))
    theta2 = float(centers[i2])

    a = np.array([np.cos(theta1), np.sin(theta1)], dtype=np.float64)
    b = np.array([np.cos(theta2), np.sin(theta2)], dtype=np.float64)

    # orthonormalize (deterministic)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b - a * float(np.dot(a, b))
    b = b / (np.linalg.norm(b) + 1e-12)

    # deterministic signs
    if a[0] < 0:
        a = -a
    if b[1] < 0:
        b = -b

    return a, b


def _sort_circles_rowmajor(
    circles: np.ndarray,
    row_tol_px: float = 13.0,   # kept for API compatibility; no longer used
) -> np.ndarray:
    """
    Robust row-major sorting for tilted grid.

    - Estimate grid axes (a,b) from kNN vectors.
    - Choose a as u-axis (more x-like), b as v-axis (more y-like).
    - Use directional pitch along v-axis to quantize rows:
          i = round((v - v0) / pitch_v)
    - Within each row, sort by u.

    Returns
    -------
    (N,3) float32 circles in approximate row-major order.
    """
    if circles is None or len(circles) == 0:
        return circles

    c = np.asarray(circles, dtype=np.float32).reshape(-1, 3)
    xy = c[:, :2].astype(np.float64)
    N = xy.shape[0]
    if N < 5:
        return c[np.lexsort((c[:, 0], c[:, 1]))].astype(np.float32)

    a, b = _estimate_grid_axes_from_knn(xy, k=6)

    # choose u as more "x-like"
    xhat = np.array([1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(a, xhat))) < abs(float(np.dot(b, xhat))):
        a, b = b, a

    # deterministic signs: u -> +x, v -> +y (down)
    if float(np.dot(a, xhat)) < 0:
        a = -a
    yhat = np.array([0.0, 1.0], dtype=np.float64)
    if float(np.dot(b, yhat)) < 0:
        b = -b

    mu = xy.mean(axis=0, keepdims=True)
    X = xy - mu
    u = (X @ a.reshape(2, 1)).ravel()
    v = (X @ b.reshape(2, 1)).ravel()

    # directional pitch along v-axis (fallback to isotropic)
    pitch_v = float(estimate_pitch_nn(xy, axis=b))
    if (not np.isfinite(pitch_v)) or pitch_v <= 1e-6:
        pitch_v = float(estimate_pitch_nn(xy))
    if (not np.isfinite(pitch_v)) or pitch_v <= 1e-6:
        return c[np.lexsort((c[:, 0], c[:, 1]))].astype(np.float32)

    v0 = float(np.median(v))
    i_idx = np.rint((v - v0) / pitch_v).astype(np.int32)

    order_i = np.argsort(i_idx, kind="mergesort")

    out_segments = []
    start = 0
    while start < N:
        ii = i_idx[order_i[start]]
        end = start + 1
        while end < N and i_idx[order_i[end]] == ii:
            end += 1

        seg = order_i[start:end]
        seg = seg[np.argsort(u[seg], kind="mergesort")]
        out_segments.append(seg)
        start = end

    out_idx = np.concatenate(out_segments) if out_segments else order_i
    return c[out_idx].astype(np.float32)


"""
def _sort_circles_rowmajor(circles: np.ndarray, row_tol_px: float = 13.0) -> np.ndarray:
    
    #Sort circles into rows (by y) and within rows by x.

    #Parameters
    #----------
    #circles : np.ndarray
    #    (N,3) array [x,y,r].
    #row_tol_px : float
    #    Tolerance in pixels for grouping points into the same row by y.

    #Returns
    #-------
    #np.ndarray
    #    (N,3) array in approximate row-major order.
    
    if circles is None or len(circles) == 0:
        return circles

    c = np.asarray(circles, dtype=np.float32).reshape(-1, 3)
    c = c[np.argsort(c[:, 1])]

    rows: List[np.ndarray] = []
    current = [c[0]]
    current_y = float(c[0, 1])

    for i in range(1, len(c)):
        y = float(c[i, 1])
        if abs(y - current_y) <= row_tol_px:
            current.append(c[i])
            current_y = float(np.mean([p[1] for p in current]))
        else:
            rows.append(np.array(current, dtype=np.float32))
            current = [c[i]]
            current_y = y

    rows.append(np.array(current, dtype=np.float32))

    rows = [row[np.argsort(row[:, 0])] for row in rows]
    rows = sorted(rows, key=lambda row: float(np.mean(row[:, 1])))

    return np.vstack(rows).astype(np.float32)
"""

def _rect_cells_from_selected(
    circles_grid: np.ndarray,
    selected_cells: Sequence[Cell],
) -> list[Cell]:
    """
    Return finite cells inside the ROI spanned by selected cells, using ONLY
    the index-rectangle in (i,j) space.

    Patch (minimal + robust):
    - Still uses index-rectangle in (i,j).
    - Additionally applies a GEOMETRIC safety filter in (u,v) space:
        * compute UV bbox from the 3 anchors
        * expand by ~0.60 * pitch_px (estimated from candidate points)
        * keep only cells whose (u,v) lies inside this expanded bbox
      This prevents "one stray point" far outside the ROI from being included.

    - Output is UNIQUE and sorted row-major by (i,j).
    """
    if circles_grid is None:
        raise ValueError("circles_grid must not be None.")
    if circles_grid.ndim != 3 or circles_grid.shape[2] != 3:
        raise ValueError("circles_grid must have shape (nrows, ncols, 3).")
    if selected_cells is None or len(selected_cells) != 3:
        raise ValueError("selected_cells must contain exactly 3 entries.")

    # bbox in index space
    si = [int(ij[0]) for ij in selected_cells]
    sj = [int(ij[1]) for ij in selected_cells]
    i_min, i_max = min(si), max(si)
    j_min, j_max = min(sj), max(sj)

    nrows, ncols, _ = circles_grid.shape

    # clamp to bounds
    i_min = max(0, i_min)
    i_max = min(nrows - 1, i_max)
    j_min = max(0, j_min)
    j_max = min(ncols - 1, j_max)

    # ------------------------------------------------------------
    # 1) Anchor UV bbox (geometric safety gate)
    # ------------------------------------------------------------
    anchors_uv: list[tuple[float, float]] = []
    for (i, j) in selected_cells:
        x, y, _ = circles_grid[int(i), int(j)]
        if np.isfinite(x) and np.isfinite(y):
            anchors_uv.append((float(x), float(y)))

    use_uv_gate = (len(anchors_uv) >= 2)
    if use_uv_gate:
        us = [u for (u, v) in anchors_uv]
        vs = [v for (u, v) in anchors_uv]
        umin_a, umax_a = min(us), max(us)
        vmin_a, vmax_a = min(vs), max(vs)
    else:
        umin_a = umax_a = vmin_a = vmax_a = 0.0  # unused

    # ------------------------------------------------------------
    # 2) Gather finite candidates in index-rectangle
    # ------------------------------------------------------------
    cand_cells: list[Cell] = []
    cand_xy: list[list[float]] = []

    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            x, y, _ = circles_grid[i, j]
            if np.isfinite(x) and np.isfinite(y):
                cand_cells.append((i, j))
                cand_xy.append([float(x), float(y)])

    # ------------------------------------------------------------
    # 3) Estimate pitch to set a reasonable UV margin
    # ------------------------------------------------------------
    margin = 0.0
    if use_uv_gate and len(cand_xy) >= 12:
        pitch_px = float(estimate_pitch_nn(np.asarray(cand_xy, dtype=np.float64)))
        if np.isfinite(pitch_px) and pitch_px > 1e-6:
            margin = 0.60 * pitch_px  # tweak 0.4..0.8 if needed

    # ------------------------------------------------------------
    # 4) Apply UV gate (if possible), else accept all candidates
    # ------------------------------------------------------------
    if not use_uv_gate:
        rect_cells = cand_cells
    else:
        umin = umin_a - margin
        umax = umax_a + margin
        vmin = vmin_a - margin
        vmax = vmax_a + margin

        rect_cells: list[Cell] = []
        for (i, j) in cand_cells:
            x, y, _ = circles_grid[i, j]
            if (umin <= float(x) <= umax) and (vmin <= float(y) <= vmax):
                rect_cells.append((i, j))

    # always include anchors (even if something weird happens / gate too strict)
    rect_cells.extend((int(i), int(j)) for (i, j) in selected_cells)

    # unique + sorted row-major
    rect_cells = sorted(set(rect_cells), key=lambda ij: (ij[0], ij[1]))

    if not rect_cells:
        raise RuntimeError("ROI selection produced no finite cells. Check circles_grid / anchors.")

    return rect_cells


def _extract_xy_from_cells(
    circles_grid: np.ndarray,
    cells: Iterable[Cell],
) -> np.ndarray:
    """
    Extract (u,v) points in PURE INDEX ORDER (row-major by (i,j)).

    No geometric re-sorting!
    """

    cells_sorted = sorted(set(cells), key=lambda ij: (ij[0], ij[1]))

    xy = []
    for (i, j) in cells_sorted:
        x, y, _ = circles_grid[i, j]
        if np.isfinite(x) and np.isfinite(y):
            xy.append([float(x), float(y)])

    if not xy:
        return np.empty((0, 2), dtype=np.float32)

    return np.asarray(xy, dtype=np.float32)


def _select_marker_roi_from_grid(
    circles_grid: np.ndarray,
    selected_cells: Sequence[Cell],
) -> Tuple[np.ndarray, list[Cell]]:
    """
    Compute ROI from circles_grid + 3 selected anchors.

    Returns
    -------
    xy_uv : np.ndarray
        (N,2) float32 array of ROI points, ordered geometrically row-wise.
    roi_cells : list[Cell]
        Unique list of (i,j) indices included in the ROI (sorted by (i,j)).
    """
    rect_cells = _rect_cells_from_selected(circles_grid, selected_cells)
    xy = _extract_xy_from_cells(circles_grid, rect_cells)
    return xy, rect_cells


def _apply_clahe(
    img_gray_u8: np.ndarray,
    *,
    use_clahe: bool,
    clip_limit: float,
    tile_grid_size: Tuple[int, int],
) -> np.ndarray:
    """Optionally apply CLAHE to a uint8 grayscale image."""
    if not use_clahe:
        return img_gray_u8
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    return clahe.apply(img_gray_u8)


def _expand_row_with_nan_gaps_u(
    row_sorted: np.ndarray,
    u_row_sorted: np.ndarray,
    *,
    pitch_px: float,
    gap_factor: float = 1.6,
    max_insert: int = 6,
    max_step_factor: float = 3.5,
) -> np.ndarray:
    """
    Insert NaN slots into a row where gaps in projected-u indicate missing detections.
    row_sorted is (M,3) [x,y,r] already sorted by u.
    u_row_sorted is (M,) projected u values in the SAME order.
    """
    row = np.asarray(row_sorted, dtype=np.float32).reshape(-1, 3)
    u = np.asarray(u_row_sorted, dtype=np.float64).ravel()
    if row.shape[0] <= 1:
        return row
    if u.shape[0] != row.shape[0]:
        raise ValueError("u_row_sorted must match row length.")

    pitch_px = float(pitch_px)
    if not np.isfinite(pitch_px) or pitch_px <= 1e-6:
        return row

    out = [row[0]]
    for k in range(1, row.shape[0]):
        d = float(u[k] - u[k - 1])

        if not np.isfinite(d) or d <= 0:
            out.append(row[k])
            continue

        if d > max_step_factor * pitch_px:
            out.append(row[k])
            continue

        n_steps = int(np.rint(d / pitch_px))
        n_steps = max(1, min(n_steps, max_insert + 1))

        if d > gap_factor * pitch_px and n_steps >= 2:
            n_missing = min(n_steps - 1, int(max_insert))
            for _ in range(n_missing):
                out.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))

        out.append(row[k])

    return np.asarray(out, dtype=np.float32)


def _refine_lattice_indices_by_centers(
    u_all: np.ndarray,
    v_all: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    *,
    pitch_px: float,
    v0: float,
    u0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine quantized lattice indices by snapping each point to the nearest
    row/col "center" (median u/v per bin), considering only neighbors {bin-1, bin, bin+1}.

    This fixes occasional ±1 bin mistakes caused by slight axis skew (v drifting with u).
    """
    u_all = np.asarray(u_all, dtype=np.float64).ravel()
    v_all = np.asarray(v_all, dtype=np.float64).ravel()
    i_idx = np.asarray(i_idx, dtype=np.int32).ravel()
    j_idx = np.asarray(j_idx, dtype=np.int32).ravel()

    if u_all.size != v_all.size or u_all.size != i_idx.size or u_all.size != j_idx.size:
        raise ValueError("u_all, v_all, i_idx, j_idx must have the same length.")

    pitch_px = float(pitch_px)
    if (not np.isfinite(pitch_px)) or pitch_px <= 1e-9:
        return i_idx, j_idx

    # --- build row centers in v ---
    v_center = {}
    for ii in np.unique(i_idx):
        vv = v_all[i_idx == ii]
        if vv.size > 0:
            v_center[int(ii)] = float(np.median(vv))

    i_ref = i_idx.copy()
    for k in range(i_ref.size):
        ii = int(i_ref[k])
        v = float(v_all[k])

        c0 = v_center.get(ii, float(v0 + ii * pitch_px))
        cm = v_center.get(ii - 1, c0 - pitch_px)
        cp = v_center.get(ii + 1, c0 + pitch_px)

        dm = abs(v - cm)
        d0 = abs(v - c0)
        dp = abs(v - cp)

        if dm < d0 and dm < dp:
            i_ref[k] = ii - 1
        elif dp < d0 and dp < dm:
            i_ref[k] = ii + 1
        else:
            i_ref[k] = ii

    # --- build col centers in u ---
    u_center = {}
    for jj in np.unique(j_idx):
        uu = u_all[j_idx == jj]
        if uu.size > 0:
            u_center[int(jj)] = float(np.median(uu))

    j_ref = j_idx.copy()
    for k in range(j_ref.size):
        jj = int(j_ref[k])
        u = float(u_all[k])

        c0 = u_center.get(jj, float(u0 + jj * pitch_px))
        cm = u_center.get(jj - 1, c0 - pitch_px)
        cp = u_center.get(jj + 1, c0 + pitch_px)

        dm = abs(u - cm)
        d0 = abs(u - c0)
        dp = abs(u - cp)

        if dm < d0 and dm < dp:
            j_ref[k] = jj - 1
        elif dp < d0 and dp < dm:
            j_ref[k] = jj + 1
        else:
            j_ref[k] = jj

    return i_ref.astype(np.int32), j_ref.astype(np.int32)


def _phase_mode_offset(vals: np.ndarray, pitch: float, nbins: int = 80) -> float:
    """
    Robust offset in [0,pitch) via histogram-mode of (vals mod pitch).
    """
    phase = np.mod(vals, pitch)
    hist, edges = np.histogram(phase, bins=nbins, range=(0.0, pitch))
    k = int(np.argmax(hist))
    return 0.5 * (edges[k] + edges[k + 1])


def _build_circles_grid(
    circles_sorted: np.ndarray,
) -> np.ndarray | None:
    """
    Build circles_grid by snapping detections to a 2D lattice in (u,v) space.

    - Robust to tilt / shear (blobs on slanted lines)
    - Ignores the ordering of circles_sorted (uses it only as a list of points)
    - No row splitting, no gap insertion, no "shift-right until free"

    Output: (nrows, ncols, 3) float32 with NaNs for missing.
    """
    if circles_sorted is None or len(circles_sorted) == 0:
        return None

    c = np.asarray(circles_sorted, dtype=np.float64).reshape(-1, 3)
    xy = c[:, :2]
    N = xy.shape[0]
    if N < 5:
        return None

    # 1) grid axes (already in your file)
    a, b = _estimate_grid_axes_from_knn(xy, k=6)

    # choose u-axis more x-like
    xhat = np.array([1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(a, xhat))) < abs(float(np.dot(b, xhat))):
        a, b = b, a

    # deterministic signs: u->+x, v->+y (image down)
    if float(np.dot(a, xhat)) < 0:
        a = -a
    yhat = np.array([0.0, 1.0], dtype=np.float64)
    if float(np.dot(b, yhat)) < 0:
        b = -b

    # 2) pitch
    pitch = float(estimate_pitch_nn(xy))
    if (not np.isfinite(pitch)) or pitch <= 1e-6:
        return None

    # 3) project to (u,v)
    mu = xy.mean(axis=0)
    X = xy - mu[None, :]
    u = (X @ a.reshape(2, 1)).ravel()
    v = (X @ b.reshape(2, 1)).ravel()

    # 4) robust offsets in the lattice
    u0 = _phase_mode_offset(u, pitch, nbins=80)
    v0 = _phase_mode_offset(v, pitch, nbins=80)

    # 5) quantize to integer indices
    j = np.rint((u - u0) / pitch).astype(int)
    i = np.rint((v - v0) / pitch).astype(int)

    # shift to non-negative grid coords
    i -= int(i.min())
    j -= int(j.min())

    nrows = int(i.max()) + 1
    ncols = int(j.max()) + 1
    grid = np.full((nrows, ncols, 3), np.nan, dtype=np.float32)

    # 6) collision resolution by residual to cell center
    # cell center in image coords: mu + (u0 + j*p)*a + (v0 + i*p)*b
    resid = np.full((nrows, ncols), np.inf, dtype=np.float64)

    for k in range(N):
        ii = int(i[k]); jj = int(j[k])

        cen = (
            mu
            + (u0 + jj * pitch) * a
            + (v0 + ii * pitch) * b
        )
        r = float(np.hypot(xy[k, 0] - cen[0], xy[k, 1] - cen[1]))

        if not np.isfinite(grid[ii, jj, 0]):
            grid[ii, jj, :] = c[k].astype(np.float32)
            resid[ii, jj] = r
        else:
            # keep the one that fits the cell center better
            if r < resid[ii, jj]:
                grid[ii, jj, :] = c[k].astype(np.float32)
                resid[ii, jj] = r
            # else: drop the worse duplicate deterministically

    return grid




# ============================================================
# Public API
# ============================================================

def run_xray_marker_detection(
    img_gray: np.ndarray,
    *,
    # preprocessing (match your page/test defaults)
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_tiles: Tuple[int, int] = (12, 12),
    use_mask: bool = True,
    # mask params (match detector_mask defaults)
    blur_ks: int = 11,
    thr_mode: str = "otsu",
    adaptive_block: int = 51,
    adaptive_C: int = -5,
    close_ks: int = 41,
    close_iter: int = 1,
    # blob detection
    hough_params=None,
    # sorting
    row_tol_px: float = 13.0,
) -> XrayMarkerDetectionResult:
    """
    Detection-only pipeline:
      CLAHE -> detector_mask -> mask image -> detect_blobs_hough -> sort (row-major) -> build grid
    """
    if img_gray.ndim != 2:
        raise ValueError("img_gray must be grayscale (H,W).")

    img_u8 = img_gray if img_gray.dtype == np.uint8 else np.clip(img_gray, 0, 255).astype(np.uint8)
    img_proc = _apply_clahe(img_u8, use_clahe=use_clahe, clip_limit=clahe_clip, tile_grid_size=clahe_tiles)

    if use_mask:
        mask = _detector_mask(
            img_proc,
            blur_ks=blur_ks,
            thr_mode=thr_mode,
            adaptive_block=adaptive_block,
            adaptive_C=adaptive_C,
            close_ks=close_ks,
            close_iter=close_iter,
        )
        img_masked = img_proc.copy()
        img_masked[mask == 0] = 0
    else:
        mask = np.full(img_proc.shape, 255, dtype=np.uint8)
        img_masked = img_proc

    circles = detect_blobs_hough(img_masked, params=hough_params)
    if circles is None or len(circles) == 0:
        return XrayMarkerDetectionResult(
            img_proc=img_proc,
            mask=mask,
            img_masked=img_masked,
            circles=None,
            circles_sorted=None,
            circles_grid=None,
        )

    circles = np.asarray(circles, dtype=np.float32).reshape(-1, 3)
    
    # debug
    base = Path(r"C:\Users\domin\Documents\Studium\Master\Masterarbeit\Projekt\Data")
    np.savez_compressed(
        base / "Circles_View4__grid.npz",
        circles=circles,
    )

    # --- derive pitch ---
    pitch_px = float(estimate_pitch_nn(circles[:, :2]))

    if not np.isfinite(pitch_px) or pitch_px <= 1e-6:
        row_tol_px_eff = float(row_tol_px)     # fallback
    else:
        row_tol_px_eff = 0.50 * pitch_px       # robust default
    
    circles_sorted = _sort_circles_rowmajor(circles, row_tol_px=row_tol_px_eff)
    
    circles_grid = _build_circles_grid(circles_sorted)
    
    # debug
    
    _debug_dump_detection(
        circles=circles,
        circles_sorted=circles_sorted,
        circles_grid=circles_grid,
        row_tol_px=row_tol_px_eff,      # <- das ist der tatsächlich benutzte Wert
    )
    

    return XrayMarkerDetectionResult(
        img_proc=img_proc,
        mask=mask,
        img_masked=img_masked,
        circles=circles,
        circles_sorted=circles_sorted,
        circles_grid=circles_grid,
    )


def compute_roi_from_grid(
    circles_grid: np.ndarray,
    selected_cells: Sequence[Cell],
) -> Tuple[np.ndarray, list[Cell]]:
    """
    Compute ROI points/cells from an already built circles_grid.

    Parameters
    ----------
    circles_grid : np.ndarray
        (nrows,ncols,3) array [x,y,r] with NaNs for missing.
    selected_cells : Sequence[Cell]
        Exactly three anchor cells (i,j) selected by the user.

    Returns
    -------
    xy_uv : np.ndarray
        (N,2) ROI points [u,v], ordered geometrically row-wise.
    roi_cells : list[Cell]
        Unique list of (i,j) indices included in ROI (sorted by (i,j)).
    """
    if circles_grid is None:
        raise ValueError("circles_grid must not be None.")
    return _select_marker_roi_from_grid(circles_grid, selected_cells)