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

from overlay.tools.blob_detection import detect_blobs_hough

Cell = Tuple[int, int]


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


def _sort_circles_rowmajor(circles: np.ndarray, row_tol_px: float = 13.0) -> np.ndarray:
    """
    Sort circles into rows (by y) and within rows by x.

    Parameters
    ----------
    circles : np.ndarray
        (N,3) array [x,y,r].
    row_tol_px : float
        Tolerance in pixels for grouping points into the same row by y.

    Returns
    -------
    np.ndarray
        (N,3) array in approximate row-major order.
    """
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


def _rect_cells_from_selected(
    circles_grid: np.ndarray,
    selected_cells: Sequence[Cell],
) -> list[Cell]:
    """
    Return finite cells inside the ROI spanned by selected cells.

    Strategy
    --------
    1) FAST PATH (recommended for your setup):
       Use the index-rectangle spanned by the selected (i,j) cells.
       This is robust to projective distortion and avoids "missing last corner".

    2) FALLBACKS (kept for safety):
       - If anchors nearly collinear -> XY bbox
       - Otherwise -> rotation-robust selection in (s,t) space
    """
    if circles_grid.ndim != 3 or circles_grid.shape[2] != 3:
        raise ValueError("circles_grid must have shape (nrows, ncols, 3).")
    if len(selected_cells) < 3:
        raise ValueError("selected_cells must contain at least 3 entries.")

    def _unique_sorted(cells: list[Cell]) -> list[Cell]:
        return sorted(set(cells), key=lambda ij: (ij[0], ij[1]))

    # ------------------------------------------------------------
    # FAST PATH: index-aligned ROI in (i,j) space
    # ------------------------------------------------------------
    si = [ij[0] for ij in selected_cells]
    sj = [ij[1] for ij in selected_cells]
    i_min, i_max = min(si), max(si)
    j_min, j_max = min(sj), max(sj)

    nrows, ncols, _ = circles_grid.shape

    # Clamp bounds
    i_min = max(0, i_min)
    i_max = min(nrows - 1, i_max)
    j_min = max(0, j_min)
    j_max = min(ncols - 1, j_max)

    rect_cells: list[Cell] = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            x, y, _ = circles_grid[i, j]
            if np.isfinite(x) and np.isfinite(y):
                rect_cells.append((i, j))

    rect_cells.extend(list(selected_cells))
    rect_cells = _unique_sorted(rect_cells)

    # If we got anything at all, we’re done.
    # (If you want to be stricter: require >= 9 or require expected size.)
    if rect_cells:
        return rect_cells

    # ------------------------------------------------------------
    # FALLBACK: original geometry-based logic (kept)
    # ------------------------------------------------------------

    # --- fetch anchor points (finite) ---
    anchors = []
    ar = []
    for (i, j) in selected_cells:
        x, y, r = circles_grid[i, j]
        if np.isfinite(x) and np.isfinite(y):
            anchors.append([float(x), float(y)])
            ar.append(float(r))

    if len(anchors) < 3:
        raise ValueError("Selected cells must refer to finite grid entries.")

    P = np.asarray(anchors, dtype=np.float64)  # (3,2)

    # --- find "corner" anchor (most orthogonal) ---
    best_k = 0
    best_score = -1.0
    for k in range(3):
        idx = [0, 1, 2]
        idx.remove(k)
        u = P[idx[0]] - P[k]
        v = P[idx[1]] - P[k]
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu < 1e-6 or nv < 1e-6:
            continue
        score = abs(u[0] * v[1] - u[1] * v[0]) / (nu * nv)
        if score > best_score:
            best_score = score
            best_k = k

    idx = [0, 1, 2]
    idx.remove(best_k)
    p0 = P[best_k]
    a = P[idx[0]] - p0
    b = P[idx[1]] - p0

    A = np.column_stack([a, b])  # (2,2)

    # ------------------------------------------------------------
    # Fallback: anchors nearly collinear -> XY bbox
    # ------------------------------------------------------------
    if abs(np.linalg.det(A)) < 1e-9:
        ax = P[:, 0].tolist()
        ay = P[:, 1].tolist()
        margin = 0.75 * float(np.median(ar)) if ar else 0.0
        xmin, xmax = min(ax) - margin, max(ax) + margin
        ymin, ymax = min(ay) - margin, max(ay) + margin

        rect_cells = []
        for i in range(nrows):
            for j in range(ncols):
                x, y, _ = circles_grid[i, j]
                if not np.isfinite(x):
                    continue
                xf, yf = float(x), float(y)
                if (xmin <= xf <= xmax) and (ymin <= yf <= ymax):
                    rect_cells.append((i, j))

        rect_cells.extend(list(selected_cells))
        return _unique_sorted(rect_cells)

    # ------------------------------------------------------------
    # Rotation-robust selection in (s,t)
    # ------------------------------------------------------------
    Ainv = np.linalg.inv(A)

    ST = (Ainv @ (P - p0).T).T
    s_min, s_max = float(np.min(ST[:, 0])), float(np.max(ST[:, 0]))
    t_min, t_max = float(np.min(ST[:, 1])), float(np.max(ST[:, 1]))

    pix_margin = 0.75 * float(np.median(ar)) if ar else 0.0
    ma = pix_margin / (np.linalg.norm(a) + 1e-12)
    mb = pix_margin / (np.linalg.norm(b) + 1e-12)

    s_min -= ma
    s_max += ma
    t_min -= mb
    t_max += mb

    rect_cells = []
    for i in range(nrows):
        for j in range(ncols):
            x, y, _ = circles_grid[i, j]
            if not np.isfinite(x):
                continue
            p = np.array([float(x), float(y)], dtype=np.float64)
            st = Ainv @ (p - p0)
            s, t = float(st[0]), float(st[1])
            if (s_min <= s <= s_max) and (t_min <= t <= t_max):
                rect_cells.append((i, j))

    rect_cells.extend(list(selected_cells))
    return _unique_sorted(rect_cells)


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


"""
def _rect_cells_from_selected(
    circles_grid: np.ndarray,
    selected_cells: Sequence[Cell],
) -> list[Cell]:
    
    #Return finite cells inside the ROI spanned by selected cells (rotation-robust).

    #Notes
    #-----
    #- Returns a UNIQUE list of (i,j) indices, sorted row-major by (i,j).
    #- This function does not modify circles_grid.
    
    if circles_grid.ndim != 3 or circles_grid.shape[2] != 3:
        raise ValueError("circles_grid must have shape (nrows, ncols, 3).")
    if len(selected_cells) < 3:
        raise ValueError("selected_cells must contain at least 3 entries.")

    # --- fetch anchor points (finite) ---
    anchors = []
    ar = []
    for (i, j) in selected_cells:
        x, y, r = circles_grid[i, j]
        if np.isfinite(x) and np.isfinite(y):
            anchors.append([float(x), float(y)])
            ar.append(float(r))

    if len(anchors) < 3:
        raise ValueError("Selected cells must refer to finite grid entries.")

    P = np.asarray(anchors, dtype=np.float64)  # (3,2)

    # --- find "corner" anchor (most orthogonal) ---
    best_k = 0
    best_score = -1.0
    for k in range(3):
        idx = [0, 1, 2]
        idx.remove(k)
        u = P[idx[0]] - P[k]
        v = P[idx[1]] - P[k]
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu < 1e-6 or nv < 1e-6:
            continue
        score = abs(u[0] * v[1] - u[1] * v[0]) / (nu * nv)
        if score > best_score:
            best_score = score
            best_k = k

    idx = [0, 1, 2]
    idx.remove(best_k)
    p0 = P[best_k]
    a = P[idx[0]] - p0
    b = P[idx[1]] - p0

    A = np.column_stack([a, b])  # (2,2)

    def _unique_sorted(cells: list[Cell]) -> list[Cell]:
        uniq = set(cells)
        return sorted(uniq, key=lambda ij: (ij[0], ij[1]))

    # ------------------------------------------------------------
    # Fallback: anchors nearly collinear -> XY bbox
    # ------------------------------------------------------------
    if abs(np.linalg.det(A)) < 1e-9:
        ax = P[:, 0].tolist()
        ay = P[:, 1].tolist()
        margin = 0.75 * float(np.median(ar)) if ar else 0.0
        xmin, xmax = min(ax) - margin, max(ax) + margin
        ymin, ymax = min(ay) - margin, max(ay) + margin

        rect_cells: list[Cell] = []
        nrows, ncols, _ = circles_grid.shape
        for i in range(nrows):
            for j in range(ncols):
                x, y, _ = circles_grid[i, j]
                if not np.isfinite(x):
                    continue
                xf, yf = float(x), float(y)
                if (xmin <= xf <= xmax) and (ymin <= yf <= ymax):
                    rect_cells.append((i, j))

        rect_cells.extend(list(selected_cells))
        return _unique_sorted(rect_cells)

    # ------------------------------------------------------------
    # Rotation-robust selection in (s,t)
    # ------------------------------------------------------------
    Ainv = np.linalg.inv(A)

    ST = (Ainv @ (P - p0).T).T
    s_min, s_max = float(np.min(ST[:, 0])), float(np.max(ST[:, 0]))
    t_min, t_max = float(np.min(ST[:, 1])), float(np.max(ST[:, 1]))

    pix_margin = 0.75 * float(np.median(ar)) if ar else 0.0
    ma = pix_margin / (np.linalg.norm(a) + 1e-12)
    mb = pix_margin / (np.linalg.norm(b) + 1e-12)

    s_min -= ma
    s_max += ma
    t_min -= mb
    t_max += mb

    rect_cells: list[Cell] = []
    nrows, ncols, _ = circles_grid.shape

    for i in range(nrows):
        for j in range(ncols):
            x, y, _ = circles_grid[i, j]
            if not np.isfinite(x):
                continue
            p = np.array([float(x), float(y)], dtype=np.float64)
            st = Ainv @ (p - p0)
            s, t = float(st[0]), float(st[1])
            if (s_min <= s <= s_max) and (t_min <= t <= t_max):
                rect_cells.append((i, j))

    rect_cells.extend(list(selected_cells))
    return _unique_sorted(rect_cells)
"""

"""
def _extract_xy_from_cells(circles_grid: np.ndarray, cells: Iterable[Cell]) -> np.ndarray:
    
    #Extract (u,v) points for provided cell indices.

    #IMPORTANT
    #---------
    #- Does NOT change any grid indices.
    #- Orders points geometrically: rows by y, within row by x.
      This matches visual grid order even if circles_grid has padding artifacts.

    #Returns
    #-------
    #np.ndarray
    #    (N,2) float32 array [u,v].
    
    cells_list = list(cells)
    if not cells_list:
        return np.empty((0, 2), dtype=np.float32)

    circles = []
    for (i, j) in cells_list:
        x, y, r = circles_grid[i, j]
        if np.isfinite(x) and np.isfinite(y):
            circles.append([float(x), float(y), float(r)])

    if not circles:
        return np.empty((0, 2), dtype=np.float32)

    circles = np.asarray(circles, dtype=np.float32).reshape(-1, 3)

    r_med = float(np.median(circles[:, 2]))
    row_tol_px = max(13.0, 2.5 * r_med)

    circles_sorted = _sort_circles_rowmajor(circles, row_tol_px=float(row_tol_px))
    return circles_sorted[:, :2].astype(np.float32)
"""

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


def _build_circles_grid(circles_sorted: np.ndarray) -> np.ndarray | None:
    """
    Build circles_grid from sorted detections (row-first strategy).

    Output grid is padded to a common column count and may contain NaNs.
    """
    if circles_sorted is None or len(circles_sorted) == 0:
        return None

    c = np.asarray(circles_sorted, dtype=np.float32).reshape(-1, 3)

    r_med = float(np.median(c[:, 2]))
    y_thresh_px = 2.5 * r_med

    current = c[:1].copy()
    y_ref = float(c[0, 1])
    rows = []

    for k in range(1, len(c)):
        pt = c[k]
        if abs(float(pt[1]) - y_ref) > y_thresh_px:
            sort_idx = np.argsort(current[:, 0])
            rows.append(current[sort_idx])
            current = pt.reshape(1, -1)
            y_ref = float(pt[1])
        else:
            current = np.vstack([current, pt])
            y_ref = float(np.mean(current[:, 1]))

    sort_idx = np.argsort(current[:, 0])
    rows.append(current[sort_idx])

    nrows = len(rows)
    center_row_idx = int(np.argmax([len(row) for row in rows]))
    ncols = len(rows[center_row_idx])

    circles_grid = np.full((nrows, ncols, 3), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        n_points = len(row)
        left_pad = (ncols - n_points) // 2
        
        # debug
        print(f"[grid] row {i:03d}: n_points={n_points:3d}  ncols={ncols:3d}  left_pad={left_pad:3d}")
        
        padded_row = np.full((ncols, 3), np.nan, dtype=np.float32)
        padded_row[left_pad:left_pad + n_points] = row
        circles_grid[i] = padded_row

    return circles_grid


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

    circles = detect_blobs_hough(img_masked, hough_params)
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
    circles_sorted = _sort_circles_rowmajor(circles, row_tol_px=float(row_tol_px))
    circles_grid = _build_circles_grid(circles_sorted)

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