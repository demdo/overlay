# marker_selection.py
# -*- coding: utf-8 -*-

"""
xray_marker_selection.py

Helpers for marker-grid selection without UI dependencies.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


Cell = Tuple[int, int]
Point = Tuple[float, float]


def fit_circle_kasa(points_xy: np.ndarray) -> Tuple[float, float, float] | None:
    """Fit circle to 2D points using Kasa's least-squares method."""
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


def detector_mask(
    img_gray: np.ndarray,
    blur_ks: int = 11,
    thr_mode: str = "otsu",
    adaptive_block: int = 51,
    adaptive_C: int = -5,
    close_ks: int = 41,
    close_iter: int = 1,
) -> np.ndarray:
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

    if np.mean(bw) > 220:
        bw = 255 - bw

    # 3) Contours
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.full((h, w), 255, dtype=np.uint8)

    def touches_border(c: np.ndarray) -> bool:
        xs = c[:, 0, 0]
        ys = c[:, 0, 1]
        return (xs.min() <= 1) or (ys.min() <= 1) or (xs.max() >= w-2) or (ys.max() >= h-2)

    # 4) Candidates
    nb = [c for c in cnts if not touches_border(c)]
    candidates = nb if len(nb) > 0 else cnts

    largest = max(candidates, key=cv2.contourArea)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

    # 6) Closing
    if close_ks > 1:
        kc = close_ks if close_ks % 2 == 1 else close_ks + 1
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            np.ones((kc, kc), np.uint8),
            iterations=close_iter
        )

    return mask


def sort_circles_grid(circles: np.ndarray, row_tol_px: float = 13.0) -> np.ndarray:
    """Sort circle detections into rows then by x within each row."""
    if circles is None or len(circles) == 0:
        return circles

    c = np.asarray(circles, dtype=np.float32)
    c = c[np.argsort(c[:, 1])]

    rows = []
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


def prepare_nearest_cell_data(circles_grid: np.ndarray):
    """Precompute finite grid coordinates for fast nearest-cell lookup."""
    coords = circles_grid[..., :2]
    finite_mask = np.isfinite(coords[..., 0]) & np.isfinite(coords[..., 1])
    if not np.any(finite_mask):
        return None
    
    rows, cols = np.nonzero(finite_mask)
    xs = coords[rows, cols, 0].astype(np.float32, copy=False)
    ys = coords[rows, cols, 1].astype(np.float32, copy=False)
    return rows, cols, xs, ys
    
      
def nearest_cell(prepared, x_click: int, y_click: int):
    """Return (i,j) of nearest finite cell and its pixel distance."""
    if prepared is None:
        return None, np.inf
    rows, cols, xs, ys = prepared
    dx = xs - float(x_click)
    dy = ys - float(y_click)
    d2 = dx * dx + dy * dy
    idx = int(np.argmin(d2))
    return (int(rows[idx]), int(cols[idx])), float(np.sqrt(d2[idx]))


def rect_cells_from_selected(circles_grid: np.ndarray, selected_cells: Sequence[Cell]):
    """Return finite cells inside the XY bounding box spanned by selected cells."""
    if circles_grid.ndim != 3 or circles_grid.shape[2] != 3:
        raise ValueError("circles_grid must have shape (nrows, ncols, 3).")
    if len(selected_cells) < 3:
        raise ValueError("selected_cells must contain at least 3 entries.")

    # Anchor coordinates (x,y)
    ax = []
    ay = []
    ar = []
    for (i, j) in selected_cells:
        x, y, r = circles_grid[i, j]
        if not np.isfinite(x):
            continue
        ax.append(float(x))
        ay.append(float(y))
        ar.append(float(r))

    if len(ax) < 3:
        raise ValueError("Selected cells must refer to finite grid entries.")

    # Small margin so you don't lose edge points
    margin = 0.75 * float(np.median(ar)) if ar else 0.0

    xmin, xmax = min(ax) - margin, max(ax) + margin
    ymin, ymax = min(ay) - margin, max(ay) + margin

    rect_cells = set()
    nrows, ncols, _ = circles_grid.shape
    for i in range(nrows):
        for j in range(ncols):
            x, y, _ = circles_grid[i, j]
            if not np.isfinite(x):
                continue
            xf = float(x)
            yf = float(y)
            if (xmin <= xf <= xmax) and (ymin <= yf <= ymax):
                rect_cells.add((i, j))

    rect_cells.update(selected_cells)
    return rect_cells


""" 
def extract_xy_from_cells(circles_grid: np.ndarray, cells: Iterable[Cell]) -> np.ndarray:
    #Return (N,2) array of x,y values for provided cell indices, ordered by (y,x).
    cells_list = list(cells)
    if not cells_list:
        return np.empty((0, 2), dtype=np.float32)

    xy = np.array(
        [(circles_grid[i, j, 0], circles_grid[i, j, 1]) for (i, j) in cells_list],
        dtype=np.float32,
    )

    # Deterministic order independent of grid padding:
    # sort by y, then x
    order = np.lexsort((xy[:, 0], xy[:, 1]))
    return xy[order]
"""

def extract_xy_from_cells(circles_grid: np.ndarray, cells: Iterable[Cell]) -> np.ndarray:
    """
    Return (N,2) array of (u,v) values for provided cell indices.

    Ordering is strictly row-major by grid index (i,j).
    This guarantees correct correspondences for PnP.
    """
    cells_list = list(cells)
    if not cells_list:
        return np.empty((0, 2), dtype=np.float32)

    # Sort by grid indices (top->bottom, left->right)
    cells_sorted = sorted(cells_list, key=lambda ij: (ij[0], ij[1]))

    xy = np.array(
        [(circles_grid[i, j, 0], circles_grid[i, j, 1]) for (i, j) in cells_sorted],
        dtype=np.float32,
    )
    return xy

def select_marker_roi_from_grid(
    circles_grid: np.ndarray,
    selected_cells: Sequence[Cell],
) -> Tuple[np.ndarray, set[Cell]]:
    """Compute ROI marker coordinates from selected grid cell indices."""
    rect_cells = rect_cells_from_selected(circles_grid, selected_cells)
    xy = extract_xy_from_cells(circles_grid, rect_cells)
    return xy, rect_cells
