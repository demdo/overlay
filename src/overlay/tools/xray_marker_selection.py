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


def detector_mask_radial(
    img_u8: np.ndarray,
    n_angles: int = 360,
    r_min_frac: float = 0.20,
    r_max_frac: float = 0.98,
    smooth_sigma: float = 2.0,
    peak_prominence: float = 0.0,
    shrink_px: int = 12,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Create a circular mask by finding the boundary via radial gradients."""
    if img_u8.ndim != 2 or img_u8.dtype != np.uint8:
        raise ValueError("detector_mask_radial expects a uint8 grayscale image.")

    h, w = img_u8.shape
    cx0, cy0 = w / 2.0, h / 2.0
    r0 = 0.5 * min(h, w)

    img_blur = cv2.GaussianBlur(img_u8, (0, 0), smooth_sigma)
    gx = cv2.Sobel(img_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    r_min = int(max(5, r_min_frac * r0))
    r_max = int(max(r_min + 10, r_max_frac * r0))

    boundary_pts: List[List[float]] = []
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False).astype(np.float32)
    rs = np.arange(r_min, r_max, dtype=np.float32)

    for th in angles:
        xs = cx0 + rs * np.cos(th)
        ys = cy0 + rs * np.sin(th)

        xs_i = np.clip(xs, 0, w - 1).astype(np.int32)
        ys_i = np.clip(ys, 0, h - 1).astype(np.int32)

        prof = grad_mag[ys_i, xs_i]
        k = int(np.argmax(prof))

        if peak_prominence > 0.0 and prof[k] < peak_prominence:
            continue

        boundary_pts.append([xs[k], ys[k]])

    boundary_pts = np.array(boundary_pts, dtype=np.float32)

    circle = fit_circle_kasa(boundary_pts)
    if circle is None:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return mask, (cx0, cy0, r0)

    cx, cy, r = circle
    r = max(1.0, r - float(shrink_px))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)

    return mask, (cx, cy, r)


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
    """Return finite cells in rectangle spanned by selected cells."""
    if circles_grid.ndim != 3 or circles_grid.shape[2] != 3:
        raise ValueError("circles_grid must have shape (nrows, ncols, 3).")
    if len(selected_cells) < 3:
        raise ValueError("selected_cells must contain at least 3 entries.")
    
    rows_sel = [p[0] for p in selected_cells]
    cols_sel = [p[1] for p in selected_cells]
    r0, r1 = int(min(rows_sel)), int(max(rows_sel))
    c0, c1 = int(min(cols_sel)), int(max(cols_sel))

    rect_cells = set()
    for i in range(r0, r1 + 1):
        for j in range(c0, c1 + 1):
            x, y, _ = circles_grid[i, j]
            if np.isfinite(x):
                rect_cells.add((i, j))

    # Ensure the 3 anchors included
    rect_cells.update(selected_cells)
    return rect_cells

    
def extract_xy_from_cells(circles_grid: np.ndarray, cells: Iterable[Cell]) -> np.ndarray:
    """Return (N,2) array of x,y values for provided cell indices."""
    cells_sorted = sorted(cells)
    if not cells_sorted:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(
        [(circles_grid[i, j, 0], circles_grid[i, j, 1]) for (i, j) in cells_sorted],
        dtype=np.float32,
    )


def select_marker_roi_from_grid(
    circles_grid: np.ndarray,
    selected_cells: Sequence[Cell],
) -> Tuple[np.ndarray, set[Cell]]:
    """Compute ROI marker coordinates from selected grid cell indices."""
    rect_cells = rect_cells_from_selected(circles_grid, selected_cells)
    xy = extract_xy_from_cells(circles_grid, rect_cells)
    return xy, rect_cells
