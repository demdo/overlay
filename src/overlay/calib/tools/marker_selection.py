# marker_selection.py
# -*- coding: utf-8 -*-

"""
marker_selection.py

Helpers for marker-grid selection without UI dependencies.
"""

import numpy as np


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


def rect_cells_from_selected(circles_grid: np.ndarray, selected_cells):
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

    
def extract_xy_from_cells(circles_grid: np.ndarray, cells):
    """Return (N,2) array of x,y values for provided cell indices."""
    if not cells:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(
        [(circles_grid[i, j, 0], circles_grid[i, j, 1]) for (i, j) in sorted(cells)],
        dtype=np.float32,
    )
