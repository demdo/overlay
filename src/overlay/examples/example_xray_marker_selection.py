# -*- coding: utf-8 -*-
"""
example_xray_marker_selection.py

Interactive test runner for marker_selection helpers.
"""

import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.xray_marker_selection import (
    detector_mask_radial,
    nearest_cell,
    prepare_nearest_cell_data,
    select_marker_roi_from_grid,
    sort_circles_grid,
)
from overlay.tools.blob_detection import HoughCircleParams, detect_blobs_hough


def _ensure_qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def open_image_file() -> str:
    _ensure_qt_app()
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select marker image",
        "",
        "Image files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*.*)",
    )
    return path


def _render_overlay(
    img_gray: np.ndarray,
    circles_grid: np.ndarray,
    highlight_cells: set,
    help_text: str = "",
) -> np.ndarray:
    """Draw circles+crosses; highlight selected cells in turquoise; everything else green."""
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    nrows, ncols, _ = circles_grid.shape

    for i in range(nrows):
        for j in range(ncols):
            x, y, rr = circles_grid[i, j]
            if not np.isfinite(x):
                continue

            color = (255, 255, 0) if (i, j) in highlight_cells else (0, 255, 0)  # BGR
            cv2.circle(vis, (int(round(x)), int(round(y))), int(round(rr)), color, 2)
            cv2.drawMarker(
                vis,
                (int(round(x)), int(round(y))),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=1,
            )

    if help_text:
        cv2.rectangle(vis, (10, 8), (10 + 1400, 44), (0, 0, 0), -1)
        cv2.putText(
            vis,
            help_text,
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def select_marker_cells_with_ui(
    img_gray: np.ndarray,
    circles_grid: np.ndarray,
    pick_radius_px: float,
    window_name: str = "Select 3 markers",
) -> list[tuple[int, int]]:
    if img_gray.ndim != 2:
        raise ValueError("img_gray must be grayscale (H,W).")
    if circles_grid.ndim != 3 or circles_grid.shape[2] != 3:
        raise ValueError("circles_grid must have shape (nrows, ncols, 3).")

    selected_cells: list[tuple[int, int]] = []
    highlight_cells = set()
    prepared = prepare_nearest_cell_data(circles_grid)
    state = {"done": False}

    def on_mouse(event, x, y, flags, userdata):
        if state["done"]:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            cell, dist = nearest_cell(prepared, x, y)
            if cell is None or dist > pick_radius_px:
                return
            if cell in highlight_cells:
                return

            selected_cells.append(cell)
            highlight_cells.add(cell)
            if len(selected_cells) == 3:
                state["done"] = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            if selected_cells:
                last = selected_cells.pop()
                highlight_cells.discard(last)

    h, w = img_gray.shape[:2]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w, h)
    cv2.setMouseCallback(window_name, on_mouse)

    help_text = "Left-click: select (3) | Right-click: undo | ESC: cancel"

    while True:
        frame = _render_overlay(img_gray, circles_grid, highlight_cells, help_text=help_text)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            selected_cells.clear()
            break
        if state["done"]:
            break

    cv2.destroyWindow(window_name)
    return selected_cells


def select_marker_roi_with_ui(
    img_gray: np.ndarray,
    circles_grid: np.ndarray,
    pick_radius_px: float,
    window_name: str = "Select 3 markers",
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    selected_cells = select_marker_cells_with_ui(
        img_gray=img_gray,
        circles_grid=circles_grid,
        pick_radius_px=pick_radius_px,
        window_name=window_name,
    )
    if len(selected_cells) < 3:
        return np.empty((0, 2), dtype=np.float32), set()

    xy, highlight_cells = select_marker_roi_from_grid(
        circles_grid=circles_grid,
        selected_cells=selected_cells,
    )

    h, w = img_gray.shape[:2]
    final_title = "ROI selected"
    cv2.namedWindow(final_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(final_title, w, h)

    rows_sel = [p[0] for p in selected_cells]
    cols_sel = [p[1] for p in selected_cells]
    r0, r1 = int(min(rows_sel)), int(max(rows_sel))
    c0, c1 = int(min(cols_sel)), int(max(cols_sel))
    roi_text = f"ROI: rows {r0}-{r1}, cols {c0}-{c1}"

    final_frame = _render_overlay(img_gray, circles_grid, highlight_cells, help_text=roi_text)
    cv2.imshow(final_title, final_frame)
    cv2.waitKey(0)
    cv2.destroyWindow(final_title)

    return xy, highlight_cells


def main():
    params = HoughCircleParams(
        min_radius=3,
        max_radius=7,
        dp=1.2,
        minDist=26,
        param1=120,
        param2=8,
        invert=True,
        median_ks=5,
    )

    use_clahe = True
    clahe_clip = 2.0
    clahe_tiles = (12, 12)

    n_angles = 360
    smooth_sigma = 2.0
    r_min_frac = 0.20
    r_max_frac = 0.98
    shrink_px = 12

    row_tol_px = 13.0

    image_path = open_image_file()
    if not image_path:
        raise RuntimeError("No image selected.")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)

    img_proc = img.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tiles)
        img_proc = clahe.apply(img_proc)

    mask, _ = detector_mask_radial(
        img_proc,
        n_angles=n_angles,
        r_min_frac=r_min_frac,
        r_max_frac=r_max_frac,
        smooth_sigma=smooth_sigma,
        peak_prominence=0.0,
        shrink_px=shrink_px,
    )

    img_masked = img_proc.copy()
    img_masked[mask == 0] = 0

    circles_out = detect_blobs_hough(img_masked, params)
    if circles_out is None or len(circles_out) == 0:
        raise RuntimeError("No circles detected.")

    circles_out = np.asarray(circles_out, dtype=np.float32)
    circles_sorted = sort_circles_grid(circles_out, row_tol_px=row_tol_px)

    r_med = float(np.median(circles_sorted[:, 2]))
    y_thresh_px = 2.5 * r_med

    c = circles_sorted.copy().astype(np.float32)
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
        padded_row = np.full((ncols, 3), np.nan, dtype=np.float32)
        padded_row[left_pad:left_pad + n_points] = row
        circles_grid[i] = padded_row

    radii = circles_grid[..., 2]
    finite_r = radii[np.isfinite(radii)]
    if finite_r.size:
        pick_radius_px = 0.6 * float(np.median(finite_r))
    else:
        pick_radius_px = 20.0

    xy, highlight_cells = select_marker_roi_with_ui(
        img_gray=img,
        circles_grid=circles_grid,
        pick_radius_px=pick_radius_px,
    )
    if xy.size == 0:
        print("Selection cancelled or incomplete.")
        return

    print("Selected ROI marker coordinates (x,y):", xy.shape)
    print(xy)
    print("Highlight cells:", sorted(highlight_cells) if highlight_cells else highlight_cells)


if __name__ == "__main__":
    main()
