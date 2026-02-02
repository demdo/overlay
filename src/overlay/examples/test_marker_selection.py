# -*- coding: utf-8 -*-
"""
test_marker_selection.py

Interactive test runner for marker_selection helpers.
"""

import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from overlay.calib.tools.marker_selection import (
    extract_xy_from_cells,
    nearest_cell,
    prepare_nearest_cell_data,
    rect_cells_from_selected,
)
from overlay.calib.tools.blob_detection import HoughCircleParams, detect_blobs_hough


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


def fit_circle_kasa(points_xy: np.ndarray):
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
):
    if img_u8.ndim != 2 or img_u8.dtype != np.uint8:
        raise ValueError("detector_mask_radial expects a uint8 grayscale image.")

    h, w = img_u8.shape
    cx0, cy0 = w / 2.0, h / 2.0
    R0 = 0.5 * min(h, w)

    img_blur = cv2.GaussianBlur(img_u8, (0, 0), smooth_sigma)
    gx = cv2.Sobel(img_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    r_min = int(max(5, r_min_frac * R0))
    r_max = int(max(r_min + 10, r_max_frac * R0))

    boundary_pts = []
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
        return mask, (cx0, cy0, R0)

    cx, cy, r = circle
    r = max(1.0, r - float(shrink_px))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)

    return mask, (cx, cy, r)


def sort_circles_grid(circles: np.ndarray, row_tol_px: float = 13.0) -> np.ndarray:
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


def _confirm_markers_dialog(selected_cells):
    _ensure_qt_app()

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Question)
    msg.setWindowTitle("Confirm selected markers")
    msg.setText("Use these 3 markers to define the ROI?")
    msg.setInformativeText(f"Selected cells: {selected_cells}")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg.setDefaultButton(QMessageBox.Yes)
    return msg.exec() == QMessageBox.Yes


def _render_overlay(img_gray: np.ndarray,
                    circles_grid: np.ndarray,
                    highlight_cells: set,
                    help_text: str = "") -> np.ndarray:
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
            cv2.LINE_AA
        )
    return vis


def select_marker_roi_from_grid(
    img_gray: np.ndarray,
    circles_grid: np.ndarray,
    pick_radius_px: float,
    window_name: str = "Select 3 markers",
):
    if img_gray.ndim != 2:
        raise ValueError("img_gray must be grayscale (H,W).")
    if circles_grid.ndim != 3 or circles_grid.shape[2] != 3:
        raise ValueError("circles_grid must have shape (nrows, ncols, 3).")

    selected_cells = []
    highlight_cells = set()
    prepared = prepare_nearest_cell_data(circles_grid)

    state = {"done": False, "confirmed": False}

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
                ok = _confirm_markers_dialog(selected_cells)
                if ok:
                    state["confirmed"] = True
                    state["done"] = True
                else:
                    selected_cells.clear()
                    highlight_cells.clear()

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
            state["done"] = True
            state["confirmed"] = False
            break
        if state["done"]:
            break

    cv2.destroyWindow(window_name)

    if not state["confirmed"]:
        return np.empty((0, 2), dtype=np.float32), set()

    rect_cells = rect_cells_from_selected(circles_grid, selected_cells)

    final_title = "ROI selected"
    cv2.namedWindow(final_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(final_title, w, h)

    rows_sel = [p[0] for p in selected_cells]
    cols_sel = [p[1] for p in selected_cells]
    r0, r1 = int(min(rows_sel)), int(max(rows_sel))
    c0, c1 = int(min(cols_sel)), int(max(cols_sel))
    roi_text = f"ROI: rows {r0}-{r1}, cols {c0}-{c1}"

    final_frame = _render_overlay(img_gray, circles_grid, rect_cells, help_text=roi_text)
    cv2.imshow(final_title, final_frame)
    cv2.waitKey(0)
    cv2.destroyWindow(final_title)

    xy = extract_xy_from_cells(circles_grid, rect_cells)
    return xy, rect_cells


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

    xy, highlight_cells = select_marker_roi_from_grid(
        img_gray=img,
        circles_grid=circles_grid,
        pick_radius_px=pick_radius_px,
    )

    print("Selected ROI marker coordinates (x,y):", xy.shape)
    print(xy)
    print("Highlight cells:", sorted(highlight_cells) if highlight_cells else highlight_cells)


if __name__ == "__main__":
    main()