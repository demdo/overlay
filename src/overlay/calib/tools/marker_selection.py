# marker_selection.py
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox


def _ensure_qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


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


def _nearest_cell(circles_grid: np.ndarray, x_click: int, y_click: int):
    """Return (i,j) of nearest finite cell and its pixel distance."""
    nrows, ncols, _ = circles_grid.shape
    best = None
    best_d2 = None

    for i in range(nrows):
        for j in range(ncols):
            x, y, _ = circles_grid[i, j]
            if not np.isfinite(x):
                continue
            dx = float(x) - float(x_click)
            dy = float(y) - float(y_click)
            d2 = dx * dx + dy * dy
            if best is None or d2 < best_d2:
                best = (i, j)
                best_d2 = d2

    return best, (np.sqrt(best_d2) if best_d2 is not None else np.inf)


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
    return_cells: bool = False,
    return_grid: bool = False,
):
    """
    Interactive selection of 3 markers on a precomputed circles_grid.

    Workflow:
    - Left click: select nearest marker (turquoise), up to 3
    - Right click: undo last selection
    - After 3 selections: confirmation dialog
    - On confirm: compute highlight_cells = all finite cells in rectangle spanned by the 3 selected cells
    - Returns xy coordinates (N,2) of ALL highlighted markers.

    Parameters
    ----------
    img_gray : np.ndarray
        Original grayscale image (H,W), used only for display.
    circles_grid : np.ndarray
        (nrows, ncols, 3) array with entries (x,y,r) or NaN.
    pick_radius_px : float
        Max pixel distance from click to a marker to accept selection.
        (Use something like 0.6*minDist from your Hough params.)
    window_name : str
        OpenCV window title.
    return_cells : bool
        If True, also return highlight_cells (set of (i,j)).
    return_grid : bool
        If True, also return circles_grid (unchanged, for convenience).

    Returns
    -------
    xy : np.ndarray
        (N,2) float32 array of (x,y) for all highlighted cells.
    (optional) highlight_cells : set[(int,int)]
    (optional) circles_grid : np.ndarray
    """
    if img_gray.ndim != 2:
        raise ValueError("img_gray must be grayscale (H,W).")
    if circles_grid.ndim != 3 or circles_grid.shape[2] != 3:
        raise ValueError("circles_grid must have shape (nrows, ncols, 3).")

    selected_cells = []
    highlight_cells = set()

    state = {"done": False, "confirmed": False}

    def on_mouse(event, x, y, flags, userdata):
        if state["done"]:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            cell, dist = _nearest_cell(circles_grid, x, y)
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

    # Create original-size interactive window
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
        # user cancelled
        xy = np.empty((0, 2), dtype=np.float32)
        out = [xy]
        if return_cells:
            out.append(set())
        if return_grid:
            out.append(circles_grid)
        return out[0] if len(out) == 1 else tuple(out)

    # Build rectangle ROI from the 3 selected cells
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

    # Final display (same window size as original)
    final_title = "ROI selected"
    cv2.namedWindow(final_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(final_title, w, h)

    roi_text = f"ROI: rows {r0}-{r1}, cols {c0}-{c1}"
    final_frame = _render_overlay(img_gray, circles_grid, rect_cells, help_text=roi_text)
    cv2.imshow(final_title, final_frame)
    cv2.waitKey(0)
    cv2.destroyWindow(final_title)

    # Extract XY coordinates
    xy = np.array(
        [(circles_grid[i, j, 0], circles_grid[i, j, 1]) for (i, j) in sorted(rect_cells)],
        dtype=np.float32
    )

    out = [xy]
    if return_cells:
        out.append(rect_cells)
    if return_grid:
        out.append(circles_grid)
    return out[0] if len(out) == 1 else tuple(out)
