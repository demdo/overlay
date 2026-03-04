# -*- coding: utf-8 -*-
"""
test_homography.py

OpenCV-only test harness:
- Load X-ray via QFileDialog
- Show X-ray in original resolution
- First LMB click: run marker detection (run_xray_marker_detection)
- Select 3 anchor markers (LMB), RMB undo, ESC reset selection
- After 3 anchors:
    * compute ROI (uv_image, roi_cells)
    * build correspondences (XY in mm, uv in px) in a GUARANTEED consistent ordering
    * estimate homography (grid -> image) and print H + reprojection error.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, List

Cell = Tuple[int, int]

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.blob_detection import HoughCircleParams
from overlay.tools.xray_marker_selection import run_xray_marker_detection, compute_roi_from_grid
from overlay.tools.homography import (
    estimate_homography_dlt,
    build_planar_correspondences,
    homography_reproj_stats,
    project_homography,
)

WIN = "test_homography (LMB detect/select, RMB undo, ESC reset selection, Q quit)"


# ============================================================
# Qt picker + load
# ============================================================

def _ensure_qt_app() -> None:
    if QApplication.instance() is None:
        QApplication(sys.argv)


def _pick_image_path() -> str:
    _ensure_qt_app()
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select X-ray image",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.ima);;All files (*.*)",
    )
    return path


def _load_xray_gray(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in (".dcm", ".ima"):
        try:
            import pydicom
        except Exception as e:
            raise RuntimeError("pydicom required for .dcm/.ima. Install: pip install pydicom") from e

        ds = pydicom.dcmread(str(p))
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img.astype(np.uint8)

    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Could not read image.")
    return img


# ============================================================
# Drawing helpers
# ============================================================

def _draw_cross(img_bgr: np.ndarray, u: int, v: int, r: int, color=(0, 0, 255), thick=2) -> None:
    cv2.line(img_bgr, (u - r, v), (u + r, v), color, thick, cv2.LINE_AA)
    cv2.line(img_bgr, (u, v - r), (u, v + r), color, thick, cv2.LINE_AA)


def _render_overlay(
    img_gray: np.ndarray,
    circles_grid: Optional[np.ndarray],
    *,
    pick_radius_px: float,
    selected_cells: List[Tuple[int, int]],
    roi_uv: Optional[np.ndarray] = None,
    corr_uv: Optional[np.ndarray] = None,
) -> np.ndarray:

    img8 = img_gray if img_gray.dtype == np.uint8 else np.clip(img_gray, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    if circles_grid is None:
        return out

    # constant drawn radius for all circles based on pick_radius
    roi_r = max(3.0, 0.35 * float(pick_radius_px))
    circle_r = int(round(roi_r))
    cross_r = int(round(0.6 * roi_r))

    # all detected circles (green circles)
    nrows, ncols, _ = circles_grid.shape
    for i in range(nrows):
        for j in range(ncols):
            x, y, _r = circles_grid[i, j]
            if not np.isfinite(x):
                continue
            u, v = int(round(float(x))), int(round(float(y)))
            cv2.circle(out, (u, v), circle_r, (0, 255, 0), 2, cv2.LINE_AA)

    # selected anchors (cyan cross)
    for (i, j) in selected_cells:
        x, y, _r = circles_grid[i, j]
        if not np.isfinite(x):
            continue
        u, v = int(round(float(x))), int(round(float(y)))
        _draw_cross(out, u, v, cross_r, color=(255, 255, 0), thick=2)

    # ROI points (red crosses)
    if roi_uv is not None and len(roi_uv) > 0:
        uv = np.asarray(roi_uv, dtype=np.float64).reshape(-1, 2)
        for (u, v) in uv:
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            _draw_cross(out, int(round(u)), int(round(v)), cross_r, color=(0, 0, 255), thick=2)

    # Correspondence uv from build_planar_correspondences (cyan rings)
    if corr_uv is not None and len(corr_uv) > 0:
        uv = np.asarray(corr_uv, dtype=np.float64).reshape(-1, 2)
        for (u, v) in uv:
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            uu, vv = int(round(u)), int(round(v))
            cv2.circle(out, (uu, vv), circle_r, (255, 255, 0), 2, cv2.LINE_AA)

    return out


# ============================================================
# Selection: nearest finite cell in circles_grid
# ============================================================

def _nearest_cell(circles_grid: np.ndarray, u_click: int, v_click: int) -> Optional[Tuple[int, int]]:
    coords = circles_grid[..., :2]
    finite = np.isfinite(coords[..., 0]) & np.isfinite(coords[..., 1])
    if not np.any(finite):
        return None

    rows, cols = np.nonzero(finite)
    xs = coords[rows, cols, 0].astype(np.float64)
    ys = coords[rows, cols, 1].astype(np.float64)

    dx = xs - float(u_click)
    dy = ys - float(v_click)
    d2 = dx * dx + dy * dy

    k = int(np.argmin(d2))
    return int(rows[k]), int(cols[k])


# ============================================================
# Main
# ============================================================

def main() -> None:
    path = _pick_image_path()
    if not path:
        print("No image selected.")
        return

    img = _load_xray_gray(path)
    Himg, Wimg = img.shape[:2]

    params = HoughCircleParams(
        min_radius=2,
        max_radius=7,
        dp=1.2,
        minDist=8,
        param1=120,
        param2=9,
        invert=True,
        median_ks=(3, 5),
    )

    circles_grid: Optional[np.ndarray] = None
    pick_radius_px: float = 20.0
    selected_cells: List[Tuple[int, int]] = []

    roi_cells: Optional[set[Tuple[int, int]]] = None
    uv_image: Optional[np.ndarray] = None
    uv_corr: Optional[np.ndarray] = None

    detected = False

    def refresh() -> None:
        overlay = _render_overlay(
            img,
            circles_grid,
            pick_radius_px=pick_radius_px,
            selected_cells=selected_cells,
            roi_uv=uv_image,
            corr_uv=uv_corr,
        )
        cv2.imshow(WIN, overlay)

    def run_detection() -> None:
        nonlocal circles_grid, pick_radius_px, detected, selected_cells, roi_cells, uv_image, uv_corr

        res = run_xray_marker_detection(
            img,
            hough_params=params,
            use_clahe=True,
            clahe_clip=2.0,
            clahe_tiles=(12, 12),
            use_mask=False,
            row_tol_px=13.0,
        )

        if res.circles_grid is None or not np.isfinite(res.circles_grid[..., 0]).any():
            print("No circles detected.")
            return

        circles_grid = res.circles_grid

        # marker_radius = median radius of finite grid entries
        radii = circles_grid[..., 2]
        finite_r = radii[np.isfinite(radii)]
        if finite_r.size:
            marker_radius_px = float(np.median(finite_r))
            pick_radius_px = 0.6 * marker_radius_px
        else:
            pick_radius_px = 20.0

        selected_cells = []
        roi_cells = None
        uv_image = None
        uv_corr = None
        detected = True

        print("Detection done. Select 3 anchor markers (LMB). RMB undo. ESC reset selection. Q quit.")
        refresh()

    def on_mouse(event, x, y, flags, userdata):
        nonlocal detected, selected_cells, uv_image, uv_corr, roi_cells

        if event == cv2.EVENT_LBUTTONDOWN:
            if not detected:
                run_detection()
                return

            if circles_grid is None:
                return

            cell = _nearest_cell(circles_grid, x, y)
            if cell is None:
                return

            if cell in selected_cells:
                return

            if len(selected_cells) < 3:
                selected_cells.append(cell)
                uv_image = None
                uv_corr = None
                roi_cells = None
                print("Selected anchors:", selected_cells)
                refresh()

                if len(selected_cells) == 3:
                    # --- compute ROI ---
                    uv_image, roi_cells = compute_roi_from_grid(
                        circles_grid=circles_grid,
                        selected_cells=selected_cells,
                    )

                    # --- build correspondences ---
                    XY, uv, _cells_kept = build_planar_correspondences(
                        circles_grid, roi_cells
                    )

                    XY = np.asarray(XY, dtype=np.float64).reshape(-1, 2)
                    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
                    
                    # --- save correspondences only (uv + XY) ---
                    """
                    npz_path = Path(path).with_suffix("").with_name(
                        Path(path).with_suffix("").name + "__uv_XY.npz"
                        )
            
                    np.savez(
                        npz_path,
                        uv=uv,
                        XY=XY,
                    )
                    
                    print(f"[OK] saved uv + XY to: {npz_path}")
                    """

                    # for cyan overlay rings
                    uv_corr = uv.copy()

                    print("XY.shape:", XY.shape)
                    print("uv.shape:", uv.shape)

                    if uv.shape[0] < 4:
                        raise RuntimeError(
                            f"Need at least 4 correspondences for homography, got {uv.shape[0]}."
                        )

                    # --- estimate homography (grid -> image) ---
                    H = estimate_homography_dlt(uv, XY)  # API: (uv_img, XY_grid)
                    
                    # --- compute full reprojection error vector ---
                    uv_proj = project_homography(H, XY)
                    
                    finite = (
                        np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) &
                        np.isfinite(uv_proj[:, 0]) & np.isfinite(uv_proj[:, 1])
                    )
                    
                    if np.any(finite):
                        e = np.linalg.norm(uv_proj[finite] - uv[finite], axis=1)
                    else:
                        e = np.empty((0,), dtype=np.float64)
                    """
                    # --- save reprojection error vector only ---
                    npz_path = Path(path).with_suffix("").with_name(
                        Path(path).with_suffix("").name + "__reproj_error.npz"
                    )
                    
                    np.savez(npz_path, e=e)
                    
                    print(f"[OK] saved reprojection error vector to: {npz_path}")  
                    """
                    
                    
                    # --- save homography only ---
                    npz_path = Path(path).with_suffix("").with_name(
                        Path(path).with_suffix("").name + "__H.npz"
                    )
                    
                    np.savez(npz_path, H=H)
                    print(f"[OK] saved homography to: {npz_path}")
                    
                    
                    mean_e, med_e, rmse_e = homography_reproj_stats(H, XY, uv)

                    np.set_printoptions(precision=6, suppress=True)
                    print("\n================= Homography (DLT) =================")
                    print("N points:", uv.shape[0])
                    print("H =\n", H)
                    print(
                        f"Reprojection error [px]: "
                        f"mean={mean_e:.3f}, median={med_e:.3f}, rmse={rmse_e:.3f}"
                    )
                    print("====================================================\n")

                    refresh()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(selected_cells) > 0:
                selected_cells.pop()
                uv_image = None
                uv_corr = None
                roi_cells = None
                print("Undo. Selected anchors:", selected_cells)
                refresh()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, Wimg, Himg)
    cv2.setMouseCallback(WIN, on_mouse)

    print("LMB: first click runs detection, subsequent clicks select anchors.")
    print("RMB: undo anchor. ESC: reset selection. Q: quit.")
    refresh()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

        if key == 27:  # ESC
            if detected:
                selected_cells = []
                roi_cells = None
                uv_image = None
                uv_corr = None
                print("Selection reset.")
                refresh()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()