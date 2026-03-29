# -*- coding: utf-8 -*-
"""
test_homography.py

OpenCV-only test harness (matches the NEW xray_marker_selection API):

- Load X-ray via QFileDialog
- Show X-ray in original resolution
- First LMB click: run marker detection (run_xray_marker_detection)
- Select 3 anchor markers (LMB), RMB undo, ESC reset selection
- After 3 anchors:
    * compute ROI (roi_uv, roi_idx, dbg)  [roi_uv is grid-ordered row-major]
    * build correspondences (XY in mm, uv in px) in a GUARANTEED consistent ordering
    * estimate homography (grid -> image) and print H + reprojection error
    * save:
        - "<image_stem>__H.npz"
        - "<image_stem>__uv.txt"   (one row per point: "u v")
        - "<image_stem>__XY.txt"   (one row per point: "X Y")

NO circles_grid, NO circles_sorted, NO cell-based selection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.blob_detection import HoughCircleParams
from overlay.tools.xray_marker_selection import run_xray_marker_detection, compute_roi_from_grid
from overlay.tools.homography import (
    estimate_homography_dlt,
    homography_reproj_stats,
    project_homography,
    build_planar_correspondences,  # NEW: now consumes (roi_uv, dbg)
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
    circles: Optional[np.ndarray],
    *,
    pick_radius_px: float,
    selected_idx: List[int],
    roi_uv: Optional[np.ndarray] = None,
    corr_uv: Optional[np.ndarray] = None,
) -> np.ndarray:
    img8 = img_gray if img_gray.dtype == np.uint8 else np.clip(img_gray, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    if circles is None or len(circles) == 0:
        return out

    roi_r = max(3.0, 0.35 * float(pick_radius_px))
    circle_r = int(round(roi_r))
    cross_r = int(round(0.6 * roi_r))

    # all detected circles (green rings)
    xy = circles[:, :2].astype(np.float64)
    for (x, y) in xy:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cv2.circle(out, (int(round(x)), int(round(y))), circle_r, (0, 255, 0), 2, cv2.LINE_AA)

    # selected anchors (cyan crosses)
    for k in selected_idx:
        x, y, _r = circles[k]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        _draw_cross(out, int(round(x)), int(round(y)), cross_r, color=(255, 255, 0), thick=2)

    # ROI points (red crosses) - as returned by compute_roi_from_grid (row-major)
    if roi_uv is not None and len(roi_uv) > 0:
        uv = np.asarray(roi_uv, dtype=np.float64).reshape(-1, 2)
        for (u, v) in uv:
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            _draw_cross(out, int(round(u)), int(round(v)), cross_r, color=(0, 0, 255), thick=2)

    # Correspondence uv used for homography (cyan rings)
    if corr_uv is not None and len(corr_uv) > 0:
        uv = np.asarray(corr_uv, dtype=np.float64).reshape(-1, 2)
        for (u, v) in uv:
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            uu, vv = int(round(u)), int(round(v))
            cv2.circle(out, (uu, vv), circle_r, (255, 255, 0), 2, cv2.LINE_AA)

    return out


# ============================================================
# Selection: nearest circle in circles (N,3)
# ============================================================

def _nearest_circle_index(circles: np.ndarray, u_click: int, v_click: int) -> Optional[int]:
    if circles is None or len(circles) == 0:
        return None

    xy = circles[:, :2].astype(np.float64)
    finite = np.isfinite(xy).all(axis=1)
    if not np.any(finite):
        return None

    xyf = xy[finite]
    idx_map = np.flatnonzero(finite)

    dx = xyf[:, 0] - float(u_click)
    dy = xyf[:, 1] - float(v_click)
    d2 = dx * dx + dy * dy

    k = int(np.argmin(d2))
    return int(idx_map[k])


# ============================================================
# Save helpers
# ============================================================

def _save_uv_txt(txt_path: Path, uv: np.ndarray) -> None:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    uv = uv[np.isfinite(uv).all(axis=1)]
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(txt_path, uv, fmt="%.2f")
    print(f"[OK] saved uv -> {txt_path}")


def _save_xy_txt(txt_path: Path, XY: np.ndarray) -> None:
    XY = np.asarray(XY, dtype=np.float64).reshape(-1, 2)
    XY = XY[np.isfinite(XY).all(axis=1)]
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(txt_path, XY, fmt="%.3f")
    print(f"[OK] saved XY -> {txt_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    path = _pick_image_path()
    if not path:
        print("No image selected.")
        return

    img_path = Path(path)
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

    pitch_mm = 2.54
    gate_tol_pitch = 0.40  # if corner missing: try 0.45; if false points: 0.35

    circles: Optional[np.ndarray] = None
    pick_radius_px: float = 20.0
    selected_idx: List[int] = []

    roi_uv: Optional[np.ndarray] = None
    roi_idx: Optional[np.ndarray] = None
    uv_corr: Optional[np.ndarray] = None

    detected = False

    def refresh() -> None:
        overlay = _render_overlay(
            img,
            circles,
            pick_radius_px=pick_radius_px,
            selected_idx=selected_idx,
            roi_uv=roi_uv,
            corr_uv=uv_corr,
        )
        cv2.imshow(WIN, overlay)

    def run_detection() -> None:
        nonlocal circles, pick_radius_px, detected, selected_idx, roi_uv, roi_idx, uv_corr

        res = run_xray_marker_detection(
            img,
            hough_params=params,
            use_clahe=True,
            clahe_clip=2.0,
            clahe_tiles=(12, 12),
            use_mask=False,
        )

        if res.circles is None or len(res.circles) == 0:
            print("No circles detected.")
            circles = None
            detected = False
            refresh()
            return

        circles = np.asarray(res.circles, dtype=np.float64).reshape(-1, 3)

        r = circles[:, 2]
        r = r[np.isfinite(r)]
        if r.size:
            marker_radius_px = float(np.median(r))
            pick_radius_px = 0.6 * marker_radius_px
        else:
            pick_radius_px = 20.0

        selected_idx = []
        roi_uv = None
        roi_idx = None
        uv_corr = None
        detected = True

        print("Detection done. Select 3 anchors (LMB). RMB undo. ESC reset selection. Q quit.")
        refresh()

    def finalize_homography() -> None:
        nonlocal roi_uv, roi_idx, uv_corr

        assert circles is not None
        assert len(selected_idx) == 3

        margin_px = 1.1 * float(pick_radius_px)

        # --- compute ROI (row-major ordered uv) ---
        roi_uv_, roi_idx_, dbg = compute_roi_from_grid(
            circles=circles,
            anchor_idx=selected_idx,
            margin_px=margin_px,
            gate_tol_pitch=gate_tol_pitch,
            min_steps=2,
        )
        roi_uv = np.asarray(roi_uv_, dtype=np.float64).reshape(-1, 2)
        roi_idx = np.asarray(roi_idx_, dtype=np.int64).reshape(-1)

        print(
            f"[ROI] keep={dbg['keep']}  in_box={dbg['in_box']}  "
            f"pitch={dbg['pitch']:.3f}  tol_px={dbg['tol_px']:.2f}  "
            f"nu0={dbg['nu0']} nv0={dbg['nv0']} -> nu={dbg['nu']} nv={dbg['nv']}"
        )

        # --- build correspondences (XY row-major + uv row-major) ---
        XY, uv, meta = build_planar_correspondences(
            roi_uv=roi_uv,
            dbg=dbg,
            pitch_mm=pitch_mm,
        )
        XY = np.asarray(XY, dtype=np.float64).reshape(-1, 2)
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        
        # save correspondences
        corr_path = img_path.with_suffix("").with_name(
            img_path.with_suffix("").name + "__corr.npz"
        )
        
        np.savez(
            corr_path,
            XY=XY.astype(np.float64),
            uv=uv.astype(np.float64),
        )
        
        print(f"[OK] saved correspondences -> {corr_path}")

        # overlay rings = points actually used for H
        uv_corr = uv.copy()

        if uv.shape[0] < 4:
            raise RuntimeError(f"Need at least 4 correspondences for homography, got {uv.shape[0]}.")

        # --- save correspondences as txt (for debugging) ---
        uv_txt = img_path.with_suffix("").with_name(img_path.with_suffix("").name + "__uv.txt")
        xy_txt = img_path.with_suffix("").with_name(img_path.with_suffix("").name + "__XY.txt")
        #_save_uv_txt(uv_txt, uv)
        #_save_xy_txt(xy_txt, XY)

        # --- estimate homography (grid -> image): XY -> uv ---
        H = estimate_homography_dlt(uv, XY)

        # --- compute reprojection error vector ---
        uv_proj = project_homography(H, XY)
        finite = (
            np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) &
            np.isfinite(uv_proj[:, 0]) & np.isfinite(uv_proj[:, 1])
        )
        if np.any(finite):
            e = np.linalg.norm(uv_proj[finite] - uv[finite], axis=1)
        else:
            e = np.empty((0,), dtype=np.float64)

        # --- save homography only ---
        npz_path = img_path.with_suffix("").with_name(img_path.with_suffix("").name + "__H.npz")
        np.savez(npz_path, H=H)
        print(f"[OK] saved homography -> {npz_path}")

        mean_e, med_e, rmse_e = homography_reproj_stats(H, XY, uv)

        np.set_printoptions(precision=6, suppress=True)
        print("\n================= Homography (DLT) =================")
        print("N points:", uv.shape[0])
        print("H =\n", H)
        print(
            f"Reprojection error [px]: "
            f"mean={mean_e:.3f}, median={med_e:.3f}, rmse={rmse_e:.3f}"
        )
        if e.size:
            print(
                f"Reproj err vector: min={float(np.min(e)):.3f}  "
                f"max={float(np.max(e)):.3f}  p95={float(np.percentile(e, 95)):.3f}"
            )
        print("====================================================\n")

        refresh()

    def on_mouse(event, x, y, flags, userdata):
        nonlocal detected, selected_idx, roi_uv, roi_idx, uv_corr

        if event == cv2.EVENT_LBUTTONDOWN:
            if not detected:
                run_detection()
                return

            if circles is None:
                return

            k = _nearest_circle_index(circles, x, y)
            if k is None or k in selected_idx:
                return

            if len(selected_idx) < 3:
                selected_idx.append(k)
                roi_uv = None
                roi_idx = None
                uv_corr = None
                print("Selected anchors idx:", selected_idx)
                refresh()

                if len(selected_idx) == 3:
                    finalize_homography()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(selected_idx) > 0:
                selected_idx.pop()
                roi_uv = None
                roi_idx = None
                uv_corr = None
                print("Undo. Selected anchors idx:", selected_idx)
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
                selected_idx = []
                roi_uv = None
                roi_idx = None
                uv_corr = None
                print("Selection reset.")
                refresh()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()