# -*- coding: utf-8 -*-
"""
test_xray_marker_selection.py

OpenCV-only test harness (matches the NEW xray_marker_selection API):

- Load X-ray via QFileDialog
- Show X-ray in original resolution
- First LMB click: run marker detection (run_xray_marker_detection)
- Select 3 anchor markers (LMB), RMB undo, ESC reset selection
- After 3 anchors: compute ROI (roi_uv, roi_idx, dbg)
- Save the selected ROI uv points to a txt file (one row per point, 2 decimals).
  Format per line: "u v"

NO homography, NO XY.
NO circles_grid, NO circles_sorted.
ROI output is NOT row-wise ordered (keeps the original circles order, filtered).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.blob_detection import HoughCircleParams
from overlay.tools.xray_marker_selection import run_xray_marker_detection, compute_roi_from_grid


WIN = "test_xray_marker_selection (LMB detect/select, RMB undo, ESC reset, Q quit)"
DBG_PROC = "DBG img_proc (after CLAHE)"
DBG_MASK = "DBG mask (255=keep, 0=drop)"
DBG_MASKED = "DBG img_masked (fed into Hough)"
OUT_DIR = Path(r"C:\Users\domin\Documents\Studium\Master\Masterarbeit\Projekt\Data")


# ============================================================
# Debug windows
# ============================================================

def _show_detection_debug(res) -> None:
    cv2.imshow(DBG_PROC, res.img_proc)
    cv2.imshow(DBG_MASK, res.mask)
    cv2.imshow(DBG_MASKED, res.img_masked)

    keep = int(np.count_nonzero(res.mask))
    total = int(res.mask.size)
    print(f"[mask] keep={keep}/{total} ({100.0*keep/total:.1f}%)")
    print(f"[img_masked] nonzero={int(np.count_nonzero(res.img_masked))}")


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

    # ROI points (red crosses)
    if roi_uv is not None and len(roi_uv) > 0:
        uv = np.asarray(roi_uv, dtype=np.float64).reshape(-1, 2)
        for (u, v) in uv:
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            _draw_cross(out, int(round(u)), int(round(v)), cross_r, color=(0, 0, 255), thick=2)

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
# Save ROI UV to TXT
# ============================================================

def _save_roi_uv_txt(txt_path: Path, uv_image: np.ndarray) -> None:
    uv = np.asarray(uv_image, dtype=np.float64).reshape(-1, 2)
    uv = uv[np.isfinite(uv).all(axis=1)]

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        for (u, v) in uv:
            f.write(f"{u:.2f} {v:.2f}\n")

    print(f"[OK] Saved ROI uv points: {uv.shape[0]} rows -> {txt_path}")


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

    circles: Optional[np.ndarray] = None
    pick_radius_px: float = 20.0
    selected_idx: List[int] = []
    roi_uv: Optional[np.ndarray] = None
    roi_idx: Optional[np.ndarray] = None
    detected = False

    gate_tol_pitch = 0.40  # if a corner point is missing: try 0.45; if false points: 0.35

    def refresh() -> None:
        overlay = _render_overlay(
            img,
            circles,
            pick_radius_px=pick_radius_px,
            selected_idx=selected_idx,
            roi_uv=roi_uv,
        )
        cv2.imshow(WIN, overlay)

    def run_detection() -> None:
        nonlocal circles, pick_radius_px, detected, selected_idx, roi_uv, roi_idx

        res = run_xray_marker_detection(
            img,
            hough_params=params,
            use_clahe=True,
            clahe_clip=2.0,
            clahe_tiles=(12, 12),
            use_mask=False,
        )

        _show_detection_debug(res)

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
        detected = True

        print("Detection done. Select 3 anchors (LMB). RMB undo. ESC reset selection. Q quit.")
        refresh()

    def finalize_roi_and_save() -> None:
        nonlocal roi_uv, roi_idx

        assert circles is not None
        assert len(selected_idx) == 3

        margin_px = 1.1 * float(pick_radius_px)

        roi_uv_, roi_idx_, dbg = compute_roi_from_grid(
            circles=circles,
            anchor_idx=selected_idx,
            margin_px=margin_px,
            gate_tol_pitch=gate_tol_pitch,
            min_steps=2,
        )

        roi_uv = roi_uv_
        roi_idx = roi_idx_

        print(
            f"[ROI] keep={dbg['keep']}  in_box={dbg['in_box']}  "
            f"pitch={dbg['pitch']:.3f}  tol_px={dbg['tol_px']:.2f}  "
            f"nu0={dbg['nu0']} nv0={dbg['nv0']} -> nu={dbg['nu']} nv={dbg['nv']}"
        )

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        txt_path = OUT_DIR / f"{img_path.stem}__roi_uv.txt"
        _save_roi_uv_txt(txt_path, roi_uv)

        refresh()

    def on_mouse(event, x, y, flags, userdata):
        nonlocal detected, selected_idx, roi_uv, roi_idx

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
                print("Selected anchors idx:", selected_idx)
                refresh()

                if len(selected_idx) == 3:
                    finalize_roi_and_save()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(selected_idx) > 0:
                selected_idx.pop()
                roi_uv = None
                roi_idx = None
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
                print("Selection reset.")
                refresh()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()