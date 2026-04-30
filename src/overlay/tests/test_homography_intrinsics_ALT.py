# -*- coding: utf-8 -*-
"""
test_homography_intrinsics_xray_order.py

Build X-ray homography/intrinsics correspondences using the SAME X-ray marker
ordering as the later Cam2X / pose-estimation pipeline.

Important:
- Uses overlay.tools.xray_marker_selection.compute_roi_from_grid(...)
- Therefore the X-ray-specific ordering is applied there:
      j_xray = nu - j_cam
      i_xray = i_cam

Goal:
      board_xyz_canonical[k] <-> uv_xray_order[k]

No image flip is applied here. Only the correspondence ordering is made
consistent with the later Cam2X pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.blob_detection import HoughCircleParams
from overlay.tools.xray_marker_selection import (
    run_xray_marker_detection,
    compute_roi_from_grid,
)
from overlay.tools.homography import (
    estimate_homography,
    homography_reproj_stats,
    project_homography,
    build_board_xyz_canonical,
)

WIN = "test_homography XRAY ORDER (LMB detect/select, RMB undo, ESC reset, Q quit)"


# ============================================================
# Config
# ============================================================

PITCH_MM = 2.54
GATE_TOL_PITCH = 0.40

HOMOGRAPHY_METHOD = "dlt"
HOMOGRAPHY_THRESH_PX = 0.8
HOMOGRAPHY_MAX_ITERS = 10000
HOMOGRAPHY_CONFIDENCE = 0.999
HOMOGRAPHY_REFINE_WITH_INLIERS = True


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
            raise RuntimeError(
                "pydicom required for .dcm/.ima. Install: pip install pydicom"
            ) from e

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

def _draw_cross(
    img_bgr: np.ndarray,
    u: int,
    v: int,
    r: int,
    color=(0, 0, 255),
    thick=2,
) -> None:
    cv2.line(out := img_bgr, (u - r, v), (u + r, v), color, thick, cv2.LINE_AA)
    cv2.line(out, (u, v - r), (u, v + r), color, thick, cv2.LINE_AA)


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

    xy = circles[:, :2].astype(np.float64)

    for (x, y) in xy:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cv2.circle(
            out,
            (int(round(x)), int(round(y))),
            circle_r,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for k in selected_idx:
        x, y, _r = circles[k]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        _draw_cross(
            out,
            int(round(x)),
            int(round(y)),
            cross_r,
            color=(255, 255, 0),
            thick=2,
        )

    if roi_uv is not None and len(roi_uv) > 0:
        uv = np.asarray(roi_uv, dtype=np.float64).reshape(-1, 2)
        for (u, v) in uv:
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            _draw_cross(
                out,
                int(round(u)),
                int(round(v)),
                cross_r,
                color=(0, 0, 255),
                thick=2,
            )

    if corr_uv is not None and len(corr_uv) > 0:
        uv = np.asarray(corr_uv, dtype=np.float64).reshape(-1, 2)
        for (u, v) in uv:
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            uu, vv = int(round(u)), int(round(v))
            cv2.circle(out, (uu, vv), circle_r, (255, 255, 0), 2, cv2.LINE_AA)

    return out


# ============================================================
# Selection helper
# ============================================================

def _nearest_circle_index(
    circles: np.ndarray,
    u_click: int,
    v_click: int,
) -> Optional[int]:
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
# Main
# ============================================================

def main() -> None:
    path = _pick_image_path()
    if not path:
        print("No image selected.")
        return

    img_path = Path(path)

    img_raw = _load_xray_gray(path)
    Himg, Wimg = img_raw.shape[:2]

    print(f"Loaded RAW image: {Wimg}x{Himg}")
    print("XRAY ORDER mode:")
    print("  - no image flip")
    print("  - uses compute_roi_from_grid(...) from xray_marker_selection.py")
    print("  - therefore same X-ray point ordering as later Cam2X")

    params = HoughCircleParams(
        min_radius=2,
        max_radius=7,
        dp=1.2,
        minDist=8,
        param1=120,
        param2=7,
        invert=True,
        median_ks=(3, 5),
    )

    circles: Optional[np.ndarray] = None
    pick_radius_px: float = 20.0
    selected_idx: List[int] = []

    roi_uv: Optional[np.ndarray] = None
    roi_idx: Optional[np.ndarray] = None
    uv_corr: Optional[np.ndarray] = None

    detected = False

    def refresh() -> None:
        overlay = _render_overlay(
            img_raw,
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
            img_raw,
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

        print(f"Detection done: {len(circles)} circles on RAW image.")
        print("Select 3 anchors in CAMERA VIEW semantics:")
        print("  1) TL")
        print("  2) TR")
        print("  3) BL")
        print("compute_roi_from_grid(...) will then apply the X-ray ordering internally.")

        refresh()

    def finalize_homography() -> None:
        nonlocal roi_uv, roi_idx, uv_corr

        assert circles is not None
        assert len(selected_idx) == 3

        margin_px = 1.1 * float(pick_radius_px)

        uv_xray_, roi_idx_, dbg = compute_roi_from_grid(
            circles=circles,
            anchor_idx=selected_idx,
            margin_px=margin_px,
            gate_tol_pitch=GATE_TOL_PITCH,
            min_steps=2,
        )

        roi_uv = np.asarray(uv_xray_, dtype=np.float64).reshape(-1, 2)
        roi_idx = np.asarray(roi_idx_, dtype=np.int64).reshape(-1)

        print(
            f"[ROI XRAY ORDER] keep={dbg['keep']}  in_box={dbg['in_box']}  "
            f"pitch={dbg['pitch']:.3f}  "
            f"nu0={dbg['nu0']} nv0={dbg['nv0']} -> nu={dbg['nu']} nv={dbg['nv']}"
        )

        if not dbg.get("orientation_ok", True):
            print("\n[WARNING] Anchor orientation check failed:")
            print(dbg.get("orientation_warning", "No warning text available."))
            print("Orientation diagnostics:")
            print(f"  ux      = {dbg.get('orientation_ux')}")
            print(f"  uy      = {dbg.get('orientation_uy')}")
            print(f"  vx      = {dbg.get('orientation_vx')}")
            print(f"  vy      = {dbg.get('orientation_vy')}")
            print(f"  cross_z = {dbg.get('orientation_cross_z')}")
            print()

        board_xyz = build_board_xyz_canonical(
            nu=int(dbg["nu"]),
            nv=int(dbg["nv"]),
            pitch_mm=PITCH_MM,
        )

        XY = np.asarray(board_xyz[:, :2], dtype=np.float64).reshape(-1, 2)
        uv_used = np.asarray(roi_uv, dtype=np.float64).reshape(-1, 2)

        if XY.shape[0] != uv_used.shape[0]:
            raise RuntimeError(
                f"Point count mismatch: board XY has {XY.shape[0]} points, "
                f"uv_xray has {uv_used.shape[0]} points."
            )

        corr_path = img_path.with_suffix("").with_name(
            img_path.with_suffix("").name + "__corr_XRAY_ORDER.npz"
        )

        np.savez(
            corr_path,
            XY=XY.astype(np.float64),
            uv=uv_used.astype(np.float64),
            homography_method=np.array(HOMOGRAPHY_METHOD),
            grid_i=np.asarray(dbg["grid_i"], dtype=np.int32),
            grid_j=np.asarray(dbg["grid_j"], dtype=np.int32),
            xray_order=np.array(True, dtype=bool),
            no_image_flip=np.array(True, dtype=bool),
            anchor_idx=np.asarray(selected_idx, dtype=np.int32),
        )

        print(f"[OK] saved correspondences XRAY ORDER -> {corr_path}")

        uv_corr = uv_used.copy()

        H = estimate_homography(
            uv_used,
            XY,
            method=HOMOGRAPHY_METHOD,
            ransac_reproj_threshold_px=HOMOGRAPHY_THRESH_PX,
            max_iters=HOMOGRAPHY_MAX_ITERS,
            confidence=HOMOGRAPHY_CONFIDENCE,
            refine_with_all_inliers=HOMOGRAPHY_REFINE_WITH_INLIERS,
        )

        uv_proj = project_homography(H, XY)
        finite = (
            np.isfinite(uv_used[:, 0])
            & np.isfinite(uv_used[:, 1])
            & np.isfinite(uv_proj[:, 0])
            & np.isfinite(uv_proj[:, 1])
        )

        if np.any(finite):
            e = np.linalg.norm(uv_proj[finite] - uv_used[finite], axis=1)
        else:
            e = np.empty((0,), dtype=np.float64)

        H_path = img_path.with_suffix("").with_name(
            img_path.with_suffix("").name + "__H_XRAY_ORDER.npz"
        )

        np.savez(
            H_path,
            H=H,
            homography_method=np.array(HOMOGRAPHY_METHOD),
            xray_order=np.array(True, dtype=bool),
            no_image_flip=np.array(True, dtype=bool),
        )

        print(f"[OK] saved homography XRAY ORDER -> {H_path}")

        mean_e, med_e, rmse_e = homography_reproj_stats(H, XY, uv_used)

        np.set_printoptions(precision=6, suppress=True)
        print(f"\n================= Homography ({HOMOGRAPHY_METHOD}, XRAY ORDER) =================")
        print("N points:", uv_used.shape[0])
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

        print("======================================================================\n")

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
                if len(selected_idx) == 1:
                    print("  Anchor 1 = TL")
                elif len(selected_idx) == 2:
                    print("  Anchor 2 = TR")
                elif len(selected_idx) == 3:
                    print("  Anchor 3 = BL")

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

    print("RAW image — no rotation, no image flip.")
    print("LMB: first click runs detection, subsequent clicks select anchors.")
    print("Select anchors in CAMERA VIEW semantics:")
    print("  1) TL")
    print("  2) TR")
    print("  3) BL")
    print("This script uses compute_roi_from_grid(...), i.e. same X-ray ordering as Cam2X.")
    print("Saved files:")
    print("  *__corr_XRAY_ORDER.npz")
    print("  *__H_XRAY_ORDER.npz")
    print("Q: quit. ESC: reset selection. RMB: undo anchor.")

    refresh()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        if key == 27:
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