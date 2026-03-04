# -*- coding: utf-8 -*-
"""
test_xray_calibration.py

Two tests:

(1) test_estimate_intrinsics_and_report_motion()
    - Select multiple .npz files that contain homographies (3x3)
    - Estimate X-ray intrinsics Kx via Zhang
    - Save Kx -> xray_intrinsics_Kx.npz
    - Decompose each homography into pose using Kx
    - Print relative board motion (angles + translation) w.r.t. View 01

(2) test_visual_projection_view()
    - Select ONE view NPZ containing a homography H (3x3)
    - Select ONE correspondences NPZ containing:
        * XY (N,2) in mm (board frame, Z=0)
        * uv (N,2) in px (detected in image)
        * optional blob_radius_px / marker_radius_px (for cross sizing)
    - Select Kx NPZ (xray_intrinsics_Kx.npz) [loaded/validated for consistency]
    - Load X-ray image (png/jpg/tif/bmp OR dicom .dcm/.ima)
    - Project XY -> uv_proj using H (planar mapping)
    - Display overlay with ZOOM:
        detected uv  = red cross
        projected uv = cyan cross
      Cross size ≈ 0.5 * blob_radius and scales with zoom.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2

from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.calib.calib_xray_intrinsics import (
    estimate_intrinsics_from_homographies,
    decompose_homography,
    relative_board_angles_deg,
    relative_shift_board_mm,
)

from overlay.tools.homography import project_homography


# ============================================================
# Qt helpers
# ============================================================

def _ensure_qt_app() -> None:
    app = QApplication.instance()
    if app is None:
        _ = QApplication(sys.argv)


def select_npz_files(title: str) -> List[Path]:
    _ensure_qt_app()
    files, _ = QFileDialog.getOpenFileNames(
        None,
        title,
        "",
        "NumPy archives (*.npz);;All files (*.*)",
    )
    return [Path(f) for f in files if f.lower().endswith(".npz")]


def select_npz_file(title: str) -> Optional[Path]:
    _ensure_qt_app()
    f, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        "NumPy archives (*.npz);;All files (*.*)",
    )
    if not f:
        return None
    return Path(f)


def select_image_file(title: str) -> Optional[Path]:
    _ensure_qt_app()
    f, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.dcm *.ima);;All files (*.*)",
    )
    if not f:
        return None
    return Path(f)


# ============================================================
# Image loading (PNG + DICOM) - like your snippet
# ============================================================

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
# NPZ helpers
# ============================================================

def _find_first_matrix33(npz: np.lib.npyio.NpzFile) -> Optional[np.ndarray]:
    for key in npz.files:
        arr = np.asarray(npz[key])
        if arr.shape == (3, 3):
            return arr.astype(np.float64)
    return None


def _find_homographies_in_npz(npz: np.lib.npyio.NpzFile) -> List[np.ndarray]:
    hits: List[np.ndarray] = []
    for key in npz.files:
        arr = np.asarray(npz[key])
        if arr.shape == (3, 3):
            hits.append(arr.astype(np.float64))
        elif arr.ndim == 3 and arr.shape[1:] == (3, 3):
            for i in range(arr.shape[0]):
                hits.append(arr[i].astype(np.float64))
    return hits


def _load_required_points_from_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as npz:
        if "XY" not in npz.files or "uv" not in npz.files:
            raise KeyError(
                f"{path.name} must contain keys 'XY' and 'uv'. Found: {npz.files}"
            )
        XY = np.asarray(npz["XY"], dtype=np.float64)
        uv = np.asarray(npz["uv"], dtype=np.float64)

    if XY.ndim != 2 or XY.shape[1] != 2:
        raise ValueError(f"{path.name}: XY has shape {XY.shape}, expected (N,2)")
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"{path.name}: uv has shape {uv.shape}, expected (N,2)")
    if XY.shape[0] != uv.shape[0]:
        raise ValueError(f"{path.name}: XY and uv must have same N. XY={XY.shape[0]}, uv={uv.shape[0]}")
    return XY, uv


def _load_kx_from_npz(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as npz:
        if "Kx" not in npz.files:
            raise KeyError(f"{path.name} must contain key 'Kx'. Found keys: {npz.files}")
        K = np.asarray(npz["Kx"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"{path.name}: Kx has shape {K.shape}, expected (3,3)")
    return K


def _load_blob_radius_px_from_npz(path: Path, default_px: float = 12.0) -> float:
    """
    Optional: blob radius for cross sizing.

    We try common keys. If missing, use default.
    """
    keys = ("blob_radius_px", "marker_radius_px", "radius_px", "r_px")
    with np.load(str(path), allow_pickle=False) as npz:
        for k in keys:
            if k in npz.files:
                val = np.asarray(npz[k]).astype(np.float64)
                return float(val.item()) if val.size == 1 else float(np.nanmedian(val))
    return float(default_px)


# ============================================================
# Drawing helpers (DISPLAY space)
# ============================================================

def _draw_cross_disp(img_bgr: np.ndarray, xd: float, yd: float,
                     size: int, color_bgr, thick: int = 2) -> None:
    x = int(round(xd))
    y = int(round(yd))
    cv2.line(img_bgr, (x - size, y), (x + size, y), color_bgr, thick, cv2.LINE_AA)
    cv2.line(img_bgr, (x, y - size), (x, y + size), color_bgr, thick, cv2.LINE_AA)


# ============================================================
# Zoomable overlay viewer (square-zoom like your blob script)
# ============================================================

class ZoomOverlayViewer:
    """
    - LMB drag      : square zoom
    - RMB / dblLMB  : reset zoom (full image)
    - ESC           : quit

    Crosses are drawn in DISPLAY coordinates and therefore scale with zoom.
    """

    def __init__(self, img_gray: np.ndarray,
                 uv_det: np.ndarray,
                 uv_proj: np.ndarray,
                 cross_size_img_px: float,
                 win_name: str = "X-ray (Detected=red, Projected=cyan)"):

        if img_gray.ndim != 2:
            raise ValueError("img_gray must be grayscale (H,W).")

        self.win = win_name
        self.img = img_gray
        self.h, self.w = img_gray.shape[:2]

        self.uv_det = np.asarray(uv_det, dtype=np.float64).reshape(-1, 2)
        self.uv_proj = np.asarray(uv_proj, dtype=np.float64).reshape(-1, 2)

        # cross size in IMAGE pixels -> scaled by zoom to DISPLAY pixels
        self.cross_size_img_px = float(cross_size_img_px)

        # ROI state (x,y,w,h) in image coords
        self.base_roi = (0, 0, self.w, self.h)
        self.roi = self.base_roi

        # Fixed display size (constant window size while zooming)
        self.disp_w = self.w
        self.disp_h = self.h

        # drag state
        self.dragging = False
        self.start = None
        self.sel = None

    def _img_to_disp_many(self, uv: np.ndarray) -> np.ndarray:
        rx, ry, rw, rh = self.roi
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        xd = (uv[:, 0] - rx) * (self.disp_w / float(rw))
        yd = (uv[:, 1] - ry) * (self.disp_h / float(rh))
        return np.stack([xd, yd], axis=1)

    def reset_zoom(self) -> None:
        self.roi = self.base_roi
        self.dragging = False
        self.start = None
        self.sel = None
        print("[zoom] reset to full view")

    def _mouse(self, event, x, y, flags, param):
        x = max(0, min(self.disp_w - 1, int(x)))
        y = max(0, min(self.disp_h - 1, int(y)))

        # reset
        if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_LBUTTONDBLCLK:
            self.reset_zoom()
            return

        # zoom drag start
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.sel = (x, y, x, y)
            return

        # update square selection
        if event == cv2.EVENT_MOUSEMOVE and self.dragging and self.start is not None:
            x0, y0 = self.start
            dx = x - x0
            dy = y - y0
            side = max(abs(dx), abs(dy))
            sx = 1 if dx >= 0 else -1
            sy = 1 if dy >= 0 else -1
            x1 = x0 + sx * side
            y1 = y0 + sy * side

            xa, xb = sorted([x0, x1])
            ya, yb = sorted([y0, y1])

            xa = max(0, min(self.disp_w - 1, xa))
            xb = max(0, min(self.disp_w - 1, xb))
            ya = max(0, min(self.disp_h - 1, ya))
            yb = max(0, min(self.disp_h - 1, yb))

            side2 = min(xb - xa, yb - ya)
            xb = xa + side2
            yb = ya + side2

            self.sel = (int(xa), int(ya), int(xb), int(yb))
            return

        # commit zoom
        if event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            if self.sel is None:
                return

            x0d, y0d, x1d, y1d = self.sel
            if (x1d - x0d) < 10:
                self.sel = None
                return

            # map selection corners back to image coords
            rx, ry, rw, rh = self.roi
            sx = rw / float(self.disp_w)
            sy = rh / float(self.disp_h)

            u0 = rx + x0d * sx
            v0 = ry + y0d * sy
            u1 = rx + x1d * sx
            v1 = ry + y1d * sy

            new_x = int(np.floor(min(u0, u1)))
            new_y = int(np.floor(min(v0, v1)))
            new_w = int(np.ceil(abs(u1 - u0)))
            new_h = int(np.ceil(abs(v1 - v0)))

            side = max(1, min(new_w, new_h))
            new_x = max(0, min(self.w - 1, new_x))
            new_y = max(0, min(self.h - 1, new_y))
            side = min(side, self.w - new_x, self.h - new_y)

            self.roi = (new_x, new_y, int(side), int(side))
            self.sel = None
            self.start = None
            print(f"[zoom] ROI = (x={self.roi[0]}, y={self.roi[1]}, w={self.roi[2]}, h={self.roi[3]})")

    def _render(self) -> np.ndarray:
        rx, ry, rw, rh = self.roi
        roi_gray = self.img[ry:ry + rh, rx:rx + rw]

        # Nearest neighbor keeps pixel blocks crisp
        vis_gray = cv2.resize(roi_gray, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST)
        vis = cv2.cvtColor(vis_gray, cv2.COLOR_GRAY2BGR)

        # Cross scaling: display_px_per_image_px
        pixel_size = self.disp_w / float(rw)
        size_disp = int(round(self.cross_size_img_px * pixel_size))
        size_disp = max(2, min(size_disp, 80))  # clamp

        det_disp = self._img_to_disp_many(self.uv_det)
        proj_disp = self._img_to_disp_many(self.uv_proj)

        # detected: red
        for xd, yd in det_disp:
            _draw_cross_disp(vis, xd, yd, size=size_disp, color_bgr=(0, 0, 255), thick=2)

        # projected: cyan
        for xd, yd in proj_disp:
            _draw_cross_disp(vis, xd, yd, size=size_disp, color_bgr=(255, 255, 0), thick=2)

        # selection rect
        if self.sel is not None:
            x0, y0, x1, y1 = self.sel
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 2, cv2.LINE_AA)

        return vis

    def run(self) -> None:
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.disp_w, self.disp_h)
        cv2.setMouseCallback(self.win, self._mouse)

        print("\n=== Visual Test (Zoom) ===")
        print("LMB drag      : square zoom")
        print("RMB / dblLMB  : reset zoom")
        print("ESC           : quit\n")

        while True:
            cv2.imshow(self.win, self._render())
            key = cv2.waitKey(15) & 0xFF
            if key == 27:
                break

        cv2.destroyWindow(self.win)


# ============================================================
# TEST 1 — Intrinsics + Motion
# ============================================================

def test_estimate_intrinsics_and_report_motion() -> None:
    # ------------------------------------------------------------
    # 1) Load Kx (already estimated before)
    # ------------------------------------------------------------
    kx_path = select_npz_file("Select xray_intrinsics_Kx.npz (contains Kx)")
    if kx_path is None:
        print("No Kx .npz selected.")
        return

    try:
        Kx = _load_kx_from_npz(kx_path)
    except Exception as e:
        print(f"ERROR loading Kx: {e}")
        return

    np.set_printoptions(precision=8, suppress=True)

    print("\n==============================================")
    print("Loaded X-ray Intrinsics Kx")
    print("==============================================")
    print(Kx)
    print(f"\nKx NPZ: {kx_path.name}")

    # ------------------------------------------------------------
    # 2) Select exactly TWO view NPZs (each must contain a homography H)
    #    - first selected -> reference view
    #    - second selected -> test view
    # ------------------------------------------------------------
    view_paths = select_npz_files("Select TWO VIEW .npz files (Ref first, then Test)")

    if len(view_paths) != 2:
        print(f"ERROR: Please select exactly 2 view .npz files. You selected {len(view_paths)}.")
        return

    ref_path, test_path = view_paths[0], view_paths[1]

    # ------------------------------------------------------------
    # 3) Load H_ref and H_test
    # ------------------------------------------------------------
    def _load_H_from_view_npz(p: Path) -> Optional[np.ndarray]:
        with np.load(str(p), allow_pickle=False) as npz_view:
            H = _find_first_matrix33(npz_view)
            if H is None:
                print(f"ERROR: Could not find a (3,3) homography in {p.name}. Keys: {npz_view.files}")
                return None
            return H.astype(np.float64)

    H_ref = _load_H_from_view_npz(ref_path)
    if H_ref is None:
        return

    H_test = _load_H_from_view_npz(test_path)
    if H_test is None:
        return

    # ------------------------------------------------------------
    # 4) Decompose into poses using FIXED Kx
    # ------------------------------------------------------------
    pose_ref = decompose_homography(Kx, H_ref)
    pose_test = decompose_homography(Kx, H_test)

    R_ref, t_ref = pose_ref.R, pose_ref.t
    R_test, t_test = pose_test.R, pose_test.t

    # ------------------------------------------------------------
    # 5) Relative board motion (test w.r.t. reference)
    # ------------------------------------------------------------
    tilt_xg, tilt_yg, inplane = relative_board_angles_deg(
        R_ref,
        R_test,
        xray_axes=True,
    )

    dt = relative_shift_board_mm(
        R_ref, t_ref,
        R_test, t_test,
        xray_axes=True,
    )
    dx_g, dy_g, dz_g = dt.tolist()

    print("\n==============================================")
    print("Relative Board Motion (Test w.r.t. Reference)")
    print("==============================================")
    print(f"Reference view: {ref_path.name}")
    print(f"Test view     : {test_path.name}")
    print("\nAngles in deg, translation in mm (xray_axes=True):")
    print("--------------------------------------------------------------------------")
    print("tilt_xg   tilt_yg   inplane   |   dx_g (mm)   dy_g (mm)   dz_g (mm)")
    print("--------------------------------------------------------------------------")
    print(
        f"{tilt_xg:8.3f}  {tilt_yg:8.3f}  {inplane:8.3f}  | "
        f"{dx_g:10.3f}  {dy_g:10.3f}  {dz_g:10.3f}"
    )
    print("--------------------------------------------------------------------------")


# ============================================================
# TEST 2 — Visual Projection Check (zoomable)
# ============================================================

def test_visual_projection_view() -> None:
    corr_path = select_npz_file("Select CORRESPONDENCES .npz (contains XY + uv)")
    if corr_path is None:
        print("No correspondences .npz selected.")
        return

    kx_path = select_npz_file("Select xray_intrinsics_Kx.npz")
    if kx_path is None:
        print("No Kx .npz selected.")
        return

    # Load Kx (THIS is what we want to validate)
    try:
        Kx = _load_kx_from_npz(kx_path)
    except Exception as e:
        print(f"ERROR loading Kx: {e}")
        return

    # Load XY + uv (for THIS new view)
    try:
        XY, uv = _load_required_points_from_npz(corr_path)
    except Exception as e:
        print(f"ERROR loading correspondences: {e}")
        return

    # Load X-ray image (for visualization only)
    img_path = select_image_file("Select X-ray image (png/jpg/tif/bmp/dcm/ima)")
    if img_path is None:
        print("No image selected.")
        return

    try:
        gray = _load_xray_gray(str(img_path))
    except Exception as e:
        print(f"ERROR loading image: {e}")
        return

    # ------------------------------------------------------------
    # 1) Estimate planar homography H from (XY -> uv) for this view
    # ------------------------------------------------------------
    # cv2.findHomography expects src,dst as float32
    XY32 = XY.astype(np.float32)
    uv32 = uv.astype(np.float32)

    H, inliers = cv2.findHomography(XY32, uv32, method=0)  # DLT (no RANSAC)
    if H is None:
        print("ERROR: cv2.findHomography failed.")
        return
    H = H.astype(np.float64)

    # ------------------------------------------------------------
    # 2) Decompose H into pose using the FIXED Kx
    # ------------------------------------------------------------
    pose = decompose_homography(Kx, H)   # your existing function
    R = pose.R
    t = pose.t.reshape(3, 1)

    # ------------------------------------------------------------
    # 3) Direct projection: x ~ Kx (R [X,Y,0]^T + t)
    # ------------------------------------------------------------
    N = XY.shape[0]
    X3 = np.zeros((N, 3), dtype=np.float64)
    X3[:, 0:2] = XY  # (X,Y)
    # Z stays 0

    # camera coords
    Xc = (R @ X3.T) + t  # (3,N)
    x = (Kx @ Xc).T      # (N,3)
    uv_proj = x[:, :2] / x[:, 2:3]

    # ------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------
    err = np.linalg.norm(uv_proj - uv, axis=1)
    err_rms = float(np.sqrt(np.mean(err ** 2)))
    err_med = float(np.median(err))
    err_p95 = float(np.percentile(err, 95))

    n_inl = int(inliers.sum()) if inliers is not None else N

    print("\n==============================================")
    print("Visual Kx Check (fixed Kx, pose from H, direct projection)")
    print("==============================================")
    print(f"Corr NPZ : {corr_path.name}")
    print(f"Kx NPZ   : {kx_path.name}")
    print(f"N points : {N} (H inliers: {n_inl})")
    print(f"Error px : RMS={err_rms:.3f}  MED={err_med:.3f}  P95={err_p95:.3f}")

    # Cross size: 0.15 * blob radius (in IMAGE px)
    blob_r = _load_blob_radius_px_from_npz(corr_path, default_px=12.0)
    cross_size_img_px = 0.15 * blob_r
    print(f"\n[cross] blob_radius_px={blob_r:.2f} -> cross_size_img_px={cross_size_img_px:.2f} (image px)")

    # Viewer (zoomable)
    viewer = ZoomOverlayViewer(
        img_gray=gray,
        uv_det=uv,
        uv_proj=uv_proj,
        cross_size_img_px=cross_size_img_px,
    )
    viewer.run()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    test_estimate_intrinsics_and_report_motion()
    #test_visual_projection_view()


if __name__ == "__main__":
    main()