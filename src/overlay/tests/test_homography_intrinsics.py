# -*- coding: utf-8 -*-
"""
test_homography_intrinsics.py

- Load X-ray via QFileDialog
- Show X-ray as-is (RAW, NO rotation)
- First LMB click: run marker detection on the RAW image
- Select 3 anchor markers on the RAW image in BOARD semantics:
    1) TL
    2) TR
    3) BL
- After 3 anchors:
    * compute ROI on RAW-image detections
    * build uv_xray directly in CANONICAL BOARD row-major order
    * build canonical board XY from build_board_xyz_canonical(...)
    * save correspondences and homography

Important
---------
- No build_planar_correspondences(...)
- No raw/final flip logic
- uv_xray is stored in the SAME order as canonical board_xyz:
    first row TL -> TR, then next row, ..., last row BL -> BR
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.blob_detection import HoughCircleParams, estimate_pitch_nn
from overlay.tools.xray_marker_selection import run_xray_marker_detection
from overlay.tools.homography import (
    estimate_homography_dlt,
    homography_reproj_stats,
    project_homography,
    build_board_xyz_canonical,
)

WIN = "test_homography (LMB detect/select, RMB undo, ESC reset, Q quit)"


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

def _draw_cross(
    img_bgr: np.ndarray,
    u: int,
    v: int,
    r: int,
    color=(0, 0, 255),
    thick=2,
) -> None:
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

    xy = circles[:, :2].astype(np.float64)

    # all detected circles (green rings)
    for (x, y) in xy:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cv2.circle(out, (int(round(x)), int(round(y))), circle_r, (0, 255, 0), 2, cv2.LINE_AA)

    # selected anchors (yellow crosses)
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

    # Correspondence uv actually used for H (cyan rings)
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
# New ROI / ordering helper
# ============================================================

def build_uv_xray_board_order(
    circles: np.ndarray,
    anchor_idx: list[int],
    *,
    margin_px: float,
    gate_tol_pitch: float = 0.40,
    min_steps: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build uv_xray directly in canonical BOARD row-major order.

    Anchor convention
    -----------------
    The user clicks anchors in BOARD semantics:

        1) TL
        2) TR
        3) BL

    Board ordering convention
    -------------------------
    Returned uv_xray is ordered exactly like canonical board_xyz:

        row 0: TL -> TR
        row 1: left -> right
        ...
        last row: BL -> BR

    No flips. No xray-specific final ordering.
    """
    c = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    if c.shape[0] < 3:
        raise ValueError("circles must contain at least 3 entries.")
    if anchor_idx is None or len(anchor_idx) != 3:
        raise ValueError("anchor_idx must contain exactly 3 indices.")

    idx = np.asarray(anchor_idx, dtype=int)
    if np.any(idx < 0) or np.any(idx >= c.shape[0]):
        raise ValueError("anchor_idx contains out-of-range indices.")

    xy = c[:, :2]
    finite = np.isfinite(xy).all(axis=1)
    if not np.any(finite):
        raise ValueError("No finite circle centers available.")

    if not np.all(finite):
        map_back = np.flatnonzero(finite)
        xy = xy[finite]
        inv = {int(old): int(new) for new, old in enumerate(map_back)}
        idx_eff = np.array([inv[int(i)] for i in idx], dtype=int)
    else:
        map_back = None
        idx_eff = idx

    # anchors in BOARD semantics: TL, TR, BL
    p_tl = xy[idx_eff[0], :]
    p_tr = xy[idx_eff[1], :]
    p_bl = xy[idx_eff[2], :]

    u = p_tr - p_tl   # +y_b direction in canonical camera-like board drawing
    v = p_bl - p_tl   # +x_b direction in canonical camera-like board drawing

    Lu = float(np.linalg.norm(u))
    Lv = float(np.linalg.norm(v))
    if Lu <= 1e-9 or Lv <= 1e-9:
        raise ValueError("Anchor geometry is degenerate.")

    A = np.stack([u, v], axis=1)  # 2x2
    det = float(np.linalg.det(A))
    if abs(det) < 1e-9:
        raise ValueError("Anchor basis is nearly singular / collinear.")

    Ainv = np.linalg.inv(A)
    D = (xy - p_tl[None, :]).T
    AB = (Ainv @ D).T
    alpha = AB[:, 0]
    beta = AB[:, 1]

    pitch = float(estimate_pitch_nn(xy))
    if (not np.isfinite(pitch)) or pitch <= 1e-6:
        pitch = 10.0

    tol_px = float(gate_tol_pitch) * float(pitch)
    mu = float((margin_px + tol_px) / (Lu + 1e-12))
    mv = float((margin_px + tol_px) / (Lv + 1e-12))

    in_box = (
        (alpha >= -mu) & (alpha <= 1.0 + mu) &
        (beta  >= -mv) & (beta  <= 1.0 + mv)
    )

    nu0 = int(np.clip(np.rint(Lu / pitch), int(min_steps), 10_000))
    nv0 = int(np.clip(np.rint(Lv / pitch), int(min_steps), 10_000))

    best_keep = None
    best = (nu0, nv0, -1)

    for nu in (max(min_steps, nu0 - 1), nu0, nu0 + 1):
        for nv in (max(min_steps, nv0 - 1), nv0, nv0 + 1):
            du = Lu / float(nu)
            dv = Lv / float(nv)

            au = alpha * Lu
            bv = beta * Lv

            ku = np.rint(au / du)
            kv = np.rint(bv / dv)

            ru = np.abs(au - ku * du)
            rv = np.abs(bv - kv * dv)

            keep_tmp = in_box & (ru <= tol_px) & (rv <= tol_px)
            score = int(np.count_nonzero(keep_tmp))

            if score > best[2]:
                best = (int(nu), int(nv), score)
                best_keep = keep_tmp

    keep = best_keep
    if keep is None:
        raise RuntimeError("ROI gating failed unexpectedly.")

    roi_idx_local_all = np.flatnonzero(keep).astype(np.int64)

    alpha_roi = alpha[roi_idx_local_all]
    beta_roi = beta[roi_idx_local_all]

    nu = int(best[0])  # along TL -> TR
    nv = int(best[1])  # along TL -> BL

    # canonical row-major board ordering:
    # first row TL -> TR, then next row, ..., last row BL -> BR
    j_board = np.rint(alpha_roi * nu).astype(np.int32)
    i_board = np.rint(beta_roi * nv).astype(np.int32)

    j_board = np.clip(j_board, 0, nu)
    i_board = np.clip(i_board, 0, nv)

    order = np.lexsort((j_board, i_board))
    roi_idx_local = roi_idx_local_all[order]
    uv_xray = xy[roi_idx_local].astype(np.float64)

    if map_back is None:
        roi_idx = roi_idx_local
    else:
        roi_idx = map_back[roi_idx_local].astype(np.int64)

    dbg = dict(
        pitch=float(pitch),
        Lu=float(Lu),
        Lv=float(Lv),
        nu0=int(nu0),
        nv0=int(nv0),
        nu=int(nu),
        nv=int(nv),
        tol_px=float(tol_px),
        in_box=int(np.count_nonzero(in_box)),
        keep=int(np.count_nonzero(keep)),
        gate_tol_pitch=float(gate_tol_pitch),
        anchor_idx=np.asarray(anchor_idx, dtype=int).tolist(),
        anchor_role=["TL", "TR", "BL"],
        grid_i=i_board[order].tolist(),
        grid_j=j_board[order].tolist(),
        uv_board_order=uv_xray.tolist(),
    )

    return uv_xray, roi_idx, dbg


# ============================================================
# Main
# ============================================================

def main() -> None:
    path = _pick_image_path()
    if not path:
        print("No image selected.")
        return

    img_path = Path(path)

    # RAW image — no rotation
    img_raw = _load_xray_gray(path)
    Himg, Wimg = img_raw.shape[:2]
    print(f"Loaded RAW image: {Wimg}x{Himg}")

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
    gate_tol_pitch = 0.40

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
        print("Select 3 anchors in BOARD semantics (TL, TR, BL).")
        print("RMB undo. ESC reset selection. Q quit.")
        refresh()

    def finalize_homography() -> None:
        nonlocal roi_uv, roi_idx, uv_corr

        assert circles is not None
        assert len(selected_idx) == 3

        margin_px = 1.1 * float(pick_radius_px)

        uv_xray_, roi_idx_, dbg = build_uv_xray_board_order(
            circles=circles,
            anchor_idx=selected_idx,
            margin_px=margin_px,
            gate_tol_pitch=gate_tol_pitch,
            min_steps=2,
        )
        roi_uv = np.asarray(uv_xray_, dtype=np.float64).reshape(-1, 2)
        roi_idx = np.asarray(roi_idx_, dtype=np.int64).reshape(-1)

        print(
            f"[ROI] keep={dbg['keep']}  in_box={dbg['in_box']}  "
            f"pitch={dbg['pitch']:.3f}  "
            f"nu0={dbg['nu0']} nv0={dbg['nv0']} -> nu={dbg['nu']} nv={dbg['nv']}"
        )

        board_xyz = build_board_xyz_canonical(
            nu=int(dbg["nu"]),
            nv=int(dbg["nv"]),
            pitch_mm=pitch_mm,
        )
        XY = np.asarray(board_xyz[:, :2], dtype=np.float64).reshape(-1, 2)
        uv_raw = np.asarray(roi_uv, dtype=np.float64).reshape(-1, 2)

        # Save correspondences in RAW coordinates, but in canonical BOARD order
        corr_path = img_path.with_suffix("").with_name(
            img_path.with_suffix("").name + "__corr.npz"
        )
        np.savez(
            corr_path,
            XY=XY.astype(np.float64),
            uv=uv_raw.astype(np.float64),
        )
        print(f"[OK] saved correspondences (RAW coords, board order) -> {corr_path}")

        uv_corr = uv_raw.copy()

        if uv_raw.shape[0] < 4:
            raise RuntimeError(f"Need at least 4 correspondences for homography, got {uv_raw.shape[0]}.")

        # optional txt dumps
        uv_txt = img_path.with_suffix("").with_name(img_path.with_suffix("").name + "__uv.txt")
        xy_txt = img_path.with_suffix("").with_name(img_path.with_suffix("").name + "__XY.txt")
        #_save_uv_txt(uv_txt, uv_raw)
        #_save_xy_txt(xy_txt, XY)

        # Homography in RAW image coordinates
        H = estimate_homography_dlt(uv_raw, XY)

        uv_proj = project_homography(H, XY)
        finite = (
            np.isfinite(uv_raw[:, 0]) & np.isfinite(uv_raw[:, 1]) &
            np.isfinite(uv_proj[:, 0]) & np.isfinite(uv_proj[:, 1])
        )
        if np.any(finite):
            e = np.linalg.norm(uv_proj[finite] - uv_raw[finite], axis=1)
        else:
            e = np.empty((0,), dtype=np.float64)

        npz_path = img_path.with_suffix("").with_name(img_path.with_suffix("").name + "__H.npz")
        np.savez(npz_path, H=H)
        print(f"[OK] saved homography (RAW coords) -> {npz_path}")

        mean_e, med_e, rmse_e = homography_reproj_stats(H, XY, uv_raw)

        np.set_printoptions(precision=6, suppress=True)
        print("\n================= Homography (DLT, RAW image coords) =================")
        print("N points:", uv_raw.shape[0])
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
                    print("  Anchor 1 = TL (BOARD semantics)")
                elif len(selected_idx) == 2:
                    print("  Anchor 2 = TR (BOARD semantics)")
                elif len(selected_idx) == 3:
                    print("  Anchor 3 = BL (BOARD semantics)")
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

    print("RAW image — no rotation.")
    print("LMB: first click runs detection, subsequent clicks select anchors.")
    print("Select anchors in BOARD semantics:")
    print("  1) TL")
    print("  2) TR")
    print("  3) BL")
    print("RMB: undo anchor. ESC: reset selection. Q: quit.")
    print("Saved uv / H are in RAW image coordinates.")
    print("uv is stored in canonical BOARD row-major order.")
    print("=> K_x calibration must use these RAW-coord homographies.")
    print("=> Later pose estimation should use the SAME board ordering.")
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