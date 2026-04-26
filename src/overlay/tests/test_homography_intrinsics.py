# -*- coding: utf-8 -*-
"""
test_homography_intrinsics.py

Same as before, but with local compute_roi_from_grid_no_flip(...).

Important:
- overlay.tools.xray_marker_selection.compute_roi_from_grid(...) is NOT modified.
- This script uses a local copy without the X-ray left-right flip.
- Therefore:
      board_xyz_canonical[k] <-> uv_raw_no_flip[k]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Sequence, Tuple

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.blob_detection import (
    HoughCircleParams,
    estimate_pitch_nn,
)
from overlay.tools.xray_marker_selection import run_xray_marker_detection
from overlay.tools.homography import (
    estimate_homography,
    homography_reproj_stats,
    project_homography,
    build_board_xyz_canonical,
)

WIN = "test_homography NO FLIP (LMB detect/select, RMB undo, ESC reset, Q quit)"


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
# Local ROI helper: NO FLIP
# ============================================================

def _solve_alpha_beta(
    P: np.ndarray,
    p0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    A = np.stack([u, v], axis=1)

    det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    if abs(det) < 1e-9:
        raise ValueError("Degenerate ROI basis (u and v nearly colinear).")

    Ainv = np.linalg.inv(A)
    D = (np.asarray(P, float) - np.asarray(p0, float)[None, :]).T
    return (Ainv @ D).T


def _lattice_gate_keep(
    *,
    alpha: np.ndarray,
    beta: np.ndarray,
    in_box: np.ndarray,
    Lu: float,
    Lv: float,
    pitch: float,
    nu: int,
    nv: int,
    tol_px: float,
) -> np.ndarray:
    du = Lu / float(nu)
    dv = Lv / float(nv)

    au = alpha * Lu
    bv = beta * Lv

    ku = np.rint(au / du)
    kv = np.rint(bv / dv)

    ru = np.abs(au - ku * du)
    rv = np.abs(bv - kv * dv)

    on = (ru <= tol_px) & (rv <= tol_px)
    return in_box & on


def compute_roi_from_grid_no_flip(
    circles: np.ndarray,
    anchor_idx: Sequence[int],
    *,
    margin_px: float,
    gate_tol_pitch: float = 0.40,
    min_steps: int = 2,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Local copy of compute_roi_from_grid(...), but WITHOUT X-ray-specific flip.

    The selected anchors are interpreted as:
        [TL, TR, BL]

    The affine model is:
        p = p_tl + alpha*u + beta*v

    Ordering:
        i = beta direction
        j = alpha direction

    NO:
        j_xray = nu - j_cam

    Instead:
        j_final = j_cam
        i_final = i_cam
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
    if not np.isfinite(xy).all():
        finite = np.isfinite(xy).all(axis=1)
        xy = xy[finite]
        map_back = np.flatnonzero(finite)
    else:
        map_back = None

    if xy.shape[0] < 3:
        raise ValueError("Not enough finite circle centers for ROI computation.")

    if map_back is None:
        idx_eff = idx
    else:
        inv = {int(old): int(new) for new, old in enumerate(map_back)}
        idx_eff = np.array([inv[int(i)] for i in idx], dtype=int)

    p_tl = xy[idx_eff[0], :]
    p_tr = xy[idx_eff[1], :]
    p_bl = xy[idx_eff[2], :]

    u = p_tr - p_tl
    v = p_bl - p_tl

    Lu = float(np.linalg.norm(u))
    Lv = float(np.linalg.norm(v))

    if Lu <= 1e-9 or Lv <= 1e-9:
        raise ValueError("Anchor geometry is degenerate.")

    pitch = float(estimate_pitch_nn(xy))
    if (not np.isfinite(pitch)) or pitch <= 1e-6:
        pitch = 10.0

    AB = _solve_alpha_beta(xy, p_tl, u, v)
    alpha = AB[:, 0]
    beta = AB[:, 1]

    tol_px = float(gate_tol_pitch) * float(pitch)
    mu = float((margin_px + tol_px) / (Lu + 1e-12))
    mv = float((margin_px + tol_px) / (Lv + 1e-12))

    in_box = (
        (alpha >= -mu) & (alpha <= 1.0 + mu) &
        (beta >= -mv) & (beta <= 1.0 + mv)
    )

    nu0 = int(np.clip(np.rint(Lu / pitch), int(min_steps), 10_000))
    nv0 = int(np.clip(np.rint(Lv / pitch), int(min_steps), 10_000))

    best_keep = None
    best = (nu0, nv0, -1)

    for nu in (max(min_steps, nu0 - 1), nu0, nu0 + 1):
        for nv in (max(min_steps, nv0 - 1), nv0, nv0 + 1):
            keep_tmp = _lattice_gate_keep(
                alpha=alpha,
                beta=beta,
                in_box=in_box,
                Lu=Lu,
                Lv=Lv,
                pitch=pitch,
                nu=int(nu),
                nv=int(nv),
                tol_px=tol_px,
            )
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

    nu = int(best[0])
    nv = int(best[1])

    j_cam = np.rint(alpha_roi * nu).astype(np.int32)
    i_cam = np.rint(beta_roi * nv).astype(np.int32)

    j_cam = np.clip(j_cam, 0, nu)
    i_cam = np.clip(i_cam, 0, nv)

    # ------------------------------------------------------------
    # NO FLIP HERE
    # ------------------------------------------------------------
    j_final = j_cam
    i_final = i_cam

    order_final = np.lexsort((j_final, i_final))
    roi_idx_local = roi_idx_local_all[order_final]

    grid_i_final = i_final[order_final].astype(np.int32)
    grid_j_final = j_final[order_final].astype(np.int32)

    uv_final = xy[roi_idx_local].astype(np.float64)

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
        nu=int(best[0]),
        nv=int(best[1]),
        mu=float(mu),
        mv=float(mv),
        tol_px=float(tol_px),
        in_box=int(np.count_nonzero(in_box)),
        keep=int(np.count_nonzero(keep)),
        gate_tol_pitch=float(gate_tol_pitch),
        anchor_idx=np.asarray(anchor_idx, dtype=int).tolist(),
        anchor_idx_effective=np.asarray(idx_eff, dtype=int).tolist(),
        anchor_role=["TL", "TR", "BL"],
        grid_i=grid_i_final.tolist(),
        grid_j=grid_j_final.tolist(),
        debug_uv_final=uv_final.tolist(),
        no_flip=True,
    )

    return uv_final, roi_idx, dbg


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

    for (x, y) in xy:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cv2.circle(out, (int(round(x)), int(round(y))), circle_r, (0, 255, 0), 2, cv2.LINE_AA)

    for k in selected_idx:
        x, y, _r = circles[k]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        _draw_cross(out, int(round(x)), int(round(y)), cross_r, color=(255, 255, 0), thick=2)

    if roi_uv is not None and len(roi_uv) > 0:
        uv = np.asarray(roi_uv, dtype=np.float64).reshape(-1, 2)
        for (u, v) in uv:
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            _draw_cross(out, int(round(u)), int(round(v)), cross_r, color=(0, 0, 255), thick=2)

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
    print("NO FLIP mode: ROI ordering is directly from selected anchors.")

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
        print("Select 3 anchors: TL, TR, BL.")
        print("This script will NOT apply any X-ray ordering flip.")
        refresh()

    def finalize_homography() -> None:
        nonlocal roi_uv, roi_idx, uv_corr

        assert circles is not None
        assert len(selected_idx) == 3

        margin_px = 1.1 * float(pick_radius_px)

        uv_xray_, roi_idx_, dbg = compute_roi_from_grid_no_flip(
            circles=circles,
            anchor_idx=selected_idx,
            margin_px=margin_px,
            gate_tol_pitch=GATE_TOL_PITCH,
            min_steps=2,
        )

        roi_uv = np.asarray(uv_xray_, dtype=np.float64).reshape(-1, 2)
        roi_idx = np.asarray(roi_idx_, dtype=np.int64).reshape(-1)

        print(
            f"[ROI NO FLIP] keep={dbg['keep']}  in_box={dbg['in_box']}  "
            f"pitch={dbg['pitch']:.3f}  "
            f"nu0={dbg['nu0']} nv0={dbg['nv0']} -> nu={dbg['nu']} nv={dbg['nv']}"
        )

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
            img_path.with_suffix("").name + "__corr_NO_FLIP.npz"
        )
        np.savez(
            corr_path,
            XY=XY.astype(np.float64),
            uv=uv_used.astype(np.float64),
            homography_method=np.array(HOMOGRAPHY_METHOD),
            grid_i=np.asarray(dbg["grid_i"], dtype=np.int32),
            grid_j=np.asarray(dbg["grid_j"], dtype=np.int32),
            no_flip=np.array(True, dtype=bool),
        )
        print(f"[OK] saved correspondences NO FLIP -> {corr_path}")

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
            np.isfinite(uv_used[:, 0]) & np.isfinite(uv_used[:, 1]) &
            np.isfinite(uv_proj[:, 0]) & np.isfinite(uv_proj[:, 1])
        )

        if np.any(finite):
            e = np.linalg.norm(uv_proj[finite] - uv_used[finite], axis=1)
        else:
            e = np.empty((0,), dtype=np.float64)

        npz_path = img_path.with_suffix("").with_name(
            img_path.with_suffix("").name + "__H_NO_FLIP.npz"
        )
        np.savez(
            npz_path,
            H=H,
            homography_method=np.array(HOMOGRAPHY_METHOD),
            no_flip=np.array(True, dtype=bool),
        )
        print(f"[OK] saved homography NO FLIP -> {npz_path}")

        mean_e, med_e, rmse_e = homography_reproj_stats(H, XY, uv_used)

        np.set_printoptions(precision=6, suppress=True)
        print(f"\n================= Homography ({HOMOGRAPHY_METHOD}, RAW NO FLIP) =================")
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

    print("RAW image — no rotation.")
    print("LMB: first click runs detection, subsequent clicks select anchors.")
    print("Select anchors:")
    print("  1) TL")
    print("  2) TR")
    print("  3) BL")
    print("NO X-ray flip is applied in this script.")
    print("Saved files:")
    print("  *__corr_NO_FLIP.npz")
    print("  *__H_NO_FLIP.npz")
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