# -*- coding: utf-8 -*-
"""
test_homography_intrinsics_xray_uv_transform.py

Build X-ray homography/intrinsics correspondences in a consistent
X-ray WORKING IMAGE SPACE.

Main change compared to test_homography_intrinsics_xray_order.py:
- We do NOT reorder points with j_xray = nu - j_cam.
- Instead, we keep canonical board order and transform the measured RAW UVs
  into a horizontally flipped X-ray working image space:

      u_work = W - 1 - u_raw
      v_work = v_raw

Goal:
      board_xyz_canonical[k] <-> uv_xray_working[k]

The RAW image is only used for detection and anchor selection.
The saved UVs and homography live in the transformed working image space.
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

from overlay.tools.xray_marker_selection import (
    run_xray_marker_detection,
)

from overlay.tools.homography import (
    estimate_homography,
    homography_reproj_stats,
    project_homography,
    build_board_xyz_canonical,
)


WIN_RAW = "RAW X-ray selection (LMB detect/select, RMB undo, ESC reset, Q quit)"
WIN_WORK = "X-ray WORKING SPACE preview"


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
# UV transform
# ============================================================

def _flip_uv_horizontal(
    uv: np.ndarray,
    *,
    width: int,
) -> np.ndarray:
    """
    Transform RAW X-ray image coordinates into the horizontally flipped
    X-ray working image space.

    RAW:
        u_raw increases to the right in the original detector image.

    WORKING:
        u_work = W - 1 - u_raw
        v_work = v_raw
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    out = uv.copy()
    out[:, 0] = float(width - 1) - out[:, 0]
    return out


def _flip_circles_horizontal(
    circles: Optional[np.ndarray],
    *,
    width: int,
) -> Optional[np.ndarray]:
    """
    Transform detected circles from RAW image space to WORKING image space.
    """
    if circles is None:
        return None

    c = np.asarray(circles, dtype=np.float64).reshape(-1, 3).copy()
    c[:, 0] = float(width - 1) - c[:, 0]
    return c


def _flip_image_horizontal(img_gray: np.ndarray) -> np.ndarray:
    """
    Create the visual working-space image.
    """
    return cv2.flip(img_gray, 1)


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

    roi_r = max(3.0, 0.35 * float(pick_radius_px))
    circle_r = int(round(roi_r))
    cross_r = int(round(0.6 * roi_r))

    if circles is not None and len(circles) > 0:
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
            if k < 0 or k >= len(circles):
                continue
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
# Local ROI helpers
# ============================================================

def _solve_alpha_beta(
    P: np.ndarray,
    p0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Solve alpha,beta in:

        P = p0 + alpha*u + beta*v

    for each P.
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)

    A = np.stack([u, v], axis=1)

    det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    if abs(det) < 1e-9:
        raise ValueError("Degenerate ROI basis: u and v are nearly colinear.")

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
    nu: int,
    nv: int,
    tol_px: float,
) -> np.ndarray:
    """
    Lattice proximity gate in pixel units along u and v.
    """
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


def _check_anchor_orientation(
    p_tl: np.ndarray,
    p_tr: np.ndarray,
    p_bl: np.ndarray,
) -> tuple[bool, str | None, dict]:
    """
    Check expected anchor orientation in RAW X-ray image.

    The user selects anchors in CAMERA/TOP-VIEW semantics:
        1) TL
        2) TR
        3) BL

    Because the RAW X-ray observes the board from the opposite side,
    this often appears horizontally mirrored in the RAW image.

    For the current setup, the expected RAW orientation is usually:
        u = TR - TL  -> u_x < 0
        v = BL - TL  -> v_y < 0
        cross_z      -> > 0
    """
    u = p_tr - p_tl
    v = p_bl - p_tl

    ux = float(u[0])
    uy = float(u[1])
    vx = float(v[0])
    vy = float(v[1])

    cross_z = float(ux * vy - uy * vx)

    cond_ux = ux < 0.0
    cond_vy = vy < 0.0
    cond_cross = cross_z > 0.0

    dbg = dict(
        ux=ux,
        uy=uy,
        vx=vx,
        vy=vy,
        cross_z=cross_z,
        cond_ux=cond_ux,
        cond_vy=cond_vy,
        cond_cross=cond_cross,
    )

    if abs(cross_z) < 1e-9:
        return (
            False,
            "The selected anchor markers are nearly collinear.",
            dbg,
        )

    ok = cond_ux and cond_vy and cond_cross

    if ok:
        return True, None, dbg

    return (
        False,
        "The selected anchors do not match the expected RAW X-ray orientation.\n"
        "This may still be okay if the image orientation/setup changed, but check carefully.",
        dbg,
    )


def compute_roi_from_grid_uv_transform(
    circles: np.ndarray,
    anchor_idx: Sequence[int],
    *,
    image_width: int,
    margin_px: float,
    gate_tol_pitch: float = 0.40,
    min_steps: int = 2,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute marker ROI from circle detections using 3 ordered anchor indices.

    Anchors:
        [TL, TR, BL] in CAMERA/TOP-VIEW semantics.

    Important:
    - The ROI is detected and gated in RAW image space.
    - The final ordering remains canonical board order:
          i_cam, j_cam
    - No j_xray = nu - j_cam reordering is applied.
    - Instead, the measured RAW UV coordinates are transformed into the
      horizontally flipped X-ray working image space.

    Returns
    -------
    uv_working : np.ndarray
        ROI points in transformed X-ray working image space, shape (N,2).

    roi_idx : np.ndarray
        Indices of selected ROI detections in original circles array.

    dbg : dict
        Debug information.
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

    orientation_ok, orientation_warning, orientation_dbg = _check_anchor_orientation(
        p_tl,
        p_tr,
        p_bl,
    )

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
        (alpha >= -mu)
        & (alpha <= 1.0 + mu)
        & (beta >= -mv)
        & (beta <= 1.0 + mv)
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

    # Canonical board/grid indices in camera/top-view semantics.
    # This is kept as-is.
    j_cam = np.rint(alpha_roi * nu).astype(np.int32)
    i_cam = np.rint(beta_roi * nv).astype(np.int32)

    j_cam = np.clip(j_cam, 0, nu)
    i_cam = np.clip(i_cam, 0, nv)

    # Canonical board order:
    # row-major over i_cam, j_cam.
    order_canonical = np.lexsort((j_cam, i_cam))

    roi_idx_local = roi_idx_local_all[order_canonical]

    grid_i = i_cam[order_canonical].astype(np.int32)
    grid_j = j_cam[order_canonical].astype(np.int32)

    uv_raw_ordered = xy[roi_idx_local].astype(np.float64)

    # Real pixel-space transform:
    # RAW X-ray image space -> horizontally flipped X-ray working image space.
    uv_working = _flip_uv_horizontal(
        uv_raw_ordered,
        width=int(image_width),
    )

    if map_back is None:
        roi_idx = roi_idx_local
    else:
        roi_idx = map_back[roi_idx_local].astype(np.int64)

    debug_npz_path = "xray_marker_selection_debug_uv_transform.npz"
    debug_npz_saved = False
    debug_npz_error = None

    try:
        np.savez(
            debug_npz_path,
            uv_raw_ordered=uv_raw_ordered,
            uv_working=uv_working,
            roi_idx=roi_idx,
            anchor_idx=np.asarray(anchor_idx, dtype=int),
            anchor_idx_effective=np.asarray(idx_eff, dtype=int),
            grid_i=grid_i,
            grid_j=grid_j,
            image_width=np.array(int(image_width), dtype=np.int32),
            uv_transform=np.array("horizontal_flip", dtype="<U32"),
            semantic_reordering=np.array(False, dtype=bool),
        )
        debug_npz_saved = True
    except Exception as e:
        debug_npz_error = str(e)

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

        grid_i=grid_i.tolist(),
        grid_j=grid_j.tolist(),

        orientation_ok=bool(orientation_ok),
        orientation_warning=orientation_warning,
        orientation_ux=float(orientation_dbg["ux"]),
        orientation_uy=float(orientation_dbg["uy"]),
        orientation_vx=float(orientation_dbg["vx"]),
        orientation_vy=float(orientation_dbg["vy"]),
        orientation_cross_z=float(orientation_dbg["cross_z"]),
        orientation_cond_ux=bool(orientation_dbg["cond_ux"]),
        orientation_cond_vy=bool(orientation_dbg["cond_vy"]),
        orientation_cond_cross=bool(orientation_dbg["cond_cross"]),

        image_width=int(image_width),
        uv_transform="horizontal_flip",
        uv_transform_formula="u_work = W - 1 - u_raw, v_work = v_raw",
        semantic_reordering=False,

        debug_uv_raw_ordered=uv_raw_ordered.tolist(),
        debug_uv_working=uv_working.tolist(),

        debug_npz_path=debug_npz_path,
        debug_npz_saved=bool(debug_npz_saved),
        debug_npz_error=debug_npz_error,
    )

    return uv_working, roi_idx, dbg


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
    print()
    print("UV TRANSFORM mode:")
    print("  - detection and anchor selection happen on RAW image")
    print("  - ROI support/gating happens in RAW image space")
    print("  - final UVs are transformed into X-ray WORKING image space")
    print("  - transform: u_work = W - 1 - u_raw, v_work = v_raw")
    print("  - NO semantic j_xray = nu - j_cam reordering is applied")
    print()

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

    roi_uv_raw_for_display: Optional[np.ndarray] = None
    roi_idx: Optional[np.ndarray] = None

    uv_working: Optional[np.ndarray] = None
    circles_working: Optional[np.ndarray] = None

    detected = False

    def refresh_raw() -> None:
        overlay = _render_overlay(
            img_raw,
            circles,
            pick_radius_px=pick_radius_px,
            selected_idx=selected_idx,
            roi_uv=roi_uv_raw_for_display,
            corr_uv=None,
        )
        cv2.imshow(WIN_RAW, overlay)

    def show_working_preview() -> None:
        if uv_working is None:
            return

        img_work = _flip_image_horizontal(img_raw)

        overlay = _render_overlay(
            img_work,
            circles_working,
            pick_radius_px=pick_radius_px,
            selected_idx=[],
            roi_uv=None,
            corr_uv=uv_working,
        )

        cv2.imshow(WIN_WORK, overlay)

    def run_detection() -> None:
        nonlocal circles
        nonlocal circles_working
        nonlocal pick_radius_px
        nonlocal detected
        nonlocal selected_idx
        nonlocal roi_uv_raw_for_display
        nonlocal roi_idx
        nonlocal uv_working

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
            circles_working = None
            detected = False
            refresh_raw()
            return

        circles = np.asarray(res.circles, dtype=np.float64).reshape(-1, 3)
        circles_working = _flip_circles_horizontal(
            circles,
            width=Wimg,
        )

        r = circles[:, 2]
        r = r[np.isfinite(r)]

        if r.size:
            marker_radius_px = float(np.median(r))
            pick_radius_px = 0.6 * marker_radius_px
        else:
            pick_radius_px = 20.0

        selected_idx = []
        roi_uv_raw_for_display = None
        roi_idx = None
        uv_working = None
        detected = True

        print(f"Detection done: {len(circles)} circles on RAW image.")
        print()
        print("Select 3 anchors in CAMERA/TOP-VIEW semantics:")
        print("  1) TL")
        print("  2) TR")
        print("  3) BL")
        print()
        print("After ROI extraction, UVs will be transformed into working space.")
        print()

        refresh_raw()

    def finalize_homography() -> None:
        nonlocal roi_uv_raw_for_display
        nonlocal roi_idx
        nonlocal uv_working

        assert circles is not None
        assert len(selected_idx) == 3

        margin_px = 1.1 * float(pick_radius_px)

        uv_working_, roi_idx_, dbg = compute_roi_from_grid_uv_transform(
            circles=circles,
            anchor_idx=selected_idx,
            image_width=Wimg,
            margin_px=margin_px,
            gate_tol_pitch=GATE_TOL_PITCH,
            min_steps=2,
        )

        uv_working = np.asarray(uv_working_, dtype=np.float64).reshape(-1, 2)
        roi_idx = np.asarray(roi_idx_, dtype=np.int64).reshape(-1)

        roi_uv_raw_for_display = circles[roi_idx, :2].astype(np.float64)

        print(
            f"[ROI UV TRANSFORM] keep={dbg['keep']}  in_box={dbg['in_box']}  "
            f"pitch={dbg['pitch']:.3f}  "
            f"nu0={dbg['nu0']} nv0={dbg['nv0']} -> nu={dbg['nu']} nv={dbg['nv']}"
        )

        if not dbg.get("orientation_ok", True):
            print()
            print("[WARNING] Anchor orientation check failed:")
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
        uv_used = np.asarray(uv_working, dtype=np.float64).reshape(-1, 2)

        if XY.shape[0] != uv_used.shape[0]:
            raise RuntimeError(
                f"Point count mismatch: board XY has {XY.shape[0]} points, "
                f"uv_working has {uv_used.shape[0]} points."
            )

        corr_path = img_path.with_suffix("").with_name(
            img_path.with_suffix("").name + "__corr_XRAY_UV_TRANSFORM.npz"
        )

        np.savez(
            corr_path,
            XY=XY.astype(np.float64),
            uv=uv_used.astype(np.float64),

            uv_raw_ordered=np.asarray(dbg["debug_uv_raw_ordered"], dtype=np.float64),
            uv_working=np.asarray(dbg["debug_uv_working"], dtype=np.float64),

            homography_method=np.array(HOMOGRAPHY_METHOD),
            grid_i=np.asarray(dbg["grid_i"], dtype=np.int32),
            grid_j=np.asarray(dbg["grid_j"], dtype=np.int32),

            uv_space=np.array("XRAY_WORKING_FLIPPED_UV", dtype="<U64"),
            raw_image_space=np.array("XRAY_RAW", dtype="<U64"),
            uv_transform=np.array("horizontal_flip", dtype="<U32"),
            uv_transform_formula=np.array("u_work = W - 1 - u_raw, v_work = v_raw", dtype="<U64"),

            semantic_reordering=np.array(False, dtype=bool),
            no_j_xray_reordering=np.array(True, dtype=bool),
            raw_image_not_flipped_for_detection=np.array(True, dtype=bool),

            image_width=np.array(Wimg, dtype=np.int32),
            image_height=np.array(Himg, dtype=np.int32),

            anchor_idx=np.asarray(selected_idx, dtype=np.int32),
        )

        print(f"[OK] saved correspondences in WORKING UV space -> {corr_path}")

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
            img_path.with_suffix("").name + "__H_XRAY_UV_TRANSFORM.npz"
        )

        np.savez(
            H_path,
            H=H,
            homography_method=np.array(HOMOGRAPHY_METHOD),

            uv_space=np.array("XRAY_WORKING_FLIPPED_UV", dtype="<U64"),
            raw_image_space=np.array("XRAY_RAW", dtype="<U64"),
            uv_transform=np.array("horizontal_flip", dtype="<U32"),
            uv_transform_formula=np.array("u_work = W - 1 - u_raw, v_work = v_raw", dtype="<U64"),

            semantic_reordering=np.array(False, dtype=bool),
            no_j_xray_reordering=np.array(True, dtype=bool),

            image_width=np.array(Wimg, dtype=np.int32),
            image_height=np.array(Himg, dtype=np.int32),
        )

        print(f"[OK] saved homography in WORKING UV space -> {H_path}")

        mean_e, med_e, rmse_e = homography_reproj_stats(H, XY, uv_used)

        np.set_printoptions(precision=6, suppress=True)

        print()
        print("================= Homography UV TRANSFORM =================")
        print(f"method      : {HOMOGRAPHY_METHOD}")
        print(f"UV space    : XRAY_WORKING_FLIPPED_UV")
        print(f"transform   : u_work = W - 1 - u_raw, v_work = v_raw")
        print(f"reordering  : False")
        print(f"N points    : {uv_used.shape[0]}")
        print("H =")
        print(H)
        print(
            f"Reprojection error [px]: "
            f"mean={mean_e:.3f}, median={med_e:.3f}, rmse={rmse_e:.3f}"
        )

        if e.size:
            print(
                f"Reproj err vector: min={float(np.min(e)):.3f}  "
                f"max={float(np.max(e)):.3f}  p95={float(np.percentile(e, 95)):.3f}"
            )

        print("============================================================")
        print()

        refresh_raw()
        show_working_preview()

    def on_mouse(event, x, y, flags, userdata):
        nonlocal detected
        nonlocal selected_idx
        nonlocal roi_uv_raw_for_display
        nonlocal roi_idx
        nonlocal uv_working

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

                roi_uv_raw_for_display = None
                roi_idx = None
                uv_working = None

                print("Selected anchors idx:", selected_idx)

                if len(selected_idx) == 1:
                    print("  Anchor 1 = TL")
                elif len(selected_idx) == 2:
                    print("  Anchor 2 = TR")
                elif len(selected_idx) == 3:
                    print("  Anchor 3 = BL")

                refresh_raw()

                if len(selected_idx) == 3:
                    finalize_homography()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(selected_idx) > 0:
                selected_idx.pop()

                roi_uv_raw_for_display = None
                roi_idx = None
                uv_working = None

                print("Undo. Selected anchors idx:", selected_idx)

                refresh_raw()

    cv2.namedWindow(WIN_RAW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_RAW, Wimg, Himg)
    cv2.setMouseCallback(WIN_RAW, on_mouse)

    print("RAW image — no rotation, no visual flip for selection.")
    print("LMB: first click runs detection, subsequent clicks select anchors.")
    print()
    print("Select anchors in CAMERA/TOP-VIEW semantics:")
    print("  1) TL")
    print("  2) TR")
    print("  3) BL")
    print()
    print("Saved files:")
    print("  *__corr_XRAY_UV_TRANSFORM.npz")
    print("  *__H_XRAY_UV_TRANSFORM.npz")
    print()
    print("Q: quit. ESC: reset selection. RMB: undo anchor.")
    print()

    refresh_raw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        if key == 27:
            if detected:
                selected_idx = []
                roi_uv_raw_for_display = None
                roi_idx = None
                uv_working = None

                print("Selection reset.")

                refresh_raw()

                try:
                    cv2.destroyWindow(WIN_WORK)
                except Exception:
                    pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()