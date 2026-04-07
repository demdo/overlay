# -*- coding: utf-8 -*-
"""
test_pose_cam2x_ippe_topdown_toggle_crop_xy_only.py

Strikt XY-basierte Kandidatenansicht auf einem X-ray-ähnlichen Top-down-Hintergrund.

Was gezeichnet wird
-------------------
- cam  = Kamera-Ursprung        -> (x_c, y_c) = (0, 0)
- sol0 = X-ray-Ursprung cand0   -> (t_xc0[0], t_xc0[1])
- sol1 = X-ray-Ursprung cand1   -> (t_xc1[0], t_xc1[1])

Wichtig
-------
Die Markerpositionen werden NUR aus den rohen Kamera-x/y-Werten bestimmt.
z geht NICHT in die Marker-Positionen ein.

Der Hintergrund bleibt eine X-ray-ähnliche Top-down-View der Ebene.
Die Marker sind also bewusst eine separate XY-Darstellung auf diesem Hintergrund.

Bedienung
---------
t   -> toggle original / warped
z   -> im Warp-Modus ROI wählen, finalen Crop erzeugen und Marker einzeichnen
q   -> quit
ESC -> quit
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tracking.pose_solvers import solve_pose


# ============================================================
# Qt helpers
# ============================================================

def _qt() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _pick_file(title: str, filt: str) -> Path | None:
    _qt()
    path, _ = QFileDialog.getOpenFileName(None, title, "", filt)
    return Path(path) if path else None


# ============================================================
# Hardcoded K_rgb
# ============================================================

K_rgb = np.array(
    [
        [1360.41301,   0.0,       976.230766],
        [0.0,       1361.74342,   547.474129],
        [0.0,          0.0,         1.0     ],
    ],
    dtype=np.float64,
)


# ============================================================
# IO
# ============================================================

def load_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not load image: {path}")
    return img


def load_xyz(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as z:
        for k in ("points_xyz_camera_filt", "points_xyz_camera"):
            if k in z.files:
                pts = np.asarray(z[k], dtype=np.float64)
                if pts.ndim != 2 or pts.shape[1] != 3:
                    raise RuntimeError(f"XYZ array '{k}' has invalid shape: {pts.shape}")
                if np.mean(np.abs(pts)) < 10.0:
                    pts = pts * 1000.0
                return pts
    raise RuntimeError("No XYZ found in NPZ.")


def load_kx(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as z:
        for key in ("K", "Kx", "K_xray"):
            if key in z.files:
                K = np.asarray(z[key], dtype=np.float64)
                if K.shape != (3, 3):
                    raise RuntimeError(f"{key} has invalid shape: {K.shape}")
                return K
    raise RuntimeError("No key 'K', 'Kx', or 'K_xray' found in NPZ.")


def load_uv_xray(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as z:
        if "uv_xray" in z.files:
            uv = np.asarray(z["uv_xray"], dtype=np.float64)
            if uv.ndim == 2 and uv.shape[1] == 2:
                return uv

        for key in z.files:
            arr = np.asarray(z[key], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr

    raise RuntimeError("No (N,2) uv array found in NPZ.")


# ============================================================
# Geometry helpers
# ============================================================

def fit_plane_svd(points_xyz_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_xyz_mm, dtype=np.float64)
    center = np.mean(pts, axis=0)
    A = pts - center
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    n = vt[-1]
    n /= np.linalg.norm(n)
    return n, center


def ensure_plane_faces_camera(n: np.ndarray, center: np.ndarray) -> np.ndarray:
    to_cam = -center
    if float(np.dot(n, to_cam)) < 0.0:
        n = -n
    return n


def project_vector_onto_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.asarray(n, dtype=np.float64).reshape(3)
    n /= np.linalg.norm(n)
    return v - np.dot(v, n) * n


def build_plane_visual_basis(
    n_c: np.ndarray,
    *,
    flip_up: bool = False,
    flip_180: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x_cam = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    e_x = project_vector_onto_plane(x_cam, n_c)
    if np.linalg.norm(e_x) < 1e-9:
        raise RuntimeError("Projected x_cam onto plane is degenerate.")
    e_x /= np.linalg.norm(e_x)

    e_y = np.cross(n_c, e_x)
    if np.linalg.norm(e_y) < 1e-9:
        raise RuntimeError("Failed to construct e_y.")
    e_y /= np.linalg.norm(e_y)

    if flip_up:
        e_y = -e_y

    if flip_180:
        e_x = -e_x
        e_y = -e_y

    return e_x, e_y


def build_plane_patch_corners_mm(
    center_c: np.ndarray,
    e_x: np.ndarray,
    e_y: np.ndarray,
    out_w_px: int,
    out_h_px: int,
    mm_per_px: float,
) -> np.ndarray:
    half_w_mm = 0.5 * out_w_px * mm_per_px
    half_h_mm = 0.5 * out_h_px * mm_per_px

    p_tl = center_c + half_h_mm * e_y - half_w_mm * e_x
    p_tr = center_c + half_h_mm * e_y + half_w_mm * e_x
    p_br = center_c - half_h_mm * e_y + half_w_mm * e_x
    p_bl = center_c - half_h_mm * e_y - half_w_mm * e_x
    return np.vstack([p_tl, p_tr, p_br, p_bl])


def project_points(K: np.ndarray, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    uvw = (K @ P.T).T
    return uvw[:, :2] / uvw[:, 2:3]


# ============================================================
# Strict XY mapping for markers
# ============================================================

def point3d_to_xy_only(
    P_c: np.ndarray,
    xy_center_c: np.ndarray,
) -> np.ndarray:
    """
    Marker mapping uses ONLY raw camera x/y.
    z is ignored completely.
    """
    P = np.asarray(P_c, dtype=np.float64).reshape(3)
    C = np.asarray(xy_center_c, dtype=np.float64).reshape(2)
    x_mm = float(P[0] - C[0])
    y_mm = float(P[1] - C[1])
    return np.array([x_mm, y_mm], dtype=np.float64)


def xy_only_to_uv(
    xy_mm: np.ndarray,
    out_w_px: int,
    out_h_px: int,
    mm_per_px: float,
) -> np.ndarray:
    u = (out_w_px * 0.5) + (float(xy_mm[0]) / mm_per_px)
    v = (out_h_px * 0.5) - (float(xy_mm[1]) / mm_per_px)
    return np.array([u, v], dtype=np.float64)


# ============================================================
# Pose helpers
# ============================================================

def rvec_to_R(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_pose_cx_to_xc(R_cx: np.ndarray, t_cx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_xc = R_cx.T
    t_xc = -R_xc @ np.asarray(t_cx, dtype=np.float64).reshape(3)
    return R_xc, t_xc


def pose_to_dict(rvec, tvec, cand_idx, reproj_mean_px=None) -> dict:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)

    R_cx = rvec_to_R(rvec)
    R_xc, t_xc = invert_pose_cx_to_xc(R_cx, tvec)

    return {
        "candidate_index": int(cand_idx),
        "rvec": rvec,
        "tvec": tvec,
        "R_cx": R_cx,
        "T_cx": make_T(R_cx, tvec),
        "R_xc": R_xc,
        "t_xc": t_xc,
        "T_xc": make_T(R_xc, t_xc),
        "reproj_mean_px": None if reproj_mean_px is None else float(reproj_mean_px),
    }


def solve_both_candidates(
    xyz_mm: np.ndarray,
    uv_xray: np.ndarray,
    K_xray: np.ndarray,
) -> tuple[dict | None, dict | None]:
    res = solve_pose(
        object_points_xyz=xyz_mm,
        image_points_uv=uv_xray,
        K=K_xray,
        pose_method="ippe",
        refine_with_iterative=False,
    )

    chosen = pose_to_dict(
        res.rvec,
        res.tvec,
        getattr(res, "candidate_index", 0),
        getattr(res, "reproj_mean_px", None),
    )

    other = None
    if hasattr(res, "all_candidates") and res.all_candidates is not None:
        other_idx = 1 - chosen["candidate_index"]
        if other_idx < len(res.all_candidates):
            c = res.all_candidates[other_idx]
            other = pose_to_dict(
                c.rvec,
                c.tvec,
                other_idx,
                getattr(c, "reproj_mean_px", None),
            )

    if chosen["candidate_index"] == 0:
        cand0, cand1 = chosen, other
    else:
        cand0, cand1 = other, chosen

    return cand0, cand1


def print_pose_block(name: str, pose: dict | None) -> None:
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    if pose is None:
        print("None")
        return

    print(f"candidate_index : {pose['candidate_index']}")
    print(f"reproj_mean_px  : {pose['reproj_mean_px']}")
    print()

    print("--- T_cx (camera -> xray) ---")
    print("rvec [rad]  :", np.array2string(pose["rvec"], precision=6, suppress_small=False))
    print(
        "tvec [mm]   : tx={:+10.3f}  ty={:+10.3f}  tz={:+10.3f}".format(
            pose["tvec"][0], pose["tvec"][1], pose["tvec"][2]
        )
    )
    print("R_cx =")
    print(np.array2string(pose["R_cx"], precision=6, suppress_small=False))
    print("T_cx =")
    print(np.array2string(pose["T_cx"], precision=6, suppress_small=False))
    print()

    print("--- T_xc (xray -> camera) ---")
    print(
        "t_xc [mm]   : tx={:+10.3f}  ty={:+10.3f}  tz={:+10.3f}".format(
            pose["t_xc"][0], pose["t_xc"][1], pose["t_xc"][2]
        )
    )
    print("R_xc =")
    print(np.array2string(pose["R_xc"], precision=6, suppress_small=False))
    print("T_xc =")
    print(np.array2string(pose["T_xc"], precision=6, suppress_small=False))


# ============================================================
# Drawing helpers
# ============================================================

def draw_hq_cross_circle(
    img: np.ndarray,
    pt_xy: tuple[float, float],
    color_bgr: tuple[int, int, int],
    circle_radius_px: float = 7.0,
    cross_half_len_px: float = 5.0,
    thickness_px: float = 2.0,
    upscale: int = 8,
) -> None:
    x, y = float(pt_xy[0]), float(pt_xy[1])

    pad = int(np.ceil((circle_radius_px + cross_half_len_px + thickness_px + 4) * upscale))
    cx = int(round(x * upscale))
    cy = int(round(y * upscale))

    x0 = cx - pad
    y0 = cy - pad
    x1 = cx + pad + 1
    y1 = cy + pad + 1

    H, W = img.shape[:2]
    rx0 = max(0, x0)
    ry0 = max(0, y0)
    rx1 = min(W * upscale, x1)
    ry1 = min(H * upscale, y1)

    if rx0 >= rx1 or ry0 >= ry1:
        return

    bw = rx1 - rx0
    bh = ry1 - ry0

    canvas = np.zeros((bh, bw, 3), dtype=np.uint8)
    alpha = np.zeros((bh, bw), dtype=np.uint8)

    local_cx = cx - rx0
    local_cy = cy - ry0

    r = int(round(circle_radius_px * upscale))
    l = int(round(cross_half_len_px * upscale))
    t = max(1, int(round(thickness_px * upscale)))

    cv2.circle(alpha, (local_cx, local_cy), r, 255, t, cv2.LINE_AA)
    cv2.line(alpha, (local_cx - l, local_cy), (local_cx + l, local_cy), 255, t, cv2.LINE_AA)
    cv2.line(alpha, (local_cx, local_cy - l), (local_cx, local_cy + l), 255, t, cv2.LINE_AA)

    canvas[:, :] = np.array(color_bgr, dtype=np.uint8)

    small_canvas = cv2.resize(
        canvas,
        ((bw + upscale - 1) // upscale, (bh + upscale - 1) // upscale),
        interpolation=cv2.INTER_AREA,
    )
    small_alpha = cv2.resize(
        alpha,
        ((bw + upscale - 1) // upscale, (bh + upscale - 1) // upscale),
        interpolation=cv2.INTER_AREA,
    )

    sx0 = rx0 // upscale
    sy0 = ry0 // upscale
    sx1 = sx0 + small_canvas.shape[1]
    sy1 = sy0 + small_canvas.shape[0]

    sx1 = min(sx1, W)
    sy1 = min(sy1, H)

    small_canvas = small_canvas[: sy1 - sy0, : sx1 - sx0]
    small_alpha = small_alpha[: sy1 - sy0, : sx1 - sx0]

    roi = img[sy0:sy1, sx0:sx1].astype(np.float32)
    ov = small_canvas.astype(np.float32)
    a = (small_alpha.astype(np.float32) / 255.0)[..., None]

    out = a * ov + (1.0 - a) * roi
    img[sy0:sy1, sx0:sx1] = np.clip(out, 0, 255).astype(np.uint8)


def crop_point(uv: np.ndarray, roi_xywh: tuple[int, int, int, int]) -> np.ndarray:
    x, y, _, _ = roi_xywh
    return np.array([float(uv[0]) - x, float(uv[1]) - y], dtype=np.float64)


def save_crop_images(raw_crop: np.ndarray, annotated_crop: np.ndarray) -> tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path.cwd()
    raw_path = out_dir / f"topview_crop_raw_{ts}.png"
    ann_path = out_dir / f"topview_crop_annotated_{ts}.png"

    ok1 = cv2.imwrite(str(raw_path), raw_crop)
    ok2 = cv2.imwrite(str(ann_path), annotated_crop)
    if not ok1 or not ok2:
        raise RuntimeError("Failed to save crop image(s).")

    return raw_path, ann_path


# ============================================================
# Main
# ============================================================

def main() -> None:
    img_path = _pick_file("Select camera image PNG", "PNG image (*.png)")
    if img_path is None:
        print("No image selected.")
        return

    xyz_path = _pick_file("Select XYZ camera NPZ", "NPZ (*.npz)")
    if xyz_path is None:
        print("No XYZ NPZ selected.")
        return

    kx_path = _pick_file("Select K_xray NPZ", "NPZ (*.npz)")
    if kx_path is None:
        print("No K_xray NPZ selected.")
        return

    uvx_path = _pick_file("Select uv_xray NPZ", "NPZ (*.npz)")
    if uvx_path is None:
        print("No uv_xray NPZ selected.")
        return

    img = load_image_bgr(img_path)
    xyz_mm = load_xyz(xyz_path)
    K_x = load_kx(kx_path)
    uv_xray = load_uv_xray(uvx_path)

    if xyz_mm.shape[0] != uv_xray.shape[0]:
        raise RuntimeError(
            f"Point count mismatch: xyz={xyz_mm.shape[0]}, uv_xray={uv_xray.shape[0]}"
        )

    # --------------------------------------------------------
    # Plane and top-down background basis
    # --------------------------------------------------------
    n_c, plane_center_c = fit_plane_svd(xyz_mm)
    n_c = ensure_plane_faces_camera(n_c, plane_center_c)

    e_x, e_y = build_plane_visual_basis(
        n_c,
        flip_up=False,
        flip_180=False,
    )

    # --------------------------------------------------------
    # IPPE
    # --------------------------------------------------------
    cand0, cand1 = solve_both_candidates(xyz_mm, uv_xray, K_x)

    print_pose_block("RESULTS FOR CANDIDATE 0", cand0)
    print_pose_block("RESULTS FOR CANDIDATE 1", cand1)

    # --------------------------------------------------------
    # Marker points: STRICT XY ONLY
    # --------------------------------------------------------
    cam_point_c = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    sol0_point_c = None if cand0 is None else cand0["t_xc"].copy()
    sol1_point_c = None if cand1 is None else cand1["t_xc"].copy()

    print("\n" + "=" * 70)
    print("3D POINTS USED FOR MARKERS [camera frame, mm]")
    print("=" * 70)
    print("cam_point_c  =", np.array2string(cam_point_c, precision=3, suppress_small=False))
    if sol0_point_c is not None:
        print("sol0_point_c =", np.array2string(sol0_point_c, precision=3, suppress_small=False))
    if sol1_point_c is not None:
        print("sol1_point_c =", np.array2string(sol1_point_c, precision=3, suppress_small=False))

    # --------------------------------------------------------
    # Marker framing: use ONLY raw x/y from camera frame
    # --------------------------------------------------------
    xy_points = [cam_point_c[:2][None, :]]
    if sol0_point_c is not None:
        xy_points.append(sol0_point_c[:2][None, :])
    if sol1_point_c is not None:
        xy_points.append(sol1_point_c[:2][None, :])

    all_marker_xy_raw = np.vstack(xy_points)

    x_min = float(np.min(all_marker_xy_raw[:, 0]))
    x_max = float(np.max(all_marker_xy_raw[:, 0]))
    y_min = float(np.min(all_marker_xy_raw[:, 1]))
    y_max = float(np.max(all_marker_xy_raw[:, 1]))

    margin_mm = 120.0
    x_min -= margin_mm
    x_max += margin_mm
    y_min -= margin_mm
    y_max += margin_mm

    xy_center_c = np.array(
        [0.5 * (x_min + x_max), 0.5 * (y_min + y_max)],
        dtype=np.float64,
    )

    mm_per_px = 2.0
    out_w_px = int(np.ceil((x_max - x_min) / mm_per_px))
    out_h_px = int(np.ceil((y_max - y_min) / mm_per_px))

    out_w_px = max(out_w_px, 700)
    out_h_px = max(out_h_px, 700)

    print("\n" + "=" * 70)
    print("XY-ONLY FRAMING")
    print("=" * 70)
    print(f"x_range_mm   : [{x_min:.3f}, {x_max:.3f}]")
    print(f"y_range_mm   : [{y_min:.3f}, {y_max:.3f}]")
    print(f"xy_center_c  : {xy_center_c}")
    print(f"out_w_px     : {out_w_px}")
    print(f"out_h_px     : {out_h_px}")
    print(f"mm_per_px    : {mm_per_px}")

    # --------------------------------------------------------
    # Background warp
    # We keep the background centered around the plane center.
    # Just make it large enough to match the XY-only marker canvas.
    # --------------------------------------------------------
    patch_center_c = plane_center_c.copy()

    corners_c = build_plane_patch_corners_mm(
        center_c=patch_center_c,
        e_x=e_x,
        e_y=e_y,
        out_w_px=out_w_px,
        out_h_px=out_h_px,
        mm_per_px=mm_per_px,
    )

    src = project_points(K_rgb, corners_c).astype(np.float32)
    dst = np.array(
        [
            [0, 0],
            [out_w_px - 1, 0],
            [out_w_px - 1, out_h_px - 1],
            [0, out_h_px - 1],
        ],
        dtype=np.float32,
    )

    H_warp = cv2.getPerspectiveTransform(src, dst)
    vis_warp = cv2.warpPerspective(img, H_warp, (out_w_px, out_h_px))

    # --------------------------------------------------------
    # Marker positions: strict XY-only
    # --------------------------------------------------------
    cam_xy = point3d_to_xy_only(cam_point_c, xy_center_c)
    sol0_xy = None if sol0_point_c is None else point3d_to_xy_only(sol0_point_c, xy_center_c)
    sol1_xy = None if sol1_point_c is None else point3d_to_xy_only(sol1_point_c, xy_center_c)

    uv_cam = xy_only_to_uv(cam_xy, out_w_px, out_h_px, mm_per_px)
    uv_sol0 = None if sol0_xy is None else xy_only_to_uv(sol0_xy, out_w_px, out_h_px, mm_per_px)
    uv_sol1 = None if sol1_xy is None else xy_only_to_uv(sol1_xy, out_w_px, out_h_px, mm_per_px)

    print("\n" + "=" * 70)
    print("XY-ONLY COORDS")
    print("=" * 70)
    print("cam_xy  =", np.array2string(cam_xy, precision=3, suppress_small=False))
    print("sol0_xy =", None if sol0_xy is None else np.array2string(sol0_xy, precision=3, suppress_small=False))
    print("sol1_xy =", None if sol1_xy is None else np.array2string(sol1_xy, precision=3, suppress_small=False))

    print("\n" + "=" * 70)
    print("TOP-DOWN UV")
    print("=" * 70)
    print("uv_cam  =", np.array2string(uv_cam, precision=3, suppress_small=False))
    print("uv_sol0 =", None if uv_sol0 is None else np.array2string(uv_sol0, precision=3, suppress_small=False))
    print("uv_sol1 =", None if uv_sol1 is None else np.array2string(uv_sol1, precision=3, suppress_small=False))

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------
    vis_orig = img.copy()
    cv2.polylines(vis_orig, [np.int32(src)], True, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        vis_orig,
        "View: ORIGINAL  |  t=toggle",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    vis_warp_preview = vis_warp.copy()

    for uv, color, label in [
        (uv_cam,  (255, 0, 0), "cam"),
        (uv_sol0, (0, 0, 255), "sol0"),
        (uv_sol1, (0, 255, 0), "sol1"),
    ]:
        if uv is not None:
            u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
            if 0 <= u < out_w_px and 0 <= v < out_h_px:
                cv2.circle(vis_warp_preview, (u, v), 8, color, 2, cv2.LINE_AA)
                cv2.drawMarker(
                    vis_warp_preview,
                    (u, v),
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=14,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
                cv2.putText(
                    vis_warp_preview,
                    label,
                    (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

    cv2.putText(
        vis_warp_preview,
        "View: WARPED  |  z=select final crop  |  t=toggle",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    show_warp = False
    win = "IPPE top-down XY-only"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(win, vis_warp_preview if show_warp else vis_orig)
        k = cv2.waitKey(0) & 0xFF

        if k in (27, ord("q")):
            break

        elif k == ord("t"):
            show_warp = not show_warp

        elif k == ord("z") and show_warp:
            roi = cv2.selectROI("Select final crop", vis_warp_preview, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select final crop")

            x, y, w, h = [int(v) for v in roi]
            if w <= 0 or h <= 0:
                print("[INFO] Empty ROI, cancelled.")
                continue

            pts = []
            labels = []

            if uv_cam is not None:
                pts.append(np.asarray(uv_cam, dtype=np.float64))
                labels.append("cam")
            if uv_sol0 is not None:
                pts.append(np.asarray(uv_sol0, dtype=np.float64))
                labels.append("sol0")
            if uv_sol1 is not None:
                pts.append(np.asarray(uv_sol1, dtype=np.float64))
                labels.append("sol1")

            outside = []
            for lbl, pt in zip(labels, pts):
                u, v = float(pt[0]), float(pt[1])
                inside = (x <= u < x + w) and (y <= v < y + h)
                if not inside:
                    outside.append((lbl, pt.copy()))

            if outside:
                print("\n[WARN] The following points are outside the selected ROI:")
                for lbl, pt in outside:
                    print(f"  {lbl:5s}: uv = [{pt[0]:.3f}, {pt[1]:.3f}]")
                print("      -> Crop will hide these point(s).")

            auto_expand_roi = True
            roi_pad_px = 20

            if auto_expand_roi and len(pts) > 0:
                P = np.vstack(pts)
                px_min = int(np.floor(np.min(P[:, 0]))) - roi_pad_px
                py_min = int(np.floor(np.min(P[:, 1]))) - roi_pad_px
                px_max = int(np.ceil(np.max(P[:, 0]))) + roi_pad_px
                py_max = int(np.ceil(np.max(P[:, 1]))) + roi_pad_px

                new_x0 = min(x, px_min)
                new_y0 = min(y, py_min)
                new_x1 = max(x + w, px_max)
                new_y1 = max(y + h, py_max)

                H_img, W_img = vis_warp.shape[:2]
                new_x0 = max(0, new_x0)
                new_y0 = max(0, new_y0)
                new_x1 = min(W_img, new_x1)
                new_y1 = min(H_img, new_y1)

                x, y = new_x0, new_y0
                w, h = new_x1 - new_x0, new_y1 - new_y0

                print(f"[INFO] ROI expanded to include all points: x={x}, y={y}, w={w}, h={h}")

            raw_crop_small = vis_warp[y:y + h, x:x + w].copy()

            crop_scale = 4
            raw_crop = cv2.resize(
                raw_crop_small,
                None,
                fx=crop_scale,
                fy=crop_scale,
                interpolation=cv2.INTER_CUBIC,
            )
            annotated_crop = raw_crop.copy()

            marker_radius = 7.0
            marker_cross = 5.0
            marker_thickness = 2.0

            if uv_cam is not None:
                uv_cam_crop = crop_point(uv_cam, (x, y, w, h)) * crop_scale
                draw_hq_cross_circle(
                    annotated_crop,
                    (uv_cam_crop[0], uv_cam_crop[1]),
                    (255, 0, 0),
                    circle_radius_px=marker_radius,
                    cross_half_len_px=marker_cross,
                    thickness_px=marker_thickness,
                    upscale=8,
                )

            if uv_sol0 is not None:
                uv_sol0_crop = crop_point(uv_sol0, (x, y, w, h)) * crop_scale
                draw_hq_cross_circle(
                    annotated_crop,
                    (uv_sol0_crop[0], uv_sol0_crop[1]),
                    (0, 0, 255),
                    circle_radius_px=marker_radius,
                    cross_half_len_px=marker_cross,
                    thickness_px=marker_thickness,
                    upscale=8,
                )

            if uv_sol1 is not None:
                uv_sol1_crop = crop_point(uv_sol1, (x, y, w, h)) * crop_scale
                draw_hq_cross_circle(
                    annotated_crop,
                    (uv_sol1_crop[0], uv_sol1_crop[1]),
                    (0, 255, 0),
                    circle_radius_px=marker_radius,
                    cross_half_len_px=marker_cross,
                    thickness_px=marker_thickness,
                    upscale=8,
                )

            raw_path, ann_path = save_crop_images(raw_crop, annotated_crop)
            print(f"[INFO] Saved raw crop      : {raw_path}")
            print(f"[INFO] Saved annotated crop: {ann_path}")

            cv2.imshow("Final crop raw", raw_crop)
            cv2.imshow("Final crop annotated", annotated_crop)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()