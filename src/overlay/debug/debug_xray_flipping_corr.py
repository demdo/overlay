from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.homography import (
    build_board_xyz_canonical,
    estimate_plane_induced_homography,
)
from overlay.tools.warp import blend_xray_overlay


# ============================================================
# Config
# ============================================================

PITCH_MM = 2.54
STEPS_PER_EDGE = 10
DIST_ZERO = np.zeros((8, 1), dtype=np.float64)
ALPHA = 0.45


# ============================================================
# Qt helpers
# ============================================================

def select_file(title: str, file_filter: str) -> Path | None:
    app = QApplication.instance()
    owns_app = app is None

    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(None, title, "", file_filter)

    if owns_app:
        app.quit()

    return Path(path) if path else None


# ============================================================
# Transform helpers
# ============================================================

def as_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rotation_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def rotation_deviation_deg(R_ref: np.ndarray, R: np.ndarray) -> float:
    R_rel = (
        np.asarray(R_ref, dtype=np.float64).reshape(3, 3).T
        @ np.asarray(R, dtype=np.float64).reshape(3, 3)
    )
    return rotation_angle_deg(R_rel)


def print_matrix(label: str, M: np.ndarray) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    with np.printoptions(precision=6, suppress=True):
        print(np.asarray(M, dtype=np.float64))


def print_pose(label: str, T: np.ndarray, unit: str = "mm") -> None:
    R = T[:3, :3]
    t = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)

    print_matrix(label, T)
    print(f"t [{unit}] = [{t[0]:+.6f}, {t[1]:+.6f}, {t[2]:+.6f}]")
    print(f"rvec      = [{rvec[0,0]:+.6f}, {rvec[1,0]:+.6f}, {rvec[2,0]:+.6f}]")
    print(f"z-axis    = [{R[0,2]:+.6f}, {R[1,2]:+.6f}, {R[2,2]:+.6f}]")


# ============================================================
# Load helpers
# ============================================================

def load_K_from_npz(path: Path, preferred_keys: list[str]) -> np.ndarray:
    data = np.load(path, allow_pickle=True)

    for key in preferred_keys:
        if key in data.files:
            K = np.asarray(data[key], dtype=np.float64)
            if K.shape == (3, 3):
                return K

    for key in data.files:
        arr = np.asarray(data[key])
        if arr.shape == (3, 3):
            print(f"[INFO] {path.name}: using fallback K key '{key}'")
            return np.asarray(arr, dtype=np.float64)

    raise KeyError(f"No 3x3 K matrix found in {path}")


def load_rgb_corners_from_overlay(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "checkerboard_corners_uv" not in data.files:
        raise KeyError("Overlay NPZ does not contain 'checkerboard_corners_uv'.")

    corners = np.asarray(data["checkerboard_corners_uv"], dtype=np.float64).reshape(-1, 2)

    if corners.shape != (3, 2):
        raise ValueError(
            f"Expected checkerboard_corners_uv with shape (3,2), got {corners.shape}"
        )

    print("[OK] RGB anchors loaded from checkerboard_corners_uv [TL, TR, BL]")
    return corners


def interpolate_rgb_points_from_corners(corners_uv: np.ndarray) -> np.ndarray:
    p_tl, p_tr, p_bl = np.asarray(corners_uv, dtype=np.float64).reshape(3, 2)

    step_x = (p_tr - p_tl) / float(STEPS_PER_EDGE)
    step_y = (p_bl - p_tl) / float(STEPS_PER_EDGE)

    pts2d_rgb = np.array(
        [
            p_tl + alpha * step_x + beta * step_y
            for beta in range(STEPS_PER_EDGE + 1)
            for alpha in range(STEPS_PER_EDGE + 1)
        ],
        dtype=np.float64,
    )

    print(f"[OK] RGB points interpolated: {pts2d_rgb.shape}")
    return pts2d_rgb


def load_xray_points_from_uv_npz(path: Path) -> tuple[np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)

    if "points_uv" not in data.files:
        raise KeyError("X-ray UV NPZ must contain key 'points_uv'.")

    pts = np.asarray(data["points_uv"], dtype=np.float64).reshape(-1, 2)

    meta = {}
    for key in data.files:
        arr = data[key]
        if arr.shape == ():
            try:
                meta[key] = arr.item()
            except Exception:
                meta[key] = arr

    return pts, meta


def load_depth_points_from_overlay(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "xray_points_xyz_c" not in data.files:
        raise KeyError("Overlay NPZ does not contain 'xray_points_xyz_c'.")

    pts = np.asarray(data["xray_points_xyz_c"], dtype=np.float64).reshape(-1, 3)

    if pts.shape[0] != 121:
        raise ValueError(f"Expected 121 depth/camera 3D points, got {pts.shape[0]}.")

    print("[OK] depth/camera 3D points loaded from xray_points_xyz_c")
    return pts


def load_rgb_image_from_overlay(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, str]:
    if "snapshot_rgb_bgr" in data.files:
        return np.asarray(data["snapshot_rgb_bgr"], dtype=np.uint8), "snapshot_rgb_bgr"

    if "snapshot_rgb_with_tip_bgr" in data.files:
        return np.asarray(data["snapshot_rgb_with_tip_bgr"], dtype=np.uint8), "snapshot_rgb_with_tip_bgr"

    raise KeyError(
        "Overlay NPZ does not contain 'snapshot_rgb_bgr' or "
        "'snapshot_rgb_with_tip_bgr'."
    )


# ============================================================
# Reprojection / IPPE helpers
# ============================================================

def compute_reprojection_errors(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    uv_proj, _ = cv2.projectPoints(
        np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        np.asarray(dist, dtype=np.float64).reshape(-1, 1),
    )
    uv_proj = uv_proj.reshape(-1, 2)
    img = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    return np.linalg.norm(uv_proj - img, axis=1)


def solve_ippe_candidates(
    object_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    *,
    label: str,
) -> list[dict]:
    obj = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)

    ok, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
        objectPoints=obj,
        imagePoints=img,
        cameraMatrix=K,
        distCoeffs=DIST_ZERO,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not ok or rvecs is None or tvecs is None:
        raise RuntimeError(f"{label}: solvePnPGeneric(IPPE) failed.")

    candidates = []

    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

        R, _ = cv2.Rodrigues(rvec)
        T = as_transform(R, tvec.reshape(3))

        err = compute_reprojection_errors(obj, img, rvec, tvec, K, DIST_ZERO)

        err_cv = (
            float(np.asarray(reproj_errs[i]).reshape(-1)[0])
            if reproj_errs is not None
            else float("nan")
        )

        candidates.append(
            {
                "index": int(i),
                "rvec": rvec,
                "tvec": tvec,
                "R": R,
                "T": T,
                "reproj_mean": float(np.mean(err)),
                "reproj_median": float(np.median(err)),
                "reproj_max": float(np.max(err)),
                "reproj_cv": err_cv,
            }
        )

    return candidates


# ============================================================
# RGB depth-based selection
# ============================================================

def rigid_fit_kabsch(board_xyz_mm: np.ndarray, cam_xyz_mm: np.ndarray) -> dict:
    A = np.asarray(board_xyz_mm, dtype=np.float64).reshape(-1, 3)
    B = np.asarray(cam_xyz_mm, dtype=np.float64).reshape(-1, 3)

    if A.shape != B.shape:
        raise ValueError(f"Kabsch shape mismatch: {A.shape} vs {B.shape}")

    ca = np.mean(A, axis=0)
    cb = np.mean(B, axis=0)

    A0 = A - ca
    B0 = B - cb

    H = A0.T @ B0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = cb - R @ ca
    T = as_transform(R, t)

    B_hat = (R @ A.T).T + t
    errs = np.linalg.norm(B_hat - B, axis=1)

    return {
        "R": R,
        "t": t,
        "T": T,
        "mean_mm": float(np.mean(errs)),
        "median_mm": float(np.median(errs)),
        "rms_mm": float(np.sqrt(np.mean(errs ** 2))),
        "max_mm": float(np.max(errs)),
        "det_R": float(np.linalg.det(R)),
    }


def trust_region_from_depth_rms(
    rms_mm: float,
    *,
    half_board_extent_mm: float = 12.7,
    trans_scale: float = 100.0,
    rot_scale: float = 50.0,
    trans_min_mm: float = 1.0,
    trans_max_mm: float = 10.0,
    rot_min_deg: float = 1.0,
    rot_max_deg: float = 15.0,
) -> dict:
    gamma_t_raw = trans_scale * float(rms_mm)
    gamma_t = min(max(gamma_t_raw, trans_min_mm), trans_max_mm)

    gamma_r_raw = float(np.degrees(rot_scale * float(rms_mm) / float(half_board_extent_mm)))
    gamma_r = min(max(gamma_r_raw, rot_min_deg), rot_max_deg)

    return {
        "gamma_t_mm": float(gamma_t),
        "gamma_r_deg": float(gamma_r),
        "gamma_t_mm_raw": float(gamma_t_raw),
        "gamma_r_deg_raw": float(gamma_r_raw),
    }


def select_rgb_candidate_with_depth(
    rgb_candidates: list[dict],
    T_bc_depth: np.ndarray,
    gamma_t_mm: float,
    gamma_r_deg: float,
) -> tuple[int, list[dict]]:
    scores = []

    for cand in rgb_candidates:
        T_ippe = cand["T"]

        dt = float(np.linalg.norm(T_ippe[:3, 3] - T_bc_depth[:3, 3]))
        dr = rotation_deviation_deg(T_bc_depth[:3, :3], T_ippe[:3, :3])

        feasible = bool(dt <= gamma_t_mm and dr <= gamma_r_deg)
        score = (dt / gamma_t_mm) ** 2 + (dr / gamma_r_deg) ** 2

        scores.append(
            {
                "idx": cand["index"],
                "delta_t_mm": dt,
                "delta_r_deg": dr,
                "feasible": feasible,
                "score": float(score),
                "reproj_mean": cand["reproj_mean"],
            }
        )

    print("\n" + "=" * 72)
    print("RGB IPPE SELECTION — depth trust region")
    print("=" * 72)
    print(f"gamma_t_mm  = {gamma_t_mm:.6f}")
    print(f"gamma_r_deg = {gamma_r_deg:.6f}")

    for s in scores:
        print(f"\nRGB candidate {s['idx']}:")
        print(f"  delta_t_mm   = {s['delta_t_mm']:.6f}")
        print(f"  delta_r_deg  = {s['delta_r_deg']:.6f}")
        print(f"  feasible     = {s['feasible']}")
        print(f"  score        = {s['score']:.6f}")
        print(f"  reproj mean  = {s['reproj_mean']:.6f}")

    feasible = [s for s in scores if s["feasible"]]

    if len(feasible) == 1:
        return int(feasible[0]["idx"]), scores

    if len(feasible) == 2:
        best = min(feasible, key=lambda s: s["score"])
        return int(best["idx"]), scores

    print("\n[WARN] No RGB candidate inside trust region. Falling back to lowest depth score.")
    best = min(scores, key=lambda s: s["score"])
    return int(best["idx"]), scores


# ============================================================
# X-ray selection
# ============================================================

def select_xray_candidate_antiparallel(xray_candidates: list[dict]) -> tuple[int, list[dict]]:
    z_b = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    infos = []

    for cand in xray_candidates:
        R_bx = cand["T"][:3, :3]
        R_xb = R_bx.T
        z_x_in_b = R_xb[:, 2]
        dot_z = float(np.dot(z_b, z_x_in_b))

        infos.append(
            {
                "idx": cand["index"],
                "z_x_in_b": z_x_in_b,
                "dot_z": dot_z,
                "reproj_mean": cand["reproj_mean"],
            }
        )

    print("\n" + "=" * 72)
    print("X-RAY IPPE SELECTION — z-axis antiparallel rule")
    print("=" * 72)

    for info in infos:
        print(f"\nX-ray candidate {info['idx']}:")
        print(f"  z_x_in_b       = {info['z_x_in_b']}")
        print(f"  dot(z_b,z_x)   = {info['dot_z']:+.6f}")
        print(f"  reproj mean px = {info['reproj_mean']:.6f}")

    if abs(infos[0]["dot_z"] - infos[1]["dot_z"]) > 1e-12:
        best = min(infos, key=lambda s: s["dot_z"])
    else:
        best = min(infos, key=lambda s: s["reproj_mean"])

    return int(best["idx"]), infos


# ============================================================
# Homography / depth helpers
# ============================================================

def extract_d_x_from_tip(
    T_cx_m: np.ndarray,
    T_tc_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    T_cx_mm = np.asarray(T_cx_m, dtype=np.float64).reshape(4, 4).copy()
    T_cx_mm[:3, 3] *= 1e3

    T_tc_mm = np.asarray(T_tc_mm, dtype=np.float64).reshape(4, 4)

    T_tx_mm = T_cx_mm @ T_tc_mm
    tip_xyz_x_mm = T_tx_mm[:3, 3].copy()
    d_x_mm = float(tip_xyz_x_mm[2])

    return T_tx_mm, tip_xyz_x_mm, d_x_mm


def compute_and_print_H_xc(
    *,
    label: str,
    T_cx_m: np.ndarray,
    T_tc_mm: np.ndarray,
    K_rgb: np.ndarray,
    K_xray: np.ndarray,
) -> np.ndarray:
    T_cx_m = np.asarray(T_cx_m, dtype=np.float64).reshape(4, 4)
    T_xc_m = invert_transform(T_cx_m)

    R_xc = T_xc_m[:3, :3]
    t_xc = T_xc_m[:3, 3]

    T_tx_mm, tip_xyz_x_mm, d_x_mm = extract_d_x_from_tip(
        T_cx_m=T_cx_m,
        T_tc_mm=T_tc_mm,
    )

    H_xc = estimate_plane_induced_homography(
        K_c=K_rgb,
        R_xc=R_xc,
        t_xc=t_xc,
        K_x=K_xray,
        d_x=d_x_mm,
        normalize=True,
    )

    print("\n" + "=" * 72)
    print(f"H_xc RESULT — {label}")
    print("=" * 72)

    print_pose(f"{label}: T_cx [m]", T_cx_m, unit="m")
    print_pose(f"{label}: T_xc [m]", T_xc_m, unit="m")
    print_matrix(f"{label}: T_tx = tip -> xray [mm]", T_tx_mm)

    print(
        f"{label}: tip_xyz_x_mm = "
        f"[{tip_xyz_x_mm[0]:+.6f}, {tip_xyz_x_mm[1]:+.6f}, {tip_xyz_x_mm[2]:+.6f}]"
    )
    print(f"{label}: d_x_mm = {d_x_mm:+.6f}")

    print_matrix(f"{label}: H_xc = xray image -> RGB image", H_xc)

    return H_xc


# ============================================================
# Difference helper
# ============================================================

def print_transform_difference(
    label: str,
    T_ref_mm: np.ndarray,
    T_test_mm: np.ndarray,
) -> None:
    dT = invert_transform(T_ref_mm) @ T_test_mm
    dt = T_test_mm[:3, 3] - T_ref_mm[:3, 3]
    dR = dT[:3, :3]

    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    print(f"translation delta [mm] = [{dt[0]:+.6f}, {dt[1]:+.6f}, {dt[2]:+.6f}]")
    print(f"translation norm [mm]  = {np.linalg.norm(dt):.6f}")
    print(f"rotation delta [deg]   = {rotation_angle_deg(dR):.6f}")


# ============================================================
# Warp helpers
# ============================================================

def warp_and_save_overlay(
    *,
    label: str,
    overlay_path: Path,
    rgb_bgr: np.ndarray,
    xray_gray_u8: np.ndarray,
    H_xc: np.ndarray,
    alpha: float,
    out_name: str,
) -> np.ndarray:
    out_bgr, _cache = blend_xray_overlay(
        camera_bgr=rgb_bgr,
        xray_gray_u8=xray_gray_u8,
        H_xc=H_xc,
        alpha=alpha,
    )

    out_path = overlay_path.parent / out_name
    cv2.imwrite(str(out_path), out_bgr)

    print(f"[OK] saved {label} overlay -> {out_path}")

    return out_bgr


# ============================================================
# Main
# ============================================================

def main() -> None:
    overlay_path = select_file("Select overlay NPZ", "NPZ files (*.npz);;All files (*)")
    if overlay_path is None:
        return

    xray_uv_path = select_file("Select new X-ray UV NPZ with points_uv", "NPZ files (*.npz);;All files (*)")
    if xray_uv_path is None:
        return

    kx_path = select_file("Select NEW K_xray NPZ", "NPZ files (*.npz);;All files (*)")
    if kx_path is None:
        return

    overlay = np.load(overlay_path, allow_pickle=True)

    if "T_tc" not in overlay.files:
        raise KeyError("Overlay NPZ does not contain 'T_tc' required for d_x extraction.")
    if "xray_gray_u8" not in overlay.files:
        raise KeyError("Overlay NPZ does not contain 'xray_gray_u8'.")

    T_tc_mm = np.asarray(overlay["T_tc"], dtype=np.float64).reshape(4, 4)

    K_rgb = np.asarray(overlay["K_rgb"], dtype=np.float64).reshape(3, 3)
    K_xray = load_K_from_npz(kx_path, ["K", "K_xray", "K_x"])

    rgb_bgr, rgb_key = load_rgb_image_from_overlay(overlay)
    xray_gray_u8 = np.asarray(overlay["xray_gray_u8"], dtype=np.uint8)

    rgb_corners_uv = load_rgb_corners_from_overlay(overlay)
    rgb_points_uv = interpolate_rgb_points_from_corners(rgb_corners_uv)

    xray_points_uv, uv_meta = load_xray_points_from_uv_npz(xray_uv_path)
    cam_points_xyz_m = load_depth_points_from_overlay(overlay)

    print("\nLoaded X-ray UV metadata:")
    for k, v in uv_meta.items():
        print(f"  {k}: {v}")

    if xray_points_uv.shape[0] != 121:
        raise ValueError(f"Expected 121 X-ray points, got {xray_points_uv.shape[0]}.")
    if rgb_points_uv.shape[0] != 121:
        raise ValueError(f"Expected 121 RGB points, got {rgb_points_uv.shape[0]}.")
    if cam_points_xyz_m.shape[0] != 121:
        raise ValueError(f"Expected 121 camera/depth points, got {cam_points_xyz_m.shape[0]}.")

    board_xyz_mm = build_board_xyz_canonical(
        nu=STEPS_PER_EDGE,
        nv=STEPS_PER_EDGE,
        pitch_mm=PITCH_MM,
    )

    cam_points_xyz_mm = cam_points_xyz_m * 1000.0

    print_matrix("K_rgb", K_rgb)
    print_matrix("K_xray", K_xray)
    print_pose("T_tc = tip -> camera [mm]", T_tc_mm)

    print("\nImages:")
    print(f"RGB image key = {rgb_key}")
    print(f"RGB shape     = {rgb_bgr.shape}")
    print(f"X-ray shape   = {xray_gray_u8.shape}")

    print("\nImportant indices:")
    for idx in [0, 10, 110, 120]:
        print(
            f"idx {idx:3d}: "
            f"board=({board_xyz_mm[idx,0]:7.3f}, {board_xyz_mm[idx,1]:7.3f}, {board_xyz_mm[idx,2]:7.3f})  "
            f"rgb=({rgb_points_uv[idx,0]:8.3f}, {rgb_points_uv[idx,1]:8.3f})  "
            f"xray=({xray_points_uv[idx,0]:8.3f}, {xray_points_uv[idx,1]:8.3f})  "
            f"cam3d=({cam_points_xyz_mm[idx,0]:8.3f}, {cam_points_xyz_mm[idx,1]:8.3f}, {cam_points_xyz_mm[idx,2]:8.3f})"
        )

    # ============================================================
    # Depth reference for RGB candidate selection
    # ============================================================

    depth_fit = rigid_fit_kabsch(board_xyz_mm, cam_points_xyz_mm)
    T_bc_depth = depth_fit["T"]

    print("\n" + "=" * 72)
    print("DEPTH REFERENCE T_bc")
    print("=" * 72)
    print_pose("T_bc_depth = board -> camera from Kabsch", T_bc_depth)
    print(f"depth fit mean   = {depth_fit['mean_mm']:.6f} mm")
    print(f"depth fit median = {depth_fit['median_mm']:.6f} mm")
    print(f"depth fit rms    = {depth_fit['rms_mm']:.6f} mm")
    print(f"depth fit max    = {depth_fit['max_mm']:.6f} mm")
    print(f"det(R)           = {depth_fit['det_R']:.6f}")

    tr = trust_region_from_depth_rms(depth_fit["rms_mm"])

    # ============================================================
    # IPPE candidates + selection
    # ============================================================

    rgb_candidates = solve_ippe_candidates(
        board_xyz_mm,
        rgb_points_uv,
        K_rgb,
        label="RGB IPPE",
    )

    xray_candidates = solve_ippe_candidates(
        board_xyz_mm,
        xray_points_uv,
        K_xray,
        label="X-ray IPPE",
    )

    best_rgb_idx, _ = select_rgb_candidate_with_depth(
        rgb_candidates,
        T_bc_depth,
        gamma_t_mm=tr["gamma_t_mm"],
        gamma_r_deg=tr["gamma_r_deg"],
    )

    best_xray_idx, _ = select_xray_candidate_antiparallel(xray_candidates)

    best_rgb = rgb_candidates[best_rgb_idx]
    best_xray = xray_candidates[best_xray_idx]

    T_bc = best_rgb["T"]
    T_bx = best_xray["T"]

    # ============================================================
    # Main hand-eye result: selected RGB IPPE + selected X-ray IPPE
    # ============================================================

    T_cx_mm = T_bx @ invert_transform(T_bc)
    T_cx_m = T_cx_mm.copy()
    T_cx_m[:3, 3] *= 1e-3

    # ============================================================
    # Alternative hand-eye result: depth T_bc + selected X-ray IPPE
    # ============================================================

    T_cx_depth_mm = T_bx @ invert_transform(T_bc_depth)
    T_cx_depth_m = T_cx_depth_mm.copy()
    T_cx_depth_m[:3, 3] *= 1e-3

    # ============================================================
    # Compute plane-induced homographies for both variants
    # ============================================================

    H_xc_ippe = compute_and_print_H_xc(
        label="selected IPPE handeye",
        T_cx_m=T_cx_m,
        T_tc_mm=T_tc_mm,
        K_rgb=K_rgb,
        K_xray=K_xray,
    )

    H_xc_depth = compute_and_print_H_xc(
        label="selected T_bx + depth T_bc",
        T_cx_m=T_cx_depth_m,
        T_tc_mm=T_tc_mm,
        K_rgb=K_rgb,
        K_xray=K_xray,
    )

    # ============================================================
    # Warp X-ray to RGB for both H_xc variants
    # 1) NO X-ray image flip
    # ============================================================

    print("\n" + "=" * 72)
    print("WARP X-RAY TO RGB — NO X-RAY IMAGE FLIP")
    print("=" * 72)
    print(f"alpha       = {ALPHA:.3f}")
    print("xray flip   = False")

    overlay_ippe_bgr = warp_and_save_overlay(
        label="IPPE handeye no flip",
        overlay_path=overlay_path,
        rgb_bgr=rgb_bgr,
        xray_gray_u8=xray_gray_u8,
        H_xc=H_xc_ippe,
        alpha=ALPHA,
        out_name="debug_overlay_H_xc_ippe_no_xray_flip.png",
    )

    overlay_depth_bgr = warp_and_save_overlay(
        label="depth T_bc no flip",
        overlay_path=overlay_path,
        rgb_bgr=rgb_bgr,
        xray_gray_u8=xray_gray_u8,
        H_xc=H_xc_depth,
        alpha=ALPHA,
        out_name="debug_overlay_H_xc_depth_no_xray_flip.png",
    )

    # ============================================================
    # Warp X-ray to RGB for both H_xc variants
    # 2) WITH visually flipped X-ray image
    # IMPORTANT: H_xc is unchanged
    # ============================================================

    print("\n" + "=" * 72)
    print("WARP X-RAY TO RGB — WITH X-RAY IMAGE FLIP")
    print("=" * 72)
    print(f"alpha       = {ALPHA:.3f}")
    print("xray flip   = True")
    print("H_xc        = unchanged")

    xray_gray_u8_flipped = cv2.flip(xray_gray_u8, 1)

    overlay_ippe_flip_bgr = warp_and_save_overlay(
        label="IPPE handeye WITH xray flip",
        overlay_path=overlay_path,
        rgb_bgr=rgb_bgr,
        xray_gray_u8=xray_gray_u8_flipped,
        H_xc=H_xc_ippe,
        alpha=ALPHA,
        out_name="debug_overlay_H_xc_ippe_WITH_xray_flip.png",
    )

    overlay_depth_flip_bgr = warp_and_save_overlay(
        label="depth T_bc WITH xray flip",
        overlay_path=overlay_path,
        rgb_bgr=rgb_bgr,
        xray_gray_u8=xray_gray_u8_flipped,
        H_xc=H_xc_depth,
        alpha=ALPHA,
        out_name="debug_overlay_H_xc_depth_WITH_xray_flip.png",
    )

    # ============================================================
    # Output summary
    # ============================================================

    print("\n" + "=" * 72)
    print("FINAL SELECTED IPPE HAND-EYE RESULT")
    print("=" * 72)
    print(f"selected RGB candidate   = {best_rgb_idx}")
    print(f"selected X-ray candidate = {best_xray_idx}")

    print_pose("Selected T_bc = board -> camera [mm]", T_bc)
    print(
        f"RGB reproj [px]: mean={best_rgb['reproj_mean']:.6f}, "
        f"median={best_rgb['reproj_median']:.6f}, "
        f"max={best_rgb['reproj_max']:.6f}"
    )

    print_pose("Selected T_bx = board -> xray [mm]", T_bx)
    print(
        f"X-ray reproj [px]: mean={best_xray['reproj_mean']:.6f}, "
        f"median={best_xray['reproj_median']:.6f}, "
        f"max={best_xray['reproj_max']:.6f}"
    )

    print_pose("FINAL T_cx = selected T_bx @ inv(selected T_bc) [mm]", T_cx_mm)
    print_pose("FINAL T_cx = selected T_bx @ inv(selected T_bc) [m]", T_cx_m, unit="m")

    print("\n" + "=" * 72)
    print("ALTERNATIVE HAND-EYE USING DEPTH T_bc")
    print("=" * 72)
    print("T_cx_depth = selected T_bx @ inv(T_bc_depth)")

    print_pose("T_cx_depth = camera -> xray [mm]", T_cx_depth_mm)
    print_pose("T_cx_depth = camera -> xray [m]", T_cx_depth_m, unit="m")

    print_transform_difference(
        "DIFFERENCE: T_cx_depth vs selected-IPPE T_cx",
        T_ref_mm=T_cx_mm,
        T_test_mm=T_cx_depth_mm,
    )

    print("\n" + "=" * 72)
    print("H_xc SUMMARY")
    print("=" * 72)
    print_matrix("H_xc_ippe   = selected IPPE handeye", H_xc_ippe)
    print_matrix("H_xc_depth  = selected T_bx + depth T_bc", H_xc_depth)

    cv2.imshow("H_xc IPPE handeye - no xray flip", overlay_ippe_bgr)
    cv2.imshow("H_xc depth T_bc - no xray flip", overlay_depth_bgr)
    cv2.imshow("H_xc IPPE handeye - WITH xray flip", overlay_ippe_flip_bgr)
    cv2.imshow("H_xc depth T_bc - WITH xray flip", overlay_depth_flip_bgr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nDone. Saved four debug PNG overlays.")


if __name__ == "__main__":
    main()