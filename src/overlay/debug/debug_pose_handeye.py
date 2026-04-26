# -*- coding: utf-8 -*-
"""
debug_pose_handeye_affine_uv.py

Load one overlay_preview_pose*.npz file via Qt file dialog,
compute a second affine-regularized X-ray UV set from the stored
xray_points_uv, run hand-eye calibration for both variants, recompute d_x,
compute H_xc exactly like the module, and show both final overlays.

Expected NPZ keys
-----------------
Required:
    - K_xray
    - K_rgb
    - xray_points_uv
    - xray_points_xyz_c
    - checkerboard_corners_uv
    - snapshot_rgb_bgr
    - xray_gray_u8
    - T_tc

Optional:
    - T_cx
    - T_xc
    - alpha
    - tip_uv_c
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.calib.calib_camera_to_xray import calibrate_camera_to_xray
from overlay.calib.calib_xray_to_pointer import extract_depth
from overlay.tools.homography import estimate_plane_induced_homography
from overlay.tools.warp import blend_xray_overlay


# ============================================================
# Config
# ============================================================

PITCH_MM = 2.54
STEPS_PER_EDGE = 10
SAVE_DEBUG_NPZ = True


# ============================================================
# Formatting helpers
# ============================================================

def fmt_vec(v: np.ndarray) -> str:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x:+.6f}" for x in v) + "]"


def fmt_T(T: np.ndarray, indent: str = "    ") -> str:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    lines = []
    for row in T:
        lines.append(indent + "  ".join(f"{x:+.6f}" for x in row))
    return "\n".join(lines)


def rotation_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def compare_transforms(T_ref: np.ndarray, T_test: np.ndarray) -> dict:
    T_ref = np.asarray(T_ref, dtype=np.float64).reshape(4, 4)
    T_test = np.asarray(T_test, dtype=np.float64).reshape(4, 4)

    R_ref = T_ref[:3, :3]
    t_ref = T_ref[:3, 3]

    R_test = T_test[:3, :3]
    t_test = T_test[:3, 3]

    R_rel = R_ref.T @ R_test
    delta_R_deg = rotation_angle_deg(R_rel)
    delta_t = float(np.linalg.norm(t_test - t_ref))

    return {
        "delta_R_deg": delta_R_deg,
        "delta_t": delta_t,
    }


def ensure_bgr_u8(img: np.ndarray, name: str) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H,W,3), got {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def ensure_gray_u8(img: np.ndarray, name: str) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"{name} must have shape (H,W), got {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ============================================================
# Qt file picker
# ============================================================

def pick_npz_file() -> str:
    app = QApplication.instance()
    owns_app = False

    if app is None:
        app = QApplication(sys.argv)
        owns_app = True

    path, _ = QFileDialog.getOpenFileName(
        None,
        "Open overlay preview NPZ",
        "",
        "NumPy archive (*.npz);;All files (*)",
    )

    if owns_app:
        app.quit()

    return path


# ============================================================
# Loading helpers
# ============================================================

def require_key(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing required key '{key}' in NPZ.")
    return data[key]


def load_pose_npz(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = np.load(str(path), allow_pickle=True)

    result = {
        "path": str(path),
        "path_obj": path,
        "K_xray": np.asarray(require_key(data, "K_xray"), dtype=np.float64).reshape(3, 3),
        "K_rgb": np.asarray(require_key(data, "K_rgb"), dtype=np.float64).reshape(3, 3),
        "xray_points_uv": np.asarray(require_key(data, "xray_points_uv"), dtype=np.float64).reshape(-1, 2),
        "xray_points_xyz_c": np.asarray(require_key(data, "xray_points_xyz_c"), dtype=np.float64).reshape(-1, 3),
        "checkerboard_corners_uv": np.asarray(require_key(data, "checkerboard_corners_uv"), dtype=np.float64).reshape(3, 2),
        "snapshot_rgb_bgr": ensure_bgr_u8(require_key(data, "snapshot_rgb_bgr"), "snapshot_rgb_bgr"),
        "xray_gray_u8": ensure_gray_u8(require_key(data, "xray_gray_u8"), "xray_gray_u8"),
        "T_tc": np.asarray(require_key(data, "T_tc"), dtype=np.float64).reshape(4, 4),
    }

    if "tip_uv_c" in data:
        result["tip_uv_c"] = np.asarray(data["tip_uv_c"], dtype=np.float64).reshape(2)
    else:
        result["tip_uv_c"] = None

    if "alpha" in data:
        result["alpha"] = float(np.asarray(data["alpha"], dtype=np.float64).reshape(()))
    else:
        result["alpha"] = 0.22

    if "T_cx" in data:
        result["T_cx_saved"] = np.asarray(data["T_cx"], dtype=np.float64).reshape(4, 4)
    else:
        result["T_cx_saved"] = None

    if "T_xc" in data:
        result["T_xc_saved"] = np.asarray(data["T_xc"], dtype=np.float64).reshape(4, 4)
    else:
        result["T_xc_saved"] = None

    return result


# ============================================================
# Affine UV regularization
# ============================================================

def infer_grid_indices_row_major(n_points: int, steps_per_edge: int = 10) -> tuple[np.ndarray, np.ndarray]:
    side = steps_per_edge + 1
    expected = side * side
    if n_points != expected:
        raise ValueError(
            f"Expected {expected} points for a {side}x{side} grid, got {n_points}."
        )

    grid_i = np.array(
        [i for i in range(side) for _j in range(side)],
        dtype=np.int32,
    )
    grid_j = np.array(
        [j for _i in range(side) for j in range(side)],
        dtype=np.int32,
    )
    return grid_i, grid_j


class AffineGridModel:
    def __init__(self, p0: np.ndarray, u: np.ndarray, v: np.ndarray):
        self.p0 = np.asarray(p0, dtype=np.float64).reshape(2)
        self.u = np.asarray(u, dtype=np.float64).reshape(2)
        self.v = np.asarray(v, dtype=np.float64).reshape(2)

    def predict(self, grid_i: np.ndarray, grid_j: np.ndarray) -> np.ndarray:
        i = np.asarray(grid_i, dtype=np.float64).reshape(-1)
        j = np.asarray(grid_j, dtype=np.float64).reshape(-1)
        return self.p0[None, :] + j[:, None] * self.u[None, :] + i[:, None] * self.v[None, :]


def fit_affine_grid(points_uv: np.ndarray, grid_i: np.ndarray, grid_j: np.ndarray) -> AffineGridModel:
    uv = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    i = np.asarray(grid_i, dtype=np.float64).reshape(-1)
    j = np.asarray(grid_j, dtype=np.float64).reshape(-1)

    if not (len(uv) == len(i) == len(j)):
        raise ValueError("points_uv, grid_i, grid_j must have same length.")

    n = len(uv)
    A = np.zeros((2 * n, 6), dtype=np.float64)
    b = np.zeros((2 * n,), dtype=np.float64)

    # x = p0x + j*ux + i*vx
    A[0::2, 0] = 1.0
    A[0::2, 2] = j
    A[0::2, 4] = i
    b[0::2] = uv[:, 0]

    # y = p0y + j*uy + i*vy
    A[1::2, 1] = 1.0
    A[1::2, 3] = j
    A[1::2, 5] = i
    b[1::2] = uv[:, 1]

    x, *_ = np.linalg.lstsq(A, b, rcond=None)

    p0 = np.array([x[0], x[1]], dtype=np.float64)
    u = np.array([x[2], x[3]], dtype=np.float64)
    v = np.array([x[4], x[5]], dtype=np.float64)

    return AffineGridModel(p0=p0, u=u, v=v)


def build_affine_regularized_uv(points_uv_raw: np.ndarray, steps_per_edge: int = 10) -> tuple[np.ndarray, dict]:
    points_uv_raw = np.asarray(points_uv_raw, dtype=np.float64).reshape(-1, 2)

    grid_i, grid_j = infer_grid_indices_row_major(
        n_points=points_uv_raw.shape[0],
        steps_per_edge=steps_per_edge,
    )

    model = fit_affine_grid(points_uv_raw, grid_i, grid_j)
    points_uv_aff = model.predict(grid_i, grid_j)

    disp = np.linalg.norm(points_uv_aff - points_uv_raw, axis=1)

    dbg = {
        "grid_i": grid_i,
        "grid_j": grid_j,
        "p0": model.p0,
        "u": model.u,
        "v": model.v,
        "mean_disp_px": float(np.mean(disp)),
        "median_disp_px": float(np.median(disp)),
        "max_disp_px": float(np.max(disp)),
        "per_point_disp_px": disp,
    }
    return points_uv_aff, dbg


# ============================================================
# Tip helpers
# ============================================================

def project_tip_to_rgb_from_T_tc(
    T_tc: np.ndarray,
    K_rgb: np.ndarray,
) -> np.ndarray:
    T_tc = np.asarray(T_tc, dtype=np.float64).reshape(4, 4)
    K_rgb = np.asarray(K_rgb, dtype=np.float64).reshape(3, 3)

    tip_xyz_c = T_tc[:3, 3].reshape(1, 3).astype(np.float64)

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)

    uv, _ = cv2.projectPoints(
        tip_xyz_c,
        rvec,
        tvec,
        K_rgb,
        dist,
    )
    return uv.reshape(2)


def draw_tip_on_overlay(
    overlay_bgr: np.ndarray,
    tip_uv_c: np.ndarray,
    radius_px: int = 7,
) -> np.ndarray:
    out = np.asarray(overlay_bgr, dtype=np.uint8).copy()
    uv = np.asarray(tip_uv_c, dtype=np.float64).reshape(2)

    u = int(round(float(uv[0])))
    v = int(round(float(uv[1])))

    h, w = out.shape[:2]
    if 0 <= u < w and 0 <= v < h:
        cv2.circle(out, (u, v), radius_px + 2, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(out, (u, v), radius_px, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    return out


# ============================================================
# Overlay helpers
# ============================================================

def compute_overlay_from_uv_variant(
    *,
    name: str,
    K_xray: np.ndarray,
    K_rgb: np.ndarray,
    uv_xray: np.ndarray,
    xyz_c: np.ndarray,
    checkerboard_corners_uv: np.ndarray,
    rgb_bgr: np.ndarray,
    xray_gray_u8: np.ndarray,
    T_tc: np.ndarray,
    alpha: float,
):
    calib = calibrate_camera_to_xray(
        K_xray=K_xray,
        points_xyz_camera=xyz_c,
        points_uv_xray=uv_xray,
        dist_coeffs=None,
        pose_method="ippe_handeye",
        refine_with_iterative=False,
        pitch_mm=PITCH_MM,
        checkerboard_corners_uv=checkerboard_corners_uv,
        K_rgb=K_rgb,
        steps_per_edge=STEPS_PER_EDGE,
    )

    T_cx = np.asarray(calib.T_cx, dtype=np.float64).reshape(4, 4)
    T_xc = np.asarray(calib.T_xc, dtype=np.float64).reshape(4, 4)

    depth_res = extract_depth(
        T_xc=T_xc,
        T_tc=T_tc,
    )
    d_x_mm = float(depth_res.d_x_mm)

    R_xc = T_xc[:3, :3]
    t_xc = T_xc[:3, 3]

    H_xc = estimate_plane_induced_homography(
        K_c=K_rgb,
        R_xc=R_xc,
        t_xc=t_xc,
        K_x=K_xray,
        d_x=d_x_mm,
        normalize=True,
    )

    overlay_bgr, cache = blend_xray_overlay(
        camera_bgr=rgb_bgr,
        xray_gray_u8=xray_gray_u8,
        H_xc=H_xc,
        alpha=alpha,
    )

    return {
        "name": name,
        "calib": calib,
        "T_cx": T_cx,
        "T_xc": T_xc,
        "d_x_mm": d_x_mm,
        "H_xc": H_xc,
        "overlay_bgr": overlay_bgr,
        "cache": cache,
        "depth_res": depth_res,
    }


def add_title_banner(img_bgr: np.ndarray, title: str) -> np.ndarray:
    out = np.asarray(img_bgr, dtype=np.uint8).copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def show_two_overlays(overlay_raw: np.ndarray, overlay_aff: np.ndarray) -> None:
    cv2.namedWindow("Overlay RAW UV", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Overlay AFFINE UV", cv2.WINDOW_NORMAL)

    cv2.imshow("Overlay RAW UV", overlay_raw)
    cv2.imshow("Overlay AFFINE UV", overlay_aff)

    try:
        cv2.moveWindow("Overlay RAW UV", 40, 40)
        cv2.moveWindow("Overlay AFFINE UV", 980, 40)
    except cv2.error:
        pass

    print()
    print("Press any key in one of the OpenCV windows to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Main debug routine
# ============================================================

def run_debug_pose_handeye(npz_path: str | Path) -> None:
    pose = load_pose_npz(npz_path)

    K_xray = pose["K_xray"]
    K_rgb = pose["K_rgb"]
    uv_xray_raw = pose["xray_points_uv"]
    xyz_c = pose["xray_points_xyz_c"]
    checkerboard_corners_uv = pose["checkerboard_corners_uv"]
    rgb_bgr = pose["snapshot_rgb_bgr"]
    xray_gray_u8 = pose["xray_gray_u8"]
    T_tc = pose["T_tc"]
    alpha = pose["alpha"]

    print("=" * 80)
    print("DEBUG POSE HANDEYE WITH AFFINE UV")
    print("=" * 80)
    print(f"File: {pose['path']}")
    print()
    print(f"N 2D X-ray points         : {uv_xray_raw.shape[0]}")
    print(f"N 3D camera points        : {xyz_c.shape[0]}")
    print(f"checkerboard corners      : {checkerboard_corners_uv.shape}")
    print(f"RGB image shape           : {rgb_bgr.shape}")
    print(f"X-ray image shape         : {xray_gray_u8.shape}")
    print(f"alpha                     : {alpha:.3f}")
    print()

    if uv_xray_raw.shape[0] != xyz_c.shape[0]:
        raise ValueError(
            f"Point count mismatch: uv={uv_xray_raw.shape[0]} vs xyz={xyz_c.shape[0]}"
        )

    # --------------------------------------------------------
    # Build affine-regularized UV set
    # --------------------------------------------------------
    uv_xray_aff, aff_dbg = build_affine_regularized_uv(
        uv_xray_raw,
        steps_per_edge=STEPS_PER_EDGE,
    )

    print("Affine UV regularization")
    print("-" * 80)
    print(f"mean displacement [px]    : {aff_dbg['mean_disp_px']:.6f}")
    print(f"median displacement [px]  : {aff_dbg['median_disp_px']:.6f}")
    print(f"max displacement [px]     : {aff_dbg['max_disp_px']:.6f}")
    print(f"p0                        : {fmt_vec(aff_dbg['p0'])}")
    print(f"u                         : {fmt_vec(aff_dbg['u'])}")
    print(f"v                         : {fmt_vec(aff_dbg['v'])}")

    # --------------------------------------------------------
    # RAW calibration + overlay
    # --------------------------------------------------------
    raw_res = compute_overlay_from_uv_variant(
        name="raw",
        K_xray=K_xray,
        K_rgb=K_rgb,
        uv_xray=uv_xray_raw,
        xyz_c=xyz_c,
        checkerboard_corners_uv=checkerboard_corners_uv,
        rgb_bgr=rgb_bgr,
        xray_gray_u8=xray_gray_u8,
        T_tc=T_tc,
        alpha=alpha,
    )

    # --------------------------------------------------------
    # AFFINE calibration + overlay
    # --------------------------------------------------------
    aff_res = compute_overlay_from_uv_variant(
        name="affine",
        K_xray=K_xray,
        K_rgb=K_rgb,
        uv_xray=uv_xray_aff,
        xyz_c=xyz_c,
        checkerboard_corners_uv=checkerboard_corners_uv,
        rgb_bgr=rgb_bgr,
        xray_gray_u8=xray_gray_u8,
        T_tc=T_tc,
        alpha=alpha,
    )

    calib_raw = raw_res["calib"]
    calib_aff = aff_res["calib"]

    T_cx_raw = raw_res["T_cx"]
    T_cx_aff = aff_res["T_cx"]

    print()
    print("RAW result")
    print("-" * 80)
    print(f"candidate_index_rgb       : {calib_raw.pose_result.candidate_index_rgb}")
    print(f"candidate_index_xray      : {calib_raw.pose_result.candidate_index_xray}")
    print(f"reproj mean [px]          : {calib_raw.reproj_mean_px:.6f}")
    print(f"reproj median [px]        : {calib_raw.reproj_median_px:.6f}")
    print(f"reproj max [px]           : {calib_raw.reproj_max_px:.6f}")
    print(f"d_x_mm                    : {raw_res['d_x_mm']:.6f}")
    print("T_cx_raw =")
    print(fmt_T(T_cx_raw))

    print()
    print("AFFINE result")
    print("-" * 80)
    print(f"candidate_index_rgb       : {calib_aff.pose_result.candidate_index_rgb}")
    print(f"candidate_index_xray      : {calib_aff.pose_result.candidate_index_xray}")
    print(f"reproj mean [px]          : {calib_aff.reproj_mean_px:.6f}")
    print(f"reproj median [px]        : {calib_aff.reproj_median_px:.6f}")
    print(f"reproj max [px]           : {calib_aff.reproj_max_px:.6f}")
    print(f"d_x_mm                    : {aff_res['d_x_mm']:.6f}")
    print("T_cx_aff =")
    print(fmt_T(T_cx_aff))

    cmp_raw_aff = compare_transforms(T_cx_raw, T_cx_aff)
    print()
    print("RAW vs AFFINE transform difference")
    print("-" * 80)
    print(f"delta_R [deg]             : {cmp_raw_aff['delta_R_deg']:.6f}")
    print(f"delta_t [m]               : {cmp_raw_aff['delta_t']:.6f}")

    T_cx_saved = pose.get("T_cx_saved", None)
    if T_cx_saved is not None:
        cmp_saved_raw = compare_transforms(T_cx_saved, T_cx_raw)
        cmp_saved_aff = compare_transforms(T_cx_saved, T_cx_aff)

        print()
        print("Comparison against saved T_cx in NPZ")
        print("-" * 80)
        print(f"RAW    delta_R [deg]      : {cmp_saved_raw['delta_R_deg']:.6f}")
        print(f"RAW    delta_t [m]        : {cmp_saved_raw['delta_t']:.6f}")
        print(f"AFFINE delta_R [deg]      : {cmp_saved_aff['delta_R_deg']:.6f}")
        print(f"AFFINE delta_t [m]        : {cmp_saved_aff['delta_t']:.6f}")

    # --------------------------------------------------------
    # Tip overlay
    # --------------------------------------------------------
    if pose["tip_uv_c"] is not None:
        tip_uv_c = pose["tip_uv_c"]
        tip_source = "stored tip_uv_c"
    else:
        tip_uv_c = project_tip_to_rgb_from_T_tc(T_tc=T_tc, K_rgb=K_rgb)
        tip_source = "projected from T_tc"

    overlay_raw = draw_tip_on_overlay(raw_res["overlay_bgr"], tip_uv_c, radius_px=7)
    overlay_aff = draw_tip_on_overlay(aff_res["overlay_bgr"], tip_uv_c, radius_px=7)

    overlay_raw = add_title_banner(overlay_raw, "RAW UV")
    overlay_aff = add_title_banner(overlay_aff, "AFFINE UV")

    print()
    print("Tip overlay")
    print("-" * 80)
    print(f"tip source                : {tip_source}")
    print(f"tip_uv_c [px]             : {fmt_vec(tip_uv_c)}")

    print()
    print("Warp result")
    print("-" * 80)
    print(f"RAW    overlay shape      : {overlay_raw.shape}")
    print(f"RAW    mask nonzero       : {int(np.count_nonzero(raw_res['cache'].overlay_mask))}")
    print(f"AFFINE overlay shape      : {overlay_aff.shape}")
    print(f"AFFINE mask nonzero       : {int(np.count_nonzero(aff_res['cache'].overlay_mask))}")

    # --------------------------------------------------------
    # Optional save
    # --------------------------------------------------------
    if SAVE_DEBUG_NPZ:
        out_path = pose["path_obj"].with_name(
            pose["path_obj"].stem + "_affine_uv_debug.npz"
        )
        np.savez(
            str(out_path),
            xray_points_uv_raw=uv_xray_raw,
            xray_points_uv_affine=uv_xray_aff,
            xray_points_xyz_c=xyz_c,
            grid_i=aff_dbg["grid_i"],
            grid_j=aff_dbg["grid_j"],
            affine_p0=aff_dbg["p0"],
            affine_u=aff_dbg["u"],
            affine_v=aff_dbg["v"],
            affine_per_point_disp_px=aff_dbg["per_point_disp_px"],
            T_cx_raw=T_cx_raw,
            T_xc_raw=raw_res["T_xc"],
            T_cx_affine=T_cx_aff,
            T_xc_affine=aff_res["T_xc"],
            H_xc_raw=raw_res["H_xc"],
            H_xc_affine=aff_res["H_xc"],
            d_x_mm_raw=np.array(raw_res["d_x_mm"], dtype=np.float64),
            d_x_mm_affine=np.array(aff_res["d_x_mm"], dtype=np.float64),
            overlay_raw=overlay_raw,
            overlay_affine=overlay_aff,
            tip_uv_c=tip_uv_c,
        )
        print()
        print(f"[OK] saved debug npz      : {out_path}")

    show_two_overlays(overlay_raw, overlay_aff)


def main() -> None:
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    else:
        npz_path = pick_npz_file()

    if not npz_path:
        print("No file selected.")
        return

    run_debug_pose_handeye(npz_path)


if __name__ == "__main__":
    main()