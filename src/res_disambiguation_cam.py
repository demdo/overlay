from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from overlay.tracking.pose_solvers import solve_pose


# ============================================================
# Qt helpers
# ============================================================

def _ensure_qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def pick_overlay_npz_file() -> Path | None:
    _ensure_qt_app()
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select overlay preview / overlay problem NPZ",
        "",
        "NPZ files (*.npz);;All files (*.*)",
    )
    return Path(path) if path else None


def pick_intrinsics_npz_file() -> Path | None:
    _ensure_qt_app()
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select X-ray intrinsics NPZ",
        "",
        "NPZ files (*.npz);;All files (*.*)",
    )
    return Path(path) if path else None


# ============================================================
# Basic helpers
# ============================================================

def _as_xyz(arr: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N,3), got {pts.shape}")
    return pts


def _as_uv(arr: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N,2), got {pts.shape}")
    return pts


def _as_mat33(arr: np.ndarray, name: str) -> np.ndarray:
    M = np.asarray(arr, dtype=np.float64)
    if M.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3,3), got {M.shape}")
    return M


def format_matrix(M: np.ndarray, decimals: int = 6) -> str:
    rows = []
    for row in np.asarray(M):
        rows.append("[" + "  ".join(f"{v:+.{decimals}f}" for v in row) + "]")
    return "\n".join(rows)


def make_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def load_intrinsics_npz(npz_path: Path) -> np.ndarray:
    data = np.load(str(npz_path), allow_pickle=True)
    keys = set(data.files)

    for key in ("K_xray", "K_x", "K"):
        if key in keys:
            return _as_mat33(data[key], f"{npz_path.name}:{key}")

    raise ValueError(
        f"{npz_path} does not contain any of the expected keys "
        f"('K_xray', 'K_x', 'K'). Available keys: {sorted(keys)}"
    )


# ============================================================
# Data container
# ============================================================

class OverlayData:
    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path)

        data = np.load(str(npz_path), allow_pickle=True)
        keys = set(data.files)

        required = {
            "K_rgb",
            "xray_points_xyz_c",
            "xray_points_uv",
            "checkerboard_corners_uv",
        }
        missing = required - keys
        if missing:
            raise ValueError(f"Missing required keys in overlay NPZ: {sorted(missing)}")

        self.K_rgb = _as_mat33(data["K_rgb"], "K_rgb")
        self.points_xyz_c_m = _as_xyz(data["xray_points_xyz_c"], "xray_points_xyz_c")
        self.points_uv_x = _as_uv(data["xray_points_uv"], "xray_points_uv")
        self.checkerboard_corners_uv = _as_uv(
            data["checkerboard_corners_uv"],
            "checkerboard_corners_uv",
        )


# ============================================================
# Geometry helpers
# ============================================================

def build_board_grid_object_points_mm(
    rows: int = 11,
    cols: int = 11,
    pitch_mm: float = 2.54,
) -> np.ndarray:
    pts = []
    for i in range(rows):
        for j in range(cols):
            pts.append([j * pitch_mm, i * pitch_mm, 0.0])
    return np.asarray(pts, dtype=np.float64)


def board_rectangle_corners_mm(
    rows: int = 11,
    cols: int = 11,
    pitch_mm: float = 2.54,
) -> np.ndarray:
    W = (cols - 1) * pitch_mm
    H = (rows - 1) * pitch_mm
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [W,   0.0, 0.0],
            [W,   H,   0.0],
            [0.0, H,   0.0],
        ],
        dtype=np.float64,
    )


def transform_points(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_xyz, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ pts.T).T + t[None, :]


def estimate_rigid_transform_kabsch(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
) -> np.ndarray:
    src = np.asarray(src_xyz, dtype=np.float64)
    dst = np.asarray(dst_xyz, dtype=np.float64)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(
            f"src and dst must both have shape (N,3), got {src.shape} and {dst.shape}"
        )

    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)

    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid

    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = dst_centroid - R @ src_centroid

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = R_a @ R_b.T
    c = 0.5 * (np.trace(R_rel) - 1.0)
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    radius = 0.5 * max(x_range, y_range, z_range)

    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])


def draw_board_plane(
    ax,
    T_bw_mm: np.ndarray,
    corners_b_mm: np.ndarray,
    color: str,
    label: str,
    alpha_fill: float = 0.22,
) -> np.ndarray:
    corners_w = transform_points(T_bw_mm, corners_b_mm)

    cyc = np.vstack([corners_w, corners_w[0:1]])
    ax.plot(
        cyc[:, 0],
        cyc[:, 1],
        cyc[:, 2],
        color=color,
        linewidth=2.5,
        label=label,
    )

    ax.plot_trisurf(
        corners_w[:, 0],
        corners_w[:, 1],
        corners_w[:, 2],
        color=color,
        alpha=alpha_fill,
        shade=False,
    )

    return corners_w


# ============================================================
# Main
# ============================================================

def main() -> int:
    _ensure_qt_app()

    overlay_npz_path = pick_overlay_npz_file()
    if overlay_npz_path is None:
        return 0

    intrinsics_path = pick_intrinsics_npz_file()
    if intrinsics_path is None:
        return 0

    try:
        overlay_data = OverlayData(overlay_npz_path)
        K_xray = load_intrinsics_npz(intrinsics_path)

        print("\n" + "=" * 100)
        print("RGB-SIDE POSE DISAMBIGUATION VISUALIZATION (REFERENCE/WORLD FRAME)")
        print("=" * 100)
        print(f"Overlay NPZ:    {overlay_npz_path}")
        print(f"Intrinsics NPZ: {intrinsics_path}")
        print("\nK_xray =")
        print(format_matrix(K_xray))
        print("\nK_rgb =")
        print(format_matrix(overlay_data.K_rgb))

        # Same RGB-candidate generation as before, but without selection
        res = solve_pose(
            object_points_xyz=overlay_data.points_xyz_c_m,
            image_points_uv=overlay_data.points_uv_x,
            K=K_xray,
            dist_coeffs=None,
            dist_coeffs_rgb=None,
            pose_method="ippe_handeye",
            checkerboard_corners_uv=overlay_data.checkerboard_corners_uv,
            K_rgb=overlay_data.K_rgb,
            steps_per_edge=10,
            refine_with_iterative=False,
            refine_rgb_iterative=False,
            refine_xray_iterative=False,
        )

        if res.all_candidates_rgb is None or len(res.all_candidates_rgb) != 2:
            raise RuntimeError(
                f"Expected exactly 2 RGB IPPE candidates, got "
                f"{0 if res.all_candidates_rgb is None else len(res.all_candidates_rgb)}."
            )

        rows = 11
        cols = 11
        pitch_mm = 2.54

        board_grid_b_mm = build_board_grid_object_points_mm(
            rows=rows,
            cols=cols,
            pitch_mm=pitch_mm,
        )
        board_grid_c_mm_depth = 1e3 * overlay_data.points_xyz_c_m

        if board_grid_b_mm.shape[0] != board_grid_c_mm_depth.shape[0]:
            raise RuntimeError(
                f"Point count mismatch for depth reference: "
                f"{board_grid_b_mm.shape[0]} board model points vs "
                f"{board_grid_c_mm_depth.shape[0]} reconstructed depth points."
            )

        # depth-based reference pose in camera frame
        T_bc_depth_mm = estimate_rigid_transform_kabsch(
            src_xyz=board_grid_b_mm,
            dst_xyz=board_grid_c_mm_depth,
        )

        # Define world/reference frame by the depth pose
        # T_cw = inv(T_bc_depth)
        T_cw_mm = invert_transform(T_bc_depth_mm)

        print("\n" + "#" * 100)
        print("DEPTH REFERENCE")
        print("#" * 100)
        print("T_bc_depth [mm] =")
        print(format_matrix(T_bc_depth_mm))
        print("\nT_cw = inv(T_bc_depth) [mm] =")
        print(format_matrix(T_cw_mm))

        T_bw_depth_mm = T_cw_mm @ T_bc_depth_mm
        print("\nT_bw_depth [mm] =")
        print(format_matrix(T_bw_depth_mm))

        T_bw_candidates_mm: list[np.ndarray] = []
        delta_t_list: list[float] = []
        delta_r_list: list[float] = []

        for idx, cand in enumerate(res.all_candidates_rgb):
            T_bc_mm = make_transform(cand.rvec, cand.tvec)
            T_bw_mm = T_cw_mm @ T_bc_mm
            T_bw_candidates_mm.append(T_bw_mm)

            delta_t_mm = float(np.linalg.norm(
                T_bc_mm[:3, 3] - T_bc_depth_mm[:3, 3]
            ))
            delta_r_deg = rotation_angle_deg(
                T_bc_mm[:3, :3],
                T_bc_depth_mm[:3, :3],
            )
            delta_t_list.append(delta_t_mm)
            delta_r_list.append(delta_r_deg)

            print("\n" + "#" * 100)
            print(f"RGB candidate {idx}")
            print("#" * 100)
            print(f"reproj mean [px]    = {cand.reproj_mean_px:.6f}")
            print(f"reproj median [px]  = {cand.reproj_median_px:.6f}")
            print(f"reproj max [px]     = {cand.reproj_max_px:.6f}")
            print(f"delta_t [mm]        = {delta_t_mm:.6f}")
            print(f"delta_r [deg]       = {delta_r_deg:.6f}")

            print("\nT_bc [mm] =")
            print(format_matrix(T_bc_mm))

            print("\nT_bw = inv(T_bc_depth) @ T_bc [mm] =")
            print(format_matrix(T_bw_mm))

        corners_b_mm = board_rectangle_corners_mm(
            rows=rows,
            cols=cols,
            pitch_mm=pitch_mm,
        )

        # Depth points expressed in the reference/world frame
        depth_grid_w_mm = transform_points(T_cw_mm, board_grid_c_mm_depth)

        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111, projection="3d")

        depth_corners_w = draw_board_plane(
            ax=ax,
            T_bw_mm=T_bw_depth_mm,
            corners_b_mm=corners_b_mm,
            color="black",
            label="depth ref",
            alpha_fill=0.10,
        )

        cand0_corners_w = draw_board_plane(
            ax=ax,
            T_bw_mm=T_bw_candidates_mm[0],
            corners_b_mm=corners_b_mm,
            color="tab:blue",
            label="rgb cand 0",
            alpha_fill=0.20,
        )

        cand1_corners_w = draw_board_plane(
            ax=ax,
            T_bw_mm=T_bw_candidates_mm[1],
            corners_b_mm=corners_b_mm,
            color="tab:red",
            label="rgb cand 1",
            alpha_fill=0.20,
        )

        # Optional: depth reconstructed points, now in w-frame
        ax.scatter(
            depth_grid_w_mm[:, 0],
            depth_grid_w_mm[:, 1],
            depth_grid_w_mm[:, 2],
            s=8,
            color="gray",
            alpha=0.35,
        )

        all_pts = np.vstack([
            depth_corners_w,
            cand0_corners_w,
            cand1_corners_w,
            depth_grid_w_mm,
        ])

        pad = 5.0
        ax.set_xlim(np.min(all_pts[:, 0]) - pad, np.max(all_pts[:, 0]) + pad)
        ax.set_ylim(np.min(all_pts[:, 1]) - pad, np.max(all_pts[:, 1]) + pad)
        ax.set_zlim(np.min(all_pts[:, 2]) - pad, np.max(all_pts[:, 2]) + pad)
        set_axes_equal(ax)

        ax.set_xlabel("x_w [mm]")
        ax.set_ylabel("y_w [mm]")
        ax.set_zlabel("z_w [mm]")
        ax.set_title("RGB pose disambiguation in reference/world frame")
        ax.legend()
        ax.view_init(elev=28, azim=-55)

        info_text = (
            "Depth defines world/reference frame\n"
            "w := depth-based reference pose\n\n"
            f"cand 0: Δt = {delta_t_list[0]:.3f} mm, ΔR = {delta_r_list[0]:.3f} deg\n"
            f"cand 1: Δt = {delta_t_list[1]:.3f} mm, ΔR = {delta_r_list[1]:.3f} deg"
        )
        fig.text(
            0.02,
            0.98,
            info_text,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        plt.tight_layout()
        plt.show()

        return 0

    except Exception as e:
        QMessageBox.critical(None, "res_disambiguation_cam", str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())