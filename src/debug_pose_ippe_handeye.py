from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
)

from overlay.tracking.pose_solvers import solve_pose
from overlay.tools.homography import estimate_plane_induced_homography
from overlay.tools.warp import blend_xray_overlay


# ============================================================
# Config
# ============================================================

STEPS_PER_EDGE = 10
GRID_SIZE = STEPS_PER_EDGE + 1


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

def _as_scalar(x, name: str) -> float:
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(f"{name} must be scalar-like, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def _to_uint8_bgr(img: np.ndarray, name: str) -> np.ndarray:
    img = np.asarray(img)

    if img.ndim == 2:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"{name} must be grayscale or BGR image, got shape {img.shape}")

    return img


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


def _as_transform44(arr: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(arr, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4,4), got {T.shape}")
    return T


def bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(
        img_rgb.data,
        w,
        h,
        bytes_per_line,
        QImage.Format_RGB888,
    )
    return QPixmap.fromImage(qimg.copy())


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def make_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def format_matrix(M: np.ndarray, decimals: int = 6) -> str:
    rows = []
    for row in np.asarray(M):
        rows.append("[" + "  ".join(f"{v:+.{decimals}f}" for v in row) + "]")
    return "\n".join(rows)


def _safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


# ============================================================
# Flipping helper
# ============================================================

def undo_marker_selection_left_right_flip(
    uv_final: np.ndarray,
    *,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """
    Undo the LEFT-RIGHT correspondence flip from xray_marker_selection.

    Important:
    This does NOT mirror image coordinates.
    It mirrors the point ordering within each row.

    In marker selection:
        j_xray = nu - j_cam

    So the returned xray_points_uv are already ordered with flipped j.
    To get the "without flipping" correspondences, we reverse each row again.
    """
    uv = np.asarray(uv_final, dtype=np.float64)

    expected_n = grid_size * grid_size
    if uv.shape != (expected_n, 2):
        raise ValueError(
            f"Expected uv shape ({expected_n}, 2) for a {grid_size}x{grid_size} grid, "
            f"got {uv.shape}."
        )

    uv_grid = uv.reshape(grid_size, grid_size, 2)
    uv_unflipped = uv_grid[:, ::-1, :].reshape(-1, 2)

    return uv_unflipped


# ============================================================
# Data containers
# ============================================================

class OverlayData:
    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path)

        data = np.load(str(npz_path), allow_pickle=True)
        keys = set(data.files)

        required = {
            "xray_gray_u8",
            "K_rgb",
            "xray_points_xyz_c",
            "xray_points_uv",
            "checkerboard_corners_uv",
            "T_tc",
        }
        missing = required - keys
        if missing:
            raise ValueError(f"Missing required keys in overlay NPZ: {sorted(missing)}")

        if "snapshot_rgb_with_tip_bgr" in keys:
            self.camera_bgr = _to_uint8_bgr(
                data["snapshot_rgb_with_tip_bgr"],
                "snapshot_rgb_with_tip_bgr",
            )
        elif "snapshot_rgb_bgr" in keys:
            self.camera_bgr = _to_uint8_bgr(
                data["snapshot_rgb_bgr"],
                "snapshot_rgb_bgr",
            )
        else:
            raise ValueError(
                "Overlay NPZ must contain either 'snapshot_rgb_with_tip_bgr' "
                "or 'snapshot_rgb_bgr'."
            )

        self.xray_gray_u8 = np.asarray(data["xray_gray_u8"])
        if self.xray_gray_u8.ndim != 2:
            raise ValueError(
                f"xray_gray_u8 must be grayscale, got shape {self.xray_gray_u8.shape}"
            )
        if self.xray_gray_u8.dtype != np.uint8:
            self.xray_gray_u8 = np.clip(self.xray_gray_u8, 0, 255).astype(np.uint8)

        self.K_rgb = _as_mat33(data["K_rgb"], "K_rgb")
        self.points_xyz_c_m = _as_xyz(data["xray_points_xyz_c"], "xray_points_xyz_c")
        self.points_uv_x = _as_uv(data["xray_points_uv"], "xray_points_uv")
        self.checkerboard_corners_uv = _as_uv(
            data["checkerboard_corners_uv"],
            "checkerboard_corners_uv",
        )
        self.T_tc_mm = _as_transform44(data["T_tc"], "T_tc")

        self.alpha_nominal = 0.5
        if "alpha" in keys:
            try:
                self.alpha_nominal = float(
                    np.clip(_as_scalar(data["alpha"], "alpha"), 0.0, 1.0)
                )
            except Exception:
                pass


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
# Geometry helpers
# ============================================================

def compute_T_cx_from_T_bc_T_bx(T_bc_mm: np.ndarray, T_bx_mm: np.ndarray) -> np.ndarray:
    """
    T_bc_mm : board -> camera, translation in mm
    T_bx_mm : board -> xray,   translation in mm

    Returns
    -------
    T_cx_m : camera -> xray, translation in meters
    """
    T_cx_mm = np.asarray(T_bx_mm, dtype=np.float64) @ invert_transform(T_bc_mm)
    T_cx_m = T_cx_mm.copy()
    T_cx_m[:3, 3] *= 1e-3
    return T_cx_m


def recompute_dx_from_T_xc(T_xc_m: np.ndarray, T_tc_mm: np.ndarray) -> float:
    T_xc_m = np.asarray(T_xc_m, dtype=np.float64).reshape(4, 4)
    T_tc_mm = np.asarray(T_tc_mm, dtype=np.float64).reshape(4, 4)

    T_cx_m = invert_transform(T_xc_m)
    T_cx_mm = T_cx_m.copy()
    T_cx_mm[:3, 3] *= 1e3

    T_tx = T_cx_mm @ T_tc_mm
    tip_xyz_x_mm = T_tx[:3, 3]
    return float(tip_xyz_x_mm[2])


def z_axis_angle_deg_from_T_cx(T_cx_m: np.ndarray) -> float:
    """
    Angle between camera z-axis and xray z-axis, both expressed in camera frame.

    T_cx maps camera -> xray.
    Therefore T_xc maps xray -> camera.
    The xray z-axis expressed in camera coordinates is R_xc[:, 2].
    The camera z-axis is [0, 0, 1].
    """
    T_xc_m = invert_transform(T_cx_m)
    z_x_in_c = T_xc_m[:3, 2]
    z_c = np.array([0.0, 0.0, 1.0])

    dot = float(np.dot(z_c, z_x_in_c))
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


# ============================================================
# Image-only window
# ============================================================

class OverlayImageWindow(QMainWindow):
    def __init__(self, title: str, overlay_bgr: np.ndarray):
        super().__init__()
        self.overlay_bgr = np.asarray(overlay_bgr, dtype=np.uint8)

        self.setWindowTitle(title)
        self.resize(1400, 1000)

        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMinimumSize(1200, 800)
        self.lbl_image.setStyleSheet("background: #202020; border: 1px solid #505050;")
        self.setCentralWidget(self.lbl_image)

        self._update_image()

    def _update_image(self) -> None:
        pix = bgr_to_qpixmap(self.overlay_bgr)
        pix = pix.scaled(
            self.lbl_image.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.lbl_image.setPixmap(pix)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_image()


# ============================================================
# Hand-eye run helper
# ============================================================

def run_ippe_handeye_case(
    *,
    label: str,
    uv_xray: np.ndarray,
    overlay_data: OverlayData,
    K_xray: np.ndarray,
    show_overlay: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)

    res = solve_pose(
        object_points_xyz=overlay_data.points_xyz_c_m,
        image_points_uv=uv_xray,
        K=K_xray,
        dist_coeffs=None,
        dist_coeffs_rgb=None,
        pose_method="ippe_handeye",
        checkerboard_corners_uv=overlay_data.checkerboard_corners_uv,
        K_rgb=overlay_data.K_rgb,
        steps_per_edge=STEPS_PER_EDGE,
        refine_with_iterative=False,
        refine_rgb_iterative=False,
        refine_xray_iterative=False,
    )

    if res.all_candidates_rgb is None or len(res.all_candidates_rgb) != 2:
        raise RuntimeError(
            f"{label}: Expected exactly 2 RGB IPPE candidates, got "
            f"{0 if res.all_candidates_rgb is None else len(res.all_candidates_rgb)}."
        )

    if res.all_candidates is None or len(res.all_candidates) != 2:
        raise RuntimeError(
            f"{label}: Expected exactly 2 X-ray IPPE candidates, got "
            f"{0 if res.all_candidates is None else len(res.all_candidates)}."
        )

    print("\nRGB candidates:")
    for rgb_idx, rgb_cand in enumerate(res.all_candidates_rgb):
        T_bc_mm = make_transform(rgb_cand.rvec, rgb_cand.tvec)
        print("\n" + "-" * 80)
        print(f"RGB candidate {rgb_idx}")
        print(f"reproj mean [px]    = {rgb_cand.reproj_mean_px:.6f}")
        print(f"reproj median [px]  = {rgb_cand.reproj_median_px:.6f}")
        print(f"reproj max [px]     = {rgb_cand.reproj_max_px:.6f}")
        print("T_bc [mm] =")
        print(format_matrix(T_bc_mm))

    print("\nX-ray candidates:")
    for xray_idx, xray_cand in enumerate(res.all_candidates):
        T_bx_mm = make_transform(xray_cand.rvec, xray_cand.tvec)
        print("\n" + "-" * 80)
        print(f"XRAY candidate {xray_idx}")
        print(f"reproj mean [px]    = {xray_cand.reproj_mean_px:.6f}")
        print(f"reproj median [px]  = {xray_cand.reproj_median_px:.6f}")
        print(f"reproj max [px]     = {xray_cand.reproj_max_px:.6f}")
        print("T_bx [mm] =")
        print(format_matrix(T_bx_mm))

    # ------------------------------------------------------------
    # Final selected result from solve_pose
    # ------------------------------------------------------------
    if not hasattr(res, "rvec") or not hasattr(res, "tvec"):
        raise RuntimeError(
            f"{label}: solve_pose result has no final rvec/tvec. "
            "Cannot print final T_cx."
        )

    T_cx_final = make_transform(res.rvec, res.tvec)

    # Safety: if translation looks like mm, convert to m.
    # For T_cx we expect roughly meter-scale translation, not hundreds of meters.
    if np.linalg.norm(T_cx_final[:3, 3]) > 10.0:
        T_cx_final[:3, 3] *= 1e-3

    T_xc_final = invert_transform(T_cx_final)
    angle_z = z_axis_angle_deg_from_T_cx(T_cx_final)

    print("\n" + "#" * 100)
    print(f"FINAL SELECTED RESULT: {label}")
    print("#" * 100)

    print("\nT_cx final [m] =")
    print(format_matrix(T_cx_final))

    print("\nT_xc final [m] =")
    print(format_matrix(T_xc_final))

    print(f"\nangle(z_cam, z_xray expressed in camera) [deg] = {angle_z:.6f}")

    # ------------------------------------------------------------
    # Optional overlay for final selected pose
    # ------------------------------------------------------------
    overlay_bgr = None

    if show_overlay:
        d_x_mm = recompute_dx_from_T_xc(T_xc_final, overlay_data.T_tc_mm)

        R_xc = T_xc_final[:3, :3]
        t_xc = T_xc_final[:3, 3]

        H_xc = estimate_plane_induced_homography(
            K_c=overlay_data.K_rgb,
            R_xc=R_xc,
            t_xc=t_xc,
            K_x=K_xray,
            d_x=d_x_mm,
        )

        overlay_bgr, _ = blend_xray_overlay(
            camera_bgr=overlay_data.camera_bgr,
            xray_gray_u8=overlay_data.xray_gray_u8,
            H_xc=H_xc,
            alpha=overlay_data.alpha_nominal,
        )

        print(f"\nd_x final [mm] = {d_x_mm:+.6f}")
        print("\nH_xc final =")
        print(format_matrix(H_xc))

    return T_cx_final, overlay_bgr


# ============================================================
# Main
# ============================================================

def main() -> int:
    app = _ensure_qt_app()

    overlay_npz_path = pick_overlay_npz_file()
    if overlay_npz_path is None:
        return 0

    intrinsics_path = pick_intrinsics_npz_file()
    if intrinsics_path is None:
        return 0

    try:
        overlay_data = OverlayData(overlay_npz_path)
        K_xray = load_intrinsics_npz(intrinsics_path)
        intrinsic_name = _safe_name(intrinsics_path)

        print("\n" + "=" * 100)
        print("IPPE HAND-EYE: WITH FLIPPING VS WITHOUT FLIPPING")
        print("=" * 100)
        print(f"Overlay NPZ:    {overlay_npz_path}")
        print(f"Intrinsics NPZ: {intrinsics_path}")

        print("\nK_xray =")
        print(format_matrix(K_xray))

        uv_with_flipping = overlay_data.points_uv_x
        uv_without_flipping = undo_marker_selection_left_right_flip(
            uv_with_flipping,
            grid_size=GRID_SIZE,
        )

        windows: list[OverlayImageWindow] = []

        T_cx_with_flip, overlay_with_flip = run_ippe_handeye_case(
            label="WITH FLIPPING (current xray_points_uv)",
            uv_xray=uv_with_flipping,
            overlay_data=overlay_data,
            K_xray=K_xray,
            show_overlay=True,
        )

        T_cx_without_flip, overlay_without_flip = run_ippe_handeye_case(
            label="WITHOUT FLIPPING (row-wise unflipped correspondences)",
            uv_xray=uv_without_flipping,
            overlay_data=overlay_data,
            K_xray=K_xray,
            show_overlay=True,
        )

        print("\n" + "=" * 100)
        print("FINAL COMPARISON")
        print("=" * 100)

        print("\nT_cx WITH flipping [m] =")
        print(format_matrix(T_cx_with_flip))

        print("\nT_cx WITHOUT flipping [m] =")
        print(format_matrix(T_cx_without_flip))

        angle_with = z_axis_angle_deg_from_T_cx(T_cx_with_flip)
        angle_without = z_axis_angle_deg_from_T_cx(T_cx_without_flip)

        print(f"\nangle(z_cam, z_xray) WITH flipping    [deg] = {angle_with:.6f}")
        print(f"angle(z_cam, z_xray) WITHOUT flipping [deg] = {angle_without:.6f}")

        if overlay_with_flip is not None:
            win = OverlayImageWindow(
                f"{intrinsic_name} | WITH flipping",
                overlay_with_flip,
            )
            win.show()
            windows.append(win)

        if overlay_without_flip is not None:
            win = OverlayImageWindow(
                f"{intrinsic_name} | WITHOUT flipping",
                overlay_without_flip,
            )
            win.show()
            windows.append(win)

        app._overlay_windows = windows
        return app.exec()

    except Exception as e:
        QMessageBox.critical(None, "IPPE hand-eye flip comparison", str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())