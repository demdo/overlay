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

from overlay.calib.calib_camera_to_xray import calibrate_camera_to_xray
from overlay.tools.homography import estimate_plane_induced_homography
from overlay.tools.warp import blend_xray_overlay


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
        "Select overlay preview NPZ",
        "",
        "NPZ files (*.npz);;All files (*.*)",
    )
    return Path(path) if path else None


def pick_intrinsics_npz_file() -> Path | None:
    _ensure_qt_app()
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select intrinsics NPZ",
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
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = R_a.T @ R_b
    trace_val = np.trace(R_rel)
    cos_theta = np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def format_matrix(M: np.ndarray, decimals: int = 6) -> str:
    rows = []
    for row in np.asarray(M):
        rows.append("[" + "  ".join(f"{v:+.{decimals}f}" for v in row) + "]")
    return "\n".join(rows)


# ============================================================
# Depth recompute
# ============================================================

def recompute_dx(T_xc_m: np.ndarray, T_tc_mm: np.ndarray) -> float:
    T_cx_m = invert_transform(T_xc_m)
    T_cx_mm = T_cx_m.copy()
    T_cx_mm[:3, 3] *= 1e3  # m -> mm
    T_tx = T_cx_mm @ T_tc_mm
    tip_xyz_x_mm = T_tx[:3, 3]
    return float(tip_xyz_x_mm[2])


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
                "Overlay NPZ must contain either 'snapshot_rgb_with_tip_bgr' or 'snapshot_rgb_bgr'."
            )

        self.xray_gray_u8 = np.asarray(data["xray_gray_u8"])
        if self.xray_gray_u8.ndim != 2:
            raise ValueError(
                f"xray_gray_u8 must be grayscale, got shape {self.xray_gray_u8.shape}"
            )
        if self.xray_gray_u8.dtype != np.uint8:
            self.xray_gray_u8 = np.clip(self.xray_gray_u8, 0, 255).astype(np.uint8)

        self.K_rgb = np.asarray(data["K_rgb"], dtype=np.float64)
        if self.K_rgb.shape != (3, 3):
            raise ValueError(f"K_rgb must be (3,3), got {self.K_rgb.shape}")

        self.points_xyz_c_m = _as_xyz(data["xray_points_xyz_c"], "xray_points_xyz_c")
        self.points_uv_x = _as_uv(data["xray_points_uv"], "xray_points_uv")
        self.checkerboard_corners_uv = np.asarray(data["checkerboard_corners_uv"], dtype=np.float64)
        self.T_tc_mm = np.asarray(data["T_tc"], dtype=np.float64)

        if self.checkerboard_corners_uv.shape != (3, 2):
            raise ValueError(
                f"checkerboard_corners_uv must have shape (3,2), got {self.checkerboard_corners_uv.shape}"
            )
        if self.T_tc_mm.shape != (4, 4):
            raise ValueError(f"T_tc must be (4,4), got {self.T_tc_mm.shape}")

        self.alpha_nominal = 0.5
        if "alpha" in keys:
            try:
                self.alpha_nominal = float(
                    np.clip(_as_scalar(data["alpha"], "alpha"), 0.0, 1.0)
                )
            except Exception:
                pass


class IntrinsicsData:
    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path)

        data = np.load(str(npz_path), allow_pickle=True)
        self._data = data
        self.K_entries = self._extract_all_k_matrices()

        if not self.K_entries:
            raise ValueError("No valid K_* matrices found in intrinsics NPZ.")

    def _extract_all_k_matrices(self) -> list[tuple[str, np.ndarray]]:
        preferred_order = [
            "K_zhang",
            "K_refined",
            "K_refined_fixed_pp",
            "K_refined_fixed_center",
        ]

        found: list[tuple[str, np.ndarray]] = []
        used_names: set[str] = set()

        for key in preferred_order:
            if key in self._data.files:
                arr = np.asarray(self._data[key], dtype=np.float64)
                if arr.shape == (3, 3):
                    found.append((key, arr))
                    used_names.add(key)

        for key in self._data.files:
            if key in used_names:
                continue
            if not key.startswith("K_"):
                continue
            arr = np.asarray(self._data[key], dtype=np.float64)
            if arr.shape == (3, 3):
                found.append((key, arr))
                used_names.add(key)

        return found


# ============================================================
# Overlay computation
# ============================================================

def compute_overlay_result(
    overlay_data: OverlayData,
    K_xray: np.ndarray,
    label: str,
    *,
    refine_with_iterative: bool = False,
    refine_rgb_iterative: bool = False,
    refine_xray_iterative: bool = False,
) -> dict:
    result = calibrate_camera_to_xray(
        K_xray=K_xray,
        points_xyz_camera=overlay_data.points_xyz_c_m,
        points_uv_xray=overlay_data.points_uv_x,
        pose_method="ippe_handeye",
        refine_with_iterative=refine_with_iterative,
        refine_rgb_iterative=refine_rgb_iterative,
        refine_xray_iterative=refine_xray_iterative,
        ransac_reprojection_error_px=3.0,
        checkerboard_corners_uv=overlay_data.checkerboard_corners_uv,
        K_rgb=overlay_data.K_rgb,
        pitch_mm=2.54,
        steps_per_edge=10,
    )

    T_xc = np.asarray(result.T_xc, dtype=np.float64)
    d_x_mm = recompute_dx(T_xc, overlay_data.T_tc_mm)

    R_xc = T_xc[:3, :3]
    t_xc = T_xc[:3, 3]

    H_xc = estimate_plane_induced_homography(
        K_c=overlay_data.K_rgb,
        R_xc=R_xc,
        t_xc=t_xc,
        K_x=K_xray,
        d_x=float(d_x_mm),
    )

    overlay_bgr, _ = blend_xray_overlay(
        camera_bgr=overlay_data.camera_bgr,
        xray_gray_u8=overlay_data.xray_gray_u8,
        H_xc=H_xc,
        alpha=overlay_data.alpha_nominal,
    )

    return {
        "label": label,
        "overlay_bgr": overlay_bgr,
        "T_xc": T_xc,
        "d_x_mm": d_x_mm,
        "reproj_mean_px": float(result.reproj_mean_px),
        "reproj_median_px": float(result.reproj_median_px),
        "reproj_max_px": float(result.reproj_max_px),
    }


# ============================================================
# Image-only window
# ============================================================

class OverlayImageWindow(QMainWindow):
    def __init__(self, title: str, overlay_bgr: np.ndarray):
        super().__init__()
        self.overlay_bgr = overlay_bgr

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
# Main
# ============================================================

def main() -> int:
    app = _ensure_qt_app()

    if len(sys.argv) > 1:
        overlay_npz_path = Path(sys.argv[1])
    else:
        overlay_npz_path = pick_overlay_npz_file()

    if overlay_npz_path is None:
        return 0

    if len(sys.argv) > 2:
        intrinsics_npz_path = Path(sys.argv[2])
    else:
        intrinsics_npz_path = pick_intrinsics_npz_file()

    if intrinsics_npz_path is None:
        return 0

    try:
        overlay_data = OverlayData(overlay_npz_path)
        intr_data = IntrinsicsData(intrinsics_npz_path)

        windows: list[OverlayImageWindow] = []

        print("\n============================================================")
        print("Loaded intrinsics from file")
        print("============================================================")
        for key, K in intr_data.K_entries:
            print(f"{key}:")
            print(K)
            print()

        print("\n============================================================")
        print("Running full overlay computation for every K_*")
        print("Modes:")
        print("  1) IPPE")
        print("  2) only T_bx refined")
        print("  3) T_bc and T_bx refined")
        print("============================================================")

        for key, K in intr_data.K_entries:
            result_ippe = compute_overlay_result(
                overlay_data=overlay_data,
                K_xray=K,
                label=f"{key} | IPPE",
                refine_with_iterative=False,
                refine_rgb_iterative=False,
                refine_xray_iterative=False,
            )

            result_bx_only = compute_overlay_result(
                overlay_data=overlay_data,
                K_xray=K,
                label=f"{key} | only T_bx refined",
                refine_with_iterative=False,
                refine_rgb_iterative=False,
                refine_xray_iterative=True,
            )

            result_both = compute_overlay_result(
                overlay_data=overlay_data,
                K_xray=K,
                label=f"{key} | T_bc and T_bx refined",
                refine_with_iterative=False,
                refine_rgb_iterative=True,
                refine_xray_iterative=True,
            )

            print(f"\n============================================================")
            print(f"{key}")
            print("============================================================")
            print("K_xray =")
            print(format_matrix(K))

            print("\n--- 1) IPPE ---")
            print(f"d_x_mm             : {result_ippe['d_x_mm']:.6f}")
            print(f"reproj mean [px]   : {result_ippe['reproj_mean_px']:.6f}")
            print(f"reproj median [px] : {result_ippe['reproj_median_px']:.6f}")
            print(f"reproj max [px]    : {result_ippe['reproj_max_px']:.6f}")
            print("T_xc =")
            print(format_matrix(result_ippe["T_xc"]))

            print("\n--- 2) only T_bx refined ---")
            print(f"d_x_mm             : {result_bx_only['d_x_mm']:.6f}")
            print(f"reproj mean [px]   : {result_bx_only['reproj_mean_px']:.6f}")
            print(f"reproj median [px] : {result_bx_only['reproj_median_px']:.6f}")
            print(f"reproj max [px]    : {result_bx_only['reproj_max_px']:.6f}")
            print("T_xc =")
            print(format_matrix(result_bx_only["T_xc"]))

            print("\n--- 3) T_bc and T_bx refined ---")
            print(f"d_x_mm             : {result_both['d_x_mm']:.6f}")
            print(f"reproj mean [px]   : {result_both['reproj_mean_px']:.6f}")
            print(f"reproj median [px] : {result_both['reproj_median_px']:.6f}")
            print(f"reproj max [px]    : {result_both['reproj_max_px']:.6f}")
            print("T_xc =")
            print(format_matrix(result_both["T_xc"]))

            # Differences relative to raw IPPE
            R_ippe = result_ippe["T_xc"][:3, :3]
            t_ippe_mm = result_ippe["T_xc"][:3, 3] * 1e3

            R_bx_only = result_bx_only["T_xc"][:3, :3]
            t_bx_only_mm = result_bx_only["T_xc"][:3, 3] * 1e3
            dt_bx_only_mm = t_bx_only_mm - t_ippe_mm
            dR_bx_only_deg = rotation_angle_deg(R_ippe, R_bx_only)

            R_both = result_both["T_xc"][:3, :3]
            t_both_mm = result_both["T_xc"][:3, 3] * 1e3
            dt_both_mm = t_both_mm - t_ippe_mm
            dR_both_deg = rotation_angle_deg(R_ippe, R_both)

            print("\n--- Difference (only T_bx refined) - (IPPE) ---")
            print(f"Δt_x [mm]          : {dt_bx_only_mm[0]:+.6f}")
            print(f"Δt_y [mm]          : {dt_bx_only_mm[1]:+.6f}")
            print(f"Δt_z [mm]          : {dt_bx_only_mm[2]:+.6f}")
            print(f"ΔR [deg]           : {dR_bx_only_deg:+.6f}")
            print(f"Δd_x [mm]          : {result_bx_only['d_x_mm'] - result_ippe['d_x_mm']:+.6f}")

            print("\n--- Difference (T_bc and T_bx refined) - (IPPE) ---")
            print(f"Δt_x [mm]          : {dt_both_mm[0]:+.6f}")
            print(f"Δt_y [mm]          : {dt_both_mm[1]:+.6f}")
            print(f"Δt_z [mm]          : {dt_both_mm[2]:+.6f}")
            print(f"ΔR [deg]           : {dR_both_deg:+.6f}")
            print(f"Δd_x [mm]          : {result_both['d_x_mm'] - result_ippe['d_x_mm']:+.6f}")

            win_ippe = OverlayImageWindow(
                f"Overlay - {key} - IPPE",
                result_ippe["overlay_bgr"],
            )
            win_bx_only = OverlayImageWindow(
                f"Overlay - {key} - only T_bx refined",
                result_bx_only["overlay_bgr"],
            )
            win_both = OverlayImageWindow(
                f"Overlay - {key} - T_bc and T_bx refined",
                result_both["overlay_bgr"],
            )

            win_ippe.show()
            win_bx_only.show()
            win_both.show()

            windows.extend([win_ippe, win_bx_only, win_both])

        app._overlay_windows = windows
        return app.exec()

    except Exception as e:
        QMessageBox.critical(None, "Overlay comparison", str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())