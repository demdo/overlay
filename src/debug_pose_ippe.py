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
    T_cx_m[:3, 3] *= 1e-3  # mm -> m
    return T_cx_m


def recompute_dx_from_T_xc(T_xc_m: np.ndarray, T_tc_mm: np.ndarray) -> float:
    """
    Exact same logic as in the working overlay script:
        T_cx = inv(T_xc)
        convert T_cx translation m -> mm
        T_tx = T_cx_mm @ T_tc_mm
        d_x = z-component in X-ray frame (mm)
    """
    T_xc_m = np.asarray(T_xc_m, dtype=np.float64).reshape(4, 4)
    T_tc_mm = np.asarray(T_tc_mm, dtype=np.float64).reshape(4, 4)

    T_cx_m = invert_transform(T_xc_m)
    T_cx_mm = T_cx_m.copy()
    T_cx_mm[:3, 3] *= 1e3  # m -> mm

    T_tx = T_cx_mm @ T_tc_mm
    tip_xyz_x_mm = T_tx[:3, 3]
    return float(tip_xyz_x_mm[2])


def compute_overlay(
    camera_bgr: np.ndarray,
    xray_gray_u8: np.ndarray,
    K_rgb: np.ndarray,
    K_xray: np.ndarray,
    T_xc_m: np.ndarray,
    T_tc_mm: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    d_x_mm = recompute_dx_from_T_xc(T_xc_m, T_tc_mm)

    R_xc = T_xc_m[:3, :3]
    t_xc = T_xc_m[:3, 3]

    H_xc = estimate_plane_induced_homography(
        K_c=K_rgb,
        R_xc=R_xc,
        t_xc=t_xc,
        K_x=K_xray,
        d_x=d_x_mm,
    )

    overlay_bgr, _ = blend_xray_overlay(
        camera_bgr=camera_bgr,
        xray_gray_u8=xray_gray_u8,
        H_xc=H_xc,
        alpha=alpha,
    )

    return overlay_bgr, d_x_mm, H_xc


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

        windows: list[OverlayImageWindow] = []

        print("\n" + "=" * 120)
        print("OVERLAY COMPARISON: DIRECT IPPE + IPPE HANDEYE (ALL CANDIDATES)")
        print("=" * 120)
        print(f"Overlay NPZ   : {overlay_npz_path}")
        print(f"Intrinsics NPZ: {intrinsics_path}")

        print("\nK_xray =")
        print(format_matrix(K_xray))

        # ========================================================
        # 1) DIRECT IPPE
        # ========================================================
        print("\n" + "=" * 120)
        print("1) DIRECT IPPE")
        print("=" * 120)

        res_direct = solve_pose(
            object_points_xyz=overlay_data.points_xyz_c_m,
            image_points_uv=overlay_data.points_uv_x,
            K=K_xray,
            dist_coeffs=None,
            pose_method="ippe",
            refine_with_iterative=False,
            use_xray_ippe_selection_rule=False,
        )

        if res_direct.all_candidates is None or len(res_direct.all_candidates) != 2:
            raise RuntimeError("Expected exactly 2 direct IPPE candidates.")

        if res_direct.candidate_index is not None:
            print(f"selected direct candidate index = {res_direct.candidate_index}")

        for cand_idx, cand in enumerate(res_direct.all_candidates):
            T_cx_m = make_transform(cand.rvec, cand.tvec)
            T_xc_m = invert_transform(T_cx_m)

            overlay_bgr, d_x_mm, H_xc = compute_overlay(
                camera_bgr=overlay_data.camera_bgr,
                xray_gray_u8=overlay_data.xray_gray_u8,
                K_rgb=overlay_data.K_rgb,
                K_xray=K_xray,
                T_xc_m=T_xc_m,
                T_tc_mm=overlay_data.T_tc_mm,
                alpha=overlay_data.alpha_nominal,
            )

            print("\n" + "-" * 120)
            print(f"DIRECT IPPE CANDIDATE {cand_idx}")
            print("-" * 120)
            print(f"reproj mean [px]    = {cand.reproj_mean_px:.6f}")
            print(f"reproj median [px]  = {cand.reproj_median_px:.6f}")
            print(f"reproj max [px]     = {cand.reproj_max_px:.6f}")

            print("\nT_cx [m] =")
            print(format_matrix(T_cx_m))

            print("\nT_xc [m] =")
            print(format_matrix(T_xc_m))

            print(f"\nd_x [mm] = {d_x_mm:+.6f}")

            print("\nH_xc =")
            print(format_matrix(H_xc))

            title = f"{intrinsic_name} | Direct IPPE | candidate {cand_idx}"
            win = OverlayImageWindow(title, overlay_bgr)
            win.show()
            windows.append(win)

        # ========================================================
        # 2) IPPE HANDEYE: 2 RGB × 2 XRAY = 4 overlays
        # ========================================================
        print("\n" + "=" * 120)
        print("2) IPPE HANDEYE (ALL CANDIDATES)")
        print("=" * 120)

        res_handeye = solve_pose(
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

        if res_handeye.all_candidates_rgb is None or len(res_handeye.all_candidates_rgb) != 2:
            raise RuntimeError("Expected exactly 2 RGB IPPE candidates for handeye.")
        if res_handeye.all_candidates is None or len(res_handeye.all_candidates) != 2:
            raise RuntimeError("Expected exactly 2 X-ray IPPE candidates for handeye.")

        if res_handeye.candidate_index_rgb is not None:
            print(f"selected RGB candidate index    = {res_handeye.candidate_index_rgb}")
        if res_handeye.candidate_index_xray is not None:
            print(f"selected X-ray candidate index  = {res_handeye.candidate_index_xray}")

        print("\n" + "#" * 120)
        print("RGB CANDIDATES")
        print("#" * 120)
        for rgb_idx, rgb_cand in enumerate(res_handeye.all_candidates_rgb):
            T_bc_mm = make_transform(rgb_cand.rvec, rgb_cand.tvec)
            print(f"\nRGB candidate {rgb_idx}")
            print(f"reproj mean [px]    = {rgb_cand.reproj_mean_px:.6f}")
            print(f"reproj median [px]  = {rgb_cand.reproj_median_px:.6f}")
            print(f"reproj max [px]     = {rgb_cand.reproj_max_px:.6f}")
            print("T_bc [mm] =")
            print(format_matrix(T_bc_mm))

        print("\n" + "#" * 120)
        print("XRAY CANDIDATES")
        print("#" * 120)
        for xray_idx, xray_cand in enumerate(res_handeye.all_candidates):
            T_bx_mm = make_transform(xray_cand.rvec, xray_cand.tvec)
            print(f"\nXRAY candidate {xray_idx}")
            print(f"reproj mean [px]    = {xray_cand.reproj_mean_px:.6f}")
            print(f"reproj median [px]  = {xray_cand.reproj_median_px:.6f}")
            print(f"reproj max [px]     = {xray_cand.reproj_max_px:.6f}")
            print("T_bx [mm] =")
            print(format_matrix(T_bx_mm))

        for rgb_idx, rgb_cand in enumerate(res_handeye.all_candidates_rgb):
            T_bc_mm = make_transform(rgb_cand.rvec, rgb_cand.tvec)

            print("\n" + "=" * 120)
            print(f"RGB CANDIDATE {rgb_idx}")
            print("=" * 120)
            print(f"reproj mean [px]    = {rgb_cand.reproj_mean_px:.6f}")
            print(f"reproj median [px]  = {rgb_cand.reproj_median_px:.6f}")
            print(f"reproj max [px]     = {rgb_cand.reproj_max_px:.6f}")
            print("T_bc [mm] =")
            print(format_matrix(T_bc_mm))

            for xray_idx, xray_cand in enumerate(res_handeye.all_candidates):
                T_bx_mm = make_transform(xray_cand.rvec, xray_cand.tvec)
                T_xb_mm = invert_transform(T_bx_mm)

                T_cx_m = compute_T_cx_from_T_bc_T_bx(T_bc_mm, T_bx_mm)
                T_xc_m = invert_transform(T_cx_m)

                overlay_bgr, d_x_mm, H_xc = compute_overlay(
                    camera_bgr=overlay_data.camera_bgr,
                    xray_gray_u8=overlay_data.xray_gray_u8,
                    K_rgb=overlay_data.K_rgb,
                    K_xray=K_xray,
                    T_xc_m=T_xc_m,
                    T_tc_mm=overlay_data.T_tc_mm,
                    alpha=overlay_data.alpha_nominal,
                )

                print("\n" + "-" * 120)
                print(f"HANDEYE RGB {rgb_idx} + XRAY {xray_idx}")
                print("-" * 120)
                print(f"XRAY reproj mean [px]   = {xray_cand.reproj_mean_px:.6f}")
                print(f"XRAY reproj median [px] = {xray_cand.reproj_median_px:.6f}")
                print(f"XRAY reproj max [px]    = {xray_cand.reproj_max_px:.6f}")

                print("\nT_bx [mm] =")
                print(format_matrix(T_bx_mm))

                print("\nT_xb [mm] =")
                print(format_matrix(T_xb_mm))

                print("\nDerived T_cx [m] =")
                print(format_matrix(T_cx_m))

                print("\nDerived T_xc [m] =")
                print(format_matrix(T_xc_m))

                print(f"\nd_x [mm] = {d_x_mm:+.6f}")

                print("\nH_xc =")
                print(format_matrix(H_xc))

                title = f"{intrinsic_name} | Handeye | RGB {rgb_idx} | XRAY {xray_idx}"
                win = OverlayImageWindow(title, overlay_bgr)
                win.show()
                windows.append(win)

        # ========================================================
        # 3) HANDEYE: SELECTED OVERLAY ONLY
        # ========================================================
        print("\n" + "=" * 120)
        print("3) HANDEYE SELECTED OVERLAY")
        print("=" * 120)

        if res_handeye.candidate_index_rgb is None:
            raise RuntimeError("Handeye did not return a selected RGB candidate index.")
        if res_handeye.candidate_index_xray is None:
            raise RuntimeError("Handeye did not return a selected X-ray candidate index.")

        rgb_sel_idx = int(res_handeye.candidate_index_rgb)
        xray_sel_idx = int(res_handeye.candidate_index_xray)

        rgb_sel_cand = res_handeye.all_candidates_rgb[rgb_sel_idx]
        xray_sel_cand = res_handeye.all_candidates[xray_sel_idx]

        T_bc_sel_mm = make_transform(rgb_sel_cand.rvec, rgb_sel_cand.tvec)
        T_bx_sel_mm = make_transform(xray_sel_cand.rvec, xray_sel_cand.tvec)
        T_xb_sel_mm = invert_transform(T_bx_sel_mm)

        T_cx_sel_m = compute_T_cx_from_T_bc_T_bx(T_bc_sel_mm, T_bx_sel_mm)
        T_xc_sel_m = invert_transform(T_cx_sel_m)

        overlay_bgr_sel, d_x_mm_sel, H_xc_sel = compute_overlay(
            camera_bgr=overlay_data.camera_bgr,
            xray_gray_u8=overlay_data.xray_gray_u8,
            K_rgb=overlay_data.K_rgb,
            K_xray=K_xray,
            T_xc_m=T_xc_sel_m,
            T_tc_mm=overlay_data.T_tc_mm,
            alpha=overlay_data.alpha_nominal,
        )

        print("\n" + "-" * 120)
        print("SELECTED HANDEYE OVERLAY")
        print("-" * 120)
        print(f"selected RGB candidate index   = {rgb_sel_idx}")
        print(f"selected X-ray candidate index = {xray_sel_idx}")

        print("\nSelected T_bc [mm] =")
        print(format_matrix(T_bc_sel_mm))

        print("\nSelected T_bx [mm] =")
        print(format_matrix(T_bx_sel_mm))

        print("\nSelected T_xb [mm] =")
        print(format_matrix(T_xb_sel_mm))

        print("\nDerived T_cx [m] =")
        print(format_matrix(T_cx_sel_m))

        print("\nDerived T_xc [m] =")
        print(format_matrix(T_xc_sel_m))

        print(f"\nd_x [mm] = {d_x_mm_sel:+.6f}")

        print("\nH_xc =")
        print(format_matrix(H_xc_sel))

        title = (
            f"{intrinsic_name} | Handeye | SELECTED "
            f"| RGB {rgb_sel_idx} | XRAY {xray_sel_idx}"
        )
        win = OverlayImageWindow(title, overlay_bgr_sel)
        win.show()
        windows.append(win)

        app._overlay_windows = windows
        return app.exec()

    except Exception as e:
        QMessageBox.critical(None, "Overlay comparison", str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())