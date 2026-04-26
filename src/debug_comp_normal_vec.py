from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.homography import estimate_plane_induced_homography
from overlay.tools.warp import blend_xray_overlay


# ============================================================
# Qt helper
# ============================================================

def _ensure_qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def pick_file(title: str) -> Path | None:
    _ensure_qt_app()
    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        "NPZ files (*.npz);;All files (*.*)",
    )
    return Path(path) if path else None


# ============================================================
# Math helpers
# ============================================================

def normalize(v: np.ndarray, name: str = "vector") -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.size != 3:
        raise ValueError(f"{name} must have 3 elements, got shape {v.shape}")
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError(f"{name} has invalid norm: {n}")
    return v / n


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a, "a")
    b = normalize(b, "b")
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def align_sign(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    v = normalize(v, "v")
    ref = normalize(ref, "ref")
    return v if np.dot(v, ref) >= 0.0 else -v


def rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return rotation matrix R such that:
        b ~= R @ a
    for unit vectors a, b.
    """
    a = normalize(a, "a")
    b = normalize(b, "b")

    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = np.linalg.norm(v)

    # nearly identical
    if s < 1e-12 and c > 0.999999999:
        return np.eye(3, dtype=np.float64)

    # nearly opposite
    if s < 1e-12 and c < -0.999999999:
        # choose any axis orthogonal to a
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(a, helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        axis = np.cross(a, helper)
        axis = normalize(axis, "axis_180")
        rvec = axis * np.pi
        R, _ = cv2.Rodrigues(rvec)
        return R.astype(np.float64)

    axis = v / s
    theta = np.arctan2(s, c)

    rvec = axis * theta
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64)


# ============================================================
# Data loading
# ============================================================

def load_overlay_data(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True)

    required = [
        "snapshot_rgb_with_tip_bgr",
        "xray_gray_u8",
        "K_rgb",
        "K_xray",
        "T_xc",
        "d_x",
    ]
    for k in required:
        if k not in data:
            raise RuntimeError(f"Overlay file missing key: {k}")

    out = {
        "camera_bgr": np.asarray(data["snapshot_rgb_with_tip_bgr"]),
        "xray_gray_u8": np.asarray(data["xray_gray_u8"]),
        "K_rgb": np.asarray(data["K_rgb"], dtype=np.float64),
        "K_xray": np.asarray(data["K_xray"], dtype=np.float64),
        "T_xc": np.asarray(data["T_xc"], dtype=np.float64),
        "d_x_mm": float(np.asarray(data["d_x"]).reshape(-1)[0]),
        "alpha": 0.5,
    }

    if "alpha" in data:
        out["alpha"] = float(np.asarray(data["alpha"]).reshape(-1)[0])

    if out["camera_bgr"].dtype != np.uint8:
        out["camera_bgr"] = np.clip(out["camera_bgr"], 0, 255).astype(np.uint8)

    if out["xray_gray_u8"].dtype != np.uint8:
        out["xray_gray_u8"] = np.clip(out["xray_gray_u8"], 0, 255).astype(np.uint8)

    if out["T_xc"].shape != (4, 4):
        raise RuntimeError(f"T_xc must be (4,4), got {out['T_xc'].shape}")

    return out


def load_plane_normal_from_filt(path: Path) -> np.ndarray:
    data = np.load(str(path), allow_pickle=True)

    if "planes_filt" not in data:
        raise RuntimeError("Plane file missing key: planes_filt")

    planes_filt = np.asarray(data["planes_filt"], dtype=np.float64)
    if planes_filt.ndim != 2 or planes_filt.shape[1] != 4:
        raise RuntimeError(f"planes_filt must have shape (N,4), got {planes_filt.shape}")

    normals_c = []
    for i, plane in enumerate(planes_filt):
        n_c = normalize(plane[:3], name=f"planes_filt[{i},:3]")
        normals_c.append(n_c)

    normals_c = np.asarray(normals_c, dtype=np.float64)

    # mean in camera frame
    n_c_meas_mean = normalize(np.mean(normals_c, axis=0), "n_c_meas_mean")
    return n_c_meas_mean


# ============================================================
# Overlay helpers
# ============================================================

def compute_H_xc(
    K_rgb: np.ndarray,
    K_xray: np.ndarray,
    R_xc: np.ndarray,
    t_xc: np.ndarray,
    d_x_mm: float,
) -> np.ndarray:
    return estimate_plane_induced_homography(
        K_c=K_rgb,
        R_xc=R_xc,
        t_xc=t_xc,
        K_x=K_xray,
        d_x=d_x_mm,
    )


def make_overlay(
    camera_bgr: np.ndarray,
    xray_gray_u8: np.ndarray,
    H_xc: np.ndarray,
    alpha: float,
) -> np.ndarray:
    out_bgr, _ = blend_xray_overlay(
        camera_bgr=camera_bgr,
        xray_gray_u8=xray_gray_u8,
        H_xc=H_xc,
        alpha=alpha,
    )
    return out_bgr


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ============================================================
# Main
# ============================================================

def main() -> int:
    _ensure_qt_app()

    print("\nSelect overlay preview file...")
    overlay_path = pick_file("Select overlay preview NPZ")
    if overlay_path is None:
        print("No overlay file selected.")
        return 0

    print("\nSelect plane fitting session file...")
    plane_path = pick_file("Select plane fitting session NPZ")
    if plane_path is None:
        print("No plane file selected.")
        return 0

    try:
        overlay = load_overlay_data(overlay_path)
        n_c_meas = load_plane_normal_from_filt(plane_path)

        T_xc = overlay["T_xc"]
        R_xc = T_xc[:3, :3].copy()
        t_xc = T_xc[:3, 3].copy()

        # ----------------------------------------------------
        # Ideal and measured normals in X-ray frame
        # ----------------------------------------------------
        n_x_ideal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        n_x_meas = R_xc.T @ n_c_meas
        n_x_meas = normalize(n_x_meas, "n_x_meas")
        n_x_meas = align_sign(n_x_meas, n_x_ideal)

        angle_before_deg = angle_deg(n_x_meas, n_x_ideal)

        # ----------------------------------------------------
        # Correction rotation in X-ray frame
        # ----------------------------------------------------
        # We want: n_x_ideal ~= R_corr @ n_x_meas
        R_corr = rotation_from_a_to_b(n_x_meas, n_x_ideal)

        # Apply correction consistently to pose:
        # R_xc_corr = R_xc @ R_corr^T
        R_xc_corr = R_xc @ R_corr.T

        # sanity check
        n_x_meas_after = R_xc_corr.T @ n_c_meas
        n_x_meas_after = normalize(n_x_meas_after, "n_x_meas_after")
        n_x_meas_after = align_sign(n_x_meas_after, n_x_ideal)
        angle_after_deg = angle_deg(n_x_meas_after, n_x_ideal)

        # ----------------------------------------------------
        # Original overlay
        # ----------------------------------------------------
        H_xc_orig = compute_H_xc(
            K_rgb=overlay["K_rgb"],
            K_xray=overlay["K_xray"],
            R_xc=R_xc,
            t_xc=t_xc,
            d_x_mm=overlay["d_x_mm"],
        )

        overlay_orig = make_overlay(
            camera_bgr=overlay["camera_bgr"],
            xray_gray_u8=overlay["xray_gray_u8"],
            H_xc=H_xc_orig,
            alpha=overlay["alpha"],
        )

        # ----------------------------------------------------
        # Corrected overlay
        # ----------------------------------------------------
        H_xc_corr = compute_H_xc(
            K_rgb=overlay["K_rgb"],
            K_xray=overlay["K_xray"],
            R_xc=R_xc_corr,
            t_xc=t_xc,
            d_x_mm=overlay["d_x_mm"],
        )

        overlay_corr = make_overlay(
            camera_bgr=overlay["camera_bgr"],
            xray_gray_u8=overlay["xray_gray_u8"],
            H_xc=H_xc_corr,
            alpha=overlay["alpha"],
        )

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------
        print("\n============================================================")
        print("Normal-based pose correction")
        print("============================================================")
        print(f"Overlay file : {overlay_path}")
        print(f"Plane file   : {plane_path}")
        print()
        print(f"n_c_meas     = [{n_c_meas[0]: .6f}, {n_c_meas[1]: .6f}, {n_c_meas[2]: .6f}]")
        print(f"n_x_meas     = [{n_x_meas[0]: .6f}, {n_x_meas[1]: .6f}, {n_x_meas[2]: .6f}]")
        print(f"n_x_ideal    = [{n_x_ideal[0]: .6f}, {n_x_ideal[1]: .6f}, {n_x_ideal[2]: .6f}]")
        print()
        print(f"Angle before correction = {angle_before_deg:.4f} deg")
        print(f"Angle after correction  = {angle_after_deg:.4f} deg")
        print()
        print("R_corr")
        print(R_corr)
        print("============================================================\n")

        # ----------------------------------------------------
        # Show original vs corrected
        # ----------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(bgr_to_rgb(overlay_orig))
        axes[0].set_title(f"Original overlay\nnormal angle = {angle_before_deg:.2f} deg")
        axes[0].axis("off")

        axes[1].imshow(bgr_to_rgb(overlay_corr))
        axes[1].set_title(f"Corrected overlay\nnormal angle = {angle_after_deg:.2f} deg")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

        return 0

    except Exception as e:
        print("\nERROR")
        print("-----")
        print(str(e))
        print()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())