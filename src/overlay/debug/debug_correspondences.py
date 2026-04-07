# -*- coding: utf-8 -*-
"""
debug_correspondences_four_corners.py

Visual debug for the four corner indices of the 11x11 grid:
    0, 10, 110, 120

Loads via Qt file dialogs:
- xyz_c npz
- RGB image
- K_rgb npz
- uv npz
- X-ray image

Then draws the four corresponding points in BOTH images:
- red cross
- red index label

No RealSense, no marker detection, no cropping, no reordering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom

from PySide6.QtWidgets import QApplication, QFileDialog


# ============================================================
# Config
# ============================================================
WINDOW_RGB = "debug_correspondences - RGB corners"
WINDOW_XRAY = "debug_correspondences - Xray corners"

DISPLAY_MAX_W = 1400
DISPLAY_MAX_H = 1000

CORNER_INDICES = [0, 10, 110, 120]


# ============================================================
# Qt helpers
# ============================================================
def _get_qapp() -> tuple[QApplication, bool]:
    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True
    return app, created


def pick_open_file_qt(title: str, file_filter: str) -> str | None:
    app, created = _get_qapp()

    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        file_filter,
    )

    if created:
        app.quit()

    path = path.strip()
    return path if path else None


# ============================================================
# Loaders
# ============================================================
def load_xray_or_gray_image(path: str) -> np.ndarray:
    path_l = path.lower()

    if path_l.endswith((".dcm", ".ima")):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img.astype(np.uint8)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read grayscale image: {path}")
    return img


def load_rgb_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return img


def load_xyz_npz(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        for key in ("points_xyz_camera", "points_xyz", "xyz"):
            if key in npz.files:
                xyz = np.asarray(npz[key], dtype=np.float64).reshape(-1, 3)
                print(f"[INFO] Loaded xyz from key '{key}'")
                return xyz

    raise KeyError(
        f"{Path(path).name}: expected one of keys "
        f"['points_xyz_camera', 'points_xyz', 'xyz']"
    )


def load_uv_npz(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        if "points_uv" not in npz.files:
            raise KeyError(
                f"{Path(path).name}: expected key 'points_uv', found {list(npz.files)}"
            )
        uv = np.asarray(npz["points_uv"], dtype=np.float64).reshape(-1, 2)
    return uv


def load_k_npz(path: str, name: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        for key in ("K_rgb", "K_xray", "Kx", "K"):
            if key in npz.files:
                K = np.asarray(npz[key], dtype=np.float64).reshape(3, 3)
                print(f"[INFO] Loaded {name} from key '{key}'")
                return K

    raise KeyError(
        f"{Path(path).name}: expected one of keys "
        f"['K_rgb', 'K_xray', 'Kx', 'K']"
    )


# ============================================================
# Geometry
# ============================================================
def project_points_camera_to_image(
    points_xyz_camera: np.ndarray,
    K_rgb: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_xyz_camera, dtype=np.float64).reshape(-1, 3)

    fx = float(K_rgb[0, 0])
    fy = float(K_rgb[1, 1])
    cx = float(K_rgb[0, 2])
    cy = float(K_rgb[1, 2])

    uv = np.empty((pts.shape[0], 2), dtype=np.float64)

    for i, (x, y, z) in enumerate(pts):
        if z <= 1e-12:
            uv[i] = np.array([np.nan, np.nan], dtype=np.float64)
            continue
        uv[i, 0] = fx * (x / z) + cx
        uv[i, 1] = fy * (y / z) + cy

    return uv


# ============================================================
# Drawing
# ============================================================
def draw_cross(
    img: np.ndarray,
    u: float,
    v: float,
    color: tuple[int, int, int],
    size: int = 12,
    thickness: int = 2,
) -> None:
    uu = int(round(u))
    vv = int(round(v))
    cv2.line(img, (uu - size, vv), (uu + size, vv), color, thickness, cv2.LINE_AA)
    cv2.line(img, (uu, vv - size), (uu, vv + size), color, thickness, cv2.LINE_AA)


def draw_indexed_points(
    image_bgr: np.ndarray,
    uv_points: np.ndarray,
    indices: list[int],
    *,
    title: str,
) -> np.ndarray:
    vis = image_bgr.copy()

    h, w = vis.shape[:2]
    red = (0, 0, 255)

    for idx in indices:
        if idx < 0 or idx >= uv_points.shape[0]:
            print(f"[WARN] Index {idx} out of range for point array with N={uv_points.shape[0]}")
            continue

        u, v = uv_points[idx]
        if not np.isfinite(u) or not np.isfinite(v):
            print(f"[WARN] Index {idx} has non-finite coordinates.")
            continue

        if not (0 <= u < w and 0 <= v < h):
            print(f"[WARN] Index {idx} projects outside image: ({u:.2f}, {v:.2f})")
            continue

        draw_cross(vis, u, v, color=red, size=12, thickness=2)

        cv2.putText(
            vis,
            str(idx),
            (int(round(u)) + 8, int(round(v)) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            str(idx),
            (int(round(u)) + 8, int(round(v)) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            red,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        title,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return vis


# ============================================================
# Display helpers
# ============================================================
def make_display_transform(
    img_shape: tuple[int, int],
    max_w: int,
    max_h: int,
) -> tuple[float, int, int, int, int]:
    h, w = img_shape[:2]
    scale = min(max_w / float(w), max_h / float(h))
    if scale <= 0:
        scale = 1.0

    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))

    off_x = max(0, (max_w - disp_w) // 2)
    off_y = max(0, (max_h - disp_h) // 2)

    return scale, disp_w, disp_h, off_x, off_y


def render_for_display(
    img_bgr: np.ndarray,
    max_w: int,
    max_h: int,
) -> np.ndarray:
    scale, disp_w, disp_h, off_x, off_y = make_display_transform(img_bgr.shape, max_w, max_h)

    if abs(scale - 1.0) < 1e-12:
        resized = img_bgr.copy()
    else:
        resized = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    canvas[off_y:off_y + disp_h, off_x:off_x + disp_w] = resized
    return canvas


# ============================================================
# Main
# ============================================================
def main() -> None:
    xyz_path = pick_open_file_qt(
        "Load xyz_c NPZ",
        "NumPy NPZ (*.npz);;All files (*)",
    )
    if not xyz_path:
        print("[INFO] No xyz npz selected. Abort.")
        return

    rgb_img_path = pick_open_file_qt(
        "Load RGB image",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
    )
    if not rgb_img_path:
        print("[INFO] No RGB image selected. Abort.")
        return

    k_rgb_path = pick_open_file_qt(
        "Load K_rgb NPZ",
        "NumPy NPZ (*.npz);;All files (*)",
    )
    if not k_rgb_path:
        print("[INFO] No K_rgb npz selected. Abort.")
        return

    uv_path = pick_open_file_qt(
        "Load UV NPZ",
        "NumPy NPZ (*.npz);;All files (*)",
    )
    if not uv_path:
        print("[INFO] No uv npz selected. Abort.")
        return

    xray_img_path = pick_open_file_qt(
        "Load X-ray image",
        "X-ray / DICOM (*.dcm *.ima *.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)",
    )
    if not xray_img_path:
        print("[INFO] No X-ray image selected. Abort.")
        return

    print("\n" + "=" * 72)
    print("LOAD DATA")
    print("=" * 72)

    xyz = load_xyz_npz(xyz_path)
    rgb_img = load_rgb_image(rgb_img_path)
    K_rgb = load_k_npz(k_rgb_path, "K_rgb")
    uv_xray = load_uv_npz(uv_path)
    xray_gray = load_xray_or_gray_image(xray_img_path)
    xray_bgr = cv2.cvtColor(xray_gray, cv2.COLOR_GRAY2BGR)

    print(f"[INFO] xyz shape     : {xyz.shape}")
    print(f"[INFO] uv_xray shape : {uv_xray.shape}")
    print(f"[INFO] RGB image     : {rgb_img.shape}")
    print(f"[INFO] Xray image    : {xray_gray.shape}")
    print("[INFO] K_rgb:")
    print(K_rgb)

    if xyz.shape[0] <= max(CORNER_INDICES):
        raise ValueError(
            f"xyz has only {xyz.shape[0]} points, but need index {max(CORNER_INDICES)}"
        )
    if uv_xray.shape[0] <= max(CORNER_INDICES):
        raise ValueError(
            f"uv has only {uv_xray.shape[0]} points, but need index {max(CORNER_INDICES)}"
        )

    uv_rgb = project_points_camera_to_image(xyz, K_rgb)

    print("\n" + "=" * 72)
    print("FOUR CORNER POINTS")
    print("=" * 72)
    for idx in CORNER_INDICES:
        print(
            f"[IDX {idx:3d}] "
            f"RGB uv=({uv_rgb[idx,0]:8.2f}, {uv_rgb[idx,1]:8.2f})   "
            f"Xray uv=({uv_xray[idx,0]:8.2f}, {uv_xray[idx,1]:8.2f})"
        )

    rgb_vis = draw_indexed_points(
        rgb_img,
        uv_rgb,
        CORNER_INDICES,
        title=f"RGB - {Path(rgb_img_path).name}",
    )

    xray_vis = draw_indexed_points(
        xray_bgr,
        uv_xray,
        CORNER_INDICES,
        title=f"X-ray - {Path(xray_img_path).name}",
    )

    cv2.namedWindow(WINDOW_RGB, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_XRAY, cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(WINDOW_RGB, render_for_display(rgb_vis, DISPLAY_MAX_W, DISPLAY_MAX_H))
        cv2.imshow(WINDOW_XRAY, render_for_display(xray_vis, DISPLAY_MAX_W, DISPLAY_MAX_H))

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()