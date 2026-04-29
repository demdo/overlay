from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog


# ============================================================
# Config
# ============================================================

STEPS_PER_EDGE = 10
IMPORTANT_IDX = [0, 10, 110, 120]


# ============================================================
# File dialogs
# ============================================================

def select_file(title: str, file_filter: str) -> Path | None:
    app = QApplication.instance()
    owns_app = app is None

    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        file_filter,
    )

    if owns_app:
        app.quit()

    if not path:
        return None

    return Path(path)


# ============================================================
# Drawing helpers
# ============================================================

def to_bgr_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)

    if img.ndim == 2:
        img_u8 = img
        if img_u8.dtype != np.uint8:
            img_u8 = cv2.normalize(img_u8, None, 0, 255, cv2.NORM_MINMAX)
            img_u8 = img_u8.astype(np.uint8)
        return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 3:
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)
        return img.copy()

    raise ValueError(f"Unsupported image shape: {img.shape}")


def draw_indexed_points(
    image_bgr: np.ndarray,
    points_uv: np.ndarray,
    *,
    title: str,
) -> np.ndarray:
    vis = image_bgr.copy()
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)

    # Draw grid lines
    for row in range(STEPS_PER_EDGE + 1):
        row_idx = [row * (STEPS_PER_EDGE + 1) + col for col in range(STEPS_PER_EDGE + 1)]
        row_pts = pts[row_idx].astype(np.int32)
        cv2.polylines(vis, [row_pts], False, (180, 180, 180), 1, cv2.LINE_AA)

    for col in range(STEPS_PER_EDGE + 1):
        col_idx = [row * (STEPS_PER_EDGE + 1) + col for row in range(STEPS_PER_EDGE + 1)]
        col_pts = pts[col_idx].astype(np.int32)
        cv2.polylines(vis, [col_pts], False, (180, 180, 180), 1, cv2.LINE_AA)

    # Draw all points and indices
    for i, (u, v) in enumerate(pts):
        p = (int(round(u)), int(round(v)))

        cv2.circle(vis, p, 4, (0, 255, 255), -1, cv2.LINE_AA)

        if i % 10 == 0 or i in IMPORTANT_IDX:
            cv2.putText(
                vis,
                str(i),
                (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Highlight important corners
    colors = {
        0: (0, 0, 255),       # red
        10: (0, 255, 0),      # green
        110: (255, 0, 0),     # blue
        120: (255, 0, 255),   # magenta
    }

    for idx in IMPORTANT_IDX:
        u, v = pts[idx]
        p = (int(round(u)), int(round(v)))
        cv2.circle(vis, p, 12, colors[idx], 3, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"{idx}",
            (p[0] + 14, p[1] + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colors[idx],
            2,
            cv2.LINE_AA,
        )

    # Draw outer square explicitly
    outer = pts[[0, 10, 120, 110, 0]].astype(np.int32)
    cv2.polylines(vis, [outer], False, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(
        vis,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return vis


# ============================================================
# Main
# ============================================================

def main() -> None:
    npz_path = select_file(
        "Select Overlay / Cam2X NPZ containing xray_points_uv",
        "NPZ files (*.npz)",
    )
    if npz_path is None:
        print("No NPZ selected.")
        return

    img_path = select_file(
        "Select PCB X-ray image",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
    )
    if img_path is None:
        print("No X-ray image selected.")
        return

    data = np.load(npz_path, allow_pickle=True)

    print("\nLoaded NPZ:")
    print(npz_path)
    print("\nAvailable keys:")
    for k in data.files:
        arr = data[k]
        print(f"  {k:30s} shape={arr.shape} dtype={arr.dtype}")

    if "xray_points_uv" not in data.files:
        raise KeyError("NPZ does not contain 'xray_points_uv'.")

    xray_points_uv = np.asarray(data["xray_points_uv"], dtype=np.float64).reshape(-1, 2)

    expected_n = (STEPS_PER_EDGE + 1) ** 2
    if xray_points_uv.shape[0] != expected_n:
        raise ValueError(
            f"Expected {expected_n} X-ray points for {STEPS_PER_EDGE + 1}x{STEPS_PER_EDGE + 1} grid, "
            f"got {xray_points_uv.shape[0]}."
        )

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    xray_bgr = to_bgr_image(img)

    print("\nX-ray image:")
    print(img_path)
    print(f"  shape = {img.shape}")
    print(f"  dtype = {img.dtype}")

    print("\nImportant X-ray points:")
    for idx in IMPORTANT_IDX:
        u, v = xray_points_uv[idx]
        print(f"  idx {idx:3d}: u={u:9.3f}, v={v:9.3f}")

    vis = draw_indexed_points(
        xray_bgr,
        xray_points_uv,
        title="X-ray correspondence order: xray_points_uv",
    )

    out_path = npz_path.with_name(npz_path.stem + "_debug_xray_flipping_corr.png")
    cv2.imwrite(str(out_path), vis)

    print("\nSaved debug image:")
    print(out_path)

    cv2.imshow("debug_xray_flipping_corr", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()