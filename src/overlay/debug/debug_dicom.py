from __future__ import annotations

import sys

import cv2
import numpy as np
import pydicom
from PySide6.QtWidgets import QApplication, QFileDialog


def _get_app() -> tuple[QApplication, bool]:
    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True
    return app, owns_app


def _select_file() -> str:
    app, owns_app = _get_app()

    dlg = QFileDialog(None, "Select X-ray image (RAW, no rotation)")
    dlg.setFileMode(QFileDialog.ExistingFile)
    dlg.setNameFilter("Images (*.dcm *.ima *.png *.jpg *.bmp *.tif *.tiff);;All Files (*)")
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)

    if dlg.exec() != QFileDialog.Accepted:
        if owns_app:
            app.quit()
        raise RuntimeError("No file selected.")

    files = dlg.selectedFiles()
    path = files[0] if files else ""
    if not path:
        if owns_app:
            app.quit()
        raise RuntimeError("No file selected.")

    if owns_app:
        app.quit()

    return path


def _load_gray(path: str) -> np.ndarray:
    if path.lower().endswith((".dcm", ".ima")):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img.astype(np.uint8)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read: {path}")
    return img


def _draw_cross(img_bgr: np.ndarray, u: int, v: int, color, size: int = 20, thickness: int = 2) -> None:
    cv2.line(img_bgr, (u - size, v), (u + size, v), color, thickness, cv2.LINE_AA)
    cv2.line(img_bgr, (u, v - size), (u, v + size), color, thickness, cv2.LINE_AA)


def main() -> None:
    print("=" * 72)
    print("debug_vaxis — v-axis direction check")
    print("=" * 72)
    print()
    print("This script draws crosses at fixed u, increasing v.")
    print("Look at the image window and answer:")
    print("  Do the crosses go from TOP to BOTTOM?  → v grows downward (OpenCV OK)")
    print("  Do the crosses go from BOTTOM to TOP?  → v grows upward   (RAW is flipped!)")
    print()

    path = _select_file()
    img = _load_gray(path)
    h, w = img.shape[:2]

    print(f"Image size: {w} x {h}")
    print()

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Fixed u = image center
    u = w // 2

    # 5 crosses at increasing v: 10%, 25%, 50%, 75%, 90% of height
    v_fractions = [0.10, 0.25, 0.50, 0.75, 0.90]
    colors = [
        (0,   255, 0),    # green   — v smallest (should be top if OK)
        (0,   255, 255),  # yellow
        (0,   165, 255),  # orange
        (0,   0,   255),  # red
        (255, 0,   255),  # magenta — v largest (should be bottom if OK)
    ]
    labels = ["v=10%", "v=25%", "v=50%", "v=75%", "v=90%"]

    for frac, color, label in zip(v_fractions, colors, labels):
        v = int(round(frac * (h - 1)))
        _draw_cross(img_bgr, u, v, color, size=24, thickness=3)
        cv2.putText(
            img_bgr,
            f"{label}  (u={u}, v={v})",
            (u + 30, v + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        print(f"  {label}: u={u}, v={v}  → color {color}")

    print()
    print("GREEN   = smallest v")
    print("MAGENTA = largest v")
    print()
    print("If GREEN is at the TOP    → v grows downward → OpenCV convention OK ✓")
    print("If GREEN is at the BOTTOM → v grows upward   → image is flipped!  ✗")
    print()
    print("Press any key to close.")

    win = "debug_vaxis — GREEN=small v, MAGENTA=large v (press any key to close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 900), min(h, 900))
    cv2.imshow(win, img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()