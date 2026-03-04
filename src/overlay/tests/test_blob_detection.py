# -*- coding: utf-8 -*-
"""
test_blob_detection.py

Pure blob detection viewer:
- Load X-ray via QFileDialog
- Apply SAME preprocessing as before (e.g. CLAHE)
- Run detect_blobs_hough(...)
- Show overlay with detected circles
- Q / ESC quits
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.blob_detection import HoughCircleParams, detect_blobs_hough

WIN = "test_blob_detection (Q/ESC quit)"


def _ensure_qt_app() -> None:
    if QApplication.instance() is None:
        QApplication(sys.argv)


def _pick_image_path() -> str:
    _ensure_qt_app()
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select X-ray image",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.ima);;All files (*.*)",
    )
    return path


def _load_xray_gray_u8(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in (".dcm", ".ima"):
        try:
            import pydicom
        except Exception as e:
            raise RuntimeError("pydicom required for .dcm/.ima. Install: pip install pydicom") from e

        ds = pydicom.dcmread(str(p))
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img.astype(np.uint8)

    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Could not read image.")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _apply_clahe(img_u8: np.ndarray, clip: float = 2.0, tiles=(12, 12)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tiles))
    return clahe.apply(img_u8)


def _render_overlay(img_u8: np.ndarray, circles: Optional[np.ndarray]) -> np.ndarray:
    out = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    if circles is None or len(circles) == 0:
        return out

    for (x, y, r) in np.asarray(circles, dtype=np.float64).reshape(-1, 3):
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(r)):
            continue
        u, v = int(round(x)), int(round(y))
        rr = max(2, int(round(r)))
        cv2.circle(out, (u, v), rr, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(out, (u, v), 1, (0, 255, 0), -1, cv2.LINE_AA)
    return out


def main() -> None:
    path = _pick_image_path()
    if not path:
        print("No image selected.")
        return

    img = _load_xray_gray_u8(path)
    H, W = img.shape[:2]

    # --- SAME behavior as before: CLAHE ON ---
    img_proc = _apply_clahe(img, clip=2.0, tiles=(12, 12))

    # --- your params (as you used before) ---
    params = HoughCircleParams(
        min_radius=2,
        max_radius=7,
        dp=1.2,
        minDist=8,
        param1=120,
        param2=9,
        invert=True,
        median_ks=(3, 5),
    )

    circles = detect_blobs_hough(img_proc, params=params)
    if circles is None:
        circles = np.empty((0, 3), dtype=np.float32)
            
    # debug
    base = Path(r"C:\Users\domin\Documents\Studium\Master\Masterarbeit\Projekt\Data")
    np.savez_compressed(
        base / "Circles_View4__grid.npz",
        circles=circles,
    )

    overlay = _render_overlay(img, circles)  # draw on ORIGINAL (wie vorher “X-ray + overlay”)

    print(f"[detect] circles: {len(circles)}")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)
    cv2.imshow(WIN, overlay)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()