# -*- coding: utf-8 -*-
"""
test_overlay.py

- load X-ray + Camera RGB images
- convert to grayscale
- apply preprocessing (CLAHE)
- show:
    * original + blobs
    * preprocessed images (used for detection)

Controls
--------
Q / ESC : quit
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.blob_detection import HoughCircleParams, detect_blobs_hough


WIN_XRAY = "X-ray blobs"
WIN_CAM = "Camera blobs"
WIN_XRAY_PROC = "X-ray preproc"
WIN_CAM_PROC = "Camera preproc"


# ============================================================
# Qt helpers
# ============================================================

def _ensure_qt_app() -> None:
    if QApplication.instance() is None:
        QApplication(sys.argv)


def _pick_image_path(title: str) -> str:
    _ensure_qt_app()
    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
    )
    return path


# ============================================================
# IO
# ============================================================

def _load_rgb_u8(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def _to_gray_u8(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.uint8)


# ============================================================
# Preprocessing
# ============================================================

def _apply_clahe(img_u8: np.ndarray, clip=2.0, tiles=(12, 12)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    return clahe.apply(img_u8)


def _preprocess_xray(img_gray: np.ndarray) -> np.ndarray:
    return _apply_clahe(img_gray, clip=2.0, tiles=(12, 12))


def _preprocess_camera(img_gray: np.ndarray) -> np.ndarray:
    return _apply_clahe(img_gray, clip=2.0, tiles=(12, 12))


# ============================================================
# Visualization
# ============================================================

def _render_overlay(img_bgr, circles, label):
    out = img_bgr.copy()

    if circles is not None and len(circles) > 0:
        for (x, y, r) in circles:
            u, v = int(round(x)), int(round(y))
            rr = max(2, int(round(r)))
            cv2.circle(out, (u, v), rr, (0, 255, 0), 2)
            cv2.circle(out, (u, v), 1, (0, 255, 0), -1)

    text = f"{label}: {0 if circles is None else len(circles)} blobs"
    cv2.putText(out, text, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return out


# ============================================================
# Main
# ============================================================

def main():

    xray_path = _pick_image_path("Select X-ray image")
    if not xray_path:
        return

    cam_path = _pick_image_path("Select camera image")
    if not cam_path:
        return

    # --- load ---
    img_xray = _load_rgb_u8(xray_path)
    img_cam = _load_rgb_u8(cam_path)

    # --- gray ---
    gray_xray = _to_gray_u8(img_xray)
    gray_cam = _to_gray_u8(img_cam)

    # --- preprocessing ---
    proc_xray = _preprocess_xray(gray_xray)
    proc_cam = _preprocess_camera(gray_cam)

    # --- params ---
    xray_params = HoughCircleParams(
        min_radius=2,
        max_radius=6,
        dp=1.2,
        minDist=12,
        param1=120,
        param2=12,
        invert=True,
        median_ks=(3,),
    )

    cam_params = HoughCircleParams(
        min_radius=2,
        max_radius=4,
        dp=1.2,
        minDist=12,
        param1=120,
        param2=12,
        invert=True,
        median_ks=(3,),
    )

    # --- detection ---
    circles_xray = detect_blobs_hough(proc_xray, xray_params)
    
    
    #circles_cam = detect_blobs_hough(proc_cam, cam_params)

    if circles_xray is None:
        circles_xray = np.empty((0, 3), dtype=np.float32)
    #if circles_cam is None:
    #    circles_cam = np.empty((0, 3), dtype=np.float32)

    print(f"[X-ray ] {len(circles_xray)} blobs")
    #print(f"[Camera] {len(circles_cam)} blobs")

    # --- overlays ---
    overlay_xray = _render_overlay(img_xray, circles_xray, "X-ray")
    #overlay_cam = _render_overlay(img_cam, circles_cam, "Camera")

    # ============================================================
    # WINDOWS
    # ============================================================

    cv2.namedWindow(WIN_XRAY, cv2.WINDOW_NORMAL)
    #cv2.namedWindow(WIN_CAM, cv2.WINDOW_NORMAL)
    #cv2.namedWindow(WIN_XRAY_PROC, cv2.WINDOW_NORMAL)
    #cv2.namedWindow(WIN_CAM_PROC, cv2.WINDOW_NORMAL)

    cv2.imshow(WIN_XRAY, overlay_xray)
    #cv2.imshow(WIN_CAM, overlay_cam)
    #cv2.imshow(WIN_XRAY_PROC, proc_xray)
    #cv2.imshow(WIN_CAM_PROC, proc_cam)

    # Layout
    cv2.moveWindow(WIN_XRAY, 50, 50)
    #cv2.moveWindow(WIN_CAM, 800, 50)
    #cv2.moveWindow(WIN_XRAY_PROC, 50, 500)
    #cv2.moveWindow(WIN_CAM_PROC, 800, 500)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()