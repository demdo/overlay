# -*- coding: utf-8 -*-
"""
debug_cam_warp_perspective_xyz.py

- Lädt XYZ-Punkte (camera frame)
- fittet Ebene (SVD)
- erzeugt Top-View Homographie
- zeigt links Live RGB, rechts Draufsicht
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QFileDialog


# ============================================================
# Qt
# ============================================================

def _qt():
    return QApplication.instance() or QApplication(sys.argv)


def pick_npz(title):
    _qt()
    path, _ = QFileDialog.getOpenFileName(None, title, "", "NPZ (*.npz)")
    return Path(path) if path else None


# ============================================================
# RealSense
# ============================================================

class RealSenseRGB:
    def __init__(self):
        self.pipeline = None
        self.K = None

    def start(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)

        for _ in range(30):
            self.pipeline.wait_for_frames()

        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = stream.get_intrinsics()

        self.K = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float64)

    def frame(self):
        frames = self.pipeline.wait_for_frames()
        return np.asanyarray(frames.get_color_frame().get_data())

    def stop(self):
        self.pipeline.stop()


# ============================================================
# Load XYZ
# ============================================================

def load_xyz(path):
    with np.load(str(path)) as z:
        for k in ["points_xyz_camera_filt", "points_xyz_camera"]:
            if k in z:
                pts = np.asarray(z[k], float)
                if np.mean(np.abs(pts)) < 10:
                    pts *= 1000  # m → mm
                return pts
    raise RuntimeError("No XYZ found")


# ============================================================
# Plane fit (SVD)
# ============================================================

def fit_plane(pts):
    centroid = np.mean(pts, axis=0)
    A = pts - centroid
    _, _, Vt = np.linalg.svd(A)
    normal = Vt[-1]
    normal /= np.linalg.norm(normal)
    return normal, centroid


# ============================================================
# Basis (top view)
# ============================================================

def project_to_plane(v, n):
    return v - np.dot(v, n) * n


def build_basis(n):
    x = np.array([1, 0, 0], float)
    y = np.array([0, 1, 0], float)

    e_back = project_to_plane(x, n)
    e_right = project_to_plane(y, n)

    e_back /= np.linalg.norm(e_back)
    e_right -= np.dot(e_right, e_back) * e_back
    e_right /= np.linalg.norm(e_right)

    return e_right, e_back


# ============================================================
# Projection
# ============================================================

def project(K, P):
    uv = (K @ P.T).T
    return uv[:, :2] / uv[:, 2:3]


# ============================================================
# Main
# ============================================================

def main():
    xyz_path = pick_npz("Select XYZ NPZ")
    if xyz_path is None:
        return

    pts = load_xyz(xyz_path)

    n, center = fit_plane(pts)
    e_right, e_back = build_basis(n)

    cam = RealSenseRGB()
    cam.start()

    K = cam.K

    out_w, out_h = 800, 800
    scale = 2.0  # mm per pixel

    win = "debug_cam_warp_perspective"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        img = cam.frame()

        # --- plane patch ---
        hw = out_w * scale / 2
        hh = out_h * scale / 2

        p_tl = center + hh*e_back - hw*e_right
        p_tr = center + hh*e_back + hw*e_right
        p_br = center - hh*e_back + hw*e_right
        p_bl = center - hh*e_back - hw*e_right

        corners = np.vstack([p_tl, p_tr, p_br, p_bl])

        try:
            src = project(K, corners).astype(np.float32)
        except:
            continue

        dst = np.array([
            [0, 0],
            [out_w-1, 0],
            [out_w-1, out_h-1],
            [0, out_h-1]
        ], np.float32)

        H = cv2.getPerspectiveTransform(src, dst)
        warp = cv2.warpPerspective(img, H, (out_w, out_h))
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)

        # draw quad in live image
        cv2.polylines(img, [np.int32(src)], True, (0,255,255), 2)

        # stack
        h = max(img.shape[0], warp.shape[0])
        canvas = np.hstack([
            cv2.copyMakeBorder(img, 0, h-img.shape[0], 0, 0, 0),
            cv2.copyMakeBorder(warp, 0, h-warp.shape[0], 0, 0, 0)
        ])

        cv2.imshow(win, canvas)

        k = cv2.waitKey(1)
        if k == 27 or k == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()