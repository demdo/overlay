# -*- coding: utf-8 -*-
"""
checkerboard_corner_detection.py

Pure checkerboard detection helpers
"""

from dataclasses import dataclass
import cv2
import numpy as np


# ============================================================
# Helpers: scale / detection
# ============================================================
def resize_keep_aspect(gray, target_width):
    h, w = gray.shape[:2]
    if w <= target_width:
        return gray, 1.0
    s = target_width / float(w)
    small = cv2.resize(gray, (target_width, int(round(h * s))), interpolation=cv2.INTER_AREA)
    return small, s


def detect_classic_downscaled(gray_full, pattern_size, det_width=640):
    """
    Fast detector: classic findChessboardCorners on downscaled image.
    Returns:
      found (bool),
      corners_full_init (N,1,2) float32 in FULL-res coordinates (initial, not subpixel-refined)
    """
    if gray_full.size == 0:
        return False, None
    if pattern_size[0] <= 0 or pattern_size[1] <= 0:
        raise ValueError("pattern_size must be positive.")
        
    gray_small, s = resize_keep_aspect(gray_full, det_width)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners_small = cv2.findChessboardCorners(gray_small, pattern_size, flags=flags)

    if not found:
        return False, None

    # Map corners back to full-res coordinates
    corners_small = corners_small.astype(np.float32)
    corners_full = corners_small.copy()
    corners_full[:, 0, 0] /= s
    corners_full[:, 0, 1] /= s

    return True, corners_full


def refine_subpix(gray_full, corners_full):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    cv2.cornerSubPix(gray_full, corners_full, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)
    return corners_full


def detect_snapshot_full(color_bgr, pattern_size=(3, 3), det_width=640):
    """
    Snapshot detection:
    1) classic on downscale -> map to full -> subpix refine (fast + consistent with live)
    2) fallback: SB on full-res if classic fails
    Returns (found, corners_full_subpix)
    """
    if color_bgr.size == 0:
        return False, None
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)

    found, corners_full = detect_classic_downscaled(gray, pattern_size, det_width=det_width)
    if found and corners_full is not None:
        corners_full = refine_subpix(gray, corners_full)
        return True, corners_full

    # Fallback: SB (more expensive)
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found_sb, corners_sb = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags_sb)
    if not found_sb:
        return False, None

    corners_sb = corners_sb.astype(np.float32)
    corners_sb = refine_subpix(gray, corners_sb)
    return True, corners_sb


# ============================================================
# Extreme corners (geometric, robust to ordering/rotation)
# ============================================================
def get_extreme_corners_geometric(corners_full):
    """
    corners_full: (N,1,2)

    Robust extremes based on geometry:
      TL = min(x+y)
      TR = max(x-y)
      BL = min(x-y)

    Returns dict of 3 corners (u,v) float tuples.
    """
    pts = corners_full.reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]

    s = x + y
    d = x - y

    tl = pts[np.argmin(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]

    return {
        "top_left": (float(tl[0]), float(tl[1])),
        "top_right": (float(tr[0]), float(tr[1])),
        "bottom_left": (float(bl[0]), float(bl[1])),
    }



