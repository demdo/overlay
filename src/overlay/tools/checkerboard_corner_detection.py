# -*- coding: utf-8 -*-
"""
checkerboard_corner_detection.py

Pure checkerboard detection helpers
"""

from dataclasses import dataclass
import cv2
import numpy as np

from overlay.tools.homography import estimate_homography_dlt, project_homography


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


def _sort_corners_row_major(corners_9x2: np.ndarray) -> np.ndarray:
    """
    Sorts 9 corners into row-major order (top-to-bottom, left-to-right)
    regardless of OpenCV's detection order.
    """
    pts = corners_9x2.reshape(9, 2)
    # Sort by v (row), then by u (col) within each row
    idx = np.lexsort((pts[:, 0], pts[:, 1]))  # primary: v, secondary: u
    return pts[idx]


# ============================================================
# Extreme corners (geometric, robust to ordering/rotation)
# ============================================================
def get_extreme_corners_geometric(corners_full):
    """
    corners_full: (N,1,2)

    Determine TL / TR / BL directly in image coordinates:

      TL = top-left
      TR = top-right
      BL = bottom-left

    Strategy
    --------
    For a checkerboard with pattern_size = (w, h), OpenCV returns N = w*h corners.
    Here we only need the three geometric extreme corners in the IMAGE:

    1) sort all detected corners by image y-coordinate
    2) take the top row and bottom row
    3) sort each of those by image x-coordinate
    4) define:
         TL = leftmost point of top row
         TR = rightmost point of top row
         BL = leftmost point of bottom row

    This matches the visual interpretation used in plane fitting and PnP.

    Returns
    -------
    dict
        {
            "top_left": (u, v),
            "top_right": (u, v),
            "bottom_left": (u, v),
        }
    """
    pts = np.asarray(corners_full, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]

    if n < 4:
        raise ValueError("At least 4 checkerboard corners are required.")

    # Infer checkerboard width/height from the detected point count.
    # For the current use case this is 3x3 -> 9 points, so row_len = 3.
    row_len = int(round(np.sqrt(n)))
    if row_len * row_len != n:
        raise ValueError(
            f"Could not infer a square checkerboard layout from {n} corners."
        )

    # Sort by image y to separate top and bottom rows
    idx_sorted_y = np.argsort(pts[:, 1])
    top = pts[idx_sorted_y[:row_len]]
    bottom = pts[idx_sorted_y[-row_len:]]

    # Sort within each row by image x
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    tl = top[0]
    tr = top[-1]
    bl = bottom[0]
    br = bottom[-1]

    return {
        "top_left":     (float(tl[0]), float(tl[1])),
        "top_right":    (float(tr[0]), float(tr[1])),
        "bottom_left":  (float(bl[0]), float(bl[1])),
        "bottom_right": (float(br[0]), float(br[1])),
    }


# ============================================================
# Grid interpolation from 9 checkerboard corners (no depth)
# ============================================================
def interpolate_grid_uv(
    corners_uv: np.ndarray,
    nrows: int = 11,
    ncols: int = 11,
    pitch_mm: float = 2.54,
    corner_step: int = 5,
) -> np.ndarray:
    """
    Interpolate a full nrows x ncols grid in UV image space from 9
    checkerboard corners. No depth used.

    A homography H is estimated (least-squares over all 9 corners) mapping
    board coordinates (mm) to image UV (pixels). H is then used to map
    all nrows*ncols grid points. This is exact under perspective projection
    regardless of the camera viewing angle.

    Board geometry
    --------------
    The 9 corners lie at grid indices 0, corner_step, 2*corner_step on
    both axes. Their physical board coordinates are:

        corner (row_c, col_c)  ->  board_xy = (col_c * corner_step * pitch_mm,
                                               row_c * corner_step * pitch_mm)

    Parameters
    ----------
    corners_uv : (9, 2) ndarray
        The 9 subpixel-refined checkerboard corners in image UV coordinates.
        Row-major order: corners_uv[0]=TL, corners_uv[2]=TR,
                         corners_uv[6]=BL, corners_uv[8]=BR.
    nrows : int
        Number of grid rows (default 11).
    ncols : int
        Number of grid columns (default 11).
    pitch_mm : float
        Physical pitch between adjacent grid points in mm (default 2.54).
    corner_step : int
        Grid steps between adjacent checkerboard corners (default 5).
        Corner spacing in mm = corner_step * pitch_mm = 12.7 mm.

    Returns
    -------
    grid_uv : (nrows * ncols, 2) ndarray, float64
        UV coordinates in row-major order: point (i, j) at index i*ncols + j.
    """
    
    corners_uv = np.asarray(corners_uv, dtype=np.float64).reshape(9, 2)
    corners_uv = _sort_corners_row_major(np.asarray(corners_uv).reshape(9, 2))

    # Board coordinates of the 9 checkerboard corners (row-major, Z=0 plane)
    corner_board_xy = np.array(
        [
            [col_c * corner_step * pitch_mm, row_c * corner_step * pitch_mm]
            for row_c in range(3)
            for col_c in range(3)
        ],
        dtype=np.float64,
    )  # (9, 2)

    # H maps board_mm -> image_uv (least-squares, all 9 points)
    H = estimate_homography_dlt(uv_img=corners_uv, XY_grid=corner_board_xy)

    # All nrows*ncols board coordinates
    all_board_xy = np.array(
        [
            [j * pitch_mm, i * pitch_mm]
            for i in range(nrows)
            for j in range(ncols)
        ],
        dtype=np.float64,
    )  # (121, 2)

    return project_homography(H, all_board_xy)