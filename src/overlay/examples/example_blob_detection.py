# -*- coding: utf-8 -*-
"""
example_blob_detection.py

Test for blob detection helpers.
"""

import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.blob_detection import HoughCircleParams, detect_blobs_hough


# ============================================================
# UI: open file dialog (Qt / PySide6) - Spyder compatible
# ============================================================
def open_image_file() -> str:
    """
    Open a file dialog to select an image file.

    Returns
    -------
    path : str
        Selected file path.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select X-ray calibration image",
        "",
        "Image files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*.*)"
    )
    return path


# ============================================================
# Display helper
# ============================================================
def show(img: np.ndarray, title: str, scale: float = 1.0, wait: bool = False) -> None:
    """
    Show an image with a window title. Optionally resize for convenience.

    Parameters
    ----------
    img : np.ndarray
        Image to display (uint8 grayscale/BGR or float).
    title : str
        Window title.
    scale : float
        Resize factor for the window (not image resampling, window resize).
    wait : bool
        If True, block with cv2.waitKey(0) after showing.
    """
    disp = img
    if disp.dtype != np.uint8:
        disp = disp.astype(np.float32)
        disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
        disp = disp.astype(np.uint8)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    if scale != 1.0:
        h, w = disp.shape[:2]
        cv2.resizeWindow(title, int(w * scale), int(h * scale))

    cv2.imshow(title, disp)
    if wait:
        cv2.waitKey(0)


# ============================================================
# Circle fit (Kåsa least squares)
# ============================================================
def fit_circle_kasa(points_xy: np.ndarray):
    """
    Algebraic circle fit (Kåsa):
      x^2 + y^2 = 2xc x + 2yc y + c

    Parameters
    ----------
    points_xy : np.ndarray
        Array of points of shape (N, 2).

    Returns
    -------
    circle : tuple[float, float, float] or None
        (xc, yc, r) or None if fitting fails / insufficient points.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] < 30:
        return None

    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2

    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    xc, yc, c = sol
    r2 = c + xc**2 + yc**2
    if r2 <= 0:
        return None

    r = np.sqrt(r2)
    return float(xc), float(yc), float(r)


# ============================================================
# Tight detector mask via radial gradient sampling
# ============================================================
def detector_mask_radial(
    img_u8: np.ndarray,
    n_angles: int = 360,
    r_min_frac: float = 0.20,
    r_max_frac: float = 0.98,
    smooth_sigma: float = 2.0,
    peak_prominence: float = 0.0,
    shrink_px: int = 12
):
    """
    Estimate the detector circle boundary by sampling gradient magnitude along rays.

    Parameters
    ----------
    img_u8 : np.ndarray
        Input grayscale uint8 image.
    n_angles : int
        Number of rays (directions) sampled.
    r_min_frac : float
        Radial search start as fraction of min(image_dim)/2.
    r_max_frac : float
        Radial search end as fraction of min(image_dim)/2.
    smooth_sigma : float
        Gaussian blur sigma before computing gradients.
    peak_prominence : float
        Optional threshold; discard rays where max gradient < this value.
    shrink_px : int
        Shrink fitted radius by this many pixels to remove detector rim.

    Returns
    -------
    mask_u8 : np.ndarray
        Binary mask (0/255) of shape (H, W).
    circle : tuple[float, float, float]
        (cx, cy, r) fitted detector circle after shrinking.
    grad_mag : np.ndarray
        Gradient magnitude image used for boundary detection (float32).
    boundary_pts : np.ndarray
        Boundary points used for circle fitting, shape (M, 2).
    """
    if img_u8.ndim != 2 or img_u8.dtype != np.uint8:
        raise ValueError("detector_mask_radial expects a uint8 grayscale image.")

    h, w = img_u8.shape
    cx0, cy0 = w / 2.0, h / 2.0
    R0 = 0.5 * min(h, w)

    img_blur = cv2.GaussianBlur(img_u8, (0, 0), smooth_sigma)
    gx = cv2.Sobel(img_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    r_min = int(max(5, r_min_frac * R0))
    r_max = int(max(r_min + 10, r_max_frac * R0))

    boundary_pts = []
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False).astype(np.float32)
    rs = np.arange(r_min, r_max, dtype=np.float32)

    for th in angles:
        xs = cx0 + rs * np.cos(th)
        ys = cy0 + rs * np.sin(th)

        xs_i = np.clip(xs, 0, w - 1).astype(np.int32)
        ys_i = np.clip(ys, 0, h - 1).astype(np.int32)

        prof = grad_mag[ys_i, xs_i]
        k = int(np.argmax(prof))

        if peak_prominence > 0.0 and prof[k] < peak_prominence:
            continue

        boundary_pts.append([xs[k], ys[k]])

    boundary_pts = np.array(boundary_pts, dtype=np.float32)

    circle = fit_circle_kasa(boundary_pts)
    if circle is None:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return mask, (cx0, cy0, R0), grad_mag, boundary_pts

    cx, cy, r = circle
    r = max(1.0, r - float(shrink_px))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)

    return mask, (cx, cy, r), grad_mag, boundary_pts


# ============================================================
# Main
# ============================================================
def main():
    # -----------------------------
    # Your final Hough settings
    # -----------------------------
    params = HoughCircleParams(
        min_radius=3,
        max_radius=7,
        dp=1.2,
        minDist=26,
        param1=120,
        param2=8,
        invert=True,
        median_ks=5,
    )

    # -----------------------------
    # Preprocessing
    # -----------------------------
    use_clahe = True
    clahe_clip = 2.0
    clahe_tiles = (12, 12)

    # -----------------------------
    # Detector mask extraction
    # -----------------------------
    n_angles = 360
    smooth_sigma = 2.0
    r_min_frac = 0.20
    r_max_frac = 0.98
    shrink_px = 12  # adjust 8..20

    # -----------------------------
    # Load image
    # -----------------------------
    path = open_image_file()
    if not path:
        raise RuntimeError("No image selected.")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    show(img, "Step 1: Original X-ray image", scale=0.9)

    # -----------------------------
    # Step 2: CLAHE
    # -----------------------------
    img_proc = img.copy()
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tiles)
        img_proc = clahe.apply(img_proc)
    show(img_proc, "Step 2: CLAHE contrast normalization", scale=0.9)

    # -----------------------------
    # Step 3+4: Gradient magnitude + tight detector mask
    # -----------------------------
    mask, (cx, cy, r), grad_mag, boundary_pts = detector_mask_radial(
        img_proc,
        n_angles=n_angles,
        r_min_frac=r_min_frac,
        r_max_frac=r_max_frac,
        smooth_sigma=smooth_sigma,
        peak_prominence=0.0,
        shrink_px=shrink_px,
    )

    show(grad_mag, "Step 3: Gradient magnitude (detector boundary)", scale=0.9)
    show(mask, "Step 4: Detector mask (tight circular ROI)", scale=0.9)

    # -----------------------------
    # Step 5: Masked image
    # -----------------------------
    img_masked = img_proc.copy()
    img_masked[mask == 0] = 0
    show(img_masked, "Step 5: Masked image (inside detector only)", scale=0.9)

    # -----------------------------
    # Step 6+7: Hough detection via your modular function
    # -----------------------------
    circles_out = detect_blobs_hough(img_masked, params)

    # Visualization
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if circles_out is not None:
        for (x, y, rr) in circles_out:
            cv2.circle(vis, (int(round(x)), int(round(y))), int(round(rr)), (0, 255, 0), 1)
            cv2.drawMarker(
                vis,
                (int(round(x)), int(round(y))),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=1,
            )

    show(vis, "Step 7: Final marker detections (Hough circles)", scale=0.9)

    print(f"Detector circle: cx={cx:.1f}, cy={cy:.1f}, r={r:.1f} (shrink_px={shrink_px})")
    print("Detected circles:", 0 if circles_out is None else len(circles_out))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()






























