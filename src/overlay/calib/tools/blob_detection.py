import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HoughCircleParams:
    """
    Parameter set for Hough-based circular blob detection.

    Attributes
    ----------
    min_radius : int
        Minimum circle radius in pixels.
    max_radius : int
        Maximum circle radius in pixels.
    dp : float
        Inverse ratio of the accumulator resolution to the image resolution.
    minDist : float
        Minimum distance between the centers of detected circles.
    param1 : float
        Higher threshold for the internal Canny edge detector.
    param2 : float
        Accumulator threshold for circle center detection.
        Smaller values lead to more false positives.
    invert : bool
        If True, invert the grayscale image before detection.
    median_ks : int
        Kernel size for median filtering (must be odd).
    """
    min_radius: int = 3
    max_radius: int = 7
    dp: float = 1.2
    minDist: float = 26
    param1: float = 120
    param2: float = 8
    invert: bool = True
    median_ks: int = 5


def detect_blobs_hough(
    image: np.ndarray,
    params: HoughCircleParams,
) -> Optional[np.ndarray]:
    """
    Detect circular blobs in a grayscale image using the Hough Circle Transform.

    This function performs a minimal and reusable Hough-based circle detection.
    It optionally applies image inversion and median filtering before calling
    OpenCV's HoughCircles routine.

    Parameters
    ----------
    image : np.ndarray
        Input image as a single-channel uint8 array.
    params : HoughCircleParams
        Parameter set controlling the Hough circle detection.

    Returns
    -------
    circles : np.ndarray or None
        Detected circles as an array of shape (N, 3) with entries (x, y, r),
        where (x, y) are the circle centers in pixel coordinates and r is
        the radius in pixels. Returns None if no circles are detected.

    Raises
    ------
    ValueError
        If the input image is not a single-channel uint8 array.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be single-channel (grayscale).")
    if image.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8.")

    img = image.copy()

    if params.invert:
        img = 255 - img

    if params.median_ks and params.median_ks > 1:
        k = params.median_ks if params.median_ks % 2 == 1 else params.median_ks + 1
        img = cv2.medianBlur(img, k)

    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=params.dp,
        minDist=params.minDist,
        param1=params.param1,
        param2=params.param2,
        minRadius=params.min_radius,
        maxRadius=params.max_radius,
    )

    if circles is None:
        return None

    return circles[0].astype(np.float32)
