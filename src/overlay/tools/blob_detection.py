import numpy as np
import cv2
from dataclasses import dataclass
from typing import Union, Sequence, Optional



# ============================================================
# Parameter definition
# ============================================================

@dataclass(frozen=True)
class HoughCircleParams:
    """
    Parameter set for Hough-based circular blob detection.
    """

    min_radius: int = 3
    max_radius: int = 7
    dp: float = 1.2
    minDist: float = 26
    param1: float = 120
    param2: float = 8
    invert: bool = True
    median_ks: Union[int, Sequence[int]] = (3, 5)


def _validate_params(params: HoughCircleParams) -> None:
    """Sanity checks for Hough parameters."""
    if params.dp <= 0:
        raise ValueError("dp must be > 0.")
    if params.minDist <= 0:
        raise ValueError("minDist must be > 0.")
    if params.min_radius < 0 or params.max_radius < 0:
        raise ValueError("Radii must be >= 0.")
    if params.max_radius and params.min_radius > params.max_radius:
        raise ValueError("min_radius must be <= max_radius.")
    if params.param1 <= 0 or params.param2 <= 0:
        raise ValueError("param1 and param2 must be > 0.")
    if isinstance(params.median_ks, (list, tuple)):
        for ks in params.median_ks:
            if ks < 0:
                raise ValueError("median_ks values must be >= 0.")
    else:
        if params.median_ks < 0:
            raise ValueError("median_ks must be >= 0.")


# ============================================================
# Blob scoring
# ============================================================

def _circle_blob_score(
    img_u8: np.ndarray,
    x: float,
    y: float,
    r: float,
    *,
    rin_factor: float = 0.60,
    rout_factor: float = 1.40,
) -> float:
    """
    Compute a local blobness score for a circle candidate.

    The score is defined as:

        mean(inner_disk) - mean(ring)

    Returns
    -------
    float
        Positive values indicate strong bright-blob contrast.
        Returns -1e9 if the local support is invalid.
    """
    if img_u8.ndim != 2 or img_u8.dtype != np.uint8:
        raise ValueError("_circle_blob_score expects uint8 grayscale image.")

    if r < 1.0 or not np.isfinite(x) or not np.isfinite(y):
        return -1e9

    rin = max(1.0, rin_factor * r)
    rout = max(rin + 1.0, rout_factor * r)

    h, w = img_u8.shape

    x0 = int(max(0, np.floor(x - rout - 2)))
    x1 = int(min(w, np.ceil(x + rout + 2)))
    y0 = int(max(0, np.floor(y - rout - 2)))
    y1 = int(min(h, np.ceil(y + rout + 2)))

    roi = img_u8[y0:y1, x0:x1]
    if roi.size == 0:
        return -1e9

    yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
    cx = x - x0
    cy = y - y0
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2

    disk = d2 <= rin ** 2
    ring = (d2 > rin ** 2) & (d2 <= rout ** 2)

    disk_px = disk.sum()
    ring_px = ring.sum()

    exp_disk = np.pi * rin ** 2
    exp_ring = np.pi * (rout ** 2 - rin ** 2)

    disk_min_px = max(3, int(0.30 * exp_disk))
    ring_min_px = max(4, int(0.15 * exp_ring))

    if disk_px < disk_min_px or ring_px < ring_min_px:
        return -1e9

    return float(np.mean(roi[disk]) - np.mean(roi[ring]))


# ============================================================
# Geometry helpers
# ============================================================

def _enforce_min_spacing(
    c_xy: np.ndarray,
    scores: np.ndarray,
    min_sep: float,
) -> np.ndarray:
    """
    Enforce minimum spacing between detections.

    Keeps highest scoring detections first and removes
    neighbors closer than min_sep.
    """
    order = np.argsort(-scores)
    keep = []
    min2 = min_sep ** 2

    for idx in order:
        xi, yi = c_xy[idx, 0], c_xy[idx, 1]
        ok = True

        for j in keep:
            dx = xi - c_xy[j, 0]
            dy = yi - c_xy[j, 1]
            if dx * dx + dy * dy < min2:
                ok = False
                break

        if ok:
            keep.append(idx)

    return c_xy[keep]


# ============================================================
# Duplicate suppression
# ============================================================

def _merge_near_duplicates(
    circles: np.ndarray,
    *,
    dist_thr: float,
    score_img: np.ndarray,
) -> np.ndarray:
    """
    Remove duplicate detections using score-based NMS
    and pitch-based minimum spacing enforcement.
    """
    c = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    if c.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    scores = np.array(
        [_circle_blob_score(score_img, x, y, r) for (x, y, r) in c],
        dtype=np.float64,
    )

    valid = scores > -1e-8
    c = c[valid]
    scores = scores[valid]

    if c.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    order = np.argsort(-scores)
    thr2 = dist_thr ** 2
    kept = []

    for idx in order:
        xi, yi = c[idx, :2]
        ok = True

        for j in kept:
            dx = xi - c[j, 0]
            dy = yi - c[j, 1]
            if dx * dx + dy * dy <= thr2:
                ok = False
                break

        if ok:
            kept.append(idx)

    c_kept = c[kept]
    scores_kept = scores[kept]
    pitch_px = estimate_pitch_nn(c_kept[:, :2])

    c_final = _enforce_min_spacing(c_kept, scores_kept, min_sep=0.75 * pitch_px)
    return c_final.astype(np.float32)


# ============================================================
# Public API
# ============================================================


def estimate_pitch_nn(c_xy: np.ndarray) -> float:
    """
    Estimate grid pitch from nearest-neighbor distances (isotropic).

    Parameters
    ----------
    c_xy : (N,2)
        Point coordinates.

    Returns
    -------
    float
        Estimated pitch in pixels. Returns NaN if not enough valid distances.
    """
    import numpy as np

    xy = np.asarray(c_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("c_xy must have shape (N,2).")

    xy = xy[:, :2]
    n = xy.shape[0]
    if n < 2:
        return np.nan

    eps = 1e-6
    dmin = np.empty(n, dtype=np.float64)

    for i in range(n):
        dx = xy[:, 0] - xy[i, 0]
        dy = xy[:, 1] - xy[i, 1]

        d2 = dx * dx + dy * dy
        d2[i] = np.inf
        d2[d2 <= eps] = np.inf

        m = np.min(d2)
        dmin[i] = np.sqrt(m) if np.isfinite(m) else np.nan

    dmin = dmin[np.isfinite(dmin)]
    if dmin.size == 0:
        return np.nan

    return float(np.median(dmin))


def detect_blobs_hough(
    image: np.ndarray,
    params: Optional[HoughCircleParams] = None,
) -> Optional[np.ndarray]:
    """
    Detect circular blobs using HoughCircles with
    robust duplicate suppression.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Input must be single-channel uint8 image.")

    if params is None:
        params = HoughCircleParams()

    _validate_params(params)

    img = np.ascontiguousarray(image)
    img_score = img.copy()

    if params.invert:
        img = 255 - img

    ks_values = (
        list(params.median_ks)
        if isinstance(params.median_ks, (list, tuple))
        else [params.median_ks]
    )

    detections = []

    for ks in ks_values:
        img_run = img

        if ks and ks > 1:
            k = ks if ks % 2 else ks + 1
            img_run = cv2.medianBlur(img_run, k)

        circles = cv2.HoughCircles(
            img_run,
            cv2.HOUGH_GRADIENT,
            dp=params.dp,
            minDist=params.minDist,
            param1=params.param1,
            param2=params.param2,
            minRadius=params.min_radius,
            maxRadius=params.max_radius,
        )

        if circles is not None:
            detections.append(circles.reshape(-1, 3))

    if not detections:
        return None

    circles = np.vstack(detections)
    
    pitch_px = estimate_pitch_nn(circles[:, :2])
    dist_thr = 0.50 * pitch_px
    
    circles = _merge_near_duplicates(
        circles,
        dist_thr=dist_thr,
        score_img=img_score,
    )
    
    return circles.astype(np.float32)