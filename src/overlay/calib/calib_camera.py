# overlay/calib/camera_calibration.py
# -*- coding: utf-8 -*-
"""
ChArUco / ArUco camera calibration utilities.

- ONLY ChArUco / ArUco
- Robust for OpenCV 4.x (uses interpolateCornersCharuco)
- Supports:
    * Intrinsics calibration from multiple RGB images
    * Pose estimation on new images
    * Average reprojection error on new images
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2


# =========================
# Data container
# =========================

@dataclass
class CharucoDetection:
    charuco_corners: Optional[np.ndarray]   # (N,1,2)
    charuco_ids: Optional[np.ndarray]       # (N,1)
    aruco_corners: List[np.ndarray]
    aruco_ids: Optional[np.ndarray]
    num_charuco: int
    num_aruco: int


# =========================
# Helpers
# =========================

def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _image_size(img: np.ndarray) -> Tuple[int, int]:
    g = _ensure_gray(img)
    h, w = g.shape[:2]
    return (w, h)


def detect_charuco(
    image: np.ndarray,
    board: Any,
    aruco_dict: Any,
    detector_params: Optional[Any] = None,
) -> CharucoDetection:
    """
    Detect ArUco markers and interpolate ChArUco corners.
    """
    gray = _ensure_gray(image)

    if detector_params is None:
        detector_params = cv2.aruco.DetectorParameters()

    # ArUco detection
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        aruco_corners, aruco_ids, _ = detector.detectMarkers(gray)
    else:
        aruco_corners, aruco_ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=detector_params
        )

    charuco_corners = None
    charuco_ids = None

    if aruco_ids is not None and len(aruco_ids) > 0:
        ret, cc, ci = cv2.aruco.interpolateCornersCharuco(
            markerCorners=aruco_corners,
            markerIds=aruco_ids,
            image=gray,
            board=board,
        )
        if ret is not None and ret > 0:
            charuco_corners, charuco_ids = cc, ci

    return CharucoDetection(
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
        aruco_corners=aruco_corners,
        aruco_ids=aruco_ids,
        num_charuco=0 if charuco_ids is None else int(len(charuco_ids)),
        num_aruco=0 if aruco_ids is None else int(len(aruco_ids)),
    )


def _charuco_object_points(board: Any, charuco_ids: np.ndarray) -> np.ndarray:
    """
    Map ChArUco IDs to 3D board points (z=0).
    """
    all_obj = board.getChessboardCorners()  # (Nc,3)
    ids = charuco_ids.reshape(-1).astype(int)
    return all_obj[ids, :].astype(np.float32)


# =========================
# Intrinsics calibration
# =========================

def calibrate_charuco_intrinsics(
    calib_images: Sequence[np.ndarray],
    board: Any,
    aruco_dict: Any,
    detector_params: Optional[Any] = None,
    min_charuco_corners: int = 12,
    flags: int = 0,
    criteria: Optional[Tuple[int, int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """
    Estimate camera intrinsics from multiple calibration images.
    """
    if len(calib_images) == 0:
        raise ValueError("No calibration images provided.")

    if criteria is None:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-6,
        )

    image_size = _image_size(calib_images[0])

    all_charuco_corners = []
    all_charuco_ids = []
    used_idx = []

    per_img_charuco = []
    per_img_aruco = []

    for i, img in enumerate(calib_images):
        if _image_size(img) != image_size:
            raise ValueError("All images must have same resolution.")

        det = detect_charuco(img, board, aruco_dict, detector_params)
        per_img_charuco.append(det.num_charuco)
        per_img_aruco.append(det.num_aruco)

        if det.charuco_ids is None or det.charuco_corners is None:
            continue
        if det.num_charuco < min_charuco_corners:
            continue

        all_charuco_corners.append(det.charuco_corners)
        all_charuco_ids.append(det.charuco_ids)
        used_idx.append(i)

    if len(all_charuco_corners) < 3:
        raise RuntimeError("Not enough valid views for calibration.")

    K_init = np.eye(3, dtype=np.float64)
    dist_init = np.zeros((5, 1), dtype=np.float64)

    rms, K, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=K_init,
        distCoeffs=dist_init,
        flags=flags,
        criteria=criteria,
    )

    stats = {
        "image_size": image_size,
        "num_images_total": len(calib_images),
        "num_images_used": len(all_charuco_corners),
        "used_indices": used_idx,
        "per_image_num_charuco": per_img_charuco,
        "per_image_num_aruco": per_img_aruco,
        "rms": float(rms),
    }

    return K, dist, float(rms), stats


# =========================
# Pose + reprojection error
# =========================

def estimate_charuco_pose(
    image: np.ndarray,
    board: Any,
    aruco_dict: Any,
    K: np.ndarray,
    dist: np.ndarray,
    detector_params: Optional[Any] = None,
    min_charuco_corners: int = 8,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], CharucoDetection, bool]:
    """
    Estimate board pose in a single image.
    """
    det = detect_charuco(image, board, aruco_dict, detector_params)

    if det.charuco_ids is None or det.charuco_corners is None:
        return None, None, det, False
    if det.num_charuco < min_charuco_corners:
        return None, None, det, False

    obj_pts = _charuco_object_points(board, det.charuco_ids)
    img_pts = det.charuco_corners.reshape(-1, 2).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )

    return rvec, tvec, det, bool(ok)


def reprojection_error_charuco(
    test_images: Sequence[np.ndarray],
    board: Any,
    aruco_dict: Any,
    K: np.ndarray,
    dist: np.ndarray,
    detector_params: Optional[Any] = None,
    min_charuco_corners: int = 8,
) -> Tuple[float, List[Optional[float]], Dict[str, Any]]:
    """
    Compute average reprojection error (px) on NEW images.
    """
    per_view = []
    all_err = []

    per_img_charuco = []
    per_img_aruco = []
    used_idx = []

    for i, img in enumerate(test_images):
        rvec, tvec, det, ok = estimate_charuco_pose(
            img, board, aruco_dict, K, dist, detector_params, min_charuco_corners
        )

        per_img_charuco.append(det.num_charuco)
        per_img_aruco.append(det.num_aruco)

        if not ok:
            per_view.append(None)
            continue

        obj_pts = _charuco_object_points(board, det.charuco_ids)
        obs = det.charuco_corners.reshape(-1, 2).astype(np.float32)

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)

        err = np.linalg.norm(obs - proj, axis=1)
        mean_err = float(np.mean(err))

        per_view.append(mean_err)
        all_err.append(mean_err)
        used_idx.append(i)

    mean_px = float(np.mean(all_err)) if all_err else float("nan")

    stats = {
        "num_images_total": len(test_images),
        "num_images_valid": len(all_err),
        "used_indices": used_idx,
        "per_image_num_charuco": per_img_charuco,
        "per_image_num_aruco": per_img_aruco,
    }

    return mean_px, per_view, stats
