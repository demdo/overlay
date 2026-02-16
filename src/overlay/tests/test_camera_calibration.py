# overlay/calib/tests/test_camera_calibration.py
# -*- coding: utf-8 -*-
"""
Parinaz-style calibration accuracy test (RGB):

Quote target:
"To assess the accuracy of our camera calibration, we used the toolbox function
cv2.projectPoints to calculate the average re-projection error for the corner points
of the ArUco Board in the RGB frame."

What this script does:
- Loads previously calibrated RGB intrinsics K from JSON (stored as plain 3x3 array).
- Starts a RealSense RGB live video.
- Live view: shows ONLY ArUco info (marker boxes + IDs + a status banner).
- On SPACE:
    * Captures a single test frame
    * Detects ChArUco corners (measured 2D points)
    * Estimates board pose (rvec, tvec) with solvePnP (dist intentionally ignored)
    * Reprojects corresponding 3D board points with cv2.projectPoints (projected 2D points)
    * Computes and prints average reprojection error (px)
    * Shows a frozen image with ONLY:
        - measured points = CYAN rings
        - projected points = MAGENTA crosses
      (no marker drawings, so colors are unambiguous)

Keys:
  SPACE      -> evaluate current frame (freeze view)
  SPACE      -> from freeze view: return to live
  ESC / q    -> quit (live or freeze)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2
import pyrealsense2 as rs


# ============================================================
# CONFIG
# ============================================================

# RealSense RGB stream
RGB_W, RGB_H, RGB_FPS = 1280, 720, 30

# Intrinsics JSON is in calib/examples (test is in calib/tests)
INTRINSICS_JSON = (Path(__file__).resolve().parents[1] / "examples" / "rgb_intrinsics.json")

# --- Confirmed ChArUco board ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
SQUARES_X = 9
SQUARES_Y = 7
SQUARE_LEN_M = 0.0254     # 25.40 mm
MARKER_LEN_M = 0.01778    # 17.78 mm

# Require enough corners for a meaningful test
MIN_CHARUCO_CORNERS = 12


# ============================================================
# HELPERS
# ============================================================

def load_K(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Intrinsics JSON not found.\nExpected at: {path}\n"
            f"(test is in calib/tests, JSON is in calib/examples)"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.array(data, dtype=float)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 K matrix in {path}, got shape {K.shape}")
    return K


def make_charuco_board():
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LEN_M, MARKER_LEN_M, ARUCO_DICT)
    return cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LEN_M, MARKER_LEN_M, ARUCO_DICT)


def detect_charuco(gray: np.ndarray, board, detector_params):
    """
    Returns:
      aruco_corners, aruco_ids, num_aruco,
      charuco_corners, charuco_ids, num_charuco
    """
    # ArUco markers
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, detector_params)
        aruco_corners, aruco_ids, _ = detector.detectMarkers(gray)
    else:
        aruco_corners, aruco_ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=detector_params)

    num_aruco = 0 if aruco_ids is None else len(aruco_ids)

    # ChArUco corners (interpolated)
    charuco_corners, charuco_ids = None, None
    num_charuco = 0
    if aruco_ids is not None and len(aruco_ids) > 0:
        ret, cc, ci = cv2.aruco.interpolateCornersCharuco(
            markerCorners=aruco_corners,
            markerIds=aruco_ids,
            image=gray,
            board=board,
        )
        if ret is not None and ret > 0 and cc is not None and ci is not None:
            charuco_corners, charuco_ids = cc, ci
            num_charuco = len(ci)

    return aruco_corners, aruco_ids, num_aruco, charuco_corners, charuco_ids, num_charuco


def charuco_object_points(board, charuco_ids: np.ndarray) -> np.ndarray:
    """
    Map ChArUco IDs to corresponding 3D board points (z=0).
    """
    all_obj = board.getChessboardCorners()  # (Nc,3)
    ids = charuco_ids.reshape(-1).astype(int)
    return all_obj[ids, :].astype(np.float32)


def compute_reprojection_error_px(measured_xy: np.ndarray, projected_xy: np.ndarray) -> Tuple[float, float]:
    """
    measured_xy, projected_xy: (N,2)
    Returns (mean_px, max_px)
    """
    err = np.linalg.norm(measured_xy - projected_xy, axis=1)
    return float(np.mean(err)), float(np.max(err))


def draw_points_freeze(img: np.ndarray, measured_xy: np.ndarray, projected_xy: np.ndarray,
                       mean_err: float, max_err: float) -> np.ndarray:
    """
    Freeze visualization:
    - measured: CYAN rings (thick)
    - projected: MAGENTA crosses
    """
    out = img.copy()

    # Measured: cyan rings
    for (u, v) in measured_xy:
        p = (int(round(u)), int(round(v)))
        cv2.circle(out, p, 6, (255, 255, 0), 2, cv2.LINE_AA)   # cyan ring
        cv2.circle(out, p, 1, (255, 255, 0), -1, cv2.LINE_AA)  # tiny center

    # Projected: magenta crosses
    for (u, v) in projected_xy:
        u, v = int(round(u)), int(round(v))
        cv2.line(out, (u - 6, v), (u + 6, v), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.line(out, (u, v - 6), (u, v + 6), (255, 0, 255), 2, cv2.LINE_AA)

    # Legend + numbers
    cv2.rectangle(out, (10, 10), (820, 92), (0, 0, 0), -1)
    cv2.putText(out, "Measured = CYAN rings   |   Projected = MAGENTA crosses",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"Avg reprojection error: {mean_err:.4f} px   |   Max: {max_err:.4f} px   |   N={len(measured_xy)}",
                (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    K = load_K(INTRINSICS_JSON)

    # Distortion intentionally ignored in your pipeline:
    # we pass zeros to OpenCV functions.
    dist = np.zeros((5, 1), dtype=float)

    print("[INFO] Loaded intrinsics K from:", INTRINSICS_JSON.resolve())
    print(K)
    print("[INFO] Distortion is ignored for this test (dist = zeros).")

    board = make_charuco_board()
    detector_params = cv2.aruco.DetectorParameters()

    # RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, RGB_W, RGB_H, rs.format.bgr8, RGB_FPS)
    pipeline.start(config)

    win = "Test camera calibration (SPACE evaluate, SPACE back, ESC quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("[INFO] Live view: ArUco boxes + IDs only.")
    print("[INFO] SPACE -> compute reprojection error and show freeze visualization.")
    print("[INFO] ESC/q -> quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            frame = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ar_corners, ar_ids, num_aruco, cc, ci, num_charuco = detect_charuco(gray, board, detector_params)
            found = num_charuco >= MIN_CHARUCO_CORNERS

            # Live view: only ArUco marker drawing + status
            vis = frame.copy()
            if ar_ids is not None and len(ar_ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, ar_corners, ar_ids)

            bg = (0, 160, 0) if found else (0, 0, 160)
            cv2.rectangle(vis, (10, 10), (700, 60), bg, -1)
            cv2.putText(vis, f"AruCo markers: {num_aruco} | ChArUco corners: {num_charuco}",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if key == 32:  # SPACE -> evaluate
                if not found or cc is None or ci is None:
                    print(f"[SKIP] Not enough ChArUco corners (have {num_charuco}, need {MIN_CHARUCO_CORNERS}).")
                    continue

                # Measured points (2D)
                img_pts = cc.reshape(-1, 2).astype(np.float32)

                # Corresponding 3D points
                obj_pts = charuco_object_points(board, ci)

                # Pose estimation
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    print("[SKIP] solvePnP failed.")
                    continue

                # Reproject
                proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
                proj = proj.reshape(-1, 2).astype(np.float32)

                mean_err, max_err = compute_reprojection_error_px(img_pts, proj)
                print(f"[RESULT] Avg reprojection error: {mean_err:.4f} px | Max: {max_err:.4f} px | N={len(img_pts)}")

                frozen = draw_points_freeze(frame, img_pts, proj, mean_err, max_err)
                cv2.imshow(win, frozen)

                # Freeze until SPACE (back to live) or ESC/q (quit)
                while True:
                    k2 = cv2.waitKey(0) & 0xFF
                    if k2 in (27, ord("q")):
                        return
                    if k2 == 32:  # SPACE -> back to live
                        break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
