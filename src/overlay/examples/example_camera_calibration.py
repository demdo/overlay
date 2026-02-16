# example_camera_calibration.py
# -*- coding: utf-8 -*-
"""
Capture N RGB images with SPACE (only if ChArUco board found),
then auto-calibrate intrinsics, close live video immediately,
and save ONLY the 3x3 intrinsic matrix as a JSON array
in the SAME DIRECTORY as this script.

JSON format:
[
  [fx,  0, cx],
  [ 0, fy, cy],
  [ 0,  0,  1]
]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import cv2
import pyrealsense2 as rs

from overlay.calib.calibration_camera import (
    calibrate_charuco_intrinsics,
    detect_charuco,
)

# ============================================================
# USER SETTINGS
# ============================================================

NUM_IMAGES_TARGET = 10

# RGB stream
RGB_W, RGB_H, RGB_FPS = 1280, 720, 30

# --- Confirmed ChArUco board ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

SQUARES_X = 9
SQUARES_Y = 7
SQUARE_LEN_M = 0.0254     # 25.40 mm
MARKER_LEN_M = 0.01778    # 17.78 mm

FOUND_THRESHOLD = 12
CALIB_MIN_CORNERS = 12

# JSON will be written next to this script
JSON_PATH = Path(__file__).with_name("rgb_intrinsics.json")


def make_charuco_board():
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LEN_M,
            MARKER_LEN_M,
            ARUCO_DICT,
        )
    return cv2.aruco.CharucoBoard_create(
        SQUARES_X,
        SQUARES_Y,
        SQUARE_LEN_M,
        MARKER_LEN_M,
        ARUCO_DICT,
    )


def draw_status(img: np.ndarray, found: bool, det, captured: int) -> None:
    bg = (0, 160, 0) if found else (0, 0, 160)
    cv2.rectangle(img, (10, 10), (760, 60), bg, -1)

    cv2.putText(
        img,
        "BOARD FOUND" if found else "BOARD NOT FOUND",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    line = f"ChArUco={det.num_charuco} | ArUco={det.num_aruco} | Captured {captured}/{NUM_IMAGES_TARGET}"
    cv2.putText(
        img,
        line,
        (10, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_overlay(vis: np.ndarray, det) -> None:
    if det.aruco_ids is not None and len(det.aruco_ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, det.aruco_corners, det.aruco_ids)

    if det.charuco_corners is not None:
        for (u, v) in det.charuco_corners.reshape(-1, 2):
            cv2.circle(vis, (int(round(u)), int(round(v))), 3, (255, 255, 0), -1)


def main():
    board = make_charuco_board()
    detector_params = cv2.aruco.DetectorParameters()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, RGB_W, RGB_H, rs.format.bgr8, RGB_FPS)
    pipeline.start(config)

    captured: List[np.ndarray] = []
    show_overlay = True

    win = "ChArUco RGB Capture (SPACE)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        # ---------------- capture loop ----------------
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue

            frame = np.asanyarray(color.get_data())

            det = detect_charuco(frame, board, ARUCO_DICT, detector_params=detector_params)
            found = det.num_charuco >= FOUND_THRESHOLD

            vis = frame.copy()
            if show_overlay:
                draw_overlay(vis, det)
            draw_status(vis, found, det, len(captured))

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                return

            if key == ord("d"):
                show_overlay = not show_overlay

            if key == 32:  # SPACE
                if not found:
                    continue

                captured.append(frame.copy())
                print(f"[CAPTURE] {len(captured)}/{NUM_IMAGES_TARGET}")

                if len(captured) >= NUM_IMAGES_TARGET:
                    break

    finally:
        # close live video immediately after last capture
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

    # ---------------- calibration ----------------
    print("[INFO] Running intrinsics calibration...")
    K, _, rms, _ = calibrate_charuco_intrinsics(
        calib_images=captured,
        board=board,
        aruco_dict=ARUCO_DICT,
        detector_params=detector_params,
        min_charuco_corners=CALIB_MIN_CORNERS,
        flags=0,
    )

    print("\nK:\n", K)
    print("RMS reprojection error [px]:", rms)

    # Save ONLY the 3x3 intrinsic matrix
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(K.tolist(), f, indent=2)

    print(f"[INFO] Saved intrinsics to {JSON_PATH.resolve()}")


if __name__ == "__main__":
    main()
