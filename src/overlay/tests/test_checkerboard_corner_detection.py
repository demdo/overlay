# -*- coding: utf-8 -*-
"""
test_corner_detection.py

Standalone test for checkerboard corner detection only.

Behavior
--------
- Opens Intel RealSense RGB stream in full HD (1920 x 1080)
- Live view runs the SAME fast checkerboard detection as in PlaneFittingPage:
    cbd.detect_classic_downscaled(...)
- Press SPACE to capture a snapshot
- On the snapshot, runs the SAME final corner detection as in PlaneFittingPage:
    cbd.detect_snapshot_full(...)
- Displays ONLY the detected corners on the captured image in full resolution
- Press R to return to live view
- Press ESC or Q to quit

Notes
-----
- This script is intentionally limited to corner detection only.
- No plane fitting, no depth processing, no state handling.
"""

from __future__ import annotations

import cv2
import numpy as np
import pyrealsense2 as rs

from overlay.tools import checkerboard_corner_detection as cbd


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
PATTERN_SIZE = (3, 3)
DET_WIDTH = 640

COLOR_SIZE = (1920, 1080)
FPS = 30

WINDOW_NAME = "Checkerboard Corner Detection"


# ------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------
def draw_live_status(frame_bgr: np.ndarray, found: bool) -> np.ndarray:
    vis = frame_bgr.copy()

    txt = "FOUND (press SPACE)" if found else "NOT FOUND"
    col = (0, 255, 0) if found else (0, 0, 255)

    cv2.putText(
        vis,
        txt,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        6,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        txt,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        col,
        2,
        cv2.LINE_AA,
    )
    return vis


def draw_snapshot_corners(frame_bgr, corners, found):

    vis = frame_bgr.copy()

    if found and corners is not None:

        pts = corners.reshape(-1,2).astype(int)

        # --- Linien (wie OpenCV) ---
        for i in range(len(pts)-1):
            cv2.line(
                vis,
                tuple(pts[i]),
                tuple(pts[i+1]),
                (255,0,0),   # BGR -> Blau
                2,
                cv2.LINE_AA
            )

        # --- Punkte ---
        for p in pts:
            cv2.circle(
                vis,
                tuple(p),
                6,
                (0,255,0),   # Grün
                -1,
                cv2.LINE_AA
            )

    return vis


# ------------------------------------------------------------
# RealSense helpers
# ------------------------------------------------------------
def start_realsense_color(
    color_size: tuple[int, int] = (1920, 1080),
    fps: int = 30,
) -> tuple[rs.pipeline, rs.config]:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        color_size[0],
        color_size[1],
        rs.format.bgr8,
        fps,
    )

    pipeline.start(config)
    return pipeline, config


def get_color_frame_bgr(pipeline: rs.pipeline) -> np.ndarray | None:
    frames = pipeline.poll_for_frames()
    if not frames:
        return None

    cf = frames.get_color_frame()
    if not cf:
        return None

    color = np.asanyarray(cf.get_data())
    return color


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    pipeline = None

    try:
        pipeline, _ = start_realsense_color(color_size=COLOR_SIZE, fps=FPS)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, COLOR_SIZE[0], COLOR_SIZE[1])

        mode = "live"   # "live" | "frozen"
        live_frame = None
        snap_frame = None
        snap_vis = None
        live_found = False

        while True:
            if mode == "live":
                frame = get_color_frame_bgr(pipeline)
                if frame is not None:
                    live_frame = frame

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    found, _ = cbd.detect_classic_downscaled(
                        gray,
                        PATTERN_SIZE,
                        det_width=DET_WIDTH,
                    )
                    live_found = bool(found)

                if live_frame is not None:
                    vis = draw_live_status(live_frame, live_found)
                    cv2.imshow(WINDOW_NAME, vis)

            elif mode == "frozen":
                if snap_vis is not None:
                    cv2.imshow(WINDOW_NAME, snap_vis)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            if mode == "live" and key == 32:  # SPACE
                if live_frame is None:
                    continue

                snap_frame = live_frame.copy()

                found_snap, corners = cbd.detect_snapshot_full(
                    snap_frame,
                    pattern_size=PATTERN_SIZE,
                    det_width=DET_WIDTH,
                )

                snap_vis = draw_snapshot_corners(snap_frame, corners, found_snap)
                mode = "frozen"

            elif mode == "frozen" and key in (ord("r"), ord("R")):
                snap_frame = None
                snap_vis = None
                mode = "live"

    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()