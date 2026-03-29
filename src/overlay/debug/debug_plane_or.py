# -*- coding: utf-8 -*-
"""
debug_plane_or.py

Debug script to verify where the FIRST 11 points of xray_points_xyz_c
lie on the checkerboard image.

Behavior
--------
- Opens Intel RealSense RGB stream in full HD
- Live view uses the SAME fast checkerboard detection as PlaneFittingPage:
    cbd.detect_classic_downscaled(...)
- Press SPACE to capture a snapshot
- On the snapshot, uses the SAME full corner detection as PlaneFittingPage:
    cbd.detect_snapshot_full(...)
- Computes the SAME 3 geometric extreme corners used in PlaneFittingPage:
    top_left, top_right, bottom_left
- Interpolates an 11x11 image grid from these 3 corners
- Draws the FIRST 11 interpolated points (indices 0..10) on the snapshot

Meaning of the shown points
---------------------------
These are the image-plane points corresponding to the same ordering later used
for xray_points_xyz_c in PlaneFittingPage:
- row-major order
- first row first
- left to right
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

WINDOW_NAME = "debug_plane_or"


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


def draw_extremes(img: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    out = img.copy()
    for (u, v) in pts_uv:
        cv2.circle(
            out,
            (int(round(u)), int(round(v))),
            10,
            (208, 224, 64),
            -1,
            cv2.LINE_AA,
        )
    return out


def draw_first_11_points(img: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    out = img.copy()

    n = min(11, len(pts_uv))
    for i in range(n):
        u, v = pts_uv[i]
        uu = int(round(float(u)))
        vv = int(round(float(v)))

        # red cross
        cv2.line(out, (uu - 8, vv), (uu + 8, vv), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(out, (uu, vv - 8), (uu, vv + 8), (0, 0, 255), 2, cv2.LINE_AA)

        # index label
        cv2.putText(
            out,
            str(i),
            (uu + 8, vv - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return out


def draw_snapshot_result(
    frame_bgr: np.ndarray,
    corners: np.ndarray | None,
    found: bool,
    ext_uv: np.ndarray | None,
    interp_uv: np.ndarray | None,
) -> np.ndarray:
    vis = frame_bgr.copy()

    if found and corners is not None:
        cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, True)

    if ext_uv is not None:
        vis = draw_extremes(vis, ext_uv)

    if interp_uv is not None:
        vis = draw_first_11_points(vis, interp_uv)

    return vis


# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------
def interpolate_grid_uv(
    corner_uv: np.ndarray,
    steps_per_edge: int,
) -> np.ndarray:
    """
    Same ordering logic as interpolate_marker_grid(...), but in image space.

    corner_uv: (3,2) = [top_left, top_right, bottom_left]
    returns: ((s+1)*(s+1), 2) in row-major order
    """
    if steps_per_edge <= 0:
        raise ValueError("steps_per_edge must be > 0.")

    corner_uv = np.asarray(corner_uv, dtype=np.float64)
    if corner_uv.shape != (3, 2):
        raise ValueError("corner_uv must have shape (3,2).")

    p_tl, p_tr, p_bl = corner_uv
    step_x = (p_tr - p_tl) / float(steps_per_edge)
    step_y = (p_bl - p_tl) / float(steps_per_edge)

    points = []
    for beta in range(steps_per_edge + 1):
        for alpha in range(steps_per_edge + 1):
            points.append(p_tl + alpha * step_x + beta * step_y)

    return np.asarray(points, dtype=np.float64)


def print_first_11_uv(pts_uv: np.ndarray) -> None:
    print("=" * 72)
    print("FIRST 11 INTERPOLATED IMAGE POINTS (same order as xyz[0:11])")
    print("=" * 72)
    for i in range(min(11, len(pts_uv))):
        u, v = pts_uv[i]
        print(f"{i:2d}: u={u:10.3f}, v={v:10.3f}")
    print()


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

        print("=" * 72)
        print("debug_plane_or")
        print("=" * 72)
        print("SPACE -> capture snapshot and show first 11 interpolated points")
        print("R     -> return to live view")
        print("ESC/Q -> quit")
        print()

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

                ext_uv = None
                interp_uv = None

                if found_snap and corners is not None:
                    ex = cbd.get_extreme_corners_geometric(corners)

                    ext_uv = np.array(
                        [
                            ex["top_left"],
                            ex["top_right"],
                            ex["bottom_left"],
                        ],
                        dtype=np.float64,
                    )

                    # IMPORTANT:
                    # Same 11x11 ordering as later used for xyz generation
                    interp_uv = interpolate_grid_uv(ext_uv, steps_per_edge=10)

                    print_first_11_uv(interp_uv)

                snap_vis = draw_snapshot_result(
                    snap_frame,
                    corners,
                    found_snap,
                    ext_uv,
                    interp_uv,
                )

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