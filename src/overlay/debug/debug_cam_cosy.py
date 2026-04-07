# -*- coding: utf-8 -*-
"""
debug_cam_cosy.py

RealSense Live-Stream mit Bildkoordinatensystem:

    u -> nach rechts
    v -> nach unten

Achsen werden im Bildzentrum eingezeichnet.
"""

from __future__ import annotations

import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# Draw axes in center
# ============================================================

def _draw_image_axes_center(
    image: np.ndarray,
    *,
    axis_len_px: int = 120,
    line_thickness: int = 2,
) -> np.ndarray:
    out = image.copy()

    h, w = out.shape[:2]
    u0 = w // 2
    v0 = h // 2

    # Ursprung
    cv2.circle(out, (u0, v0), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # u-Achse
    cv2.arrowedLine(
        out,
        (u0, v0),
        (u0 + axis_len_px, v0),
        (0, 255, 0),
        line_thickness,
        line_type=cv2.LINE_AA,
        tipLength=0.15,
    )
    cv2.putText(
        out,
        "u",
        (u0 + axis_len_px + 10, v0 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # v-Achse
    cv2.arrowedLine(
        out,
        (u0, v0),
        (u0, v0 + axis_len_px),
        (0, 200, 255),
        line_thickness,
        line_type=cv2.LINE_AA,
        tipLength=0.15,
    )
    cv2.putText(
        out,
        "v",
        (u0 - 10, v0 + axis_len_px + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )

    return out


# ============================================================
# Main
# ============================================================

def main():
    rotate_180 = False

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    profile = pipeline.start(config)

    # warmup
    for _ in range(15):
        pipeline.wait_for_frames()

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    print("\n[INFO] Intrinsics:")
    print(f"fx={intr.fx:.2f}, fy={intr.fy:.2f}, ppx={intr.ppx:.2f}, ppy={intr.ppy:.2f}")

    cv2.namedWindow("debug_cam_cosy", cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            if rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            h, w = frame.shape[:2]
            cx = w // 2
            cy = h // 2

            # Center crosshair
            cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (255, 255, 255), 1)
            cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (255, 255, 255), 1)

            # Principal point
            ppx = int(round(intr.ppx))
            ppy = int(round(intr.ppy))
            cv2.circle(frame, (ppx, ppy), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "pp",
                (ppx + 10, ppy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            # Draw axes in center
            frame = _draw_image_axes_center(frame)

            # Status
            cv2.putText(
                frame,
                f"rotate_180={rotate_180} | r toggle | q quit",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("debug_cam_cosy", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                rotate_180 = not rotate_180

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()