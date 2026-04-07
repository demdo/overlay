# -*- coding: utf-8 -*-
"""
debug_cam_kosy.py

Öffnet die RGB-Kamera der RealSense, zeigt das Livebild
und speichert bei SPACE einen Snapshot mit eingezeichnetem
Bildkoordinatensystem:

- u: nach rechts
- v: nach unten

ESC beendet das Skript.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs


def draw_image_axes(
    image_bgr: np.ndarray,
    origin: tuple[int, int] = (80, 80),
    axis_len_px: int = 140,
) -> np.ndarray:
    """
    Zeichnet das 2D-Bildkoordinatensystem in das Bild:
      u -> rechts
      v -> unten
    """
    vis = image_bgr.copy()

    ox, oy = origin

    # Ursprung markieren
    cv2.circle(vis, (ox, oy), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(vis, (ox, oy), 7, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    # u-Achse (nach rechts)
    u_end = (ox + axis_len_px, oy)
    cv2.arrowedLine(
        vis,
        (ox, oy),
        u_end,
        (0, 0, 255),   # rot
        3,
        line_type=cv2.LINE_AA,
        tipLength=0.12,
    )
    cv2.putText(
        vis,
        "u (+x img)",
        (u_end[0] + 10, u_end[1] + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # v-Achse (nach unten)
    v_end = (ox, oy + axis_len_px)
    cv2.arrowedLine(
        vis,
        (ox, oy),
        v_end,
        (0, 255, 0),   # gruen
        3,
        line_type=cv2.LINE_AA,
        tipLength=0.12,
    )
    cv2.putText(
        vis,
        "v (+y img)",
        (v_end[0] - 10, v_end[1] + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Zusatztext
    cv2.putText(
        vis,
        "Image coordinates: origin top-left",
        (30, vis.shape[0] - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        "u -> right, v -> down",
        (30, vis.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return vis


def main() -> None:
    out_dir = Path("debug_cam_kosy_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    # Wie bei dir typischerweise genutzt:
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    try:
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()

        print("=" * 60)
        print("RGB CAMERA INTRINSICS")
        print("=" * 60)
        print(f"width  = {intr.width}")
        print(f"height = {intr.height}")
        print(f"fx     = {intr.fx:.6f}")
        print(f"fy     = {intr.fy:.6f}")
        print(f"ppx    = {intr.ppx:.6f}")
        print(f"ppy    = {intr.ppy:.6f}")
        print(f"model  = {intr.model}")
        print(f"coeffs = {list(intr.coeffs)}")
        print()
        print("Bildkoordinaten-Konvention im RAW-Bild:")
        print("  u/x: nach rechts")
        print("  v/y: nach unten")
        print()

        win_title = "debug_cam_kosy"
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_title, 1280, 720)

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())

            vis = draw_image_axes(color_bgr)

            cv2.putText(
                vis,
                "SPACE = snapshot, ESC = quit",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win_title, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            elif key == 32:  # SPACE
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"cam_kosy_snapshot_{ts}.png"
                cv2.imwrite(str(out_path), vis)
                print(f"[INFO] Snapshot gespeichert: {out_path}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()