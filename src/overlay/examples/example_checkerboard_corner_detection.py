# -*- coding: utf-8 -*-
"""
example_checkerboard_corner_detection.py

Interactive test runner for checkerboard_corner_detection.
"""

import time
import cv2
import numpy as np
import pyrealsense2 as rs
import tkinter as tk
from tkinter import messagebox

from overlay.tools.checkerboard_corner_detection import (
    detect_classic_downscaled,
    detect_snapshot_full,
    get_extreme_corners_geometric,
)


LIVE_WIN = "Live RGB (SPACE=capture if FOUND, ESC=quit)"
RES_WIN = "Corner Detection Result"
WIN_W, WIN_H = 1280, 720
FULLSCREEN = False


def setup_window(name: str) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, WIN_W, WIN_H)
    if FULLSCREEN:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def ask_satisfied(
    title: str = "Corner Detection",
    msg: str = "Satisfied with the detected corners?\n\nYes: finish and keep result\nNo: return to live video",
) -> bool:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ans = messagebox.askyesno(title, msg)
    root.destroy()
    return ans


def start_realsense_rgb(width=1920, height=1080, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)

    vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
    print(f"[RealSense] RGB stream: {vsp.width()}x{vsp.height()} @ {vsp.fps()} FPS, format={vsp.format()}")

    # Avoid lag: drop old frames
    try:
        dev = profile.get_device()
        for s in dev.query_sensors():
            name = s.get_info(rs.camera_info.name)
            if "RGB" in name or "Color" in name:
                s.set_option(rs.option.frames_queue_size, 1)

                # Optional AE settle + freeze (helps stability)
                s.set_option(rs.option.enable_auto_exposure, 1)
                time.sleep(0.3)
                s.set_option(rs.option.enable_auto_exposure, 0)
                print("[RealSense] frames_queue_size=1, auto-exposure frozen.")
                break
    except Exception as e:
        print("[RealSense] Could not set options:", e)

    return pipeline


def draw_extremes(img_bgr, extremes, color=(208, 224, 64), radius=10, thickness=-1):
    out = img_bgr.copy()
    for name, (u, v) in extremes.items():
        cv2.circle(out, (int(round(u)), int(round(v))), radius, color, thickness, lineType=cv2.LINE_AA)
        cv2.putText(out, name, (int(round(u)) + 12, int(round(v)) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


def run_checkerboard_test(pattern_size=(3, 3), det_width=640, width=1920, height=1080, fps=30):
    pipeline = start_realsense_rgb(width, height, fps)
    setup_window(LIVE_WIN)

    accepted_extremes = None
    found_live = False

    try:
        while True:
            frames = pipeline.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue

            live_color = np.asanyarray(cf.get_data())
            gray_live = cv2.cvtColor(live_color, cv2.COLOR_BGR2GRAY)

            found_now, _ = detect_classic_downscaled(gray_live, pattern_size, det_width=det_width)
            found_live = found_now

            vis_live = live_color.copy()
            if found_live:
                cv2.putText(vis_live, "FOUND (press SPACE to capture)", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.putText(vis_live, "NOT FOUND", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow(LIVE_WIN, vis_live)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            if key == 32 and found_live:  # SPACE
                snap = live_color.copy()

                found, corners = detect_snapshot_full(snap, pattern_size=pattern_size, det_width=det_width)

                result = snap.copy()
                if found and corners is not None:
                    cv2.drawChessboardCorners(result, pattern_size, corners, True)

                    extremes = get_extreme_corners_geometric(corners)
                    result = draw_extremes(result, extremes, color=(208, 224, 64), radius=10, thickness=-1)

                    cv2.putText(result, "FOUND", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

                    print("\n[Extreme corners - pixel coords (u,v)]")
                    for k, v in extremes.items():
                        print(f"  {k:12s}: ({v[0]:.2f}, {v[1]:.2f})")
                else:
                    extremes = None
                    cv2.putText(result, "NOT FOUND", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

                setup_window(RES_WIN)
                cv2.imshow(RES_WIN, result)

                satisfied = ask_satisfied()

                if satisfied and found and extremes is not None:
                    accepted_extremes = extremes
                    cv2.destroyWindow(LIVE_WIN)

                    print("\n[INFO] Accepted. Final extreme corners returned.")
                    print(accepted_extremes)

                    while True:
                        cv2.imshow(RES_WIN, result)
                        k2 = cv2.waitKey(10) & 0xFF
                        if k2 == 27:
                            break
                    break
                else:
                    cv2.destroyWindow(RES_WIN)
                    setup_window(LIVE_WIN)
                    continue

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return accepted_extremes


extreme_corners_px = run_checkerboard_test()

# extreme_corners_px is either:
#   None
# or:
#   {
#     "top_left": (u, v),
#     "top_right": (u, v),
#     "bottom_left": (u, v)
#   }

if extreme_corners_px is None:
    print("[TEST] No accepted detection (cancelled or not satisfied).")
else:
    print("[TEST] Accepted extreme corners (pixel coords):")
    for k, (u, v) in extreme_corners_px.items():
        print(f"  {k:12s}: ({u:.2f}, {v:.2f})")

    # If you want them as numpy array (3x2):
    # import numpy as np
    # extreme_corners_arr = np.array([
    #     extreme_corners_px["top_left"],
    #     extreme_corners_px["top_right"],
    #     extreme_corners_px["bottom_left"],
    # ], dtype=np.float32)