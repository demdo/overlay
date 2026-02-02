# -*- coding: utf-8 -*-
"""
checkerboard_corner_detection.py

Live RGB preview with FAST status indicator (board detected or not).
- Live: detect on DOWNSCALED grayscale using classic findChessboardCorners (fast + stable).
- SPACE capture is only allowed if LIVE status == FOUND.
- Snapshot: re-use the same detection (downscaled), map corners to full-res, then refine with cornerSubPix.
  Fallback: try findChessboardCornersSB if classic fails.
- Draw all corners + highlight 3 extreme corners (turquoise).
- Return pixel coordinates of those 3 extreme corners if accepted.

Keys:
  SPACE -> capture (only if FOUND)
  ESC   -> quit
"""

import time
import cv2
import numpy as np
import pyrealsense2 as rs
import tkinter as tk
from tkinter import messagebox


# =========================
# Window config
# =========================
LIVE_WIN = "Live RGB (SPACE=capture if FOUND, ESC=quit)"
RES_WIN  = "Corner Detection Result"

WIN_W, WIN_H = 1280, 720
FULLSCREEN = False


def setup_window(name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, WIN_W, WIN_H)
    if FULLSCREEN:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# ============================================================
# Popup helper (Yes/No)
# ============================================================
def ask_satisfied(title="Corner Detection", msg="Satisfied with detected corners?"):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ans = messagebox.askyesno(title, msg)
    root.destroy()
    return ans


# ============================================================
# RealSense RGB-only setup
# ============================================================
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


# ============================================================
# Helpers: scale / detection
# ============================================================
def resize_keep_aspect(gray, target_width):
    h, w = gray.shape[:2]
    if w <= target_width:
        return gray, 1.0
    s = target_width / float(w)
    small = cv2.resize(gray, (target_width, int(round(h * s))), interpolation=cv2.INTER_AREA)
    return small, s


def detect_classic_downscaled(gray_full, pattern_size, det_width=640):
    """
    Fast detector: classic findChessboardCorners on downscaled image.
    Returns:
      found (bool),
      corners_full_init (N,1,2) float32 in FULL-res coordinates (initial, not subpixel-refined)
    """
    gray_small, s = resize_keep_aspect(gray_full, det_width)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners_small = cv2.findChessboardCorners(gray_small, pattern_size, flags=flags)

    if not found:
        return False, None

    # Map corners back to full-res coordinates
    corners_small = corners_small.astype(np.float32)
    corners_full = corners_small.copy()
    corners_full[:, 0, 0] /= s
    corners_full[:, 0, 1] /= s

    return True, corners_full


def refine_subpix(gray_full, corners_full):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    cv2.cornerSubPix(gray_full, corners_full, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)
    return corners_full


def detect_snapshot_full(color_bgr, pattern_size=(3, 3), det_width=640):
    """
    Snapshot detection:
    1) classic on downscale -> map to full -> subpix refine (fast + consistent with live)
    2) fallback: SB on full-res if classic fails
    Returns (found, corners_full_subpix)
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)

    found, corners_full = detect_classic_downscaled(gray, pattern_size, det_width=det_width)
    if found and corners_full is not None:
        corners_full = refine_subpix(gray, corners_full)
        return True, corners_full

    # Fallback: SB (more expensive)
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found_sb, corners_sb = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags_sb)
    if not found_sb:
        return False, None

    corners_sb = corners_sb.astype(np.float32)
    corners_sb = refine_subpix(gray, corners_sb)
    return True, corners_sb


# ============================================================
# Extreme corners (geometric, robust to ordering/rotation)
# ============================================================
def get_extreme_corners_geometric(corners_full):
    """
    corners_full: (N,1,2)

    Robust extremes based on geometry:
      TL = min(x+y)
      TR = max(x-y)
      BL = min(x-y)

    Returns dict of 3 corners (u,v) float tuples.
    """
    pts = corners_full.reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]

    s = x + y
    d = x - y

    tl = pts[np.argmin(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]

    return {
        "top_left": (float(tl[0]), float(tl[1])),
        "top_right": (float(tr[0]), float(tr[1])),
        "bottom_left": (float(bl[0]), float(bl[1])),
    }


def draw_extremes(img_bgr, extremes, color=(208, 224, 64), radius=10, thickness=-1):
    out = img_bgr.copy()
    for name, (u, v) in extremes.items():
        cv2.circle(out, (int(round(u)), int(round(v))), radius, color, thickness, lineType=cv2.LINE_AA)
        cv2.putText(out, name, (int(round(u)) + 12, int(round(v)) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


# ============================================================
# Main workflow
# ============================================================
def main():
    pattern_size = (3, 3)   # 4x4 squares -> 3x3 inner corners
    det_width = 640         # downscale width for live + snapshot consistency

    pipeline = start_realsense_rgb(1920, 1080, 30)
    setup_window(LIVE_WIN)

    accepted_extremes = None

    # live: keep last status to reduce flicker (optional)
    found_live = False

    try:
        while True:
            frames = pipeline.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue

            live_color = np.asanyarray(cf.get_data())
            gray_live = cv2.cvtColor(live_color, cv2.COLOR_BGR2GRAY)

            # --- FAST live status (downscaled classic)
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

            # Only allow capture if board is detected
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

                satisfied = ask_satisfied(
                    title="Corner Detection",
                    msg="Satisfied with the detected corners?\n\nYes: finish and keep result\nNo: return to live video"
                )

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


if __name__ == "__main__":
    extremes = main()
    if extremes is None:
        print("\n[INFO] No accepted detection (user cancelled or not satisfied).")
    else:
        print("\n[RETURN] extreme corners (pixel coords):")
        print(extremes)
