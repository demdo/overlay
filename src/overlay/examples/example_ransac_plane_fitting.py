# -*- coding: utf-8 -*-
"""
example_ransac_plane_fitting.py

Interactive test harness for RANSAC_plane_fitting helpers.
"""

from __future__ import annotations

import numpy as np
import cv2
import pyrealsense2 as rs
import tkinter as tk
from tkinter import messagebox

from overlay.tools import checkerboard_corner_detection as cbd
from overlay.tools import ransac_plane_fitting as rpf


LIVE_WIN = "Live RGB (SPACE=capture if FOUND, ESC=quit)"
RES_WIN = "Corner Detection Result"
PLANE_WIN = "Plane Fitting Preview (ESC=quit)"


def setup_window(name: str) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 960, 540)


def ask_corners_satisfied(
    title: str = "Corner Detection",
    msg: str = "Satisfied with the detected corners?\n\nYes: Next (plane fitting)\nNo: back to live video",
) -> bool:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ans = messagebox.askyesno(title, msg, icon="question")
    root.destroy()
    return ans


def ask_plane_satisfied(title: str = "Plane Fitting", msg: str = "Satisfied with plane fitting?\n\nYes: accept\nNo: redo") -> bool:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ans = messagebox.askyesno(title, msg, icon="question")
    root.destroy()
    return ans


def draw_extremes(img_bgr, extremes, color=(208, 224, 64), radius=10, thickness=-1):
    for name, (u, v) in extremes.items():
        cv2.circle(img_bgr, (int(u), int(v)), radius, color, thickness)
        cv2.putText(
            img_bgr,
            name,
            (int(u) + 8, int(v) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )
    return img_bgr


def start_realsense_rgbd(fps: int = 30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        dev = profile.get_device()
        for sensor in dev.query_sensors():
            try:
                sensor.set_option(rs.option.frames_queue_size, 1)
            except Exception:
                pass
    except Exception:
        pass

    return pipeline, align


def rect_from_extremes(extremes: dict[str, tuple[float, float]], img_w: int, img_h: int, pad_px: int):
    pts = np.array([extremes["top_left"], extremes["top_right"], extremes["bottom_left"]], dtype=np.float32)

    umin = int(np.floor(np.min(pts[:, 0]) - pad_px))
    umax = int(np.ceil(np.max(pts[:, 0]) + pad_px))
    vmin = int(np.floor(np.min(pts[:, 1]) - pad_px))
    vmax = int(np.ceil(np.max(pts[:, 1]) + pad_px))

    umin = max(0, min(img_w - 1, umin))
    umax = max(0, min(img_w - 1, umax))
    vmin = max(0, min(img_h - 1, vmin))
    vmax = max(0, min(img_h - 1, vmax))

    if umax < umin:
        umin, umax = umax, umin
    if vmax < vmin:
        vmin, vmax = vmax, vmin

    return (umin, vmin, umax, vmax)


def sample_points_3d_in_rect(
    depth_frame_aligned,
    rect,
    max_points: int,
    z_min: float,
    z_max: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    depth_intrin = depth_frame_aligned.profile.as_video_stream_profile().intrinsics

    umin, vmin, umax, vmax = rect
    roi_w = umax - umin + 1
    roi_h = vmax - vmin + 1
    if roi_w <= 2 or roi_h <= 2:
        return np.empty((0, 3), dtype=np.float64)

    num_pixels = roi_w * roi_h
    sample_n = min(max_points, num_pixels)
    idx = rng.choice(num_pixels, size=sample_n, replace=False)

    us = (idx % roi_w).astype(np.int32) + umin
    vs = (idx // roi_w).astype(np.int32) + vmin

    pts3d = []
    for u, v in zip(us, vs):
        z = depth_frame_aligned.get_distance(int(u), int(v))
        if z <= 0 or z < z_min or z > z_max:
            continue
        xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, [float(u), float(v)], float(z))
        pts3d.append(xyz)

    if not pts3d:
        return np.empty((0, 3), dtype=np.float64)

    return np.asarray(pts3d, dtype=np.float64)


def compute_deviations(points: np.ndarray, plane_model: np.ndarray):
    normal = plane_model[:3]
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        return np.zeros(points.shape[0], dtype=np.float64)
    return np.abs(points @ normal + plane_model[3]) / norm


def draw_axes_top_left(image: np.ndarray, origin: tuple[int, int]):
    shaft = 70
    thickness = 2
    tip_length = 0.2

    x_end = (origin[0] + shaft, origin[1])
    y_end = (origin[0], origin[1] + shaft)
    z_end = (origin[0] + int(shaft * 0.7), origin[1] + int(shaft * 0.7))

    cv2.arrowedLine(image, origin, x_end, (0, 0, 255), thickness, cv2.LINE_AA, tipLength=tip_length)
    cv2.arrowedLine(image, origin, y_end, (0, 255, 0), thickness, cv2.LINE_AA, tipLength=tip_length)
    cv2.arrowedLine(image, origin, z_end, (255, 0, 0), thickness, cv2.LINE_AA, tipLength=tip_length)

    return image


def main():
    pattern_size = (3, 3)
    det_width = 640
    rect_pad_px = 15
    max_points = 40000
    z_min, z_max = 0.15, 2.0
    ransac_thresh_m = 0.005
    ransac_n = 3
    ransac_iters = 1000

    pipeline, align = start_realsense_rgbd(fps=30)
    setup_window(LIVE_WIN)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                continue

            color = np.asanyarray(cf.get_data())
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            found_live, _ = cbd.detect_classic_downscaled(gray, pattern_size, det_width=det_width)

            vis_live = color.copy()
            if found_live:
                cv2.putText(vis_live, "FOUND (press SPACE to capture)", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.putText(vis_live, "NOT FOUND", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow(LIVE_WIN, vis_live)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key != 32 or not found_live:
                continue

            snap_color = color.copy()
            found, corners = cbd.detect_snapshot_full(snap_color, pattern_size=pattern_size, det_width=det_width)
            if not found or corners is None:
                continue

            res = snap_color.copy()
            cv2.drawChessboardCorners(res, pattern_size, corners, True)
            extremes = cbd.get_extreme_corners_geometric(corners)
            res = draw_extremes(res, extremes, color=(208, 224, 64), radius=10, thickness=-1)

            setup_window(RES_WIN)
            cv2.imshow(RES_WIN, res)

            ok_corners = ask_corners_satisfied()
            if not ok_corners:
                cv2.destroyWindow(RES_WIN)
                setup_window(LIVE_WIN)
                continue

            H, W = snap_color.shape[:2]
            rect = rect_from_extremes(extremes, W, H, pad_px=rect_pad_px)

            redo_seed = 0
            while True:
                pts3d = sample_points_3d_in_rect(
                    depth_frame_aligned=df,
                    rect=rect,
                    max_points=max_points,
                    z_min=z_min,
                    z_max=z_max,
                    seed=redo_seed,
                )

                if len(pts3d) < 800:
                    print("Not enough points for plane fit. Try again.")
                    cv2.destroyWindow(RES_WIN)
                    setup_window(LIVE_WIN)
                    break

                plane_model, inliers = rpf.fit_plane_from_points(
                    pts3d,
                    distance_threshold=ransac_thresh_m,
                    ransac_n=ransac_n,
                    num_iterations=ransac_iters,
                )

                deviations = compute_deviations(pts3d, plane_model)
                inlier_deviations = deviations[inliers] if len(inliers) else deviations
                normal = plane_model[:3]
                normal /= np.linalg.norm(normal) + 1e-12

                print("\nRANSAC plane fitting")
                print(f"Plane model (a,b,c,d): [{plane_model[0]:+.6f}, {plane_model[1]:+.6f}, {plane_model[2]:+.6f}, {plane_model[3]:+.6f}]")
                print(f"Normal: [{normal[0]:+.6f}, {normal[1]:+.6f}, {normal[2]:+.6f}]")
                print(
                    "Deviation stats (all points): "
                    f"mean={deviations.mean():.6f}, std={deviations.std():.6f}, max={deviations.max():.6f}"
                )
                print(
                    "Deviation stats (inliers): "
                    f"mean={inlier_deviations.mean():.6f}, std={inlier_deviations.std():.6f}, max={inlier_deviations.max():.6f}"
                )

                preview = snap_color.copy()
                preview = draw_extremes(preview, extremes, color=(208, 224, 64), radius=10, thickness=-1)
                top_left = extremes["top_left"]
                preview = draw_axes_top_left(preview, origin=(int(round(top_left[0])), int(round(top_left[1]))))

                cv2.imshow(PLANE_WIN, preview)
                cv2.waitKey(10)

                if ask_plane_satisfied():
                    cv2.destroyWindow(RES_WIN)
                    while True:
                        key_final = cv2.waitKey(50) & 0xFF
                        if key_final == 27:
                            cv2.destroyWindow(PLANE_WIN)
                            return

                redo_seed += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
