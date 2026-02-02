# -*- coding: utf-8 -*-
"""
RANSAC_plane_fitting.py

Return:
- plane_model (a,b,c,d) for ax+by+cz+d=0
- extremes pixel coords dict:
    {"top_left":(u,v), "top_right":(u,v), "bottom_left":(u,v)}

UI:
- corner detection -> satisfied? (Yes/No)
- plane fitting -> satisfied? (Yes/No, redo if No)
- final image: only extremes + camera axes at top_left corner + normal vector text at top-left

Dependencies:
- pyrealsense2
- open3d
- your checkerboard_corner_detection module (rewritten version)
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d
import tkinter as tk
from tkinter import messagebox

import checkerboard_corner_detection as cbd  # your module


# =========================
# Popups
# =========================
def ask_plane_satisfied(
    title: str = "Plane Fitting",
    msg: str = "Satisfied with the plane fitting?\n\nYes: accept\nNo: redo plane fitting"
) -> bool:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ans = messagebox.askyesno(title, msg, icon="question")
    root.destroy()
    return ans


# =========================
# RealSense start
# =========================
def start_realsense_rgbd(fps: int = 30):
    pipeline = rs.pipeline()
    config = rs.config()

    # D435i-safe combo
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)  # align depth to COLOR grid

    # reduce lag
    try:
        dev = profile.get_device()
        for s in dev.query_sensors():
            try:
                s.set_option(rs.option.frames_queue_size, 1)
            except Exception:
                pass
    except Exception:
        pass

    return pipeline, align


# =========================
# ROI rectangle from extremes
# =========================
def rect_from_extremes(extremes, img_w: int, img_h: int, pad_px: int):
    pts = np.array(
        [extremes["top_left"], extremes["top_right"], extremes["bottom_left"]],
        dtype=np.float32
    )

    umin = int(np.floor(np.min(pts[:, 0]) - pad_px))
    umax = int(np.ceil (np.max(pts[:, 0]) + pad_px))
    vmin = int(np.floor(np.min(pts[:, 1]) - pad_px))
    vmax = int(np.ceil (np.max(pts[:, 1]) + pad_px))

    umin = max(0, min(img_w - 1, umin))
    umax = max(0, min(img_w - 1, umax))
    vmin = max(0, min(img_h - 1, vmin))
    vmax = max(0, min(img_h - 1, vmax))

    if umax < umin:
        umin, umax = umax, umin
    if vmax < vmin:
        vmin, vmax = vmax, vmin

    return (umin, vmin, umax, vmax)


# =========================
# Sample 3D points in rectangle ROI
# =========================
def sample_points_3d_in_rect(
    depth_frame_aligned,
    rect,
    max_points: int,
    z_min: float,
    z_max: float,
    seed: int
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


# =========================
# Open3D RANSAC plane fit
# =========================
def ransac_plane_open3d(points_xyz, distance_threshold: float, ransac_n: int, num_iterations: int):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    return np.asarray(plane_model, dtype=np.float64), np.asarray(inliers, dtype=np.int64)


# =========================
# Projection depth->color
# =========================
def project_depth_point_to_color_pixel(point_d, depth_to_color_extrin, color_intrin):
    point_c = rs.rs2_transform_point_to_point(depth_to_color_extrin, point_d.tolist())
    uv = rs.rs2_project_point_to_pixel(color_intrin, point_c)
    return uv


# =========================
# Drawing primitives (no nesting)
# =========================
def draw_cross(img, center, size_px: int, thickness: int):
    x, y = center
    cv2.line(img, (x - size_px, y), (x + size_px, y), (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size_px), (x, y + size_px), (255, 255, 255), thickness, cv2.LINE_AA)


def draw_closed_arrow(img, p0, p1, bgr, thickness: int, head_len: int, head_half_width: int):
    x0, y0 = p0
    x1, y1 = p1
    cv2.line(img, (x0, y0), (x1, y1), bgr, thickness, cv2.LINE_AA)

    dx = x1 - x0
    dy = y1 - y0
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-6:
        return

    ux = dx / norm
    uy = dy / norm

    bx = x1 - head_len * ux
    by = y1 - head_len * uy

    px = -uy
    py = ux

    left = (int(round(bx + head_half_width * px)), int(round(by + head_half_width * py)))
    right = (int(round(bx - head_half_width * px)), int(round(by - head_half_width * py)))

    tri = np.array([[x1, y1], [left[0], left[1]], [right[0], right[1]]], dtype=np.int32)
    cv2.fillConvexPoly(img, tri, bgr, cv2.LINE_AA)


def normalize_to_fixed_pixel_length(p0, p1, length_px: int):
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-6:
        return p1
    ux = dx / norm
    uy = dy / norm
    return (int(round(x0 + length_px * ux)), int(round(y0 + length_px * uy)))


def draw_camera_axes_at_origin(
    img_bgr,
    origin_d,
    depth_to_color_extrin,
    color_intrin,
    axis_len_m: float,
    equal_in_image: bool,
    shaft_len_px: int,
    arrow_thickness: int,
    head_len_px: int,
    head_half_width_px: int,
    cross_size_px: int,
    cross_thickness: int
):
    H, W = img_bgr.shape[:2]
    vis = img_bgr.copy()

    o_uv = project_depth_point_to_color_pixel(origin_d, depth_to_color_extrin, color_intrin)
    ox, oy = int(round(o_uv[0])), int(round(o_uv[1]))
    if not (0 <= ox < W and 0 <= oy < H):
        return vis

    p0 = (ox, oy)
    draw_cross(vis, p0, cross_size_px, cross_thickness)

    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    px_d = origin_d + axis_len_m * ex
    py_d = origin_d + axis_len_m * ey
    pz_d = origin_d + axis_len_m * ez

    px_uv = project_depth_point_to_color_pixel(px_d, depth_to_color_extrin, color_intrin)
    py_uv = project_depth_point_to_color_pixel(py_d, depth_to_color_extrin, color_intrin)
    pz_uv = project_depth_point_to_color_pixel(pz_d, depth_to_color_extrin, color_intrin)

    pX = (int(round(px_uv[0])), int(round(px_uv[1])))
    pY = (int(round(py_uv[0])), int(round(py_uv[1])))
    pZ = (int(round(pz_uv[0])), int(round(pz_uv[1])))

    if equal_in_image:
        pX = normalize_to_fixed_pixel_length(p0, pX, shaft_len_px)
        pY = normalize_to_fixed_pixel_length(p0, pY, shaft_len_px)
        pZ = normalize_to_fixed_pixel_length(p0, pZ, shaft_len_px)

    draw_closed_arrow(vis, p0, pX, (0, 0, 255), arrow_thickness, head_len_px, head_half_width_px)  # X red
    draw_closed_arrow(vis, p0, pY, (0, 255, 0), arrow_thickness, head_len_px, head_half_width_px)  # Y green
    draw_closed_arrow(vis, p0, pZ, (255, 0, 0), arrow_thickness, head_len_px, head_half_width_px)  # Z blue

    return vis


# =========================
# Main API
# =========================
def RANSAC_plane_fitting():
    # Settings (same as before, just inlined here)
    pattern_size = (3, 3)
    det_width = 640

    rect_pad_px = 15
    max_points = 40000
    z_min, z_max = 0.15, 2.0
    ransac_thresh_m = 0.005
    ransac_n = 3
    ransac_iters = 1000

    AXIS_LEN_M = 0.0075
    EQUAL_IN_IMAGE = True
    SHAFT_LEN_PX = 80
    ARROW_THICKNESS = 3
    HEAD_LEN_PX = 16
    HEAD_HALF_WIDTH_PX = 8
    CROSS_SIZE_PX = 10
    CROSS_THICKNESS = 2

    FINAL_WIN = "Final (Extremes + Normal) - ESC to quit"

    pipeline, align = start_realsense_rgbd(fps=30)
    cbd.setup_window(cbd.LIVE_WIN)

    out_plane = None
    out_extremes = None

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

            cv2.imshow(cbd.LIVE_WIN, vis_live)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            if key != 32 or not found_live:  # SPACE only if found
                continue

            # Snapshot corner detection (using your module)
            snap_color = color.copy()
            found, corners = cbd.detect_snapshot_full(snap_color, pattern_size=pattern_size, det_width=det_width)
            if not found or corners is None:
                continue

            res = snap_color.copy()
            cv2.drawChessboardCorners(res, pattern_size, corners, True)
            extremes = cbd.get_extreme_corners_geometric(corners)
            res = cbd.draw_extremes(res, extremes, color=(208, 224, 64), radius=10, thickness=-1)

            cbd.setup_window(cbd.RES_WIN)
            cv2.imshow(cbd.RES_WIN, res)

            ok_corners = cbd.ask_satisfied(
                title="Corner Detection",
                msg="Satisfied with the detected corners?\n\nYes: Next (plane fitting)\nNo: back to live video"
            )
            if not ok_corners:
                cv2.destroyWindow(cbd.RES_WIN)
                cbd.setup_window(cbd.LIVE_WIN)
                continue

            # ROI rectangle
            H, W = snap_color.shape[:2]
            rect = rect_from_extremes(extremes, W, H, pad_px=rect_pad_px)

            # Plane redo loop
            redo_seed = 0
            while True:
                pts3d = sample_points_3d_in_rect(
                    depth_frame_aligned=df,
                    rect=rect,
                    max_points=max_points,
                    z_min=z_min,
                    z_max=z_max,
                    seed=redo_seed
                )

                if len(pts3d) < 800:
                    cv2.destroyWindow(cbd.RES_WIN)
                    cbd.setup_window(cbd.LIVE_WIN)
                    break

                plane_model, inliers = ransac_plane_open3d(
                    pts3d,
                    distance_threshold=ransac_thresh_m,
                    ransac_n=ransac_n,
                    num_iterations=ransac_iters
                )

                a, b, c, d = plane_model
                n = np.array([a, b, c], dtype=np.float64)
                n = n / (np.linalg.norm(n) + 1e-12)

                # origin at top_left extreme corner (depth frame)
                depth_intrin = df.profile.as_video_stream_profile().intrinsics
                u_tl, v_tl = extremes["top_left"]
                u_tl_i, v_tl_i = int(round(u_tl)), int(round(v_tl))
                z_tl = df.get_distance(u_tl_i, v_tl_i)

                if z_tl > 0:
                    origin_d = np.array(
                        rs.rs2_deproject_pixel_to_point(depth_intrin, [float(u_tl_i), float(v_tl_i)], float(z_tl)),
                        dtype=np.float64
                    )
                else:
                    origin_d = pts3d[inliers].mean(axis=0)

                # depth->color transform for visualization
                depth_prof = df.profile.as_video_stream_profile()
                color_prof = cf.profile.as_video_stream_profile()
                depth_to_color_extrin = depth_prof.get_extrinsics_to(color_prof)
                color_intrin = cf.profile.as_video_stream_profile().intrinsics

                # preview image (same window as RES)
                preview = snap_color.copy()
                preview = cbd.draw_extremes(preview, extremes, color=(208, 224, 64), radius=10, thickness=-1)
                preview = draw_camera_axes_at_origin(
                    preview, origin_d,
                    depth_to_color_extrin, color_intrin,
                    axis_len_m=AXIS_LEN_M,
                    equal_in_image=EQUAL_IN_IMAGE,
                    shaft_len_px=SHAFT_LEN_PX,
                    arrow_thickness=ARROW_THICKNESS,
                    head_len_px=HEAD_LEN_PX,
                    head_half_width_px=HEAD_HALF_WIDTH_PX,
                    cross_size_px=CROSS_SIZE_PX,
                    cross_thickness=CROSS_THICKNESS
                )
                cv2.putText(preview, f"n = [{n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f}]",
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

                cv2.imshow(cbd.RES_WIN, preview)
                cv2.waitKey(10)

                if ask_plane_satisfied():
                    out_plane = tuple(plane_model.tolist())
                    out_extremes = extremes

                    cv2.destroyWindow(cbd.LIVE_WIN)
                    cv2.destroyWindow(cbd.RES_WIN)

                    cbd.setup_window(FINAL_WIN)
                    while True:
                        cv2.imshow(FINAL_WIN, preview)
                        k2 = cv2.waitKey(10) & 0xFF
                        if k2 == 27:
                            return out_plane, out_extremes
                else:
                    redo_seed += 1
                    continue

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return out_plane, out_extremes


if __name__ == "__main__":
    plane, extremes = RANSAC_plane_fitting()
    print("\nReturned:")
    print("  plane (a,b,c,d):", plane)
    print("  extremes:", extremes)
