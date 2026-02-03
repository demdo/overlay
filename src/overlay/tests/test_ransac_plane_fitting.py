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
import matplotlib.pyplot as plt

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


def show_plane_fit_3d(inlier_pts3d: np.ndarray, plane_model: np.ndarray, title: str = "Plane fit (3D)"):
    """
    3D visualization:
    - inlier point cloud
    - fitted plane surface (bounded by inlier bbox)
    - local plane frame at centroid:
        X_tangent (red), Y_tangent (green), Normal (blue)
      all arrows same length and thicker.
    """

    P = np.asarray(inlier_pts3d, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
        print("show_plane_fit_3d: not enough valid inlier points.")
        return

    a, b, c, d = [float(x) for x in plane_model]
    n = np.array([a, b, c], dtype=float)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        print("show_plane_fit_3d: degenerate plane normal.")
        return
    n_unit = n / n_norm

    # Center & scale
    center = P.mean(axis=0)
    pmin = P.min(axis=0)
    pmax = P.max(axis=0)
    extent = float(np.linalg.norm(pmax - pmin))
    if extent < 1e-9:
        extent = 1.0
    L = 0.25 * extent

    # Plane mesh (stable axis)
    res = 25
    xs = np.linspace(pmin[0], pmax[0], res)
    ys = np.linspace(pmin[1], pmax[1], res)
    zs = np.linspace(pmin[2], pmax[2], res)

    absn = np.abs([a, b, c])
    solve_axis = int(np.argmax(absn))

    if solve_axis == 2 and abs(c) > 1e-12:
        XX, YY = np.meshgrid(xs, ys)
        ZZ = -(a * XX + b * YY + d) / c
    elif solve_axis == 1 and abs(b) > 1e-12:
        XX, ZZ = np.meshgrid(xs, zs)
        YY = -(a * XX + c * ZZ + d) / b
    elif solve_axis == 0 and abs(a) > 1e-12:
        YY, ZZ = np.meshgrid(ys, zs)
        XX = -(b * YY + c * ZZ + d) / a
    else:
        print("show_plane_fit_3d: cannot build plane mesh (degenerate parameters).")
        return

    # Style (MathText)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Title / labels
    ax.set_title(title)
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$Y$")
    ax.set_zlabel(r"$Z$")

    # --- plane fill first (so points are on top) ---
    ax.plot_surface(XX, YY, ZZ, alpha=0.18, linewidth=0, color="0.65")

    # plane outline (black)
    ax.plot(XX[0, :],  YY[0, :],  ZZ[0, :],  color="k", linewidth=2.0)
    ax.plot(XX[-1, :], YY[-1, :], ZZ[-1, :], color="k", linewidth=2.0)
    ax.plot(XX[:, 0],  YY[:, 0],  ZZ[:, 0],  color="k", linewidth=2.0)
    ax.plot(XX[:, -1], YY[:, -1], ZZ[:, -1], color="k", linewidth=2.0)

    # normal (blue: 0,0,255)
    ax.quiver(center[0], center[1], center[2],
              n_unit[0], n_unit[1], n_unit[2],
              length=L, normalize=True,
              color=(0.0, 0.0, 1.0),
              linewidths=3.5,
              arrow_length_ratio=0.25,
              pivot="tail")

    # annotation (bold n, column vector) – offset away from axes
    tip = center + L * n_unit
    side_offset = 0.12 * extent * np.array([-0.2, 0.2, 0.0])
    t = tip + 0.06 * L * n_unit + side_offset
    n_text = (
        r"$\mathbf{n}="
        r"\begin{bmatrix}"
        rf"{a:.3f}\\"
        rf"{b:.3f}\\"
        rf"{c:.3f}"
        r"\end{bmatrix}$"
    )
    ax.text(t[0], t[1], t[2], n_text)

    # --- inliers LAST + no depthshade so they don't disappear ---
    ax.scatter(P[:, 0], P[:, 1], P[:, 2],
               s=6, alpha=0.35, c="#2ca02c",
               depthshade=False, label=r"$\mathrm{Inliers}$")

    # limits (equal-ish)
    mid = (pmin + pmax) / 2.0
    half = (pmax - pmin) / 2.0
    max_half = float(np.max(half))
    ax.set_xlim(mid[0] - max_half, mid[0] + max_half)
    ax.set_ylim(mid[1] - max_half, mid[1] + max_half)
    ax.set_zlim(mid[2] - max_half, mid[2] + max_half)

    ax.legend(loc="upper right", frameon=True)
    plt.show(block=True)


def show_plane_fit_histogram(
    pts3d: np.ndarray,
    plane_model: np.ndarray,
    inliers: np.ndarray,
    ransac_thresh_m: float | None = None,
    bins: int = 60,
    x_quantile: float = 99.5,
    density: bool = False,
    title: str = r"RANSAC plane fit: inlier distance distribution",
):
    """
    Histogram for inlier point-to-plane distances.

    - density=False: y-axis is Count (default)
    - density=True : y-axis is Probability density (better for comparing different n)
    """

    pts3d = np.asarray(pts3d, dtype=float)
    inliers = np.asarray(inliers, dtype=int)

    # --- plane parameters ---
    a, b, c, d = [float(x) for x in plane_model]
    nvec = np.array([a, b, c], dtype=float)
    n_norm = np.linalg.norm(nvec)
    if n_norm < 1e-12:
        print("Invalid plane normal.")
        return

    # --- distances (inliers only) in mm ---
    d_perp_mm = (np.abs(pts3d[inliers] @ nvec + d) / n_norm) * 1000.0

    # --- stats ---
    N = int(d_perp_mm.size)
    d_mean = float(np.mean(d_perp_mm))
    d_med  = float(np.median(d_perp_mm))
    d_p95  = float(np.percentile(d_perp_mm, 95))
    tau_mm = (ransac_thresh_m * 1000.0) if ransac_thresh_m is not None else None

    # x-range: show at least up to tau if given, otherwise show some tail beyond P95
    if tau_mm is not None:
        xmax = max(tau_mm, d_p95 * 1.2)
    else:
        xmax = d_p95 * 1.8
    xmax = max(xmax, 1e-6)

    # --- histogram counts (manual drawing so only bars are colored) ---
    counts, edges = np.histogram(d_perp_mm, bins=bins, range=(0.0, xmax))
    widths = np.diff(edges)
    lefts = edges[:-1]
    centers = lefts + 0.5 * widths

    is_left = centers <= d_p95
    is_right = ~is_left

    # --- style (LaTeX-like) ---
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
    })

    fig, ax = plt.subplots()

    # bars
    ax.bar(lefts[is_left], counts[is_left], width=widths[is_left],
           align="edge", alpha=0.85, color="#1f77b4", edgecolor="none")
    ax.bar(lefts[is_right], counts[is_right], width=widths[is_right],
           align="edge", alpha=0.35, color="#d62728", edgecolor="none")

    # --- short P95 marker (only slightly above the P95 bin) ---
    k = int(np.clip(np.searchsorted(edges, d_p95, side="right") - 1, 0, len(counts) - 1))
    bar_h = float(counts[k])
    y0 = bar_h * 0.15          # start a bit above the bar base (looks nicer)
    y1 = bar_h * 1.10 + 1.0    # end a bit above the bar

    ax.plot([d_p95, d_p95], [y0, y1], linestyle="--", linewidth=2.5, color="black")
    ax.text(d_p95 + 0.02 * xmax, y1, r"$P_{95}$", ha="left", va="bottom")

    # labels
    ax.set_title(r"RANSAC plane fit: distribution of $d_{\perp}$")
    ax.set_xlabel(r"Point-to-plane distance $d_{\perp}\;[\mathrm{mm}]$")
    ax.set_ylabel(r"Count")

    ax.set_xlim(left=0.0, right=xmax)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.2)

    # --- stats-only legend (INSIDE upper-right, guaranteed visible) ---
    # Create dummy handles (invisible) so legend shows text lines only
    handles = [ax.plot([], [], " ")[0] for _ in range(5)]

    labels = [
        rf"$n={N}$",
        rf"$\bar{{d}}={d_mean:.3f}\,\mathrm{{mm}}$",
        rf"$\tilde{{d}}={d_med:.3f}\,\mathrm{{mm}}$",
        rf"$P_{{95}}={d_p95:.3f}\,\mathrm{{mm}}$",
        (rf"$\tau_{{\mathrm{{RANSAC}}}}={tau_mm:.1f}\,\mathrm{{mm}}$" if tau_mm is not None
         else rf"$\tau_{{\mathrm{{RANSAC}}}}=\mathrm{{n/a}}$")
    ]

    ax.legend(handles, labels, loc="upper right", frameon=True, borderpad=0.8)

    fig.tight_layout()
    plt.show(block=True)


def main():
    pattern_size = (3, 3)
    det_width = 640
    rect_pad_px = 15
    max_points = 40000
    z_min, z_max = 0.15, 2.0
    ransac_thresh_m = 0.005   # threshold based on expected depth noise
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
            accepted = False
            accepted_cache = None
            
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
                
                if len(inliers) == 0:
                    print("WARNING: No inliers returned by RANSAC. Cannot compute inlier mean/median.")
                else:
                    inlier_dev_mm = inlier_deviations * 1000.0  # m -> mm
                    mean_mm = float(np.mean(inlier_dev_mm))
                    median_mm = float(np.median(inlier_dev_mm))
                    std_mm = float(np.std(inlier_dev_mm))
                    max_mm = float(np.max(inlier_dev_mm))
                
                    print(f"Inliers: {len(inliers)}/{len(pts3d)} ({len(inliers)/len(pts3d):.3f})")
                    print(f"Inlier distance mean:   {mean_mm:.3f} mm")
                    print(f"Inlier distance median: {median_mm:.3f} mm")
                    print(f"Inlier distance std: {std_mm:.3f} mm")
                    print(f"Inlier distance max: {max_mm:.3f} mm")

                preview = snap_color.copy()
                preview = draw_extremes(preview, extremes, color=(208, 224, 64), radius=10, thickness=-1)
                top_left = extremes["top_left"]
                preview = draw_axes_top_left(preview, origin=(int(round(top_left[0])), int(round(top_left[1]))))

                cv2.imshow(PLANE_WIN, preview)
                cv2.waitKey(10)

                if ask_plane_satisfied():
                    cv2.destroyWindow(RES_WIN)
                    
                    accepted = True
                    accepted_cache = {
                        "pts3d": pts3d,
                        "inliers": inliers,         # int indices
                        "plane_model": plane_model,
                    }
                    break
                redo_seed += 1
                
            if accepted and accepted_cache is not None:
                while True:
                    k = cv2.waitKey(50) & 0xFF
                    if k == 27:
                        break

                cv2.destroyWindow(PLANE_WIN)
                inlier_pts3d = accepted_cache["pts3d"][accepted_cache["inliers"]]

                # optional downsample for clarity/speed
                max_plot = 8000
                if inlier_pts3d.shape[0] > max_plot:
                    rng = np.random.default_rng(0)
                    idx = rng.choice(inlier_pts3d.shape[0], size=max_plot, replace=False)
                    inlier_pts3d = inlier_pts3d[idx]

                show_plane_fit_3d(
                    inlier_pts3d=inlier_pts3d,
                    plane_model=accepted_cache["plane_model"],
                    title="RANSAC plane fit: inliers + fitted plane + normal"
                )
                
                show_plane_fit_histogram(
                    pts3d=accepted_cache["pts3d"],
                    plane_model=accepted_cache["plane_model"],
                    inliers=accepted_cache["inliers"],
                    ransac_thresh_m=ransac_thresh_m,   # nur für Legende
                    bins=60,
                    density=False,   # oder True
                )
                return

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
