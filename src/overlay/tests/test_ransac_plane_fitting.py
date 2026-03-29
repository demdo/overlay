# -*- coding: utf-8 -*-
"""
example_ransac_plane_fitting.py

Interactive test harness for RANSAC_plane_fitting helpers.

Features:
- Live RGB preview + live Depth preview (aligned)
- SPACE:
    - always saves a depth snapshot:
        * raw 16-bit depth PNG
        * colorized depth PNG (for quick inspection)
        * JSON metadata with key RealSense depth settings
    - if chessboard is FOUND, continues into your capture + plane fitting flow
- Optional RealSense depth tuning via flag USE_TUNED_SETTINGS
  (and when disabled, we explicitly switch back to auto/default-ish settings)
"""

from __future__ import annotations

import os
import time
import json
import numpy as np
import cv2
import pyrealsense2 as rs
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt

from overlay.tools import checkerboard_corner_detection as cbd
from overlay.tools import ransac_plane_fitting as rpf


LIVE_WIN = "Live RGB (SPACE=capture if FOUND, ESC=quit)"
DEPTH_WIN = "Live Depth (aligned)"
RES_WIN = "Corner Detection Result"
PLANE_WIN = "Plane Fitting Preview (ESC=quit)"
DEPTH_SNAP_WIN = "Depth Snapshot + ROI"

# Toggle RealSense tuning on/off (so you can A/B compare)
USE_TUNED_SETTINGS = False  # set True to enable manual depth tuning


# =========================
# Small helpers
# =========================
def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def save_preview_same_dir(img: np.ndarray, filename: str) -> str:
    """Save to same dir as this script. Returns output path."""
    out_path = os.path.join(script_dir(), filename)
    ok = cv2.imwrite(out_path, img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for: {out_path}")
    print(f"[DEBUG] Saved: {out_path}")
    return out_path


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


def ask_plane_satisfied(
    title: str = "Plane Fitting",
    msg: str = "Satisfied with plane fitting?\n\nYes: accept\nNo: redo",
) -> bool:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ans = messagebox.askyesno(title, msg, icon="question")
    root.destroy()
    return ans


def draw_extremes(img_bgr, extremes, color=(208, 224, 64), radius=10, thickness=-1, draw_labels=True):
    for name, (u, v) in extremes.items():
        cv2.circle(img_bgr, (int(u), int(v)), radius, color, thickness)
        if draw_labels:
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


def draw_support_rect(img_bgr, rect, color=(255, 255, 0), thickness=2):
    umin, vmin, umax, vmax = rect
    out = img_bgr.copy()
    cv2.rectangle(out, (umin, vmin), (umax, vmax), color, thickness, cv2.LINE_AA)
    return out


def depth_to_vis_bgr(depth_frame) -> np.ndarray:
    """
    Robust visualization of depth:
    - normalize using 2..98 percentiles for better contrast
    - invalid depth (0) shown as black
    """
    depth_img = np.asanyarray(depth_frame.get_data()).astype(np.uint16)

    nonzero = depth_img[depth_img > 0]
    if nonzero.size == 0:
        return np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)

    lo = float(np.percentile(nonzero, 2.0))
    hi = float(np.percentile(nonzero, 98.0))
    if hi <= lo:
        hi = lo + 1.0

    depth_8u = np.clip((depth_img.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    depth_bgr[depth_img == 0] = (0, 0, 0)

    txt = f"Depth vis: p2={lo:.0f}, p98={hi:.0f} (raw units)"
    cv2.putText(depth_bgr, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return depth_bgr


# =========================
# RealSense option helpers + setup
# =========================
def _rs_get(sensor: rs.sensor | None, opt: rs.option) -> float | None:
    try:
        if sensor is not None and sensor.supports(opt):
            return float(sensor.get_option(opt))
    except Exception:
        pass
    return None


def _rs_set(sensor: rs.sensor | None, opt: rs.option, val: float) -> bool:
    try:
        if sensor is not None and sensor.supports(opt):
            sensor.set_option(opt, float(val))
            return True
    except Exception:
        pass
    return False


def _rs_range(sensor: rs.sensor | None, opt: rs.option):
    try:
        if sensor is not None and sensor.supports(opt):
            return sensor.get_option_range(opt)
    except Exception:
        pass
    return None


def get_depth_settings_dict(depth_sensor: rs.sensor | None) -> dict:
    """Key parameters for reproducibility/logging."""
    return {
        "visual_preset": _rs_get(depth_sensor, rs.option.visual_preset),
        "emitter_enabled": _rs_get(depth_sensor, rs.option.emitter_enabled),
        "laser_power": _rs_get(depth_sensor, rs.option.laser_power),
        "auto_exposure": _rs_get(depth_sensor, rs.option.enable_auto_exposure),
        "exposure": _rs_get(depth_sensor, rs.option.exposure),
        "gain": _rs_get(depth_sensor, rs.option.gain),
    }


def start_realsense_rgbd(fps: int = 30):
    pipeline = rs.pipeline()
    config = rs.config()

    # Streams
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)

    profile = pipeline.start(config)
    dev = profile.get_device()

    # reduce queue latency
    try:
        for sensor in dev.query_sensors():
            try:
                sensor.set_option(rs.option.frames_queue_size, 1)
            except Exception:
                pass
    except Exception:
        pass

    # Depth sensor
    try:
        depth_sensor = dev.first_depth_sensor()
    except Exception:
        depth_sensor = None

    # --- Optional depth tuning (toggle with USE_TUNED_SETTINGS) ---
    if depth_sensor is not None:
        if USE_TUNED_SETTINGS:
            # MANUAL / tuned settings
            _rs_set(depth_sensor, rs.option.emitter_enabled, 1)
            _rs_set(depth_sensor, rs.option.enable_auto_exposure, 0)

            r = _rs_range(depth_sensor, rs.option.laser_power)
            if r is not None:
                _rs_set(depth_sensor, rs.option.laser_power, min(r.max, max(r.min, 150)))

            r = _rs_range(depth_sensor, rs.option.exposure)
            if r is not None:
                _rs_set(depth_sensor, rs.option.exposure, float(np.clip(4000, r.min, r.max)))

            r = _rs_range(depth_sensor, rs.option.gain)
            if r is not None:
                # D435i often has min gain 16
                _rs_set(depth_sensor, rs.option.gain, float(np.clip(16, r.min, r.max)))

            print("[Depth tuning enabled]")
        else:
            # AUTO / default-ish settings (so A/B really works)
            _rs_set(depth_sensor, rs.option.enable_auto_exposure, 1)

            # emitter: prefer AUTO if supported (often 2 on D435i)
            r = _rs_range(depth_sensor, rs.option.emitter_enabled)
            if r is not None:
                _rs_set(depth_sensor, rs.option.emitter_enabled, min(r.max, 2))

            # laser power: set to neutral mid (avoids stale manual power)
            r = _rs_range(depth_sensor, rs.option.laser_power)
            if r is not None:
                _rs_set(depth_sensor, rs.option.laser_power, float(np.clip(150, r.min, r.max)))

            print("[Depth tuning disabled -> auto exposure ON, emitter AUTO]")

        # Always print current key settings
        s = get_depth_settings_dict(depth_sensor)
        print("[Depth settings]")
        for k, v in s.items():
            print(f"  {k}: {v}")

    align = rs.align(rs.stream.color)
    return pipeline, align, profile, depth_sensor


def save_depth_snapshot_with_meta(
    df_aligned,
    depth_sensor: rs.sensor | None,
    use_tuned: bool,
    prefix: str = "depth",
) -> None:
    """
    Saves:
    - raw 16-bit depth PNG
    - colorized depth PNG
    - JSON meta including key RealSense settings
    """
    ts = time.strftime("%Y%m%d_%H%M%S")

    depth_raw = np.asanyarray(df_aligned.get_data())  # uint16
    depth_vis = depth_to_vis_bgr(df_aligned)

    raw_name = f"{prefix}_{ts}_raw.png"
    vis_name = f"{prefix}_{ts}_vis.png"
    meta_name = f"{prefix}_{ts}_meta.json"

    save_preview_same_dir(depth_raw, raw_name)
    save_preview_same_dir(depth_vis, vis_name)

    meta = {
        "timestamp": ts,
        "use_tuned_settings": bool(use_tuned),
        "depth_settings": get_depth_settings_dict(depth_sensor),
        # You can extend this later (e.g. intrinsics, stream sizes, etc.)
    }

    meta_path = os.path.join(script_dir(), meta_name)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[DEBUG] Saved: {meta_path}")


# =========================
# Geometry helpers
# =========================
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


# =========================
# Plot helpers (unchanged)
# =========================
def show_plane_fit_3d(
    inlier_pts3d: np.ndarray,
    plane_model: np.ndarray,
    title: str = "Plane fit (3D)",
    tuning: dict | None = None,
):
    """
    3D visualization:
    - inlier point cloud
    - fitted plane surface (bounded by inlier bbox)
    - normal vector at centroid (blue)
    - info box (figure coords, stable position):
        * USE_TUNED_SETTINGS=True  -> "Depth tuning" + key values
        * USE_TUNED_SETTINGS=False -> "No depth tuning"
      Box height follows the actual text (no padding lines).
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

    # normal (blue)
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

    # --- info box: fixed position, natural height ---
    # Place it where you circled: a bit right of the left margin, below the title.
    BOX_X = 0.25
    BOX_Y = 0.85

    if USE_TUNED_SETTINGS:
        lp = tuning.get("laser_power", None) if isinstance(tuning, dict) else None
        ex = tuning.get("exposure", None) if isinstance(tuning, dict) else None
        ae = tuning.get("auto_exposure", None) if isinstance(tuning, dict) else None

        def _fmt(x):
            try:
                return f"{float(x):.0f}"
            except Exception:
                return "n/a"

        ae_str = "n/a"
        try:
            if ae is not None:
                ae_str = "ON" if float(ae) > 0.5 else "OFF"
        except Exception:
            ae_str = "n/a"

        lines = [
            r"$\bf{Depth\ tuning}$",
            f"Laser power: {_fmt(lp)}",
            f"Exposure: {_fmt(ex)}",
            f"Auto exposure: {ae_str}",
        ]
    else:
        lines = [r"$\bf{No\ depth\ tuning}$"]

    fig.text(
        BOX_X, BOX_Y,
        "\n".join(lines),
        va="top", ha="left",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.3", alpha=0.95),
    )

    ax.legend(loc="upper right", frameon=True)
    plt.show(block=True)


def show_plane_fit_histogram(
    pts3d: np.ndarray,
    plane_model: np.ndarray,
    inliers: np.ndarray,
    ransac_thresh_m: float | None = None,
    bins: int = 60,
    density: bool = False,
):
    pts3d = np.asarray(pts3d, dtype=float)
    inliers = np.asarray(inliers, dtype=int)

    total_points = pts3d.shape[0]
    inlier_count = len(inliers)

    a, b, c, d = [float(x) for x in plane_model]
    nvec = np.array([a, b, c], dtype=float)
    n_norm = np.linalg.norm(nvec)
    if n_norm < 1e-12:
        print("Invalid plane normal.")
        return

    # --- perpendicular distances (inliers only) in mm ---
    d_perp_mm = (np.abs(pts3d[inliers] @ nvec + d) / n_norm) * 1000.0

    # --- statistics ---
    d_mean = float(np.mean(d_perp_mm))
    d_med  = float(np.median(d_perp_mm))
    d_p95  = float(np.percentile(d_perp_mm, 95))
    tau_mm = (ransac_thresh_m * 1000.0) if ransac_thresh_m is not None else None

    xmax = max((tau_mm if tau_mm is not None else 0.0), d_p95 * 1.8)
    xmax = max(xmax, 1e-6)

    counts, edges = np.histogram(d_perp_mm, bins=bins, range=(0.0, xmax))
    widths = np.diff(edges)
    lefts = edges[:-1]

    # --- LaTeX style ---
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

    # --- draw histogram ---
    bars = ax.bar(
        lefts,
        counts,
        width=widths,
        align="edge",
        alpha=0.85,
        edgecolor="none",
    )

    # --- color bins right of P95 red ---
    for left, bar in zip(lefts, bars):
        if left >= d_p95:
            bar.set_color("#d62728")  # red (upper 5%)
        else:
            bar.set_color("#1f77b4")  # blue

    # --- title with tuning state ---
    if USE_TUNED_SETTINGS:
        tuning_tag = r"$\mathrm{Depth\ tuning}$"
    else:
        tuning_tag = r"$\mathrm{No\ depth\ tuning}$"

    ax.set_title(
        r"RANSAC plane fit: distribution of $d_{\perp}$ ("
        + tuning_tag +
        r")"
    )

    # --- axis labels ---
    ax.set_xlabel(r"Point-to-plane distance $d_{\perp}\;[\mathrm{mm}]$")
    ax.set_ylabel(r"Count" if not density else r"Density")

    ax.grid(True, alpha=0.2)
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(bottom=0.0)

    # --- statistics legend ---
    handles = [ax.plot([], [], " ")[0] for _ in range(5)]
    labels = [
        rf"$n={inlier_count}/{total_points}$",
        rf"$\bar{{d}}={d_mean:.3f}\,\mathrm{{mm}}$",
        rf"$\tilde{{d}}={d_med:.3f}\,\mathrm{{mm}}$",
        rf"$P_{{95}}={d_p95:.3f}\,\mathrm{{mm}}$",
        (rf"$\tau_{{\mathrm{{RANSAC}}}}={tau_mm:.1f}\,\mathrm{{mm}}$"
         if tau_mm is not None
         else rf"$\tau_{{\mathrm{{RANSAC}}}}=\mathrm{{n/a}}$")
    ]

    ax.legend(handles, labels, loc="upper right", frameon=True, borderpad=0.8)

    fig.tight_layout()
    plt.show(block=True)


# =========================
# Main
# =========================
def main():
    pattern_size = (3, 3)
    det_width = 640
    rect_pad_px = 15

    max_points = 5000
    z_min, z_max = 0.15, 2.0

    ransac_thresh_m = 0.0015
    ransac_n = 3
    ransac_iters = 1000

    pipeline, align, profile, depth_sensor = start_realsense_rgbd(fps=30)

    setup_window(LIVE_WIN)
    setup_window(DEPTH_WIN)

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

            # --- RGB overlay ---
            vis_live = color.copy()
            if found_live:
                cv2.putText(vis_live, "FOUND (press SPACE to capture)", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.putText(vis_live, "NOT FOUND", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow(LIVE_WIN, vis_live)

            # --- Depth preview (aligned) ---
            depth_vis = depth_to_vis_bgr(df)
            cv2.putText(depth_vis, "DEPTH (aligned)", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(DEPTH_WIN, depth_vis)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            # --- SPACE: always save depth snapshot + meta ---
            if key == 32:
                save_depth_snapshot_with_meta(
                    df_aligned=df,
                    depth_sensor=depth_sensor,
                    use_tuned=USE_TUNED_SETTINGS,
                    prefix="depth",
                )

            # Capture only on SPACE when pattern is found
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

            # show depth snapshot with ROI (preview)
            depth_snap = depth_to_vis_bgr(df)
            depth_snap = draw_support_rect(depth_snap, rect, color=(0, 165, 255), thickness=2)
            cv2.imshow(DEPTH_SNAP_WIN, depth_snap)
            cv2.waitKey(10)
            save_preview_same_dir(depth_snap, "plane_fitting_depth_snapshot_roi.png")

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
                
                # DEBUG
                if len(pts3d) > 0:
                    z_vals = pts3d[:, 2]
                    print("\nSampled 3D points (camera frame)")
                    print(f"count: {len(pts3d)}")
                    print(f"z range:   {np.min(z_vals):.4f} m .. {np.max(z_vals):.4f} m")
                    print(f"z mean:    {np.mean(z_vals):.4f} m")
                    print(f"xyz min:   [{pts3d[:,0].min():+.4f}, {pts3d[:,1].min():+.4f}, {pts3d[:,2].min():+.4f}] m")
                    print(f"xyz max:   [{pts3d[:,0].max():+.4f}, {pts3d[:,1].max():+.4f}, {pts3d[:,2].max():+.4f}] m")
                #

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
                
                # DEBUG
                a, b, c, d = [float(x) for x in plane_model]
                n = np.array([a, b, c], dtype=np.float64)
                n_norm = np.linalg.norm(n)
                
                plane_dist_m = abs(d) / max(n_norm, 1e-12)
                
                print(f"Plane distance from camera origin: {plane_dist_m:.4f} m")
                print(f"Plane normal norm: {n_norm:.6f}")
                print(f"Plane normal z-component: {c / max(n_norm, 1e-12):+.6f}")
                
                corners_uv = np.array([
                    extremes["top_left"],
                    extremes["top_right"],
                    extremes["bottom_left"],
                ], dtype=np.float64)
                
                K_rgb = np.array([[1360.416961, 0, 975.507938], [0, 1362.366079, 544.699398], [0, 0, 1]])
                
                corner_xyz = rpf.intersect_corners_with_plane(
                    corners_uv=corners_uv,
                    rgb_intrinsics=K_rgb,
                    plane_model=plane_model,
                )
                
                P_tl, P_tr, P_bl = corner_xyz
                
                print("\nReconstructed extreme corners (camera frame)")
                print(f"P_tl = [{P_tl[0]:+.4f}, {P_tl[1]:+.4f}, {P_tl[2]:+.4f}] m")
                print(f"P_tr = [{P_tr[0]:+.4f}, {P_tr[1]:+.4f}, {P_tr[2]:+.4f}] m")
                print(f"P_bl = [{P_bl[0]:+.4f}, {P_bl[1]:+.4f}, {P_bl[2]:+.4f}] m")
                
                w_m = np.linalg.norm(P_tr - P_tl)
                h_m = np.linalg.norm(P_bl - P_tl)
                diag_m = np.linalg.norm(P_bl - P_tr)
                
                print(f"Top edge length     = {w_m*1000:.2f} mm")
                print(f"Left edge length    = {h_m*1000:.2f} mm")
                print(f"TR-BL distance      = {diag_m*1000:.2f} mm")
                #

                deviations = compute_deviations(pts3d, plane_model)
                inlier_deviations = deviations[inliers] if len(inliers) else deviations
                normal = plane_model[:3]
                normal /= np.linalg.norm(normal) + 1e-12

                print("\nRANSAC plane fitting")
                print(f"Plane model (a,b,c,d): [{plane_model[0]:+.6f}, {plane_model[1]:+.6f}, "
                      f"{plane_model[2]:+.6f}, {plane_model[3]:+.6f}]")
                print(f"Normal: [{normal[0]:+.6f}, {normal[1]:+.6f}, {normal[2]:+.6f}]")

                if len(inliers) == 0:
                    print("WARNING: No inliers returned by RANSAC. Cannot compute stats.")
                else:
                    inlier_dev_mm = inlier_deviations * 1000.0
                    mean_mm = float(np.mean(inlier_dev_mm))
                    median_mm = float(np.median(inlier_dev_mm))
                    std_mm = float(np.std(inlier_dev_mm))
                    max_mm = float(np.max(inlier_dev_mm))

                    print(f"Inliers: {len(inliers)}/{len(pts3d)} ({len(inliers)/len(pts3d):.3f})")
                    print(f"Inlier distance mean:   {mean_mm:.3f} mm")
                    print(f"Inlier distance median: {median_mm:.3f} mm")
                    print(f"Inlier distance std:    {std_mm:.3f} mm")
                    print(f"Inlier distance max:    {max_mm:.3f} mm")

                preview = snap_color.copy()
                preview = draw_extremes(preview, extremes, color=(208, 224, 64),
                                        radius=6, thickness=-1, draw_labels=False)
                preview = draw_support_rect(preview, rect, color=(0, 165, 255), thickness=2)
                top_left = extremes["top_left"]
                preview = draw_axes_top_left(preview, origin=(int(round(top_left[0])), int(round(top_left[1]))))

                #save_preview_same_dir(preview, "plane_fitting_01_corners_support.png")

                cv2.imshow(PLANE_WIN, preview)
                cv2.waitKey(10)

                if ask_plane_satisfied():
                    cv2.destroyWindow(RES_WIN)

                    accepted = True
                    accepted_cache = {
                        "pts3d": pts3d,
                        "inliers": inliers,
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

                max_plot = 8000
                if inlier_pts3d.shape[0] > max_plot:
                    rng = np.random.default_rng(0)
                    idx = rng.choice(inlier_pts3d.shape[0], size=max_plot, replace=False)
                    inlier_pts3d = inlier_pts3d[idx]

                show_plane_fit_3d(
                    inlier_pts3d=inlier_pts3d,
                    plane_model=accepted_cache["plane_model"],
                    title="RANSAC plane fit: inliers + fitted plane + normal",
                    tuning=get_depth_settings_dict(depth_sensor)
                )

                show_plane_fit_histogram(
                    pts3d=accepted_cache["pts3d"],
                    plane_model=accepted_cache["plane_model"],
                    inliers=accepted_cache["inliers"],
                    ransac_thresh_m=ransac_thresh_m,
                    bins=60,
                    density=False,
                )
                return

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()