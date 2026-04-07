# -*- coding: utf-8 -*-
"""
example_ransac_plane_fitting.py

Interactive test harness for RANSAC_plane_fitting helpers.

Workflow
--------
1) Live RGB preview + aligned depth preview
2) SPACE  (checkerboard must be FOUND):
   - captures RGB snapshot immediately
   - accumulates N_AVERAGE_FRAMES depth frames and averages them pixel-wise
     → reduces random depth noise by sqrt(N_AVERAGE_FRAMES)
   - saves averaged depth snapshot + metadata
   - detects checkerboard corners, defines ROI
3) Sample 3D points in ROI from the averaged depth array
4) Robust plane fit by RANSAC
5) Show final figures:
   - 3D RANSAC fit
   - histogram of RANSAC fit

Settings (all at the top of the file)
---------------------------------------
USE_TUNED_SETTINGS   – RealSense sensor tuning
USE_TEMPORAL_FILTER  – per-frame temporal filter (live preview AND averaging)
TEMPORAL_ALPHA       – temporal filter smoothing strength [0..1], lower = more
TEMPORAL_DELTA       – temporal filter depth-change tolerance (Intel default: 20)
N_AVERAGE_FRAMES     – frames averaged on SPACE capture
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


# ---------------------------------------------------------------------------
# Window names
# ---------------------------------------------------------------------------
LIVE_WIN       = "Live RGB (SPACE=capture if FOUND, ESC=quit)"
DEPTH_WIN      = "Live Depth (aligned)"
RES_WIN        = "Corner Detection Result"
PLANE_WIN      = "Plane Fitting Preview (ESC=quit)"
DEPTH_SNAP_WIN = "Depth Snapshot + ROI"

# ---------------------------------------------------------------------------
# RealSense sensor tuning
# ---------------------------------------------------------------------------
USE_TUNED_SETTINGS = True

# ---------------------------------------------------------------------------
# Temporal filter
# ---------------------------------------------------------------------------
USE_TEMPORAL_FILTER = True
TEMPORAL_ALPHA      = 0.1
TEMPORAL_DELTA      = 20

# ---------------------------------------------------------------------------
# Depth frame averaging on capture
# Noise reduces by sqrt(N_AVERAGE_FRAMES).
#   10 frames  ~0.5x noise  (~0.3 s @ 30 fps)
#   30 frames  ~0.3x noise  (~1.0 s @ 30 fps)
#   60 frames  ~0.2x noise  (~2.0 s @ 30 fps)
# ---------------------------------------------------------------------------
N_AVERAGE_FRAMES = 30


# ===========================================================================
# Small UI / IO helpers
# ===========================================================================
def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def save_preview_same_dir(img: np.ndarray, filename: str) -> str:
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
    msg:   str = "Satisfied with the detected corners?\n\nYes: Next (plane fitting)\nNo: back to live video",
) -> bool:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    ans = messagebox.askyesno(title, msg, icon="question")
    root.destroy()
    return ans


def ask_plane_satisfied(
    title: str = "Plane Fitting",
    msg:   str = "Satisfied with plane fitting?\n\nYes: accept\nNo: redo",
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
            cv2.putText(img_bgr, name, (int(u) + 8, int(v) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return img_bgr


def draw_axes_top_left(image: np.ndarray, origin: tuple[int, int]):
    shaft = 70; thickness = 2; tip = 0.2
    cv2.arrowedLine(image, origin, (origin[0] + shaft, origin[1]),
                    (0, 0, 255), thickness, cv2.LINE_AA, tipLength=tip)
    cv2.arrowedLine(image, origin, (origin[0], origin[1] + shaft),
                    (0, 255, 0), thickness, cv2.LINE_AA, tipLength=tip)
    cv2.arrowedLine(image, origin, (origin[0] + int(shaft * 0.7), origin[1] + int(shaft * 0.7)),
                    (255, 0, 0), thickness, cv2.LINE_AA, tipLength=tip)
    return image


def draw_support_rect(img_bgr, rect, color=(255, 255, 0), thickness=2):
    umin, vmin, umax, vmax = rect
    out = img_bgr.copy()
    cv2.rectangle(out, (umin, vmin), (umax, vmax), color, thickness, cv2.LINE_AA)
    return out


def depth_array_to_vis_bgr(depth_raw: np.ndarray) -> np.ndarray:
    """Visualise a uint16 or float32 depth array as a colour-mapped BGR image."""
    arr = np.asarray(depth_raw)
    if arr.dtype != np.uint16:
        arr = np.nan_to_num(arr, nan=0.0).astype(np.uint16)

    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        return np.zeros((*arr.shape, 3), dtype=np.uint8)

    lo = float(np.percentile(nonzero, 2.0))
    hi = float(np.percentile(nonzero, 98.0))
    if hi <= lo:
        hi = lo + 1.0

    depth_8u  = np.clip((arr.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    depth_bgr[arr == 0] = (0, 0, 0)
    cv2.putText(depth_bgr, f"Depth vis: p2={lo:.0f}, p98={hi:.0f} (raw units)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return depth_bgr


def depth_frame_to_vis_bgr(depth_frame) -> np.ndarray:
    return depth_array_to_vis_bgr(np.asanyarray(depth_frame.get_data()))


# ===========================================================================
# RealSense helpers
# ===========================================================================
def _rs_get(sensor, opt):
    try:
        if sensor is not None and sensor.supports(opt):
            return float(sensor.get_option(opt))
    except Exception:
        pass
    return None


def _rs_set(sensor, opt, val) -> bool:
    try:
        if sensor is not None and sensor.supports(opt):
            sensor.set_option(opt, float(val))
            return True
    except Exception:
        pass
    return False


def _rs_range(sensor, opt):
    try:
        if sensor is not None and sensor.supports(opt):
            return sensor.get_option_range(opt)
    except Exception:
        pass
    return None


def get_depth_settings_dict(depth_sensor) -> dict:
    return {
        "visual_preset":   _rs_get(depth_sensor, rs.option.visual_preset),
        "emitter_enabled": _rs_get(depth_sensor, rs.option.emitter_enabled),
        "laser_power":     _rs_get(depth_sensor, rs.option.laser_power),
        "auto_exposure":   _rs_get(depth_sensor, rs.option.enable_auto_exposure),
        "exposure":        _rs_get(depth_sensor, rs.option.exposure),
        "gain":            _rs_get(depth_sensor, rs.option.gain),
    }


def start_realsense_rgbd(fps: int = 30):
    """
    Start pipeline and return:
        pipeline, align, profile, depth_sensor, temporal_filter, depth_scale

    temporal_filter is None when USE_TEMPORAL_FILTER is False.
    depth_scale converts raw uint16 units to metres.
    """
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, 1280,  720, rs.format.z16,  fps)

    profile = pipeline.start(config)
    dev     = profile.get_device()

    try:
        for sensor in dev.query_sensors():
            try:
                sensor.set_option(rs.option.frames_queue_size, 1)
            except Exception:
                pass
    except Exception:
        pass

    try:
        depth_sensor = dev.first_depth_sensor()
    except Exception:
        depth_sensor = None

    depth_scale = 1.0
    try:
        depth_scale = float(depth_sensor.get_depth_scale())
        print(f"[Depth scale] {depth_scale:.6f} m/unit")
    except Exception:
        print("[Depth scale] could not read, defaulting to 1.0")

    if depth_sensor is not None:
        if USE_TUNED_SETTINGS:
            _rs_set(depth_sensor, rs.option.emitter_enabled, 1)
            _rs_set(depth_sensor, rs.option.enable_auto_exposure, 0)
            r = _rs_range(depth_sensor, rs.option.laser_power)
            if r:
                _rs_set(depth_sensor, rs.option.laser_power, min(r.max, max(r.min, 150)))
            r = _rs_range(depth_sensor, rs.option.exposure)
            if r:
                _rs_set(depth_sensor, rs.option.exposure, float(np.clip(4000, r.min, r.max)))
            r = _rs_range(depth_sensor, rs.option.gain)
            if r:
                _rs_set(depth_sensor, rs.option.gain, float(np.clip(16, r.min, r.max)))
            print("[Depth tuning enabled]")
        else:
            _rs_set(depth_sensor, rs.option.enable_auto_exposure, 1)
            r = _rs_range(depth_sensor, rs.option.emitter_enabled)
            if r:
                _rs_set(depth_sensor, rs.option.emitter_enabled, min(r.max, 2))
            r = _rs_range(depth_sensor, rs.option.laser_power)
            if r:
                _rs_set(depth_sensor, rs.option.laser_power, float(np.clip(150, r.min, r.max)))
            print("[Depth tuning disabled -> auto exposure ON, emitter AUTO]")

        s = get_depth_settings_dict(depth_sensor)
        print("[Depth settings]")
        for k, v in s.items():
            print(f"  {k}: {v}")

    if USE_TEMPORAL_FILTER:
        temporal_filter = rs.temporal_filter()
        temporal_filter.set_option(rs.option.filter_smooth_alpha, TEMPORAL_ALPHA)
        temporal_filter.set_option(rs.option.filter_smooth_delta, TEMPORAL_DELTA)
        print(f"[Temporal filter enabled]  alpha={TEMPORAL_ALPHA}  delta={TEMPORAL_DELTA}")
    else:
        temporal_filter = None
        print("[Temporal filter disabled]")

    align = rs.align(rs.stream.color)
    return pipeline, align, profile, depth_sensor, temporal_filter, depth_scale


# ===========================================================================
# Depth frame averaging
# ===========================================================================
def capture_averaged_depth(
    pipeline,
    align,
    temporal_filter,
    n_frames: int = N_AVERAGE_FRAMES,
) -> tuple[np.ndarray, object]:
    """
    Accumulate `n_frames` aligned depth frames and return:
        averaged_raw  – float32 (H, W) in raw depth units, NaN where no signal
        last_df       – last rs.depth_frame (used to read intrinsics)

    Zero pixels (no IR signal) are excluded from the per-pixel mean.
    The temporal filter is applied to each frame individually before accumulation
    so the filter sees the proper time-ordered sequence.
    """
    print(f"[Averaging {n_frames} depth frames — hold still...]")
    accumulator = None
    count_map   = None
    last_df     = None

    for i in range(n_frames):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        df = frames.get_depth_frame()
        if not df:
            continue

        if temporal_filter is not None:
            df = temporal_filter.process(df).as_depth_frame()

        raw = np.asanyarray(df.get_data()).astype(np.float64)

        if accumulator is None:
            accumulator = np.zeros_like(raw, dtype=np.float64)
            count_map   = np.zeros_like(raw, dtype=np.int32)

        valid = raw > 0
        accumulator[valid] += raw[valid]
        count_map[valid]   += 1
        last_df = df

        if (i + 1) % 10 == 0 or (i + 1) == n_frames:
            print(f"  frame {i + 1}/{n_frames}")

    if accumulator is None:
        raise RuntimeError("capture_averaged_depth: no valid frames received.")

    with np.errstate(invalid="ignore"):
        averaged = np.where(
            count_map > 0,
            accumulator / count_map,
            np.nan,
        ).astype(np.float32)

    valid_frac = float(np.sum(count_map > 0)) / count_map.size * 100.0
    print(
        f"[Averaging done]  valid pixels: {valid_frac:.1f}%  "
        f"raw range: {float(np.nanmin(averaged)):.0f} .. {float(np.nanmax(averaged)):.0f}"
    )
    return averaged, last_df


def save_averaged_depth_snapshot(
    averaged_raw: np.ndarray,
    depth_scale: float,
    depth_sensor,
    prefix: str = "depth",
) -> None:
    ts    = time.strftime("%Y%m%d_%H%M%S")
    u16   = np.nan_to_num(averaged_raw, nan=0.0).astype(np.uint16)
    vis   = depth_array_to_vis_bgr(u16)

    save_preview_same_dir(u16, f"{prefix}_{ts}_avg{N_AVERAGE_FRAMES}_raw.png")
    save_preview_same_dir(vis, f"{prefix}_{ts}_avg{N_AVERAGE_FRAMES}_vis.png")

    meta = {
        "timestamp":           ts,
        "n_average_frames":    N_AVERAGE_FRAMES,
        "use_temporal_filter": USE_TEMPORAL_FILTER,
        "temporal_alpha":      TEMPORAL_ALPHA if USE_TEMPORAL_FILTER else None,
        "temporal_delta":      TEMPORAL_DELTA if USE_TEMPORAL_FILTER else None,
        "use_tuned_settings":  USE_TUNED_SETTINGS,
        "depth_scale_m":       depth_scale,
        "depth_settings":      get_depth_settings_dict(depth_sensor),
    }
    meta_path = os.path.join(script_dir(), f"{prefix}_{ts}_avg{N_AVERAGE_FRAMES}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[DEBUG] Saved: {meta_path}")


# ===========================================================================
# Geometry helpers
# ===========================================================================
def rect_from_extremes(extremes, img_w, img_h, pad_px):
    pts  = np.array([extremes["top_left"], extremes["top_right"],
                     extremes["bottom_left"]], dtype=np.float32)
    umin = max(0,        int(np.floor(np.min(pts[:, 0]) - pad_px)))
    umax = min(img_w-1,  int(np.ceil (np.max(pts[:, 0]) + pad_px)))
    vmin = max(0,        int(np.floor(np.min(pts[:, 1]) - pad_px)))
    vmax = min(img_h-1,  int(np.ceil (np.max(pts[:, 1]) + pad_px)))
    if umax < umin: umin, umax = umax, umin
    if vmax < vmin: vmin, vmax = vmax, vmin
    return (umin, vmin, umax, vmax)


def sample_points_3d_from_averaged(
    averaged_raw: np.ndarray,
    depth_scale:  float,
    intrinsics,
    rect,
    max_points: int,
    z_min: float,
    z_max: float,
    seed: int,
) -> np.ndarray:
    """
    Sample up to `max_points` 3-D points from the averaged depth array.
    averaged_raw : float32 (H, W) in raw depth units, NaN = no measurement.
    depth_scale  : multiply raw units by this to get metres.
    """
    rng  = np.random.default_rng(seed)
    umin, vmin, umax, vmax = rect
    roi_w = umax - umin + 1
    roi_h = vmax - vmin + 1
    if roi_w <= 2 or roi_h <= 2:
        return np.empty((0, 3), dtype=np.float64)

    num_pixels = roi_w * roi_h
    idx = rng.choice(num_pixels, size=min(max_points, num_pixels), replace=False)
    us  = (idx % roi_w).astype(np.int32) + umin
    vs  = (idx // roi_w).astype(np.int32) + vmin

    pts3d = []
    for u, v in zip(us, vs):
        raw = averaged_raw[v, u]
        if np.isnan(raw) or raw <= 0:
            continue
        z = float(raw) * depth_scale
        if z < z_min or z > z_max:
            continue
        xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], z)
        pts3d.append(xyz)

    return np.asarray(pts3d, dtype=np.float64) if pts3d else np.empty((0, 3), dtype=np.float64)


def compute_deviations(points: np.ndarray, plane_model: np.ndarray) -> np.ndarray:
    normal = plane_model[:3]
    norm   = np.linalg.norm(normal)
    if norm <= 1e-12:
        return np.zeros(points.shape[0], dtype=np.float64)
    return np.abs(points @ normal + plane_model[3]) / norm


def plane_stats_mm(pts3d, plane_model, indices=None) -> dict:
    pts = np.asarray(pts3d, dtype=np.float64)
    if indices is not None:
        pts = pts[np.asarray(indices, dtype=int)]
    dev = compute_deviations(pts, plane_model) * 1000.0
    return {
        "count":     int(len(dev)),
        "mean_mm":   float(np.mean(dev)),
        "median_mm": float(np.median(dev)),
        "std_mm":    float(np.std(dev)),
        "p95_mm":    float(np.percentile(dev, 95)),
        "max_mm":    float(np.max(dev)),
    }


# ===========================================================================
# Plot helpers
# ===========================================================================
_LATEX_RC = {
    "text.usetex":         True,
    "font.family":         "serif",
    "mathtext.fontset":    "cm",
    "font.size":           12,
    "axes.titlesize":      14,
    "axes.labelsize":      13,
    "legend.fontsize":     11,
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def _tuning_label() -> str:
    tf  = r"temporal\ filter"    if USE_TEMPORAL_FILTER else r"no\ temporal\ filter"
    dt  = r"depth\ tuning"       if USE_TUNED_SETTINGS  else r"no\ depth\ tuning"
    avg = rf"avg\ {N_AVERAGE_FRAMES}\ frames"
    return rf"$\mathrm{{{dt},\ {tf},\ {avg}}}$"


def show_plane_fit_3d(inlier_pts3d, plane_model, title="Plane fit (3D)", tuning=None):
    P = np.asarray(inlier_pts3d, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
        print("show_plane_fit_3d: not enough points.")
        return

    a, b, c, d = [float(x) for x in plane_model]
    n      = np.array([a, b, c])
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        print("show_plane_fit_3d: degenerate normal.")
        return
    n_unit = n / n_norm

    center = P.mean(axis=0)
    pmin   = P.min(axis=0)
    pmax   = P.max(axis=0)
    extent = float(np.linalg.norm(pmax - pmin)) or 1.0
    L      = 0.25 * extent
    res    = 25

    xs = np.linspace(pmin[0], pmax[0], res)
    ys = np.linspace(pmin[1], pmax[1], res)
    zs = np.linspace(pmin[2], pmax[2], res)

    sa = int(np.argmax(np.abs([a, b, c])))
    if   sa == 2 and abs(c) > 1e-12:
        XX, YY = np.meshgrid(xs, ys); ZZ = -(a*XX + b*YY + d) / c
    elif sa == 1 and abs(b) > 1e-12:
        XX, ZZ = np.meshgrid(xs, zs); YY = -(a*XX + c*ZZ + d) / b
    elif sa == 0 and abs(a) > 1e-12:
        YY, ZZ = np.meshgrid(ys, zs); XX = -(b*YY + c*ZZ + d) / a
    else:
        print("show_plane_fit_3d: cannot build mesh."); return

    plt.rcParams.update(_LATEX_RC)
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel(r"$X$"); ax.set_ylabel(r"$Y$"); ax.set_zlabel(r"$Z$")

    ax.plot_surface(XX, YY, ZZ, alpha=0.18, linewidth=0, color="0.65")
    ax.plot(XX[0,:],  YY[0,:],  ZZ[0,:],  "k", linewidth=2)
    ax.plot(XX[-1,:], YY[-1,:], ZZ[-1,:], "k", linewidth=2)
    ax.plot(XX[:,0],  YY[:,0],  ZZ[:,0],  "k", linewidth=2)
    ax.plot(XX[:,-1], YY[:,-1], ZZ[:,-1], "k", linewidth=2)

    ax.quiver(center[0], center[1], center[2],
              n_unit[0], n_unit[1], n_unit[2],
              length=L, normalize=True, color=(0,0,1),
              linewidths=3.5, arrow_length_ratio=0.25, pivot="tail")

    t = center + L*n_unit + 0.06*L*n_unit + 0.12*extent*np.array([-0.2,0.2,0])
    ax.text(t[0], t[1], t[2],
            r"$\mathbf{n}=\begin{bmatrix}" + rf"{a:.3f}\\ {b:.3f}\\ {c:.3f}" + r"\end{bmatrix}$")

    ax.scatter(P[:,0], P[:,1], P[:,2], s=6, alpha=0.35, c="#2ca02c",
               depthshade=False, label=r"$\mathrm{Inliers}$")

    mid      = (pmin + pmax) / 2
    max_half = float(np.max((pmax - pmin) / 2))
    ax.set_xlim(mid[0]-max_half, mid[0]+max_half)
    ax.set_ylim(mid[1]-max_half, mid[1]+max_half)
    ax.set_zlim(mid[2]-max_half, mid[2]+max_half)

    def _fmt(x):
        try:    return f"{float(x):.0f}"
        except: return "n/a"

    if USE_TUNED_SETTINGS and isinstance(tuning, dict):
        ae_str = "ON" if (tuning.get("auto_exposure") or 0) > 0.5 else "OFF"
        lines  = [
            r"$\bf{Depth\ tuning}$",
            f"Laser power:     {_fmt(tuning.get('laser_power'))}",
            f"Exposure:        {_fmt(tuning.get('exposure'))}",
            f"Auto exposure:   {ae_str}",
            f"Temporal filter: {'ON' if USE_TEMPORAL_FILTER else 'OFF'}",
            f"Averaged frames: {N_AVERAGE_FRAMES}",
        ]
    else:
        lines = [
            r"$\bf{No\ depth\ tuning}$",
            f"Temporal filter: {'ON' if USE_TEMPORAL_FILTER else 'OFF'}",
            f"Averaged frames: {N_AVERAGE_FRAMES}",
        ]

    fig.text(0.25, 0.85, "\n".join(lines), va="top", ha="left", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.3", alpha=0.95))
    ax.legend(loc="upper right", frameon=True)
    plt.show(block=True)


def show_plane_fit_histogram(pts3d, plane_model, inliers, ransac_thresh_m=None, bins=60):
    pts3d   = np.asarray(pts3d,   dtype=float)
    inliers = np.asarray(inliers, dtype=int)

    a, b, c, d = [float(x) for x in plane_model]
    nvec   = np.array([a, b, c])
    n_norm = np.linalg.norm(nvec)
    if n_norm < 1e-12:
        print("Invalid plane normal."); return

    d_mm  = (np.abs(pts3d[inliers] @ nvec + d) / n_norm) * 1000.0
    d_p95 = float(np.percentile(d_mm, 95))
    tau   = (ransac_thresh_m * 1000.0) if ransac_thresh_m is not None else None

    xmax = max((tau or 0.0), d_p95 * 1.8, 1e-6)
    counts, edges = np.histogram(d_mm, bins=bins, range=(0.0, xmax))

    plt.rcParams.update(_LATEX_RC)
    fig, ax = plt.subplots()
    bars = ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
                  alpha=0.85, edgecolor="none")
    for left, bar in zip(edges[:-1], bars):
        bar.set_color("#d62728" if left >= d_p95 else "#1f77b4")

    ax.set_title(r"RANSAC plane fit: $d_{\perp}$ distribution (" + _tuning_label() + r")")
    ax.set_xlabel(r"$d_{\perp}\;[\mathrm{mm}]$")
    ax.set_ylabel(r"Count")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0.0, xmax); ax.set_ylim(bottom=0.0)

    handles = [ax.plot([], [], " ")[0] for _ in range(5)]
    labels  = [
        rf"$n={len(inliers)}/{len(pts3d)}$",
        rf"$\bar{{d}}={float(np.mean(d_mm)):.3f}\,\mathrm{{mm}}$",
        rf"$\tilde{{d}}={float(np.median(d_mm)):.3f}\,\mathrm{{mm}}$",
        rf"$P_{{95}}={d_p95:.3f}\,\mathrm{{mm}}$",
        (rf"$\tau_{{\mathrm{{RANSAC}}}}={tau:.1f}\,\mathrm{{mm}}$"
         if tau is not None else r"$\tau_{\mathrm{RANSAC}}=\mathrm{n/a}$"),
    ]
    ax.legend(handles, labels, loc="upper right", frameon=True, borderpad=0.8)
    fig.tight_layout()
    plt.show(block=True)


# ===========================================================================
# Main
# ===========================================================================
def main():
    pattern_size    = (3, 3)
    det_width       = 640
    rect_pad_px     = 15
    max_points      = 5000
    z_min, z_max    = 0.15, 2.0
    ransac_thresh_m = 0.00075
    ransac_n        = 3
    ransac_iters    = 1000

    K_rgb = np.array([
        [1360.416961, 0.,          975.507938],
        [0.,          1362.366079, 544.699398],
        [0.,          0.,          1.        ],
    ])

    pipeline, align, profile, depth_sensor, temporal_filter, depth_scale = \
        start_realsense_rgbd(fps=30)

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

            if temporal_filter is not None:
                df = temporal_filter.process(df).as_depth_frame()

            color = np.asanyarray(cf.get_data())
            gray  = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            found_live, _ = cbd.detect_classic_downscaled(gray, pattern_size, det_width=det_width)

            vis_live = color.copy()
            label    = "FOUND (press SPACE to capture)" if found_live else "NOT FOUND"
            colour   = (0, 255, 0) if found_live else (0, 0, 255)
            cv2.putText(vis_live, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3, cv2.LINE_AA)
            cv2.imshow(LIVE_WIN, vis_live)

            dvis = depth_frame_to_vis_bgr(df)
            cv2.putText(dvis, "DEPTH (aligned)", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(DEPTH_WIN, dvis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key != 32 or not found_live:
                continue

            snap_color = color.copy()

            banner = vis_live.copy()
            cv2.putText(banner, f"Averaging {N_AVERAGE_FRAMES} frames...",
                        (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3, cv2.LINE_AA)
            cv2.imshow(LIVE_WIN, banner)
            cv2.waitKey(1)

            averaged_raw, last_df = capture_averaged_depth(
                pipeline, align, temporal_filter, n_frames=N_AVERAGE_FRAMES,
            )
            intrinsics = last_df.profile.as_video_stream_profile().intrinsics

            save_averaged_depth_snapshot(averaged_raw, depth_scale, depth_sensor)

            found, corners = cbd.detect_snapshot_full(
                snap_color, pattern_size=pattern_size, det_width=det_width,
            )
            if not found or corners is None:
                print("Corner detection failed on snapshot — try again.")
                continue

            res      = snap_color.copy()
            cv2.drawChessboardCorners(res, pattern_size, corners, True)
            extremes = cbd.get_extreme_corners_geometric(corners)
            res      = draw_extremes(res, extremes, color=(208,224,64), radius=10, thickness=-1)
            setup_window(RES_WIN)
            cv2.imshow(RES_WIN, res)

            if not ask_corners_satisfied():
                cv2.destroyWindow(RES_WIN)
                setup_window(LIVE_WIN)
                continue

            H, W = snap_color.shape[:2]
            rect  = rect_from_extremes(extremes, W, H, pad_px=rect_pad_px)

            depth_snap_vis = depth_array_to_vis_bgr(
                np.nan_to_num(averaged_raw, nan=0).astype(np.uint16)
            )
            depth_snap_vis = draw_support_rect(depth_snap_vis, rect, color=(0,165,255), thickness=2)
            cv2.imshow(DEPTH_SNAP_WIN, depth_snap_vis)
            cv2.waitKey(10)
            save_preview_same_dir(depth_snap_vis, "plane_fitting_depth_snapshot_roi.png")

            redo_seed      = 0
            accepted       = False
            accepted_cache = None

            while True:
                pts3d = sample_points_3d_from_averaged(
                    averaged_raw = averaged_raw,
                    depth_scale  = depth_scale,
                    intrinsics   = intrinsics,
                    rect         = rect,
                    max_points   = max_points,
                    z_min        = z_min,
                    z_max        = z_max,
                    seed         = redo_seed,
                )

                if len(pts3d) > 0:
                    zv = pts3d[:, 2]
                    span_mm = (zv.max() - zv.min()) * 1000.0
                    print(f"\nSampled 3D points (camera frame)")
                    print(f"  count:   {len(pts3d)}")
                    print(f"  z range: {zv.min():.4f} .. {zv.max():.4f} m  (span {span_mm:.2f} mm)")
                    print(f"  z mean:  {zv.mean():.4f} m")
                    print(f"  xyz min: [{pts3d[:,0].min():+.4f}, {pts3d[:,1].min():+.4f}, {pts3d[:,2].min():+.4f}] m")
                    print(f"  xyz max: [{pts3d[:,0].max():+.4f}, {pts3d[:,1].max():+.4f}, {pts3d[:,2].max():+.4f}] m")

                if len(pts3d) < 800:
                    print("Not enough valid points — try again.")
                    cv2.destroyWindow(RES_WIN)
                    setup_window(LIVE_WIN)
                    break

                plane_model, inliers = rpf.fit_plane_from_points(
                    pts3d,
                    distance_threshold = ransac_thresh_m,
                    ransac_n           = ransac_n,
                    num_iterations     = ransac_iters,
                )

                if len(inliers) == 0:
                    print("WARNING: No inliers returned by RANSAC — redo.")
                    redo_seed += 1
                    continue

                stats = plane_stats_mm(pts3d, plane_model, inliers)

                print("\nRANSAC plane fitting")
                print(f"  Plane: [{plane_model[0]:+.6f}, {plane_model[1]:+.6f}, "
                      f"{plane_model[2]:+.6f}, {plane_model[3]:+.6f}]")

                normal = plane_model[:3] / (np.linalg.norm(plane_model[:3]) + 1e-12)
                print(f"  Normal: [{normal[0]:+.6f}, {normal[1]:+.6f}, {normal[2]:+.6f}]")
                print(f"  Inliers: {len(inliers)}/{len(pts3d)} ({len(inliers)/len(pts3d):.3f})")
                print(f"  Mean:    {stats['mean_mm']:.3f} mm")
                print(f"  Median:  {stats['median_mm']:.3f} mm")
                print(f"  Std:     {stats['std_mm']:.3f} mm")
                print(f"  P95:     {stats['p95_mm']:.3f} mm")
                print(f"  Max:     {stats['max_mm']:.3f} mm")

                a, b, c, d = [float(x) for x in plane_model]
                n_norm = np.linalg.norm([a, b, c])
                print(f"\n  Plane dist from origin: {abs(d)/max(n_norm,1e-12):.4f} m")
                print(f"  Normal z-component:     {c/max(n_norm,1e-12):+.6f}")

                corners_uv = np.array([
                    extremes["top_left"], extremes["top_right"], extremes["bottom_left"],
                ], dtype=np.float64)
                P_tl, P_tr, P_bl = rpf.intersect_corners_with_plane(
                    corners_uv=corners_uv, rgb_intrinsics=K_rgb, plane_model=plane_model,
                )
                print("\nReconstructed extreme corners (camera frame)")
                print(f"  P_tl = [{P_tl[0]:+.4f}, {P_tl[1]:+.4f}, {P_tl[2]:+.4f}] m")
                print(f"  P_tr = [{P_tr[0]:+.4f}, {P_tr[1]:+.4f}, {P_tr[2]:+.4f}] m")
                print(f"  P_bl = [{P_bl[0]:+.4f}, {P_bl[1]:+.4f}, {P_bl[2]:+.4f}] m")
                print(f"  Top edge:   {np.linalg.norm(P_tr-P_tl)*1000:.2f} mm")
                print(f"  Left edge:  {np.linalg.norm(P_bl-P_tl)*1000:.2f} mm")
                print(f"  TR-BL diag: {np.linalg.norm(P_bl-P_tr)*1000:.2f} mm")

                preview = snap_color.copy()
                preview = draw_extremes(preview, extremes, color=(208,224,64),
                                        radius=6, thickness=-1, draw_labels=False)
                preview = draw_support_rect(preview, rect, color=(0,165,255), thickness=2)
                tl = extremes["top_left"]
                preview = draw_axes_top_left(preview, (int(round(tl[0])), int(round(tl[1]))))
                cv2.imshow(PLANE_WIN, preview)
                cv2.waitKey(10)

                if ask_plane_satisfied():
                    cv2.destroyWindow(RES_WIN)
                    accepted = True
                    accepted_cache = {
                        "pts3d":       pts3d,
                        "inliers":     inliers,
                        "plane_model": plane_model,
                        "stats":       stats,
                    }
                    break

                redo_seed += 1

            if accepted and accepted_cache is not None:
                while (cv2.waitKey(50) & 0xFF) != 27:
                    pass
                cv2.destroyWindow(PLANE_WIN)

                inlier_pts = accepted_cache["pts3d"][accepted_cache["inliers"]]
                if inlier_pts.shape[0] > 8000:
                    rng = np.random.default_rng(0)
                    inlier_pts = inlier_pts[rng.choice(inlier_pts.shape[0], 8000, replace=False)]

                show_plane_fit_3d(
                    inlier_pts, accepted_cache["plane_model"],
                    title="RANSAC plane fit: inliers + plane + normal",
                    tuning=get_depth_settings_dict(depth_sensor),
                )
                show_plane_fit_histogram(
                    accepted_cache["pts3d"], accepted_cache["plane_model"],
                    accepted_cache["inliers"], ransac_thresh_m=ransac_thresh_m, bins=60,
                )
                return

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()