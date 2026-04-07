# -*- coding: utf-8 -*-
"""
debug_plane_fitting.py

Live RGB-D debug script — identische Logik wie PlaneFittingPage:
  1. Live-Stream mit Checkerboard-Detection
  2. SPACE: 30 Depth-Frames mitteln → Plane Fit → Marker-Grid
  3. Dialog: Ecken ok? → Ja: speichert GENAU EINE NPZ, Nein: zurück zu Live
  4. Q/ESC: beenden

Gespeicherte Felder (identisch mit Page):
  - points_xyz_camera  (121, 3)  Marker-Grid in Kamera-Frame [m]
  - corners_uv         (9, 2)    Schachbrett-Ecken im Bild
  - rgb_image          (1080, 1920, 3)
  - K_rgb              (3, 3)
  - K_xray             (3, 3)
  - depth_avg_raw      (1080, 1920) float32 raw depth units
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QMessageBox

from overlay.tools import checkerboard_corner_detection as cbd
from overlay.tools import ransac_plane_fitting as rpf


# ─────────────────────────────────────────────────────────────────
# Config  (identisch mit PlaneFittingPage)
# ─────────────────────────────────────────────────────────────────
COLOR_SIZE   = (1920, 1080)
DEPTH_SIZE   = (1280, 720)
FPS          = 30
WINDOW_NAME  = "Debug Plane Fitting"

PATTERN_SIZE = (3, 3)
DET_WIDTH    = 640
PAD_PX       = 15
MAX_POINTS   = 5000
Z_MIN, Z_MAX = 0.30, 2.0
MIN_PTS_FIT  = 800
THRESH_M     = 0.001   # 1 mm
RANSAC_N     = 8
ITERS        = 3000
N_AVG_FRAMES = 30
STEPS_PER_EDGE = 10

USE_TEMPORAL = True
TEMP_ALPHA   = 0.1
TEMP_DELTA   = 20

DEBUG_SNAPSHOT_DIR = Path("debug_snapshots")

K_rgb = np.array([
    [1.37115843e+03, 0.00000000e+00, 9.73363350e+02],
    [0.00000000e+00, 1.36926962e+03, 5.37849300e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
], dtype=np.float64)

K_xray = np.array([
    [5.61452151e+03, 6.62189208e+00, 3.79975010e+02],
    [0.00000000e+00, 5.60599331e+03, 5.34475381e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────────────────────────

def _qt() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)



def _depth_u16_to_vis(depth_u16: np.ndarray) -> np.ndarray:
    nz = depth_u16[depth_u16 > 0]
    if nz.size == 0:
        return np.zeros((*depth_u16.shape, 3), dtype=np.uint8)
    lo, hi = float(np.percentile(nz, 2)), float(np.percentile(nz, 98))
    if hi <= lo:
        hi = lo + 1
    d8 = np.clip((depth_u16.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    vis[depth_u16 == 0] = 0
    return vis


def _rs_set(sensor, opt, val):
    try:
        if sensor is not None and sensor.supports(opt):
            sensor.set_option(opt, float(val))
    except Exception:
        pass


def _rs_range(sensor, opt):
    try:
        if sensor is not None and sensor.supports(opt):
            return sensor.get_option_range(opt)
    except Exception:
        return None


def _apply_depth_defaults(sensor) -> None:
    _rs_set(sensor, rs.option.enable_auto_exposure, 1)
    r = _rs_range(sensor, rs.option.emitter_enabled)
    if r is not None:
        _rs_set(sensor, rs.option.emitter_enabled, 2.0 if r.min <= 2.0 <= r.max else float(np.clip(1.0, r.min, r.max)))
    r = _rs_range(sensor, rs.option.laser_power)
    if r is not None:
        _rs_set(sensor, rs.option.laser_power, 0.5 * (r.min + r.max))
    r = _rs_range(sensor, rs.option.visual_preset)
    if r is not None:
        _rs_set(sensor, rs.option.visual_preset, float(np.clip(0.0, r.min, r.max)))


def _build_temporal_filter():
    if not USE_TEMPORAL:
        return None
    try:
        tf = rs.temporal_filter()
        tf.set_option(rs.option.filter_smooth_alpha, TEMP_ALPHA)
        tf.set_option(rs.option.filter_smooth_delta, TEMP_DELTA)
        return tf
    except Exception:
        return None


def _capture_averaged_depth(pipeline, align, temporal_filter):
    """30 Depth-Frames mitteln — identisch mit PlaneFittingPage._capture_averaged_depth."""
    accumulator = count_map = last_df = None
    for _ in range(N_AVG_FRAMES):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        df = frames.get_depth_frame()
        if not df:
            continue
        if temporal_filter is not None:
            try:
                df = temporal_filter.process(df).as_depth_frame()
            except Exception:
                pass
        raw = np.asanyarray(df.get_data()).astype(np.float64)
        if accumulator is None:
            accumulator = np.zeros_like(raw, dtype=np.float64)
            count_map   = np.zeros_like(raw, dtype=np.int32)
        valid = raw > 0
        accumulator[valid] += raw[valid]
        count_map[valid]   += 1
        last_df = df
    if accumulator is None or last_df is None:
        return None, None
    with np.errstate(invalid="ignore"):
        averaged = np.where(count_map > 0, accumulator / count_map, np.nan).astype(np.float32)
    return averaged, last_df


def _sample_pts3d(averaged_raw, depth_scale_m, intrinsics, rect, seed=0):
    """Punkte aus ROI samplen — identisch mit PlaneFittingPage._sample_pts3d_from_averaged."""
    rng = np.random.default_rng(seed)
    umin, vmin, umax, vmax = rect
    roi_w, roi_h = umax - umin + 1, vmax - vmin + 1
    num_pixels = roi_w * roi_h
    idx = rng.choice(num_pixels, size=min(MAX_POINTS, num_pixels), replace=False)
    us = (idx % roi_w).astype(np.int32) + umin
    vs = (idx // roi_w).astype(np.int32) + vmin
    pts3d = []
    for u, v in zip(us, vs):
        raw = averaged_raw[v, u]
        if np.isnan(raw) or raw <= 0:
            continue
        z = float(raw) * depth_scale_m
        if z < Z_MIN or z > Z_MAX:
            continue
        pts3d.append(rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], z))
    return np.asarray(pts3d, dtype=np.float64) if pts3d else np.empty((0, 3), dtype=np.float64)


def _draw_extremes(img, pts_uv):
    out = img.copy()
    for (u, v) in pts_uv:
        cv2.circle(out, (int(round(u)), int(round(v))), 10, (208, 224, 64), -1)
    return out


def _save_snapshot(marker_xyz, corners_full, color, averaged_raw, K_rgb, K_xray) -> Path:
    DEBUG_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = DEBUG_SNAPSHOT_DIR / f"plane_fitting_snapshot_{ts}.npz"

    arrays: dict[str, np.ndarray] = {
        "points_xyz_camera": np.asarray(marker_xyz, dtype=np.float64),
        "depth_avg_raw":     np.asarray(averaged_raw, dtype=np.float32),
    }
    if corners_full is not None:
        arrays["corners_uv"] = np.asarray(corners_full, dtype=np.float64).reshape(9, 2)
    if color is not None:
        arrays["rgb_image"] = color
    if K_rgb is not None:
        arrays["K_rgb"] = np.asarray(K_rgb, dtype=np.float64)
    if K_xray is not None:
        arrays["K_xray"] = np.asarray(K_xray, dtype=np.float64)

    np.savez(str(fname), **arrays)
    return fname


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    app = _qt()



    # ── RealSense setup ──────────────────────────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, COLOR_SIZE[0], COLOR_SIZE[1], rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_SIZE[0], DEPTH_SIZE[1], rs.format.z16,  FPS)
    pipeline.start(config)

    align           = rs.align(rs.stream.color)
    temporal_filter = _build_temporal_filter()
    depth_scale_m   = 1.0
    depth_sensor    = None

    try:
        prof         = pipeline.get_active_profile()
        depth_sensor = prof.get_device().first_depth_sensor()
        depth_scale_m = float(depth_sensor.get_depth_scale())
        _apply_depth_defaults(depth_sensor)
    except Exception as e:
        print(f"[WARN] Depth sensor access failed: {e}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    live_color: np.ndarray | None = None
    found = False

    print("Live — SPACE wenn Checkerboard FOUND | Q/ESC: beenden\n")

    # ── Live-Loop ────────────────────────────────────────────────
    while True:
        frames = pipeline.poll_for_frames()
        if frames:
            frames = align.process(frames)
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()

            if df and temporal_filter is not None:
                try:
                    df = temporal_filter.process(df).as_depth_frame()
                except Exception:
                    pass

            if cf:
                live_color = np.asanyarray(cf.get_data()).copy()
                gray = cv2.cvtColor(live_color, cv2.COLOR_BGR2GRAY)
                found, _ = cbd.detect_classic_downscaled(gray, PATTERN_SIZE, det_width=DET_WIDTH)

        if live_color is not None:
            vis = live_color.copy()
            txt = "FOUND (press SPACE)" if found else "NOT FOUND"
            col = (0, 255, 0) if found else (0, 0, 255)
            cv2.putText(vis, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(vis, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

        if key == 32 and found and live_color is not None:
            # ── Capture ─────────────────────────────────────────
            color_snap = live_color.copy()

            print(f"[INFO] Averaging {N_AVG_FRAMES} depth frames …")
            averaged_raw, last_df = _capture_averaged_depth(pipeline, align, temporal_filter)
            if averaged_raw is None:
                print("[WARN] Depth averaging failed.")
                continue

            intrinsics = last_df.profile.as_video_stream_profile().intrinsics

            # Vollständige Corner-Detection (wie Page)
            found_snap, corners = cbd.detect_snapshot_full(
                color_snap, pattern_size=PATTERN_SIZE, det_width=DET_WIDTH
            )
            if not found_snap or corners is None:
                print("[WARN] Corner detection failed on snapshot.")
                continue

            ex      = cbd.get_extreme_corners_geometric(corners)
            ext_uv  = np.array([ex["top_left"], ex["top_right"], ex["bottom_left"]], dtype=np.float64)

            # Preview
            preview = color_snap.copy()
            cv2.drawChessboardCorners(preview, PATTERN_SIZE, corners, True)
            preview = _draw_extremes(preview, ext_uv)
            cv2.imshow(WINDOW_NAME, preview)
            cv2.waitKey(1)

            # Dialog
            ans = QMessageBox.question(
                None,
                "Corner Detection",
                "Ecken ok?\n\nJa: Plane Fitting + speichern\nNein: zurück zu Live",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if ans != QMessageBox.Yes:
                print("[INFO] Abgebrochen — zurück zu Live.")
                continue

            # ── Plane Fit + Grid (identisch mit Page._fit_and_grid) ──
            if K_rgb is None:
                print("[ERROR] K_rgb fehlt — kann nicht fitten.")
                continue

            h_img, w_img = color_snap.shape[:2]
            rect  = rpf.rect_from_pts(ext_uv, w_img, h_img, PAD_PX)
            pts3d = _sample_pts3d(averaged_raw, depth_scale_m, intrinsics, rect, seed=0)

            if pts3d.shape[0] < MIN_PTS_FIT:
                print(f"[WARN] Zu wenig Punkte ({pts3d.shape[0]}).")
                continue

            plane, inliers = rpf.fit_plane_from_points(
                pts3d, distance_threshold=THRESH_M, ransac_n=RANSAC_N, num_iterations=ITERS
            )

            dev    = rpf.deviations(pts3d, plane)
            dev_in = dev[inliers] if len(inliers) else dev
            print(f"\n{'='*50}")
            print(f"  Inliers : {len(inliers)}/{pts3d.shape[0]}  ({100*len(inliers)/max(pts3d.shape[0],1):.1f}%)")
            print(f"  Mean    : {np.mean(dev_in)*1000:.3f} mm")
            print(f"  Median  : {np.median(dev_in)*1000:.3f} mm")
            print(f"  P95     : {np.percentile(dev_in,95)*1000:.3f} mm")
            print(f"{'='*50}\n")

            corner_xyz = rpf.intersect_corners_with_plane(ext_uv, K_rgb, plane)
            marker_xyz = rpf.interpolate_marker_grid(corner_xyz, steps_per_edge=STEPS_PER_EDGE)

            # ── Genau eine NPZ speichern ──────────────────────────
            fname = _save_snapshot(marker_xyz, corners, color_snap, averaged_raw, K_rgb, K_xray)
            print(f"[INFO] Snapshot gespeichert: {fname}")
            break  # fertig

    # ── Cleanup ──────────────────────────────────────────────────
    try:
        _apply_depth_defaults(depth_sensor)
    except Exception:
        pass
    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()