# -*- coding: utf-8 -*-
"""
debug_depth.py

Isolates depth estimation (d_x) from pointer tracking.

Assumption:
    T_xc = Identity  → X-ray source == camera center

Goal:
    Debug instability of T_tc (camera-to-tip pose)
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from overlay.calib.calib_camera_to_pointer import (
    calibrate_camera_to_pointer,
    get_default_pointer_tool_model,
    _build_T_pt,
)
from overlay.calib.calib_xray_to_pointer import extract_depth


# ============================================================
# Utilities
# ============================================================

def load_rgb_intrinsics(npz_path: str | Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)

    if "K_rgb" not in data:
        raise KeyError("NPZ does not contain 'K_rgb'.")

    K_rgb = np.asarray(data["K_rgb"], dtype=np.float64)

    if K_rgb.shape != (3, 3):
        raise ValueError(f"K_rgb must have shape (3,3), got {K_rgb.shape}")

    return K_rgb


def draw_text_lines(
    img: np.ndarray,
    lines: list[str],
    org=(30, 40),
    line_step=32,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.8,
    color=(0, 255, 0),
    thickness=2,
) -> np.ndarray:
    out = img.copy()
    x0, y0 = org
    for i, line in enumerate(lines):
        y = y0 + i * line_step
        cv2.putText(
            out,
            line,
            (x0, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return out


def rotation_angle_deg(R: np.ndarray) -> float:
    """
    Convert rotation matrix to rotation angle (deg).
    """
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.degrees(theta))


# ============================================================
# Main
# ============================================================

def main() -> None:
    npz_path = r"overlay_debug_20260324_171105.npz"

    K_rgb = load_rgb_intrinsics(npz_path)

    # Debug assumption:
    T_xc = np.eye(4, dtype=np.float64)

    print("=" * 80)
    print("Loaded intrinsics")
    print("=" * 80)
    print("K_rgb:")
    print(K_rgb)
    print()
    print("Using T_xc = Identity")
    print()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    pipeline.start(config)

    prev_rvec = None
    prev_tvec = None

    prev_R = None
    prev_t = None

    last_print_t = 0.0

    print("Press ESC or q to quit.\n")

    pointer_model = get_default_pointer_tool_model()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img_bgr = np.asanyarray(color_frame.get_data())
            status_lines: list[str] = []

            try:
                # --------------------------------------------------
                # Camera → Pointer Tip
                # --------------------------------------------------
                result = calibrate_camera_to_pointer(
                    image_bgr=img_bgr,
                    camera_intrinsics=K_rgb,
                    dist_coeffs=None,
                    rvec_init=prev_rvec,
                    tvec_init=prev_tvec,
                    use_extrinsic_guess=(prev_rvec is not None and prev_tvec is not None),
                )

                prev_rvec = result.rvec.copy()
                prev_tvec = result.tvec.copy()

                # --------------------------------------------------
                # Depth extraction (T_xc = I)
                # --------------------------------------------------
                depth_res = extract_depth(
                    T_xc=T_xc,
                    T_tc=result.T_4x4,
                )

                # --------------------------------------------------
                # Extract poses
                # --------------------------------------------------
                T_tc = result.T_4x4
                R_tc = T_tc[:3, :3]
                t_tc = T_tc[:3, 3]

                # Board pose (pointer frame)
                T_pt = _build_T_pt(pointer_model)
                T_pc = T_tc @ T_pt

                R_pc = T_pc[:3, :3]
                t_pc = T_pc[:3, 3]

                # --------------------------------------------------
                # Frame-to-frame changes
                # --------------------------------------------------
                delta_rot = 0.0
                delta_t = 0.0

                if prev_R is not None:
                    R_rel = prev_R.T @ R_pc
                    delta_rot = rotation_angle_deg(R_rel)

                if prev_t is not None:
                    delta_t = float(np.linalg.norm(t_pc - prev_t))

                prev_R = R_pc.copy()
                prev_t = t_pc.copy()

                # --------------------------------------------------
                # Print every second
                # --------------------------------------------------
                now = time.time()
                if now - last_print_t >= 1.0:
                    last_print_t = now

                    print("=" * 80)
                    print(time.strftime("%Y-%m-%d %H:%M:%S"))

                    print(f"markers detected : {len(result.marker_ids_detected)}")
                    print(f"markers used     : {len(result.marker_ids_used)}")
                    print(f"marker ids       : {result.marker_ids_used.tolist()}")

                    print(f"reproj mean [px] : {result.reproj_mean_px:.3f}")
                    print(f"reproj max  [px] : {result.reproj_max_px:.3f}")

                    print()
                    print("BOARD (PnP)")
                    print(f"t_pc [mm]        : {np.round(t_pc, 3)}")
                    print(f"Δt_pc [mm]       : {delta_t:.3f}")
                    print(f"Δrot [deg]       : {delta_rot:.3f}")

                    print()
                    print("TIP")
                    print(f"tip_xyz_c [mm]   : {np.round(result.tip_point_camera_mm, 3)}")
                    print(f"d_x [mm]         : {depth_res.d_x_mm:.3f}")

                    # Hebelarm sichtbar machen
                    tip_to_board = np.linalg.norm(result.tip_point_camera_mm - t_pc)
                    print(f"|tip - board| mm : {tip_to_board:.3f}")

                    print()

                # --------------------------------------------------
                # Visualization
                # --------------------------------------------------
                uv = np.round(result.tip_uv).astype(int)

                cv2.drawMarker(
                    img_bgr,
                    (int(uv[0]), int(uv[1])),
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=24,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )

                cv2.circle(
                    img_bgr,
                    (int(uv[0]), int(uv[1])),
                    8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                status_lines = [
                    f"markers: {len(result.marker_ids_used)} / {len(result.marker_ids_detected)}",
                    f"reproj: {result.reproj_mean_px:.2f}px",
                    f"d_x: {depth_res.d_x_mm:.2f} mm",
                ]

            except Exception as e:
                prev_rvec = None
                prev_tvec = None
                status_lines = [
                    "pointer tracking failed",
                    str(e),
                ]

            vis = draw_text_lines(img_bgr, status_lines)
            cv2.imshow("debug_depth", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()