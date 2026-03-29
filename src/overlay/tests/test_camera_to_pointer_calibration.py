# -*- coding: utf-8 -*-
"""
test_camera_to_pointer_calibration.py

Live pointer tracking test using Intel RealSense D435i.

Displays
--------
- projected pointer origin
- projected pointer tip
- optionally:
    * detected ArUco markers with IDs
    * detected marker corner points with local corner indices 0..3

Controls
--------
SPACE : freeze frame and print transform + z-distance along optical axis
R     : reset pose initialization
M     : toggle marker overlay
C     : toggle corner-index overlay
ESC   : quit
"""

from __future__ import annotations

import cv2
import numpy as np
import pyrealsense2 as rs
from time import perf_counter

from overlay.calib.calib_camera_to_pointer import (
    calibrate_camera_to_pointer,
    get_default_pointer_tool_model,
)


# ============================================================
# Helpers
# ============================================================

def _draw_text_box(
    img_bgr: np.ndarray,
    lines: list[str],
    org=(30, 55),
    color=(255, 255, 255),
    line_gap=35,
    font_scale=1.0,
) -> np.ndarray:
    out = img_bgr.copy()
    x, y = org

    for i, t in enumerate(lines):
        yy = y + i * line_gap
        cv2.putText(
            out,
            t,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            t,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _detect_aruco(
    image: np.ndarray,
    aruco_dict,
    detector_params,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    gray = _ensure_gray(image)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=detector_params,
        )

    return corners, ids


def _draw_aruco_overlay(
    image_bgr: np.ndarray,
    corners: list[np.ndarray],
    ids: np.ndarray | None,
    *,
    show_markers: bool,
    show_corner_idx: bool,
) -> np.ndarray:
    vis = image_bgr.copy()

    if show_markers and ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

    if show_corner_idx:
        for c in corners:
            pts = np.asarray(c, dtype=np.float64).reshape(4, 2)
            for idx, (u, v) in enumerate(pts):
                uu = int(round(u))
                vv = int(round(v))
                cv2.circle(vis, (uu, vv), 4, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(
                    vis,
                    str(idx),
                    (uu + 6, vv - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

    return vis


def _draw_point(
    img: np.ndarray,
    uv: np.ndarray,
    *,
    color=(255, 0, 255),
    radius=7,
    cross_size=18,
    thickness=2,
    label: str | None = None,
) -> None:
    uv = np.asarray(uv, dtype=np.float64).reshape(2)
    u, v = np.round(uv).astype(int)

    cv2.circle(img, (u, v), radius, color, thickness, cv2.LINE_AA)
    cv2.drawMarker(
        img,
        (u, v),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=cross_size,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )

    if label is not None:
        cv2.putText(
            img,
            label,
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def _project_origin_uv(
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    origin_xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    uv, _ = cv2.projectPoints(origin_xyz, rvec, tvec, K, dist)
    return uv.reshape(2,)


def _print_result(result, origin_uv: np.ndarray) -> None:
    translation = result.translation.reshape(3)
    tip_cam = result.tip_point_camera_mm.reshape(3)

    z_origin_mm = float(translation[2])
    z_tip_mm = float(tip_cam[2])

    print("\n==============================")
    print("Camera to Pointer Calibration")
    print("==============================\n")

    print("T_4x4 (pointer -> camera):\n")
    print(np.array2string(result.T_4x4, precision=6, suppress_small=False))
    print()

    print("translation [mm]:")
    print(np.array2string(translation, precision=6, suppress_small=False))
    print()

    print("tip_point_camera_mm:")
    print(np.array2string(tip_cam, precision=6, suppress_small=False))
    print()

    print(f"origin_uv = [{origin_uv[0]:.2f}, {origin_uv[1]:.2f}]")
    print(f"tip_uv    = [{result.tip_uv[0]:.2f}, {result.tip_uv[1]:.2f}]")
    print()

    print("Distance along optical axis [mm]:")
    print(f"origin_z_mm = {z_origin_mm:.3f}")
    print(f"tip_z_mm    = {z_tip_mm:.3f}")
    print()

    print(f"marker_ids_detected = {result.marker_ids_detected.tolist()}")
    print(f"marker_ids_used     = {result.marker_ids_used.tolist()}")
    print()

    print(f"reproj_mean_px   = {result.reproj_mean_px:.4f}")
    print(f"reproj_median_px = {result.reproj_median_px:.4f}")
    print(f"reproj_max_px    = {result.reproj_max_px:.4f}")
    print(f"used_guess       = {result.used_extrinsic_guess}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    pointer_model = get_default_pointer_tool_model()

    dict_id = getattr(cv2.aruco, pointer_model.dictionary_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    if hasattr(cv2.aruco, "DetectorParameters"):
        detector_params = cv2.aruco.DetectorParameters()
    else:
        detector_params = cv2.aruco.DetectorParameters_create()

    pipeline = rs.pipeline()
    config = rs.config()

    WIDTH = 1920
    HEIGHT = 1080
    FPS = 30

    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)

    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()

    K = np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.array(intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)

    print("\nUsing RealSense intrinsics")
    print(K)
    print("dist =", dist.ravel())

    prev_rvec = None
    prev_tvec = None

    last_valid = None
    last_valid_vis = None
    last_valid_origin_uv = None

    frozen = False
    fps_value = 0.0
    last_error_text = None

    show_markers = True
    show_corner_idx = True

    window_name = "pointer_tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1080)

    t_prev = perf_counter()

    try:
        while True:
            if not frozen:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break
                    continue

                img = np.asanyarray(color_frame.get_data())

                corners, ids = _detect_aruco(
                    image=img,
                    aruco_dict=aruco_dict,
                    detector_params=detector_params,
                )

                vis = _draw_aruco_overlay(
                    img,
                    corners,
                    ids,
                    show_markers=show_markers,
                    show_corner_idx=show_corner_idx,
                )

                lines = ["NOT FOUND"]
                box_color = (0, 0, 255)

                try:
                    result = calibrate_camera_to_pointer(
                        image_bgr=img,
                        camera_intrinsics=K,
                        dist_coeffs=dist,
                        pointer_model=pointer_model,
                        rvec_init=prev_rvec,
                        tvec_init=prev_tvec,
                        use_extrinsic_guess=(prev_rvec is not None and prev_tvec is not None),
                    )

                    prev_rvec = result.rvec.copy()
                    prev_tvec = result.tvec.copy()

                    origin_uv = _project_origin_uv(
                        result.rvec,
                        result.tvec,
                        K,
                        dist,
                    )

                    _draw_point(
                        vis,
                        origin_uv,
                        color=(255, 0, 255),
                        label="origin",
                    )

                    _draw_point(
                        vis,
                        result.tip_uv,
                        color=(0, 0, 255),
                        label="tip",
                    )

                    last_valid = result
                    last_valid_vis = vis.copy()
                    last_valid_origin_uv = origin_uv.copy()
                    last_error_text = None

                    n_ids = 0 if ids is None else int(len(ids))
                    id_list = [] if ids is None else ids.reshape(-1).astype(int).tolist()

                    z_origin_mm = float(result.translation.reshape(3)[2])
                    z_tip_mm = float(result.tip_point_camera_mm.reshape(3)[2])

                    lines = [
                        "FOUND (press SPACE)",
                        f"ArUco markers: {n_ids}",
                        f"IDs: {id_list}",
                        f"Reproj: {result.reproj_mean_px:.3f} px",
                        f"Z origin: {z_origin_mm:.1f} mm",
                        f"Z tip: {z_tip_mm:.1f} mm",
                        f"Overlay: markers={'on' if show_markers else 'off'} corners={'on' if show_corner_idx else 'off'}",
                    ]
                    box_color = (0, 255, 0)

                except Exception as e:
                    prev_rvec = None
                    prev_tvec = None
                    last_error_text = str(e)

                    n_ids = 0 if ids is None else int(len(ids))
                    id_list = [] if ids is None else ids.reshape(-1).astype(int).tolist()

                    lines = [
                        "NOT FOUND",
                        f"ArUco markers: {n_ids}",
                        f"IDs: {id_list}",
                        str(e),
                        f"Overlay: markers={'on' if show_markers else 'off'} corners={'on' if show_corner_idx else 'off'}",
                    ]
                    box_color = (0, 0, 255)

                t_now = perf_counter()
                fps_value = 1.0 / max(t_now - t_prev, 1e-9)
                t_prev = t_now

                lines.append(f"FPS: {fps_value:.1f}")
                vis = _draw_text_box(vis, lines, color=box_color)

                cv2.imshow(window_name, vis)

            else:
                if last_valid is not None and last_valid_vis is not None:
                    vis = last_valid_vis.copy()

                    z_origin_mm = float(last_valid.translation.reshape(3)[2])
                    z_tip_mm = float(last_valid.tip_point_camera_mm.reshape(3)[2])

                    lines = [
                        "FROZEN",
                        f"Reproj: {last_valid.reproj_mean_px:.3f} px",
                        f"Z origin: {z_origin_mm:.1f} mm",
                        f"Z tip: {z_tip_mm:.1f} mm",
                        f"Overlay: markers={'on' if show_markers else 'off'} corners={'on' if show_corner_idx else 'off'}",
                        f"FPS: {fps_value:.1f}",
                    ]
                    vis = _draw_text_box(vis, lines, color=(0, 255, 255))
                    cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            elif key == ord("r") or key == ord("R"):
                prev_rvec = None
                prev_tvec = None
                frozen = False
                print("\nPose reset.")

            elif key == ord("m") or key == ord("M"):
                show_markers = not show_markers

            elif key == ord("c") or key == ord("C"):
                show_corner_idx = not show_corner_idx

            elif key == 32:
                frozen = not frozen
                if frozen:
                    if last_valid is not None and last_valid_origin_uv is not None:
                        _print_result(last_valid, last_valid_origin_uv)
                    else:
                        print("\nNo valid pointer pose available yet.")
                        if last_error_text is not None:
                            print("Last error:", last_error_text)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()