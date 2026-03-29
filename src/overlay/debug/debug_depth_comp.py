# -*- coding: utf-8 -*-
"""
debug_depth.py

Live comparison of pointer depth d_x using:
    - iterative + refinement
    - IPPE + refinement
    - IPPE + refinement + Kalman filtering on the 3D tip position in camera coordinates

Workflow
--------
- live RGB video
- all pose methods run every frame
- press SPACE once:
    * set angular reference from the current ITERATIVE pose
    * clear old samples
    * reset Kalman filter
    * start continuous recording
- while recording:
    * every valid frame is saved automatically
- press 's':
    * save all recorded samples to disk
    * stop recording
- press ESC / q:
    * quit

Important simplification
------------------------
- only K_rgb is loaded from disk
- T_xc is assumed to be the identity matrix
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from overlay.calib.calib_camera_to_pointer import calibrate_camera_to_pointer
from overlay.calib.calib_xray_to_pointer import extract_depth
from overlay.tracking.pose_filters import AdaptiveKalmanFilterCV3D


# ============================================================
# Helpers
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
    org: tuple[int, int] = (30, 40),
    line_step: int = 32,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.8,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
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
    Return the angle of a relative rotation matrix in degrees.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.degrees(theta))


def make_output_path() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / f"debug_depth_compare_{ts}.npz"


# ============================================================
# Save
# ============================================================

def save_results(
    out_path: Path,
    *,
    angles_deg: list[float],
    d_x_iter_mm: list[float],
    d_x_ippe_mm: list[float],
    d_x_ippe_kf_mm: list[float],
    frame_indices: list[int],
) -> None:
    np.savez(
        out_path,
        angles_deg=np.asarray(angles_deg, dtype=np.float64),
        d_x_iter_mm=np.asarray(d_x_iter_mm, dtype=np.float64),
        d_x_ippe_mm=np.asarray(d_x_ippe_mm, dtype=np.float64),
        d_x_ippe_kf_mm=np.asarray(d_x_ippe_kf_mm, dtype=np.float64),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
    )


def save_results_extended(
    out_path: Path,
    *,
    angles_deg,
    d_x_iter_mm,
    d_x_ippe_mm,
    d_x_ippe_kf_mm,
    tip_iter_xyz_mm,
    tip_ippe_xyz_mm,
    tip_ippe_kf_xyz_mm,
    reproj_iter_px,
    reproj_ippe_px,
    num_markers_iter,
    num_markers_ippe,
    ippe_solution_index,
    kf_motion_score,
    kf_tip_step_mm,
    kf_rot_step_deg,
):
    np.savez(
        out_path,
        angles_deg=np.asarray(angles_deg, dtype=np.float64),
        d_x_iter_mm=np.asarray(d_x_iter_mm, dtype=np.float64),
        d_x_ippe_mm=np.asarray(d_x_ippe_mm, dtype=np.float64),
        d_x_ippe_kf_mm=np.asarray(d_x_ippe_kf_mm, dtype=np.float64),
        tip_iter_xyz_mm=np.asarray(tip_iter_xyz_mm, dtype=np.float64),
        tip_ippe_xyz_mm=np.asarray(tip_ippe_xyz_mm, dtype=np.float64),
        tip_ippe_kf_xyz_mm=np.asarray(tip_ippe_kf_xyz_mm, dtype=np.float64),
        reproj_iter_px=np.asarray(reproj_iter_px, dtype=np.float64),
        reproj_ippe_px=np.asarray(reproj_ippe_px, dtype=np.float64),
        num_markers_iter=np.asarray(num_markers_iter, dtype=np.int32),
        num_markers_ippe=np.asarray(num_markers_ippe, dtype=np.int32),
        ippe_solution_index=np.asarray(ippe_solution_index, dtype=np.int32),
        kf_motion_score=np.asarray(kf_motion_score, dtype=np.float64),
        kf_tip_step_mm=np.asarray(kf_tip_step_mm, dtype=np.float64),
        kf_rot_step_deg=np.asarray(kf_rot_step_deg, dtype=np.float64),
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    npz_path = r"overlay_debug_20260324_171105.npz"
    K_rgb = load_rgb_intrinsics(npz_path)

    T_xc = np.eye(4, dtype=np.float64)

    print("=" * 80)
    print("Loaded setup")
    print("=" * 80)
    print("K_rgb:")
    print(K_rgb)
    print()
    print("T_xc = Identity")
    print()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    print("Starting RealSense pipeline...")
    pipeline.start(config)
    print("Pipeline started.")
    print()

    # Previous pose for iterative solvePnP
    prev_rvec_iter: np.ndarray | None = None
    prev_tvec_iter: np.ndarray | None = None

    # Reference rotation for angle computation
    R_ref_iter: np.ndarray | None = None

    # Recording state
    recording = False

    # Kalman filter for IPPE + refinement tip position in camera coordinates
    dt = 1.0 / 30.0
    kf_ippe_tip = AdaptiveKalmanFilterCV3D(
        dt=dt,
        q_pos_still=1e-4,
        q_vel_still=1e-2,
        r_still=8e-2,
        q_pos_move=5e-3,
        q_vel_move=3e-1,
        r_move=2e-2,
    )

    # Current valid live values
    current_result_iter = None
    current_result_ippe = None
    current_dx_iter_mm: float | None = None
    current_dx_ippe_mm: float | None = None
    current_dx_ippe_kf_mm: float | None = None
    current_angle_deg: float | None = None
    current_tip_ippe_kf_xyz_mm: np.ndarray | None = None

    # Optional filter debug values
    current_motion_score_ippe_kf: float | None = None
    current_tip_step_mm: float | None = None
    current_rot_step_deg: float | None = None

    # Recorded arrays
    angles_deg: list[float] = []
    d_x_iter_mm: list[float] = []
    d_x_ippe_mm: list[float] = []
    d_x_ippe_kf_mm: list[float] = []
    frame_indices: list[int] = []

    # Extended debug arrays
    tip_iter_xyz_mm: list[list[float]] = []
    tip_ippe_xyz_mm: list[list[float]] = []
    tip_ippe_kf_xyz_mm: list[list[float]] = []

    reproj_iter_px: list[float] = []
    reproj_ippe_px: list[float] = []

    num_markers_iter: list[int] = []
    num_markers_ippe: list[int] = []

    ippe_solution_index: list[int] = []
    kf_motion_score: list[float] = []
    kf_tip_step_mm: list[float] = []
    kf_rot_step_deg: list[float] = []

    frame_idx = 0
    out_path = make_output_path()

    print("Controls:")
    print("  SPACE : set reference + clear old samples + reset Kalman + start recording")
    print("  s     : save recorded samples and stop recording")
    print("  q/ESC : quit")
    print()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_idx += 1
            img_bgr = np.asanyarray(color_frame.get_data())
            status_lines: list[str] = []

            iter_error = None
            ippe_error = None

            result_iter = None
            result_ippe = None
            depth_iter = None
            depth_ippe = None

            # --------------------------------------------------
            # ITERATIVE + REFINEMENT
            # --------------------------------------------------
            try:
                use_guess = (prev_rvec_iter is not None and prev_tvec_iter is not None)

                result_iter = calibrate_camera_to_pointer(
                    image_bgr=img_bgr,
                    camera_intrinsics=K_rgb,
                    dist_coeffs=None,
                    rvec_init=prev_rvec_iter,
                    tvec_init=prev_tvec_iter,
                    use_extrinsic_guess=use_guess,
                    pose_method="iterative",
                    refine_with_iterative=True,
                )

                prev_rvec_iter = result_iter.rvec.copy()
                prev_tvec_iter = result_iter.tvec.copy()

                depth_iter = extract_depth(
                    T_xc=T_xc,
                    T_tc=result_iter.T_4x4,
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                iter_error = str(e)
                prev_rvec_iter = None
                prev_tvec_iter = None

            # --------------------------------------------------
            # IPPE + REFINEMENT
            # --------------------------------------------------
            try:
                result_ippe = calibrate_camera_to_pointer(
                    image_bgr=img_bgr,
                    camera_intrinsics=K_rgb,
                    dist_coeffs=None,
                    rvec_init=None,
                    tvec_init=None,
                    use_extrinsic_guess=False,
                    pose_method="ippe",
                    refine_with_iterative=True,
                )

                depth_ippe = extract_depth(
                    T_xc=T_xc,
                    T_tc=result_ippe.T_4x4,
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                ippe_error = str(e)

            # --------------------------------------------------
            # Angle relative to iterative reference
            # --------------------------------------------------
            if result_iter is not None:
                R_iter = result_iter.T_4x4[:3, :3]

                if R_ref_iter is None:
                    angle_deg = 0.0
                else:
                    R_rel = R_ref_iter.T @ R_iter
                    angle_deg = rotation_angle_deg(R_rel)
            else:
                angle_deg = None

            current_result_iter = result_iter
            current_result_ippe = result_ippe
            current_dx_iter_mm = float(depth_iter.d_x_mm) if depth_iter is not None else None
            current_dx_ippe_mm = float(depth_ippe.d_x_mm) if depth_ippe is not None else None
            current_angle_deg = float(angle_deg) if angle_deg is not None else None

            # --------------------------------------------------
            # IPPE + REFINEMENT + KALMAN
            # --------------------------------------------------
            current_dx_ippe_kf_mm = None
            current_tip_ippe_kf_xyz_mm = None
            current_motion_score_ippe_kf = None
            current_tip_step_mm = None
            current_rot_step_deg = None

            if result_ippe is not None:
                tip_ippe_cam_filt = kf_ippe_tip.filter(
                    measurement_mm=result_ippe.tip_point_camera_mm,
                    rotation_camera=result_ippe.rotation,
                )

                current_tip_ippe_kf_xyz_mm = np.asarray(
                    tip_ippe_cam_filt,
                    dtype=np.float64,
                ).reshape(3)

                current_dx_ippe_kf_mm = float(current_tip_ippe_kf_xyz_mm[2])

                current_motion_score_ippe_kf = getattr(
                    kf_ippe_tip,
                    "last_motion_score",
                    None,
                )
                current_tip_step_mm = getattr(
                    kf_ippe_tip,
                    "last_tip_step_mm",
                    None,
                )
                current_rot_step_deg = getattr(
                    kf_ippe_tip,
                    "last_rot_step_deg",
                    None,
                )

            # --------------------------------------------------
            # Automatic recording
            # --------------------------------------------------
            if (
                recording
                and current_angle_deg is not None
                and current_dx_iter_mm is not None
                and current_dx_ippe_mm is not None
                and current_dx_ippe_kf_mm is not None
            ):
                angles_deg.append(current_angle_deg)
                d_x_iter_mm.append(current_dx_iter_mm)
                d_x_ippe_mm.append(current_dx_ippe_mm)
                d_x_ippe_kf_mm.append(current_dx_ippe_kf_mm)
                frame_indices.append(frame_idx)

                tip_iter_xyz_mm.append(
                    np.asarray(result_iter.tip_point_camera_mm, dtype=np.float64).reshape(3).tolist()
                )
                tip_ippe_xyz_mm.append(
                    np.asarray(result_ippe.tip_point_camera_mm, dtype=np.float64).reshape(3).tolist()
                )
                tip_ippe_kf_xyz_mm.append(
                    np.asarray(current_tip_ippe_kf_xyz_mm, dtype=np.float64).reshape(3).tolist()
                )

                reproj_iter_px.append(float(result_iter.reproj_mean_px))
                reproj_ippe_px.append(float(result_ippe.reproj_mean_px))

                num_markers_iter.append(int(len(result_iter.marker_ids_used)))
                num_markers_ippe.append(int(len(result_ippe.marker_ids_used)))

                ippe_solution_index.append(
                    -1 if result_ippe.ippe_solution_index is None else int(result_ippe.ippe_solution_index)
                )
                kf_motion_score.append(
                    np.nan if current_motion_score_ippe_kf is None else float(current_motion_score_ippe_kf)
                )
                kf_tip_step_mm.append(
                    np.nan if current_tip_step_mm is None else float(current_tip_step_mm)
                )
                kf_rot_step_deg.append(
                    np.nan if current_rot_step_deg is None else float(current_rot_step_deg)
                )

            # --------------------------------------------------
            # Visualization
            # --------------------------------------------------
            if result_iter is not None:
                uv_iter = np.round(result_iter.tip_uv).astype(int)
                cv2.drawMarker(
                    img_bgr,
                    (int(uv_iter[0]), int(uv_iter[1])),
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=26,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )

            if result_ippe is not None:
                uv_ippe = np.round(result_ippe.tip_uv).astype(int)
                cv2.drawMarker(
                    img_bgr,
                    (int(uv_ippe[0]), int(uv_ippe[1])),
                    (255, 255, 0),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=26,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )

            status_lines = [
                f"frame: {frame_idx}",
                f"recording: {'ON' if recording else 'OFF'}",
                f"saved samples: {len(angles_deg)}",
                f"angle rel. reference: {current_angle_deg:.2f} deg" if current_angle_deg is not None else "angle rel. reference: n/a",
            ]

            if result_iter is not None and current_dx_iter_mm is not None:
                status_lines.append(
                    f"iter+refine      d_x: {current_dx_iter_mm:.2f} mm   reproj: {result_iter.reproj_mean_px:.2f}px"
                )
            else:
                status_lines.append(f"iter+refine      FAILED: {iter_error}")

            if result_ippe is not None and current_dx_ippe_mm is not None:
                status_lines.append(
                    f"ippe+refine      d_x: {current_dx_ippe_mm:.2f} mm   reproj: {result_ippe.reproj_mean_px:.2f}px   idx: {result_ippe.ippe_solution_index}"
                )
            else:
                status_lines.append(f"ippe+refine      FAILED: {ippe_error}")

            if current_dx_ippe_kf_mm is not None:
                status_lines.append(
                    f"ippe+refine+kf   d_x: {current_dx_ippe_kf_mm:.2f} mm"
                )
            else:
                status_lines.append("ippe+refine+kf   d_x: n/a")

            if current_motion_score_ippe_kf is not None:
                status_lines.append(
                    f"kf motion: {current_motion_score_ippe_kf:.2f}   tip_step: {current_tip_step_mm:.3f} mm   rot_step: {current_rot_step_deg:.3f} deg"
                )
            else:
                status_lines.append("kf motion: n/a")

            status_lines.append("SPACE = start recording | s = save")

            vis = draw_text_lines(img_bgr, status_lines)
            cv2.imshow("debug_depth_compare", vis)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            elif key == ord(" "):
                if current_result_iter is None:
                    print("[SPACE] cannot start recording: tracking invalid.")
                    continue

                R_ref_iter = current_result_iter.T_4x4[:3, :3].copy()

                angles_deg.clear()
                d_x_iter_mm.clear()
                d_x_ippe_mm.clear()
                d_x_ippe_kf_mm.clear()
                frame_indices.clear()

                tip_iter_xyz_mm.clear()
                tip_ippe_xyz_mm.clear()
                tip_ippe_kf_xyz_mm.clear()

                reproj_iter_px.clear()
                reproj_ippe_px.clear()

                num_markers_iter.clear()
                num_markers_ippe.clear()

                ippe_solution_index.clear()
                kf_motion_score.clear()
                kf_tip_step_mm.clear()
                kf_rot_step_deg.clear()

                kf_ippe_tip.reset()

                recording = True
                print("[SPACE] reference set. Continuous recording started.")

            elif key == ord("s"):
                save_results(
                    out_path,
                    angles_deg=angles_deg,
                    d_x_iter_mm=d_x_iter_mm,
                    d_x_ippe_mm=d_x_ippe_mm,
                    d_x_ippe_kf_mm=d_x_ippe_kf_mm,
                    frame_indices=frame_indices,
                )

                out_path_ext = out_path.with_name(out_path.stem + "_extended.npz")
                save_results_extended(
                    out_path_ext,
                    angles_deg=angles_deg,
                    d_x_iter_mm=d_x_iter_mm,
                    d_x_ippe_mm=d_x_ippe_mm,
                    d_x_ippe_kf_mm=d_x_ippe_kf_mm,
                    tip_iter_xyz_mm=tip_iter_xyz_mm,
                    tip_ippe_xyz_mm=tip_ippe_xyz_mm,
                    tip_ippe_kf_xyz_mm=tip_ippe_kf_xyz_mm,
                    reproj_iter_px=reproj_iter_px,
                    reproj_ippe_px=reproj_ippe_px,
                    num_markers_iter=num_markers_iter,
                    num_markers_ippe=num_markers_ippe,
                    ippe_solution_index=ippe_solution_index,
                    kf_motion_score=kf_motion_score,
                    kf_tip_step_mm=kf_tip_step_mm,
                    kf_rot_step_deg=kf_rot_step_deg,
                )

                recording = False
                print(f"[s] saved {len(angles_deg)} samples to: {out_path}")
                print(f"[s] extended data saved to: {out_path_ext}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()