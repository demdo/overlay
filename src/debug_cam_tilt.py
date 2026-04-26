from __future__ import annotations

import time
import math
import numpy as np
import pyrealsense2 as rs


def vec_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = vec_norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector.")
    return v / n


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_u = unit(a)
    b_u = unit(b)
    c = float(np.clip(np.dot(a_u, b_u), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def format_vec(name: str, v: np.ndarray) -> str:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return f"{name} = [{v[0]:+8.5f}  {v[1]:+8.5f}  {v[2]:+8.5f}]"


def accel_frame_to_np(accel_frame: rs.motion_frame) -> np.ndarray:
    d = accel_frame.get_motion_data()
    return np.array([d.x, d.y, d.z], dtype=np.float64)


def compute_tilt_metrics(g_cam: np.ndarray) -> dict[str, float]:
    gx, gy, gz = g_cam
    g_norm = vec_norm(g_cam)
    if g_norm < 1e-12:
        raise ValueError("Invalid gravity vector norm.")

    g_u = g_cam / g_norm
    cam_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cam_minus_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    total_tilt_deg = angle_deg(cam_z, g_u)
    tilt_from_minus_z_deg = angle_deg(cam_minus_z, g_u)

    roll_deg = math.degrees(math.atan2(gy, gz))
    pitch_deg = math.degrees(math.atan2(-gx, math.sqrt(gy * gy + gz * gz)))

    return {
        "norm": g_norm,
        "total_tilt_deg": total_tilt_deg,
        "tilt_from_minus_z_deg": tilt_from_minus_z_deg,
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
    }


def open_imu_pipeline() -> tuple[rs.pipeline, rs.pipeline_profile]:
    pipeline = rs.pipeline()
    config = rs.config()

    # Wichtiger Fix: beide IMU-Streams zusammen
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)

    profile = pipeline.start(config)
    return pipeline, profile


def close_imu_pipeline(pipeline: rs.pipeline | None) -> None:
    if pipeline is None:
        return
    try:
        pipeline.stop()
    except Exception:
        pass


def main() -> int:
    pipeline: rs.pipeline | None = None

    try:
        print("Starting RealSense IMU stream...")
        pipeline, profile = open_imu_pipeline()
        device = profile.get_device()
        print(f"Connected device: {device.get_info(rs.camera_info.name)}")
        print("Reading accelerometer and printing tilt every 3 seconds.")
        print("On timeout, the IMU pipeline is reopened automatically.")
        print("Press Ctrl+C to stop.\n")

        while True:
            t_start = time.time()
            samples: list[np.ndarray] = []

            while time.time() - t_start < 3.0:
                try:
                    frames = pipeline.wait_for_frames(1000)
                except RuntimeError as e:
                    print(f"[WARN] IMU timeout: {e}")
                    print("[INFO] Reopening IMU pipeline...\n")
                    close_imu_pipeline(pipeline)
                    time.sleep(1.0)
                    pipeline, profile = open_imu_pipeline()
                    continue

                # Aus den Frames nur accel sammeln
                accel = frames.first_or_default(rs.stream.accel)
                if accel:
                    samples.append(accel_frame_to_np(accel.as_motion_frame()))

            if not samples:
                print("[WARN] No accelerometer samples received in this window.\n")
                continue

            acc_mean = np.mean(np.stack(samples, axis=0), axis=0)
            metrics = compute_tilt_metrics(acc_mean)

            print("=" * 70)
            print(format_vec("acc_mean [m/s^2]", acc_mean))
            print(f"norm                    = {metrics['norm']:.5f} m/s^2")
            print(f"roll_deg                = {metrics['roll_deg']:+.3f}")
            print(f"pitch_deg               = {metrics['pitch_deg']:+.3f}")
            print(f"angle(camera +Z, g)     = {metrics['total_tilt_deg']:.3f} deg")
            print(f"angle(camera -Z, g)     = {metrics['tilt_from_minus_z_deg']:.3f} deg")
            print()

    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0

    finally:
        close_imu_pipeline(pipeline)


if __name__ == "__main__":
    raise SystemExit(main())