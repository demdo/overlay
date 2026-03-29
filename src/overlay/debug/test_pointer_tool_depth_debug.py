from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt

from overlay.calib.calib_camera_to_pointer import (
    calibrate_camera_to_pointer,
    get_default_pointer_tool_model,
)
from overlay.tracking.transforms import (
    as_transform,
    invert_transform,
)


# ============================================================
# Config
# ============================================================

T_CX_FILE = Path(
    r"C:\Users\domin\Documents\Studium\Master\Masterarbeit\Projekt\Overlay\src\overlay\debug\T_cx_debug_new.npz"
)

POSE_METHOD = "ippe"
REFINE_WITH_ITERATIVE = True

WINDOW_NAME = "Pointer Tool Depth Debug Capture"


# ============================================================
# Helpers
# ============================================================

def _load_first_existing_array(npz_path: Path, keys: list[str]) -> np.ndarray:
    data = np.load(npz_path)
    for k in keys:
        if k in data:
            return np.asarray(data[k], dtype=np.float64)
    raise KeyError(f"None of the keys {keys} found in: {npz_path}")


def load_T_cx(npz_path: Path) -> np.ndarray:
    T_cx = _load_first_existing_array(
        npz_path,
        keys=["T_cx", "T", "transform", "T_4x4"],
    )

    # Optionaler 180°-Fix:
    # Falls das X-ray-Bild für die Kalibrierung um 180° gedreht wurde und
    # diese Drehung geometrisch in T_cx kompensiert werden soll.
    print("\n=== ANWENDUNG 180° FIX ===")
    R_fix = np.diag([1.0, -1.0, -1.0])   # 180° um X
    T_cx_fixed = T_cx.copy()
    T_cx_fixed[:3, :3] = T_cx_fixed[:3, :3] @ R_fix

    print("\nT_cx nach 180° Fix:")
    print(T_cx_fixed)

    T_cx = T_cx_fixed
    print("\n=== 180° FIX ANGEWENDET ===\n")

    return as_transform(T_cx, "T_cx")


def intrinsics_from_rs_video_stream_profile(
    profile: rs.video_stream_profile,
) -> np.ndarray:
    intr = profile.get_intrinsics()
    return np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def draw_text_block(
    img: np.ndarray,
    lines: list[str],
    org: tuple[int, int] = (20, 35),
    line_gap: int = 28,
    font_scale: float = 0.72,
) -> np.ndarray:
    out = img.copy()
    x0, y0 = org
    for i, line in enumerate(lines):
        y = y0 + i * line_gap
        cv2.putText(
            out, line, (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (0, 0, 0), 5, cv2.LINE_AA
        )
        cv2.putText(
            out, line, (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), 2, cv2.LINE_AA
        )
    return out


def draw_tip_marker(
    img: np.ndarray,
    tip_uv: np.ndarray | None,
    color: tuple[int, int, int] = (255, 0, 255),
) -> None:
    if tip_uv is None:
        return

    tip_uv = np.asarray(tip_uv, dtype=np.float64).reshape(2)
    if not np.isfinite(tip_uv).all():
        return

    u, v = np.round(tip_uv).astype(int)
    cv2.circle(img, (u, v), 8, color, 2, cv2.LINE_AA)
    cv2.drawMarker(
        img,
        (u, v),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "tip",
        (u + 10, v - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def plot_debug_records(records: list[dict]) -> None:
    if not records:
        print("[INFO] No measurements captured. No plots created.")
        return

    t = np.array([rec["t_rel_s"] for rec in records], dtype=np.float64)

    tip_x_x = np.array([rec["tip_xyz_x_mm"][0] for rec in records], dtype=np.float64)
    tip_y_x = np.array([rec["tip_xyz_x_mm"][1] for rec in records], dtype=np.float64)
    tip_z_x = np.array([rec["tip_xyz_x_mm"][2] for rec in records], dtype=np.float64)

    d_x = np.array([rec["d_x_mm"] for rec in records], dtype=np.float64)

    cam_x = np.array([rec["tip_xyz_c_mm"][0] for rec in records], dtype=np.float64)
    cam_y = np.array([rec["tip_xyz_c_mm"][1] for rec in records], dtype=np.float64)
    cam_z = np.array([rec["tip_xyz_c_mm"][2] for rec in records], dtype=np.float64)

    n_markers = np.array([rec["n_markers"] for rec in records], dtype=np.float64)
    reproj_mean = np.array([rec["reproj_mean_px"] for rec in records], dtype=np.float64)
    reproj_median = np.array([rec["reproj_median_px"] for rec in records], dtype=np.float64)
    reproj_max = np.array([rec["reproj_max_px"] for rec in records], dtype=np.float64)

    plt.figure(figsize=(10, 5))
    plt.plot(t, cam_z, label="z_c")
    plt.plot(t, d_x, label="d_x")
    plt.xlabel("Time [s]")
    plt.ylabel("mm")
    plt.title("z_c vs d_x over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    plt.plot(t, tip_x_x, label="x_x")
    plt.plot(t, tip_y_x, label="y_x")
    plt.plot(t, tip_z_x, label="z_x")
    plt.xlabel("Time [s]")
    plt.ylabel("Tip position in X-ray frame [mm]")
    plt.title("tip_xyz_x_mm over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    plt.plot(t, cam_x, label="x_c")
    plt.plot(t, cam_y, label="y_c")
    plt.plot(t, cam_z, label="z_c")
    plt.xlabel("Time [s]")
    plt.ylabel("Tip position in camera frame [mm]")
    plt.title("tip_xyz_c_mm over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    plt.plot(t, n_markers, label="n_markers")
    plt.plot(t, reproj_mean, label="reproj_mean_px")
    plt.plot(t, reproj_median, label="reproj_median_px")
    plt.plot(t, reproj_max, label="reproj_max_px")
    plt.xlabel("Time [s]")
    plt.ylabel("Count / error")
    plt.title("Detection quality over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("[INFO] Loading T_cx...")
    T_cx = load_T_cx(T_CX_FILE)
    T_xc = invert_transform(T_cx)

    print("[INFO] T_cx:")
    print(T_cx)
    print("[INFO] T_xc = inv(T_cx):")
    print(T_xc)

    pointer_model = get_default_pointer_tool_model()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    print("[INFO] Starting RealSense color stream...")
    profile = pipeline.start(config)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    K_rgb = intrinsics_from_rs_video_stream_profile(color_profile)

    print("[INFO] K_rgb:")
    print(K_rgb)

    prev_rvec = None
    prev_tvec = None

    records: list[dict] = []
    t0: float | None = None

    last_status: str = "READY"
    last_dx_mm: float | None = None

    print("\nControls:")
    print("  SPACE  -> capture one debug measurement")
    print("  r      -> reset pose seed")
    print("  q/ESC  -> quit and plot debug curves\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            vis = img.copy()

            pose_ok = False
            tip_uv = None
            T_tc = None
            tip_xyz_c_mm = None
            tip_xyz_x_mm = None
            d_x_mm = None

            reproj_mean_px = None
            reproj_median_px = None
            reproj_max_px = None
            n_markers = None

            try:
                result = calibrate_camera_to_pointer(
                    image_bgr=img,
                    camera_intrinsics=K_rgb,
                    dist_coeffs=None,
                    pointer_model=pointer_model,
                    rvec_init=prev_rvec,
                    tvec_init=prev_tvec,
                    use_extrinsic_guess=(prev_rvec is not None and prev_tvec is not None),
                    pose_method=POSE_METHOD,
                    refine_with_iterative=REFINE_WITH_ITERATIVE,
                )

                prev_rvec = result.rvec.copy()
                prev_tvec = result.tvec.copy()

                pose_ok = True
                tip_uv = np.asarray(result.tip_uv, dtype=np.float64).reshape(2)
                T_tc = np.asarray(result.T_4x4, dtype=np.float64)
                tip_xyz_c_mm = np.asarray(result.tip_point_camera_mm, dtype=np.float64).reshape(3)

                # Originale Interpretation:
                T_tx = T_cx @ T_tc
                tip_xyz_x_mm = np.asarray(T_tx[:3, 3], dtype=np.float64).reshape(3)
                d_x_mm = float(tip_xyz_x_mm[2])

                reproj_mean_px = float(result.reproj_mean_px)
                reproj_median_px = float(result.reproj_median_px)
                reproj_max_px = float(result.reproj_max_px)
                n_markers = int(len(result.marker_ids_used))

                last_dx_mm = d_x_mm
                last_status = "POSE OK"

            except Exception as e:
                pose_ok = False
                last_status = f"POSE FAIL: {e}"

            draw_tip_marker(vis, tip_uv)

            lines = [
                f"status: {last_status}",
                f"captured measurements: {len(records)}",
                "",
                "Controls: SPACE capture | r reset | q quit+plot",
            ]

            if pose_ok:
                lines.extend([
                    "",
                    f"d_x_mm  : {d_x_mm:8.3f}",
                    f"z_x_mm  : {tip_xyz_x_mm[2]:8.2f}",
                    f"z_c_mm  : {tip_xyz_c_mm[2]:8.2f}",
                    "",
                    f"tip_xyz_x: [{tip_xyz_x_mm[0]:8.2f}, {tip_xyz_x_mm[1]:8.2f}, {tip_xyz_x_mm[2]:8.2f}]",
                    f"tip_xyz_c: [{tip_xyz_c_mm[0]:8.2f}, {tip_xyz_c_mm[1]:8.2f}, {tip_xyz_c_mm[2]:8.2f}]",
                    f"markers used: {n_markers}",
                    f"reproj mean px: {reproj_mean_px:.3f}",
                ])
            else:
                if last_dx_mm is not None:
                    lines.extend([
                        "",
                        f"last d_x_mm: {last_dx_mm:8.3f}",
                    ])

            vis = draw_text_block(vis, lines)
            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if key == ord("r"):
                prev_rvec = None
                prev_tvec = None
                last_status = "POSE SEED RESET"
                print("[INFO] Pose seed reset.")

            elif key == 32:  # SPACE
                if not pose_ok:
                    print("[WARN] Cannot capture: no valid pointer pose in current frame.")
                    continue

                t_now = time.perf_counter()
                if t0 is None:
                    t0 = t_now

                record = {
                    "t_rel_s": float(t_now - t0),
                    "d_x_mm": float(d_x_mm),
                    "tip_xyz_x_mm": tip_xyz_x_mm.copy(),
                    "tip_xyz_c_mm": tip_xyz_c_mm.copy(),
                    "T_tc": T_tc.copy(),
                    "T_cx": T_cx.copy(),
                    "T_xc": T_xc.copy(),
                    "tip_uv": tip_uv.copy() if tip_uv is not None else np.full((2,), np.nan),
                    "rvec": np.asarray(result.rvec, dtype=np.float64).reshape(-1).copy(),
                    "tvec": np.asarray(result.tvec, dtype=np.float64).reshape(-1).copy(),
                    "reproj_mean_px": float(reproj_mean_px),
                    "reproj_median_px": float(reproj_median_px),
                    "reproj_max_px": float(reproj_max_px),
                    "marker_ids_used": np.asarray(result.marker_ids_used, dtype=np.int64).copy(),
                    "n_markers": int(n_markers),
                }
                records.append(record)

                last_status = f"CAPTURED  d_x={d_x_mm:.3f} mm"
                print(
                    f"[CAPTURE] #{len(records)}  "
                    f"t = {record['t_rel_s']:.3f} s, "
                    f"d_x = {d_x_mm:.3f} mm, "
                    f"z_c = {tip_xyz_c_mm[2]:.3f} mm, "
                    f"n_markers = {record['n_markers']}, "
                    f"reproj_mean = {record['reproj_mean_px']:.3f}px"
                )

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    plot_debug_records(records)


if __name__ == "__main__":
    main()