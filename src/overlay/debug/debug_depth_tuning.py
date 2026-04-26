# debug_depth_tuning.py
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# Config
# ============================================================

COLOR_SIZE = (1920, 1080)
DEPTH_SIZE = (1280, 720)
FPS = 30

USE_TEMPORAL_FILTER = True
TEMPORAL_ALPHA = 0.1
TEMPORAL_DELTA = 20

WINDOW_NAME = "Debug Depth Tuning"

# Tuned values (wie in deiner Page)
TUNED_LASER_POWER = 150
TUNED_EXPOSURE = 4000
TUNED_GAIN = 16


# ============================================================
# Helpers
# ============================================================

@dataclass
class DepthSettingsSnapshot:
    values: Dict[rs.option, float]


def rs_get(sensor: Optional[rs.sensor], opt: rs.option) -> Optional[float]:
    try:
        if sensor is not None and sensor.supports(opt):
            return float(sensor.get_option(opt))
    except Exception:
        pass
    return None


def rs_set(sensor: Optional[rs.sensor], opt: rs.option, val: float) -> bool:
    try:
        if sensor is not None and sensor.supports(opt):
            sensor.set_option(opt, float(val))
            return True
    except Exception as e:
        print(f"[WARN] Could not set {opt}: {e}")
    return False


def rs_range(sensor: Optional[rs.sensor], opt: rs.option):
    try:
        if sensor is not None and sensor.supports(opt):
            return sensor.get_option_range(opt)
    except Exception:
        pass
    return None


def clip_to_supported(sensor: Optional[rs.sensor], opt: rs.option, value: float) -> Optional[float]:
    r = rs_range(sensor, opt)
    if r is None:
        return None
    return float(np.clip(value, r.min, r.max))


def snapshot_depth_settings(sensor: Optional[rs.sensor]) -> DepthSettingsSnapshot:
    opts = [
        rs.option.visual_preset,
        rs.option.emitter_enabled,
        rs.option.laser_power,
        rs.option.enable_auto_exposure,
        rs.option.exposure,
        rs.option.gain,
    ]
    out: Dict[rs.option, float] = {}
    for opt in opts:
        v = rs_get(sensor, opt)
        if v is not None:
            out[opt] = v
    return DepthSettingsSnapshot(values=out)


def restore_depth_settings(sensor: Optional[rs.sensor], snap: Optional[DepthSettingsSnapshot]) -> None:
    if sensor is None or snap is None:
        return
    for opt, val in snap.values.items():
        rs_set(sensor, opt, val)


def print_depth_settings(sensor: Optional[rs.sensor], title: str) -> None:
    print("")
    print("=" * 70)
    print(title)
    print("=" * 70)

    names = {
        rs.option.visual_preset: "visual_preset",
        rs.option.emitter_enabled: "emitter_enabled",
        rs.option.laser_power: "laser_power",
        rs.option.enable_auto_exposure: "enable_auto_exposure",
        rs.option.exposure: "exposure",
        rs.option.gain: "gain",
    }

    for opt in [
        rs.option.visual_preset,
        rs.option.emitter_enabled,
        rs.option.laser_power,
        rs.option.enable_auto_exposure,
        rs.option.exposure,
        rs.option.gain,
    ]:
        val = rs_get(sensor, opt)
        rng = rs_range(sensor, opt)
        if val is None:
            print(f"{names[opt]:>24}: not supported")
        elif rng is None:
            print(f"{names[opt]:>24}: {val}")
        else:
            print(
                f"{names[opt]:>24}: {val} "
                f"(range: {rng.min} .. {rng.max}, step={rng.step})"
            )
    print("")


def apply_depth_defaults(sensor: Optional[rs.sensor]) -> None:
    if sensor is None:
        return

    # So wie in eurer Page
    rs_set(sensor, rs.option.enable_auto_exposure, 1)

    r = rs_range(sensor, rs.option.emitter_enabled)
    if r is not None:
        target = 2.0
        if target < r.min or target > r.max:
            target = float(np.clip(1.0, r.min, r.max))
        rs_set(sensor, rs.option.emitter_enabled, target)

    r = rs_range(sensor, rs.option.laser_power)
    if r is not None:
        mid = 0.5 * (float(r.min) + float(r.max))
        rs_set(sensor, rs.option.laser_power, mid)

    r = rs_range(sensor, rs.option.visual_preset)
    if r is not None:
        rs_set(sensor, rs.option.visual_preset, float(np.clip(0.0, r.min, r.max)))


def apply_depth_tuning(sensor: Optional[rs.sensor]) -> None:
    if sensor is None:
        return

    # Genau wie in eurer Page
    rs_set(sensor, rs.option.emitter_enabled, 1)
    rs_set(sensor, rs.option.enable_auto_exposure, 0)

    laser = clip_to_supported(sensor, rs.option.laser_power, TUNED_LASER_POWER)
    expo = clip_to_supported(sensor, rs.option.exposure, TUNED_EXPOSURE)
    gain = clip_to_supported(sensor, rs.option.gain, TUNED_GAIN)

    if laser is not None:
        rs_set(sensor, rs.option.laser_power, laser)
    if expo is not None:
        rs_set(sensor, rs.option.exposure, expo)
    if gain is not None:
        rs_set(sensor, rs.option.gain, gain)


def make_temporal_filter(enabled: bool) -> Optional[rs.temporal_filter]:
    if not enabled:
        return None
    try:
        tf = rs.temporal_filter()
        tf.set_option(rs.option.filter_smooth_alpha, float(TEMPORAL_ALPHA))
        tf.set_option(rs.option.filter_smooth_delta, float(TEMPORAL_DELTA))
        return tf
    except Exception as e:
        print(f"[WARN] Could not create temporal filter: {e}")
        return None


def depth_u16_to_vis_bgr(depth_u16: np.ndarray) -> np.ndarray:
    depth_u16 = np.asarray(depth_u16, dtype=np.uint16)
    nonzero = depth_u16[depth_u16 > 0]
    if nonzero.size == 0:
        return np.zeros((depth_u16.shape[0], depth_u16.shape[1], 3), dtype=np.uint8)

    lo = float(np.percentile(nonzero, 2.0))
    hi = float(np.percentile(nonzero, 98.0))
    if hi <= lo:
        hi = lo + 1.0

    depth_8u = np.clip(
        (depth_u16.astype(np.float32) - lo) * (255.0 / (hi - lo)),
        0,
        255,
    ).astype(np.uint8)

    depth_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    depth_bgr[depth_u16 == 0] = (0, 0, 0)
    return depth_bgr


def draw_overlay(
    img: np.ndarray,
    *,
    mode_name: str,
    depth_scale_m: float,
    temporal_enabled: bool,
) -> np.ndarray:
    out = img.copy()

    lines = [
        f"Mode: {mode_name}",
        f"Temporal filter: {'ON' if temporal_enabled else 'OFF'}",
        f"Depth scale: {depth_scale_m:.8f} m/unit",
        "Keys: [t] tuned/default, [f] temporal, [r] reset defaults, [q]/[ESC] quit",
    ]

    y = 40
    for line in lines:
        cv2.putText(
            out, line, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            out, line, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 32

    return out


# ============================================================
# Main
# ============================================================

def main() -> int:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, COLOR_SIZE[0], COLOR_SIZE[1], rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_SIZE[0], DEPTH_SIZE[1], rs.format.z16, FPS)

    profile = None
    align = rs.align(rs.stream.color)
    sensor = None
    prev_settings = None
    temporal_filter = make_temporal_filter(USE_TEMPORAL_FILTER)
    use_tuned = False
    temporal_enabled = USE_TEMPORAL_FILTER
    depth_scale_m = 1.0

    try:
        print("[INFO] Starting RealSense pipeline ...")
        profile = pipeline.start(config)

        dev = profile.get_device()
        sensor = dev.first_depth_sensor()

        prev_settings = snapshot_depth_settings(sensor)

        try:
            depth_scale_m = float(sensor.get_depth_scale())
        except Exception:
            depth_scale_m = 1.0

        print_depth_settings(sensor, "Initial sensor settings")

        apply_depth_defaults(sensor)
        print_depth_settings(sensor, "After apply_depth_defaults()")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1400, 900)

        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            if temporal_enabled and temporal_filter is not None:
                try:
                    depth_frame = temporal_filter.process(depth_frame).as_depth_frame()
                except Exception:
                    pass

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_u16 = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
            depth_bgr = depth_u16_to_vis_bgr(depth_u16)

            mode_name = "TUNED" if use_tuned else "DEFAULT"
            depth_bgr = draw_overlay(
                depth_bgr,
                mode_name=mode_name,
                depth_scale_m=depth_scale_m,
                temporal_enabled=temporal_enabled,
            )

            # rechts klein das Farbbild zur Orientierung
            preview = color_bgr.copy()
            ph = 270
            pw = int(preview.shape[1] * (ph / preview.shape[0]))
            preview = cv2.resize(preview, (pw, ph), interpolation=cv2.INTER_AREA)

            H, W = depth_bgr.shape[:2]
            y0 = 20
            x0 = W - pw - 20
            if x0 >= 0 and y0 + ph <= H:
                depth_bgr[y0:y0 + ph, x0:x0 + pw] = preview
                cv2.rectangle(depth_bgr, (x0, y0), (x0 + pw, y0 + ph), (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, depth_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            elif key == ord("t"):
                use_tuned = not use_tuned
                if use_tuned:
                    apply_depth_tuning(sensor)
                    print_depth_settings(sensor, "After apply_depth_tuning()")
                else:
                    apply_depth_defaults(sensor)
                    print_depth_settings(sensor, "After apply_depth_defaults()")

            elif key == ord("r"):
                print("[INFO] Re-applying defaults ...")
                apply_depth_defaults(sensor)
                use_tuned = False
                print_depth_settings(sensor, "After manual reset to defaults")

            elif key == ord("f"):
                temporal_enabled = not temporal_enabled
                print(f"[INFO] Temporal filter: {'ON' if temporal_enabled else 'OFF'}")

    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    finally:
        try:
            print("[INFO] Restoring previous sensor settings ...")
            restore_depth_settings(sensor, prev_settings)
            print_depth_settings(sensor, "Restored previous sensor settings")
        except Exception as e:
            print(f"[WARN] Could not restore previous settings: {e}")

        try:
            pipeline.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())