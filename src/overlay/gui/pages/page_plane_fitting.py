# overlay/gui/pages/page_plane_fitting.py

from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np
import cv2

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QMessageBox, QSizePolicy, QWidget, QCheckBox

import pyrealsense2 as rs

from overlay.gui.state import SessionState
from overlay.gui.widgets.widget_flow_layout import FlowLayout

from overlay.tools import checkerboard_corner_detection as cbd
from overlay.tools import ransac_plane_fitting as rpf
from overlay.gui.pages.templates.templ_live_image import LiveImagePage


def _draw_extremes(img: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    out = img.copy()
    for (u, v) in pts_uv:
        cv2.circle(out, (int(round(u)), int(round(v))), 10, (208, 224, 64), -1)
    return out


class PlaneFittingPage(LiveImagePage):
    def __init__(self, state: SessionState, on_complete_changed=None, parent=None):
        self.state = state
        self.on_complete_changed = on_complete_changed

        # detection
        self.pattern_size = (3, 3)
        self.det_width = 640

        # ROI & sampling
        self.pad_px = 15
        self.max_points = 40000
        self.z_min, self.z_max = 0.15, 2.0
        self.min_points_for_fit = 800

        # RANSAC
        self.thresh_m = 0.005
        self.ransac_n = 3
        self.iters = 1000

        # local config
        self.steps_per_edge = 10

        # runtime
        self._mode = "idle"  # idle | live | frozen
        self._found = False

        self._live_color: np.ndarray | None = None
        self._live_depth = None  # keep if you still want it
        # IMPORTANT: cache depth as numpy so we can display even if rs.poll_for_frames() returns None
        self._live_depth_u16: np.ndarray | None = None
        self._live_depth_vis_bgr: np.ndarray | None = None

        self._snap_color: np.ndarray | None = None
        self._snap_vis: np.ndarray | None = None
        self._snap_depth = None

        self._ext_uv: np.ndarray | None = None  # (3,2)
        self._rect: tuple[int, int, int, int] | None = None
        self._seed = 0
        self._show_depth = False

        # RealSense depth tuning
        self.use_tuned_depth_settings = True  # Plane fitting benefits from tuning
        self._depth_sensor: rs.sensor | None = None
        self._depth_prev_settings: dict[rs.option, float] | None = None

        # stats storage
        self._stats: list[str] = []
        self._last_stats_rows: list[tuple[str, str]] | None = None

        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        self._build_controls()

        self.instructions_label.setText(
            "1) Press Start to open RGB-D\n"
            "2) Move checkerboard until FOUND\n"
            "3) Press SPACE to capture\n"
            "4) Confirm corners to run plane fitting"
        )

        self.set_viewport_background(active=False)

        # if state already has plane points, show frozen so Next works immediately
        if getattr(self.state, "plane_model_c", None) is not None and getattr(self.state, "plane_stats", None) is not None:
            self._mode = "frozen"

        self._update_buttons()
        self._update_panels()
        self.update_view()

    # ---------------- UI: controls + panels ----------------

    def _build_controls(self) -> None:
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_clicked)

        self.btn_start.setFocusPolicy(Qt.NoFocus)
        self.btn_start.setMinimumHeight(44)
        self.btn_start.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.btn_start.setMinimumWidth(0)

        self.chk_show_depth = QCheckBox("Show depth image")
        self.chk_show_depth.setChecked(False)
        self.chk_show_depth.stateChanged.connect(self._on_show_depth_changed)
        self.chk_show_depth.setFocusPolicy(Qt.NoFocus)

        wrap = QWidget()
        wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        flow = FlowLayout(wrap, spacing=10)
        wrap.setLayout(flow)

        flow.addWidget(self.btn_start)
        flow.addWidget(self.chk_show_depth)

        self.controls_content.addWidget(wrap)

    def _update_panels(self) -> None:
        rows = self.stats_rows()
        if rows != self._last_stats_rows:
            self.set_stats_rows(rows)
            self._last_stats_rows = list(rows)

    def stats_rows(self) -> list[tuple[str, str]]:
        if not self._stats:
            return [
                ("Inliers", "-"),
                ("Mean", "-"),
                ("Median", "-"),
                ("P95", "-"),
            ]

        out: list[tuple[str, str]] = []
        for s in self._stats:
            if ":" in s:
                k, v = s.split(":", 1)
                out.append((k.strip(), v.strip()))
            else:
                out.append((s.strip(), ""))
        return out

    # ---------------- template hooks ----------------

    def get_frame(self) -> np.ndarray | None:
        if self._mode == "idle":
            return None
        if self._mode == "frozen":
            return self._snap_vis

        # No pipeline yet -> return last cached frame depending on toggle
        if self.pipeline is None or self.align is None:
            if self._show_depth and self._live_depth_vis_bgr is not None:
                return self._live_depth_vis_bgr
            return self._live_color

        frames = self.pipeline.poll_for_frames()

        # If no new frames, return cached view depending on toggle
        if not frames:
            if self._show_depth and self._live_depth_vis_bgr is not None:
                return self._live_depth_vis_bgr
            return self._live_color

        # Align depth to color, so depth visualization matches RGB geometry
        frames = self.align.process(frames)

        cf = frames.get_color_frame()
        df = frames.get_depth_frame()

        # If color missing, fall back to cached
        if not cf:
            if self._show_depth and self._live_depth_vis_bgr is not None:
                return self._live_depth_vis_bgr
            return self._live_color

        # Update RGB cache always
        color = np.asanyarray(cf.get_data())
        self._live_color = color

        # Update depth caches if depth frame exists
        if df:
            self._live_depth = df
            depth_u16 = np.asanyarray(df.get_data()).astype(np.uint16)
            self._live_depth_u16 = depth_u16

            # Precompute visual once per frame (so display can fallback reliably)
            self._live_depth_vis_bgr = self._depth_u16_to_vis_bgr(depth_u16)

        # Checkerboard detection ALWAYS on RGB (even if we display depth)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        found, _ = cbd.detect_classic_downscaled(gray, self.pattern_size, det_width=self.det_width)
        self._found = bool(found)

        self._update_panels()

        # --- Display selection (toggle) ---
        if self._show_depth:
            # If we don't have a valid depth yet, show black (or last RGB if you prefer)
            if self._live_depth_vis_bgr is None:
                h, w = color.shape[:2]
                return np.zeros((h, w, 3), dtype=np.uint8)

            depth_bgr = self._live_depth_vis_bgr.copy()
            cv2.putText(
                depth_bgr,
                "DEPTH VIEW",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return depth_bgr

        return self._live_color

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        vis = frame_bgr.copy()
        if self._mode == "live":
            txt = "FOUND (press SPACE)" if self._found else "NOT FOUND"
            col = (0, 255, 0) if self._found else (0, 0, 255)
            cv2.putText(vis, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(vis, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 2, cv2.LINE_AA)
        return vis

    # ---------------- depth image helpers ----------------

    def _on_show_depth_changed(self, state: int) -> None:
        self._show_depth = (state == Qt.Checked)
        self.update_view()

    @staticmethod
    def _depth_u16_to_vis_bgr(depth_u16: np.ndarray) -> np.ndarray:
        """
        Robust depth visualization:
        - normalize using 2..98 percentiles (raw units)
        - invalid depth (0) shown as black
        """
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
            0, 255
        ).astype(np.uint8)

        depth_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        depth_bgr[depth_u16 == 0] = (0, 0, 0)
        return depth_bgr

    # ---------------- lifecycle ----------------

    def on_enter(self) -> None:
        pass

    def on_leave(self) -> None:
        try:
            self._apply_depth_defaults()
        except Exception:
            pass

        super().on_leave()

    # ---------------- RealSense depth tuning helpers ----------------

    @staticmethod
    def _rs_get(sensor: rs.sensor | None, opt: rs.option) -> float | None:
        try:
            if sensor is not None and sensor.supports(opt):
                return float(sensor.get_option(opt))
        except Exception:
            pass
        return None

    @staticmethod
    def _rs_set(sensor: rs.sensor | None, opt: rs.option, val: float) -> bool:
        try:
            if sensor is not None and sensor.supports(opt):
                sensor.set_option(opt, float(val))
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _rs_range(sensor: rs.sensor | None, opt: rs.option):
        try:
            if sensor is not None and sensor.supports(opt):
                return sensor.get_option_range(opt)
        except Exception:
            pass
        return None

    def _get_depth_sensor(self) -> rs.sensor | None:
        """Fetch & cache depth sensor from current pipeline (if running)."""
        if self.pipeline is None:
            self._depth_sensor = None
            return None
        try:
            prof = self.pipeline.get_active_profile()
            dev = prof.get_device()
            self._depth_sensor = dev.first_depth_sensor()
            return self._depth_sensor
        except Exception:
            self._depth_sensor = None
            return None

    def _snapshot_depth_settings(self, sensor: rs.sensor | None) -> Dict[rs.option, float]:
        """Store exactly what we will touch, so restore is precise."""
        opts = [
            rs.option.visual_preset,
            rs.option.emitter_enabled,
            rs.option.laser_power,
            rs.option.enable_auto_exposure,
            rs.option.exposure,
            rs.option.gain,
        ]
        snap: Dict[rs.option, float] = {}
        for o in opts:
            v = self._rs_get(sensor, o)
            if v is not None:
                snap[o] = float(v)
        return snap

    def _apply_depth_defaults(self) -> None:
        """
        Reset depth sensor to a known 'default-ish' baseline.
        (RealSense settings can persist across runs; so we force a clean baseline.)
        """
        sensor = self._get_depth_sensor()
        if sensor is None:
            return

        # Snapshot previous ONCE (so restore-on-leave is exact)
        if self._depth_prev_settings is None:
            self._depth_prev_settings = self._snapshot_depth_settings(sensor)

        # 1) Auto exposure ON
        self._rs_set(sensor, rs.option.enable_auto_exposure, 1)

        # 2) Emitter: prefer AUTO (often 2), else ON
        r = self._rs_range(sensor, rs.option.emitter_enabled)
        if r is not None:
            target = 2.0
            if target < r.min or target > r.max:
                target = float(np.clip(1.0, r.min, r.max))
            self._rs_set(sensor, rs.option.emitter_enabled, target)

        # 3) Laser power: neutral mid
        r = self._rs_range(sensor, rs.option.laser_power)
        if r is not None:
            mid = 0.5 * (float(r.min) + float(r.max))
            self._rs_set(sensor, rs.option.laser_power, mid)

        # 4) Visual preset back to 0 (clamped)
        r = self._rs_range(sensor, rs.option.visual_preset)
        if r is not None:
            self._rs_set(sensor, rs.option.visual_preset, float(np.clip(0.0, r.min, r.max)))

    def _apply_depth_tuning(self) -> None:
        """
        Apply manual/tuned RealSense depth settings for plane fitting.
        """
        if not self.use_tuned_depth_settings:
            return

        sensor = self._get_depth_sensor()
        if sensor is None:
            return

        if self._depth_prev_settings is None:
            self._depth_prev_settings = self._snapshot_depth_settings(sensor)

        self._rs_set(sensor, rs.option.emitter_enabled, 1)
        self._rs_set(sensor, rs.option.enable_auto_exposure, 0)

        r = self._rs_range(sensor, rs.option.laser_power)
        if r is not None:
            self._rs_set(sensor, rs.option.laser_power, float(np.clip(150, r.min, r.max)))

        r = self._rs_range(sensor, rs.option.exposure)
        if r is not None:
            self._rs_set(sensor, rs.option.exposure, float(np.clip(4000, r.min, r.max)))

        r = self._rs_range(sensor, rs.option.gain)
        if r is not None:
            self._rs_set(sensor, rs.option.gain, float(np.clip(16, r.min, r.max)))

    # ---------------- actions ----------------

    def start_clicked(self) -> None:
        if self.state.K_rgb is None:
            QMessageBox.information(self, "Plane Fitting", "Missing K_rgb. Run Camera Calibration first.")
            return

        try:
            if self.pipeline is None:
                self.start_realsense(
                    fps=self.FPS,
                    color_size=(1920, 1080),
                    depth_size=(1280, 720),
                    align_to="color",
                )
                self._apply_depth_defaults()
                self._apply_depth_tuning()

        except Exception as e:
            self.stop_realsense()
            QMessageBox.critical(self, "Camera", f"Could not open RealSense camera.\n\n{e}")
            self._mode = "idle"
            self.set_viewport_background(active=False)
            self._update_buttons()
            self._update_panels()
            self.update_view()
            return

        # reset outputs + snapshot state
        self.state.xray_points_xyz_c = None
        self.state.plane_confirmed = False
        self.state.plane_model_c = None
        self.state.plane_stats = None

        self._stats = []
        self._last_stats_rows = None

        self._snap_color = None
        self._snap_vis = None
        self._snap_depth = None
        self._ext_uv = None
        self._rect = None
        self._seed = 0

        self._mode = "live"
        self.set_viewport_background(active=True)
        self.start_timer(self.FPS)
        self._update_buttons()
        self._update_panels()
        self.setFocus()
        self.update_view()

        if callable(self.on_complete_changed):
            self.on_complete_changed()

    # ---------------- keyboard ----------------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self._mode == "live" and self._found:
                self._capture()
            event.accept()
            return

        super().keyPressEvent(event)

    # ---------------- workflow ----------------

    def _capture(self) -> None:
        if self._live_color is None or self._live_depth is None:
            return

        color = self._live_color.copy()
        depth = self._live_depth

        found, corners = cbd.detect_snapshot_full(
            color,
            pattern_size=self.pattern_size,
            det_width=self.det_width,
        )
        if (not found) or corners is None:
            return

        ex = cbd.get_extreme_corners_geometric(corners)
        ext_uv = np.array(
            [ex["top_left"], ex["top_right"], ex["bottom_left"]],
            dtype=np.float64,
        )

        preview = color.copy()
        cv2.drawChessboardCorners(preview, self.pattern_size, corners, True)
        preview = _draw_extremes(preview, ext_uv)

        # Freeze view while modal dialog is open
        self._mode = "frozen"
        self._snap_color = color
        self._snap_vis = preview
        self._snap_depth = depth
        self._ext_uv = ext_uv
        self.stop_timer()

        self._update_buttons()
        self._update_panels()
        self.update_view()

        ans = QMessageBox.question(
            self,
            "Corner Detection",
            "Satisfied with the detected corners?\n\nYes: run plane fitting\nNo: back to live video",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if ans != QMessageBox.Yes:
            # Back to live
            self._seed = 0
            self._mode = "live"
            self._snap_color = None
            self._snap_vis = None
            self._snap_depth = None
            self._ext_uv = None
            self._rect = None
            self._stats = []
            self._last_stats_rows = None

            self.set_viewport_background(active=True)
            self.start_timer(self.FPS)

            self._update_buttons()
            self._update_panels()
            self.setFocus()
            self.update_view()
            return

        h, w = color.shape[:2]
        self._rect = rpf.rect_from_pts(ext_uv, w, h, self.pad_px)
        self._seed = 0

        self._fit_and_grid()

        # Stay frozen (timer already stopped)
        self._update_buttons()
        self._update_panels()
        self.setFocus()
        self.update_view()

    def _fit_and_grid(self) -> None:
        assert self._snap_depth is not None and self._rect is not None and self._ext_uv is not None
        assert self.state.K_rgb is not None

        pts3d = rpf.sample_pts3d(
            self._snap_depth,
            self._rect,
            self.max_points,
            self.z_min,
            self.z_max,
            self._seed,
        )
        if pts3d.shape[0] < self.min_points_for_fit:
            QMessageBox.warning(self, "Plane Fitting", f"Not enough points ({pts3d.shape[0]}).")
            return

        plane, inliers = rpf.fit_plane_from_points(
            pts3d,
            distance_threshold=self.thresh_m,
            ransac_n=self.ransac_n,
            num_iterations=self.iters,
        )

        dev = rpf.deviations(pts3d, plane)
        dev_in = dev[inliers] if len(inliers) else dev

        mean = float(np.mean(dev_in))
        med = float(np.median(dev_in))
        p95 = float(np.percentile(dev_in, 95))

        # Stats (UI)
        self._stats = [
            f"Inliers: {int(len(inliers))}",
            f"Mean: {mean*1000.0:.3f} mm",
            f"Median: {med*1000.0:.3f} mm",
            f"P95: {p95*1000.0:.3f} mm",
        ]
        self._last_stats_rows = None  # force rebuild once after new stats

        # IMPORTANT FOR NEXT BUTTON (main.py step_complete):
        self.state.plane_model_c = np.asarray(plane, dtype=np.float64)
        self.state.plane_stats = {
            "n_inliers": int(len(inliers)),
            "mean_m": mean,
            "median_m": med,
            "p95_m": p95,
        }

        # Grid in camera frame
        corner_xyz = rpf.intersect_corners_with_plane(self._ext_uv, self.state.K_rgb, plane)
        marker_xyz = rpf.interpolate_marker_grid(corner_xyz, steps_per_edge=int(self.steps_per_edge))

        self.state.xray_points_xyz_c = marker_xyz
        self.state.plane_confirmed = True

        if callable(self.on_complete_changed):
            self.on_complete_changed()

        self._update_buttons()
        self._update_panels()
        self.update_view()

    def _update_buttons(self) -> None:
        # Start only when idle (you intentionally have no "stop")
        self.btn_start.setEnabled(self._mode == "idle")