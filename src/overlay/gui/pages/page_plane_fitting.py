from __future__ import annotations

import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QMessageBox, QGroupBox, QGridLayout, QSizePolicy
)

import pyrealsense2 as rs

from overlay.gui.state import SessionState
from overlay.tools import checkerboard_corner_detection as cbd
from overlay.tools import ransac_plane_fitting as rpf


# ============================================================
# Helpers
# ============================================================

def _bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    if img_bgr is None:
        return QPixmap()
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    img_bgr = np.ascontiguousarray(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = np.ascontiguousarray(img_rgb)

    h, w = img_rgb.shape[:2]
    qimg = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _draw_extremes(
    img_bgr: np.ndarray,
    extremes: dict[str, tuple[float, float]],
    color=(208, 224, 64),
    radius=10,
    thickness=-1,
) -> np.ndarray:
    out = img_bgr.copy()
    for name, (u, v) in extremes.items():
        cv2.circle(out, (int(round(u)), int(round(v))), radius, color, thickness)
        cv2.putText(
            out,
            name,
            (int(round(u)) + 8, int(round(v)) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _draw_axes_top_left(img_bgr: np.ndarray, origin_uv: tuple[int, int]) -> np.ndarray:
    out = img_bgr.copy()
    shaft = 70
    thickness = 2
    tip_length = 0.2

    u0, v0 = origin_uv
    x_end = (u0 + shaft, v0)
    y_end = (u0, v0 + shaft)
    z_end = (u0 + int(shaft * 0.7), v0 + int(shaft * 0.7))

    cv2.arrowedLine(out, (u0, v0), x_end, (0, 0, 255), thickness, cv2.LINE_AA, tipLength=tip_length)
    cv2.arrowedLine(out, (u0, v0), y_end, (0, 255, 0), thickness, cv2.LINE_AA, tipLength=tip_length)
    cv2.arrowedLine(out, (u0, v0), z_end, (255, 0, 0), thickness, cv2.LINE_AA, tipLength=tip_length)
    return out


def _rect_from_extremes(extremes: dict[str, tuple[float, float]], img_w: int, img_h: int, pad_px: int):
    pts = np.array(
        [extremes["top_left"], extremes["top_right"], extremes["bottom_left"]],
        dtype=np.float32
    )
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


def _sample_points_3d_in_rect(
    depth_frame_aligned,
    rect,
    max_points: int,
    z_min: float,
    z_max: float,
    seed: int,
) -> np.ndarray:
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


def _compute_deviations(points: np.ndarray, plane_model: np.ndarray) -> np.ndarray:
    normal = plane_model[:3]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return np.zeros(points.shape[0], dtype=np.float64)
    return np.abs(points @ normal + plane_model[3]) / norm


# ============================================================
# Page
# ============================================================

class PlaneFittingPage(QWidget):
    """
    Step — Plane Fitting

    Key behavior:
    - Before Start: show nothing (transparent), but reserve the same workspace height/width
    - Start: open camera and show live stream
    - Space: capture only when FOUND
    - No black bars: cover+crop
    - No "square" look: the display area is forced to 16:9 (960x540)
    """

    # Display box is fixed 16:9 to avoid "square" crops
    DISP_W = 960
    DISP_H = 540

    RIGHT_W = 260
    MARGIN = 20
    SPACING = 20

    def __init__(self, state: SessionState, on_complete_changed=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.on_complete_changed = on_complete_changed

        # ---------- params ----------
        self.pattern_size = (3, 3)
        self.det_width = 640
        self.rect_pad_px = 15

        self.max_points = 40000
        self.z_min, self.z_max = 0.15, 2.0

        self.ransac_thresh_m = 0.005
        self.ransac_n = 3
        self.ransac_iters = 1000
        self.min_points_for_fit = 800

        # ---------- realsense ----------
        self.pipeline = None
        self.align = None

        # ---------- runtime ----------
        self._mode = "idle"  # idle | live | corners | plane
        self._found_live = False
        self._live_color = None
        self._live_depth = None

        self._snap_color = None
        self._snap_depth = None
        self._corners = None
        self._extremes = None
        self._rect = None

        self._last_vis = None  # last BGR visual for redraw

        # ---------- timer ----------
        self._timer = QTimer(self)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._tick)

        # ======================================================
        # LEFT: fixed 16:9 display area (prevents "square" look)
        # ======================================================
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setFixedSize(self.DISP_W, self.DISP_H)
        self.image_label.setFocusPolicy(Qt.NoFocus)

        # before start: show nothing
        self.image_label.setStyleSheet("background-color: transparent; border-radius: 10px;")
        self.image_label.clear()

        # Put the label into a container so it can be aligned nicely
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.image_label, 0, Qt.AlignLeft | Qt.AlignTop)
        left_layout.addStretch(1)

        # ======================================================
        # RIGHT: keep your buttons logic (as before)
        # ======================================================
        self.info_label = QLabel("Press Start to begin.")
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info_label.setWordWrap(True)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_redo = QPushButton("Redo Plane Fitting")

        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_redo.clicked.connect(self.redo_plane_fit)

        # Important for SPACE: buttons should not steal focus
        self.btn_start.setFocusPolicy(Qt.NoFocus)
        self.btn_stop.setFocusPolicy(Qt.NoFocus)
        self.btn_redo.setFocusPolicy(Qt.NoFocus)

        for b in (self.btn_start, self.btn_stop, self.btn_redo):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        controls = QGroupBox("Controls")
        g = QGridLayout(controls)
        g.setContentsMargins(8, 8, 8, 8)
        g.setSpacing(8)
        g.addWidget(self.btn_start, 0, 0)
        g.addWidget(self.btn_stop, 0, 1)
        g.addWidget(self.btn_redo, 1, 0, 1, 2)
        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)

        stats_box = QGroupBox("Plane Fitting Stats")
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        stats_layout.addWidget(self.info_label)

        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        right_layout.setSpacing(12)
        right_layout.addWidget(controls)
        right_layout.addWidget(stats_box)
        right_layout.addStretch(1)

        self.right_panel = QWidget()
        self.right_panel.setLayout(right_layout)
        self.right_panel.setFixedWidth(self.RIGHT_W)
        self.right_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        # ======================================================
        # MAIN LAYOUT
        # ======================================================
        main = QHBoxLayout(self)
        main.setContentsMargins(self.MARGIN, self.MARGIN, self.MARGIN, self.MARGIN)
        main.setSpacing(self.SPACING)
        main.addWidget(left_container, 1)
        main.addWidget(self.right_panel, 0)

        # Page must receive key events (SPACE)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        self._set_ui_state_idle()

    # ======================================================
    # UI states
    # ======================================================

    def _set_ui_state_idle(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_redo.setEnabled(False)

    def _set_ui_state_live(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_redo.setEnabled(False)

    def _set_ui_state_plane(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_redo.setEnabled(True)

    # ======================================================
    # Rendering: cover+crop but anchored TOP-LEFT (x=0,y=0)
    # ======================================================

    def _show_image(self, img_bgr: np.ndarray) -> None:
        self._last_vis = img_bgr

        target_w = self.DISP_W
        target_h = self.DISP_H

        h, w = img_bgr.shape[:2]
        if h <= 0 or w <= 0:
            self.image_label.clear()
            return
        
        target_ratio = float(target_w) / float(target_h)
        src_ratio = float(w) / float(h)

        vis = img_bgr
        if src_ratio > target_ratio:
            crop_w = int(round(h * target_ratio))
            x0 = max(0, (w - crop_w) // 2)
            vis = img_bgr[:, x0:x0 + crop_w]
        elif src_ratio < target_ratio:
            crop_h = int(round(w / target_ratio))
            y0 = max(0, (h - crop_h) // 2)
            vis = img_bgr[y0:y0 + crop_h, :]
            
        vis = cv2.resize(vis, (target_w, target_h), interpolation=cv2.INTER_AREA)
        pm = _bgr_to_qpixmap(vis)
        if pm.isNull():
            self.image_label.clear()
            return

        self.image_label.setPixmap(pm)

    # ======================================================
    # Lifecycle (camera)
    # ======================================================

    def start(self):
        # show the dark background only AFTER Start
        self.image_label.setStyleSheet("background-color: #202020; border-radius: 10px;")

        if self.pipeline is None:
            try:
                self.pipeline, self.align = self._start_realsense_rgbd(fps=30)
            except Exception as e:
                self.image_label.setStyleSheet("background-color: transparent; border-radius: 10px;")
                self.image_label.clear()
                QMessageBox.critical(self, "Camera", f"Could not open RealSense camera.\n\n{e}")
                self.pipeline = None
                self.align = None
                self._mode = "idle"
                self._set_ui_state_idle()
                return

        self._mode = "live"
        self._found_live = False
        self._live_color = None
        self._live_depth = None
        self._last_vis = None

        # reset state outputs
        self.state.cb_found = False
        self.state.cb_corners_uv = None
        self.state.cb_extremes_uv = None
        self.state.cb_rect_uv = None

        self.state.pts3d_c = None
        self.state.plane_model_c = None
        self.state.plane_stats = None
        self.state.plane_redo_seed = 0

        self.info_label.setText("Live view running.\nPress SPACE only when checkerboard is FOUND.")
        self._set_ui_state_live()

        # Ensure page has focus for SPACE
        self.setFocus()
        self._timer.start()
        self._tick()

    def stop(self):
        self._timer.stop()

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        self.pipeline = None
        self.align = None

        self._mode = "idle"
        self._found_live = False
        self._live_color = None
        self._live_depth = None
        self._last_vis = None

        # reset outputs
        self.state.cb_found = False
        self.state.cb_corners_uv = None
        self.state.cb_extremes_uv = None
        self.state.cb_rect_uv = None

        self.state.pts3d_c = None
        self.state.plane_model_c = None
        self.state.plane_stats = None
        self.state.plane_redo_seed = 0

        self.info_label.setText("Press Start to begin.")
        self._set_ui_state_idle()

        # before start: show nothing
        self.image_label.setStyleSheet("background-color: transparent; border-radius: 10px;")
        self.image_label.clear()

        # regain focus
        self.setFocus()

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)

    # ======================================================
    # Keyboard (SPACE)
    # ======================================================

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.stop()
            event.accept()
            return

        if event.key() == Qt.Key_Space:
            # keep behavior like before: only capture in live + found
            if self._mode == "live" and self._found_live:
                self._capture_and_detect()
            event.accept()
            return

        super().keyPressEvent(event)

    # ======================================================
    # RealSense
    # ======================================================

    @staticmethod
    def _start_realsense_rgbd(fps: int = 30):
        pipeline = rs.pipeline()
        config = rs.config()

        # D435i color stream: 1920x1080 -> 16:9
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)

        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)

        # reduce latency if possible
        try:
            dev = profile.get_device()
            for sensor in dev.query_sensors():
                try:
                    sensor.set_option(rs.option.frames_queue_size, 1)
                except Exception:
                    pass
        except Exception:
            pass

        return pipeline, align

    # ======================================================
    # Live tick (non-blocking)
    # ======================================================

    def _tick(self):
        if self.pipeline is None or self.align is None or self._mode != "live":
            return

        frames = self.pipeline.poll_for_frames()
        if not frames:
            return

        frames = self.align.process(frames)
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            return

        color = np.asanyarray(cf.get_data())
        self._live_color = color
        self._live_depth = df

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        found_live, _ = cbd.detect_classic_downscaled(gray, self.pattern_size, det_width=self.det_width)
        self._found_live = bool(found_live)

        vis = color.copy()
        if self._found_live:
            cv2.putText(vis, "FOUND (press SPACE to capture)", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(vis, "NOT FOUND", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        self._show_image(vis)

    # ======================================================
    # Capture + plane fitting (kept from your flow)
    # ======================================================

    def _capture_and_detect(self):
        if self._live_color is None or self._live_depth is None:
            return

        snap_color = self._live_color.copy()
        snap_depth = self._live_depth

        found, corners = cbd.detect_snapshot_full(
            snap_color,
            pattern_size=self.pattern_size,
            det_width=self.det_width,
        )
        if not found or corners is None:
            return

        res = snap_color.copy()
        cv2.drawChessboardCorners(res, self.pattern_size, corners, True)

        extremes = cbd.get_extreme_corners_geometric(corners)
        res = _draw_extremes(res, extremes)

        self._mode = "corners"
        self._show_image(res)

        ans = QMessageBox.question(
            self,
            "Corner Detection",
            "Satisfied with the detected corners?\n\nYes: run plane fitting\nNo: back to live video",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if ans != QMessageBox.Yes:
            self._mode = "live"
            self.setFocus()
            return

        self._snap_color = snap_color
        self._snap_depth = snap_depth
        self._corners = corners
        self._extremes = extremes

        h, w = snap_color.shape[:2]
        self._rect = _rect_from_extremes(extremes, w, h, pad_px=self.rect_pad_px)

        self.state.cb_found = True
        self.state.cb_corners_uv = corners
        self.state.cb_extremes_uv = np.array(
            [extremes["top_left"], extremes["top_right"], extremes["bottom_left"]],
            dtype=np.float32,
        )
        self.state.cb_rect_uv = self._rect
        self.state.plane_redo_seed = 0

        self._run_plane_fit_once()
        self.setFocus()

    def _run_plane_fit_once(self) -> bool:
        if self._snap_color is None or self._snap_depth is None or self._rect is None or self._extremes is None:
            return False

        pts3d = _sample_points_3d_in_rect(
            depth_frame_aligned=self._snap_depth,
            rect=self._rect,
            max_points=self.max_points,
            z_min=self.z_min,
            z_max=self.z_max,
            seed=int(self.state.plane_redo_seed),
        )

        if pts3d.shape[0] < self.min_points_for_fit:
            QMessageBox.warning(
                self,
                "Plane Fitting",
                f"Not enough points for plane fit ({pts3d.shape[0]} < {self.min_points_for_fit}).\n"
                "Try again (move closer / improve depth / re-capture).",
            )
            self._mode = "live"
            self._set_ui_state_live()
            return False

        plane_model, inliers = rpf.fit_plane_from_points(
            pts3d,
            distance_threshold=self.ransac_thresh_m,
            ransac_n=self.ransac_n,
            num_iterations=self.ransac_iters,
        )

        deviations = _compute_deviations(pts3d, plane_model)
        inlier_devs = deviations[inliers] if len(inliers) else deviations

        mean = float(np.mean(inlier_devs)) if inlier_devs.size else float(np.mean(deviations))
        median = float(np.median(inlier_devs)) if inlier_devs.size else float(np.median(deviations))
        p95 = float(np.percentile(inlier_devs, 95)) if inlier_devs.size else float(np.percentile(deviations, 95))

        preview = self._snap_color.copy()
        preview = _draw_extremes(preview, self._extremes)
        tl = self._extremes["top_left"]
        preview = _draw_axes_top_left(preview, (int(round(tl[0])), int(round(tl[1]))))

        self._mode = "plane"
        self._show_image(preview)

        self.info_label.setText(
            "Plane fitting stats (inliers)\n"
            f"n_inliers: {int(len(inliers))}\n"
            f"mean:   {mean * 1000.0:.3f} mm\n"
            f"median: {median * 1000.0:.3f} mm\n"
            f"P95:    {p95 * 1000.0:.3f} mm\n"
        )

        self.state.pts3d_c = pts3d
        self.state.plane_model_c = plane_model
        self.state.plane_stats = {
            "n_inliers": float(len(inliers)),
            "mean_m": mean,
            "median_m": median,
            "p95_m": p95,
        }

        self._set_ui_state_plane()

        if callable(self.on_complete_changed):
            self.on_complete_changed()

        return True

    # ======================================================
    # Button handler (kept)
    # ======================================================

    def redo_plane_fit(self):
        if self._mode != "plane":
            return
        if self._snap_color is None or self._snap_depth is None or self._rect is None:
            return

        self.state.plane_redo_seed += 1
        self._run_plane_fit_once()
        self.setFocus()
