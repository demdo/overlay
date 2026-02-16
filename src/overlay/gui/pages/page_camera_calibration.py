from __future__ import annotations

import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QMessageBox, QGroupBox, QGridLayout, QSizePolicy
)

import pyrealsense2 as rs

from overlay.gui.state import SessionState
from overlay.calib import calibration_camera as camcal



# ============================================================
# Helpers (same style as page_plane_fitting.py)
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


def _draw_text_box(
    img_bgr: np.ndarray,
    lines: list[str],
    org=(30, 55),
    color=(255, 255, 255)
) -> np.ndarray:
    """
    Draw multi-line text with black outline (shadow) and colored foreground.
    color: BGR, e.g. green=(0,255,0), red=(0,0,255), white=(255,255,255)
    """
    out = img_bgr.copy()
    x, y = org
    for i, t in enumerate(lines):
        yy = y + i * 35
        cv2.putText(out, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(out, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return out


# ============================================================
# Page
# ============================================================

class CameraCalibrationPage(QWidget):
    """
    Camera Calibration (ChArUco) page.

    Implements:
    - Live view (RealSense starts ONLY after Start)
    - FOUND/NOT FOUND text: green/red
    - SPACE captures only if FOUND
    - Collect 10 images, show detected markers x/31
    - After 10 captures: enable Accuracy Test
    - Accuracy Test:
        - SNAPSHOT one frame
        - STOP live video (static image)
        - Draw measured (cyan) vs projected (red) using intrinsics from 10 images
        - Show avg reprojection error on the right
    - Redo repeats calibration
    """

    # Display fixed like PlaneFittingPage
    DISP_W = 960
    DISP_H = 540

    RIGHT_W = 260
    MARGIN = 20
    SPACING = 20

    # Board definition (as printed in your board footer)
    SQUARES_X = 9
    SQUARES_Y = 7
    SQUARE_LEN_M = 25.40e-3
    MARKER_LEN_M = 17.78e-3
    DICT_ID = cv2.aruco.DICT_5X5_50
    MAX_ARUCO = 31

    # Workflow
    N_VIEWS = 10
    MIN_CHARUCO_CAPTURE = 12
    MIN_CHARUCO_LIVE_FOUND = 8

    def __init__(self, state: SessionState, on_complete_changed=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.on_complete_changed = on_complete_changed

        # ---------- aruco / board ----------
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.DICT_ID)
        self.detector_params = cv2.aruco.DetectorParameters()

        self.board = cv2.aruco.CharucoBoard(
            (self.SQUARES_X, self.SQUARES_Y),
            self.SQUARE_LEN_M,
            self.MARKER_LEN_M,
            self.aruco_dict
        )

        # ---------- realsense ----------
        self.pipeline = None

        # ---------- runtime ----------
        self._mode = "idle"  # idle | live | test
        self._found_live = False
        self._live_color = None

        self._views: list[np.ndarray] = []
        self._last_det = None

        # Calibration results
        self._K = None
        self._dist = None
        self._rms = None

        # Accuracy test snapshot + visualization
        self._test_frame = None
        self._test_vis = None
        self._avg_reproj_test = None

        # ---------- timer ----------
        self._timer = QTimer(self)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._tick)

        # ======================================================
        # LEFT: fixed 16:9 image
        # ======================================================
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setFixedSize(self.DISP_W, self.DISP_H)
        self.image_label.setFocusPolicy(Qt.NoFocus)

        # Before start: transparent + no pixmap (like PlaneFittingPage behavior)
        self.image_label.setStyleSheet("background-color: transparent; border-radius: 10px;")
        self.image_label.clear()

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.image_label, 0, Qt.AlignLeft | Qt.AlignTop)
        left_layout.addStretch(1)

        # ======================================================
        # RIGHT: controls + status
        # ======================================================
        self.info_label = QLabel(
            "Press Start to open the camera.\n"
            "Press SPACE to capture (only when FOUND)."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.progress_label = QLabel("Captured: 0/10")
        self.marker_count_label = QLabel("Detected markers: 0/31")
        self.reproj_label = QLabel("Avg reprojection error: —")

        for lab in (self.progress_label, self.marker_count_label, self.reproj_label):
            lab.setWordWrap(True)
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_redo = QPushButton("Redo Calibration")
        self.btn_test = QPushButton("Accuracy test")

        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_redo.clicked.connect(self.redo)
        self.btn_test.clicked.connect(self.run_accuracy_test)

        # Do not steal focus -> SPACE should work
        for b in (self.btn_start, self.btn_stop, self.btn_redo, self.btn_test):
            b.setFocusPolicy(Qt.NoFocus)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        controls = QGroupBox("Controls")
        g = QGridLayout(controls)
        g.setContentsMargins(8, 8, 8, 8)
        g.setSpacing(8)
        g.addWidget(self.btn_start, 0, 0)
        g.addWidget(self.btn_stop, 0, 1)
        g.addWidget(self.btn_redo, 1, 0, 1, 2)
        g.addWidget(self.btn_test, 2, 0, 1, 2)

        status_box = QGroupBox("Status")
        v = QVBoxLayout(status_box)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)
        v.addWidget(self.info_label)
        v.addWidget(self.progress_label)
        v.addWidget(self.marker_count_label)
        v.addWidget(self.reproj_label)

        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        right_layout.setSpacing(12)
        right_layout.addWidget(controls)
        right_layout.addWidget(status_box)
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

        # Page must receive SPACE
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        self._set_ui_idle()
        self.refresh()

    # ======================================================
    # Public hook from MainWindow.update_ui
    # ======================================================

    def refresh(self):
        # If state already contains intrinsics, adopt them (but still allow redo)
        if getattr(self.state, "K_rgb", None) is not None and self._K is None:
            self._K = np.asarray(self.state.K_rgb, dtype=np.float64)
        if getattr(self.state, "dist_rgb", None) is not None and self._dist is None:
            self._dist = np.asarray(self.state.dist_rgb, dtype=np.float64)

        self._update_labels()

    # ======================================================
    # UI state helpers
    # ======================================================

    def _set_ui_idle(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_redo.setEnabled(True)
        self.btn_test.setEnabled(False)

    def _set_ui_live(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_redo.setEnabled(True)
        self.btn_test.setEnabled(self._K is not None and len(self._views) >= self.N_VIEWS)

    def _set_ui_test(self):
        # static snapshot shown, no live updates
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_redo.setEnabled(True)
        self.btn_test.setEnabled(False)

    # ======================================================
    # Rendering: cover+crop like PlaneFittingPage
    # ======================================================

    def _show_image(self, img_bgr: np.ndarray) -> None:
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
    # Lifecycle
    # ======================================================

    def start(self):
        # show dark background only after Start
        self.image_label.setStyleSheet("background-color: #202020; border-radius: 10px;")

        # If we were in test mode, drop snapshot and go back to live
        if self._mode == "test":
            self._mode = "live"
            self._test_frame = None
            self._test_vis = None

        if self.pipeline is None:
            try:
                self.pipeline = self._start_realsense_color(fps=30)
            except Exception as e:
                self.image_label.setStyleSheet("background-color: transparent; border-radius: 10px;")
                self.image_label.clear()
                QMessageBox.critical(self, "Camera", f"Could not open RealSense camera.\n\n{e}")
                self.pipeline = None
                self._mode = "idle"
                self._set_ui_idle()
                return

        self._mode = "live"
        self._found_live = False
        self._live_color = None

        self.info_label.setText("Live view running.\nPress SPACE only when FOUND.")
        self._set_ui_live()

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
        self._mode = "idle"
        self._found_live = False
        self._live_color = None

        self.info_label.setText(
            "Press Start to open the camera.\n"
            "Press SPACE to capture (only when FOUND)."
        )
        self._set_ui_idle()

        # before start: show nothing
        self.image_label.setStyleSheet("background-color: transparent; border-radius: 10px;")
        self.image_label.clear()

        self.setFocus()

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)

    def redo(self):
        # Clear only this step's outputs
        self._views.clear()
        self._K = None
        self._dist = None
        self._rms = None

        self._test_frame = None
        self._test_vis = None
        self._avg_reproj_test = None

        # clear session outputs (wizard gating)
        self.state.K_rgb = None
        self.state.dist_rgb = None  # make sure state has this field

        self._update_labels()
        self.btn_test.setEnabled(False)

        if callable(self.on_complete_changed):
            self.on_complete_changed()

        if self._mode == "live":
            self._set_ui_live()
        elif self._mode == "test":
            self._set_ui_test()
        else:
            self._set_ui_idle()

        self.setFocus()

    # ======================================================
    # Keyboard (SPACE capture)
    # ======================================================

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.stop()
            event.accept()
            return

        if event.key() == Qt.Key_Space:
            if self._mode == "live" and self._found_live:
                self._capture_view()
            event.accept()
            return

        super().keyPressEvent(event)

    # ======================================================
    # RealSense
    # ======================================================

    @staticmethod
    def _start_realsense_color(fps: int = 30):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, fps)
        profile = pipeline.start(config)

        # reduce latency / queue
        try:
            dev = profile.get_device()
            for sensor in dev.query_sensors():
                try:
                    sensor.set_option(rs.option.frames_queue_size, 1)
                except Exception:
                    pass
        except Exception:
            pass

        return pipeline

    # ======================================================
    # Live tick
    # ======================================================

    def _tick(self):
        if self.pipeline is None or self._mode != "live":
            return

        frames = self.pipeline.poll_for_frames()
        if not frames:
            return

        cf = frames.get_color_frame()
        if not cf:
            return

        color = np.asanyarray(cf.get_data())
        self._live_color = color

        vis, found, det = self._make_live_visual(color)
        self._found_live = bool(found)
        self._last_det = det

        self.marker_count_label.setText(f"Detected markers: {det.num_aruco}/{self.MAX_ARUCO}")
        self._show_image(vis)

    def _make_live_visual(self, color_bgr: np.ndarray):
        # Use your module for detection
        det = camcal.detect_charuco(
            color_bgr,
            board=self.board,
            aruco_dict=self.aruco_dict,
            detector_params=self.detector_params
        )

        vis = color_bgr.copy()

        # Draw markers (green rectangles)
        if det.aruco_ids is not None and len(det.aruco_ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, det.aruco_corners, det.aruco_ids)

        # Draw charuco corners (cyan)
        if det.charuco_corners is not None and det.charuco_ids is not None and det.num_charuco > 0:
            try:
                cv2.aruco.drawDetectedCornersCharuco(vis, det.charuco_corners, det.charuco_ids, (255, 255, 0))
            except Exception:
                pts = det.charuco_corners.reshape(-1, 2)
                for (u, v) in pts:
                    cv2.circle(vis, (int(round(u)), int(round(v))), 4, (255, 255, 0), 2)

        found = det.num_charuco >= self.MIN_CHARUCO_LIVE_FOUND

        if found:
            vis = _draw_text_box(
                vis,
                [
                    "FOUND (press SPACE)",
                    f"Captured: {len(self._views)}/{self.N_VIEWS}",
                    f"Markers: {det.num_aruco}/{self.MAX_ARUCO}",
                ],
                color=(0, 255, 0)  # GREEN
            )
        else:
            vis = _draw_text_box(
                vis,
                [
                    "NOT FOUND",
                    f"Captured: {len(self._views)}/{self.N_VIEWS}",
                    f"Markers: {det.num_aruco}/{self.MAX_ARUCO}",
                ],
                color=(0, 0, 255)  # RED
            )

        return vis, found, det

    # ======================================================
    # Capture + calibration (uses YOUR module)
    # ======================================================

    def _capture_view(self):
        if self._live_color is None:
            return
        if len(self._views) >= self.N_VIEWS:
            return

        self._views.append(self._live_color.copy())
        self._update_labels()

        # Once 10 views -> calibrate intrinsics
        if len(self._views) >= self.N_VIEWS:
            ok = self._run_calibration_from_views()
            if ok:
                self.btn_test.setEnabled(True)
                self.info_label.setText(
                    "Captured 10/10.\n"
                    "Press 'Accuracy test' to validate.\n"
                    "Redo if not satisfied."
                )
                if callable(self.on_complete_changed):
                    self.on_complete_changed()
            else:
                QMessageBox.warning(self, "Calibration", "Calibration failed. Try again (Redo).")

        self.setFocus()

    def _run_calibration_from_views(self) -> bool:
        try:
            K, dist, rms, stats = camcal.calibrate_charuco_intrinsics(
                calib_images=self._views,
                board=self.board,
                aruco_dict=self.aruco_dict,
                detector_params=self.detector_params,
                min_charuco_corners=self.MIN_CHARUCO_CAPTURE
            )
        except Exception as e:
            QMessageBox.warning(self, "Calibration", f"Calibration error:\n\n{e}")
            return False

        self._K = np.asarray(K, dtype=np.float64)
        self._dist = np.asarray(dist, dtype=np.float64)
        self._rms = float(rms)

        # write to session state (wizard gating)
        self.state.K_rgb = self._K.copy()
        self.state.dist_rgb = self._dist.copy()

        # average reprojection error shown is for the test snapshot, so keep it empty here
        self._avg_reproj_test = None
        self.reproj_label.setText("Avg reprojection error: —")

        return True

    # ======================================================
    # Accuracy test (STATIC snapshot, uses YOUR module)
    # ======================================================

    def run_accuracy_test(self):
        if self._K is None or self._dist is None:
            QMessageBox.information(self, "Accuracy test", "No intrinsics yet. Capture 10 images first.")
            return
        if self._live_color is None:
            QMessageBox.information(self, "Accuracy test", "No live frame available. Press Start.")
            return

        # 1) SNAPSHOT frame -> becomes static test image
        self._test_frame = self._live_color.copy()

        # 2) STOP live updates (freeze)
        self._timer.stop()
        self._mode = "test"
        self._found_live = False

        # 3) Compute avg reprojection error on THIS snapshot
        try:
            mean_px, per_view, stats = camcal.reprojection_error_charuco(
                test_images=[self._test_frame],
                board=self.board,
                aruco_dict=self.aruco_dict,
                K=self._K,
                dist=self._dist,
                detector_params=self.detector_params,
                min_charuco_corners=self.MIN_CHARUCO_LIVE_FOUND
            )
            self._avg_reproj_test = float(mean_px)
        except Exception as e:
            QMessageBox.warning(self, "Accuracy test", f"Reprojection test failed:\n\n{e}")
            self._avg_reproj_test = None

        # 4) Create visualization on the SAME snapshot
        vis = self._visualize_measured_vs_projected(self._test_frame)

        # Title + avg reproj error on the image (white like your screenshot)
        if self._avg_reproj_test is not None:
            lines = [
                "Measured corners (cyan) vs Projected (red)",
                f"Avg reprojection error: {self._avg_reproj_test:.4f} px",
            ]
        else:
            lines = [
                "Measured corners (cyan) vs Projected (red)",
                "Avg reprojection error: —",
            ]

        vis = _draw_text_box(vis, lines, color=(255, 255, 255))
        self._test_vis = vis

        # Show static image
        self._show_image(self._test_vis)

        # Update right panel
        if self._avg_reproj_test is not None:
            self.reproj_label.setText(f"Avg reprojection error: {self._avg_reproj_test:.4f} px")
        else:
            self.reproj_label.setText("Avg reprojection error: —")

        self.info_label.setText(
            "Accuracy test snapshot shown (static).\n"
            "Press Redo to repeat calibration or Next if satisfied."
        )

        self._set_ui_test()
        self.setFocus()

    def _visualize_measured_vs_projected(self, img_bgr: np.ndarray) -> np.ndarray:
        vis = img_bgr.copy()

        # pose estimation using your module
        try:
            rvec, tvec, det, ok = camcal.estimate_charuco_pose(
                image=img_bgr,
                board=self.board,
                aruco_dict=self.aruco_dict,
                K=self._K,
                dist=self._dist,
                detector_params=self.detector_params,
                min_charuco_corners=self.MIN_CHARUCO_LIVE_FOUND
            )
        except Exception:
            ok = False
            det = None
            rvec = None
            tvec = None

        # Draw detected markers for context
        if det is not None and det.aruco_ids is not None and len(det.aruco_ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, det.aruco_corners, det.aruco_ids)

        if (not ok) or det is None or det.charuco_corners is None or det.charuco_ids is None:
            vis = _draw_text_box(vis, ["Accuracy test failed: pose not found"], color=(0, 0, 255))
            return vis

        # measured corners (cyan circles)
        meas = det.charuco_corners.reshape(-1, 2)
        for (u, v) in meas:
            cv2.circle(vis, (int(round(u)), int(round(v))), 5, (255, 255, 0), 2)

        # projected (red crosses)
        # NOTE: uses your module's mapping helper
        obj_pts = camcal._charuco_object_points(self.board, det.charuco_ids)
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, self._K, self._dist)
        proj = proj.reshape(-1, 2)

        for (u, v) in proj:
            cv2.drawMarker(
                vis,
                (int(round(u)), int(round(v))),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
                line_type=cv2.LINE_AA
            )

        return vis

    # ======================================================
    # Labels
    # ======================================================

    def _update_labels(self):
        self.progress_label.setText(f"Captured: {len(self._views)}/{self.N_VIEWS}")

        if self._last_det is None:
            self.marker_count_label.setText(f"Detected markers: 0/{self.MAX_ARUCO}")
        else:
            self.marker_count_label.setText(f"Detected markers: {self._last_det.num_aruco}/{self.MAX_ARUCO}")

        if self._avg_reproj_test is None:
            self.reproj_label.setText("Avg reprojection error: —")
        else:
            self.reproj_label.setText(f"Avg reprojection error: {self._avg_reproj_test:.4f} px")
