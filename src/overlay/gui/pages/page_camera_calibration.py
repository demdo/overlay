# overlay/gui/pages/page_camera_calibration.py

from __future__ import annotations
from typing import Optional

import numpy as np
import cv2
import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QMessageBox, QWidget, QSizePolicy

from overlay.gui.widgets.widget_flow_layout import FlowLayout
from overlay.gui.state import SessionState
from overlay.gui.pages.templates.templ_live_image import LiveImagePage
from overlay.calib import calib_camera as camcal


# --- DEBUG / EXPORT ---
SAVE_CAMERA_CALIBRATION_TEST = False


def _draw_text_box(
    img_bgr: np.ndarray,
    lines: list[str],
    org=(30, 55),
    color=(255, 255, 255),
    line_gap=35,
    font_scale=1.0,
) -> np.ndarray:
    """
    Multi-line text with black outline + colored foreground.
    color: BGR, e.g. green=(0,255,0), red=(0,0,255), white=(255,255,255)
    """
    out = img_bgr.copy()
    x, y = org
    for i, t in enumerate(lines):
        yy = y + i * line_gap
        cv2.putText(out, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(out, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return out


class CameraCalibrationPage(LiveImagePage):
    # Board definition
    SQUARES_X = 9
    SQUARES_Y = 7
    SQUARE_LEN_M = 25.40e-3
    MARKER_LEN_M = 17.78e-3
    DICT_ID = cv2.aruco.DICT_5X5_50
    MAX_ARUCO = 31
    MAX_CHARUCO_CORNERS = (SQUARES_X - 1) * (SQUARES_Y - 1)

    # Workflow
    N_VIEWS = 10
    MIN_CHARUCO_LIVE_FOUND = 8
    MIN_CHARUCO_CAPTURE = 12

    def __init__(self, state: SessionState, on_complete_changed=None, parent=None):
        self.state = state
        self.on_complete_changed = on_complete_changed

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.DICT_ID)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.board = cv2.aruco.CharucoBoard(
            (self.SQUARES_X, self.SQUARES_Y),
            self.SQUARE_LEN_M,
            self.MARKER_LEN_M,
            self.aruco_dict,
        )

        self._mode = "idle"  # idle | live | test
        self._views: list[np.ndarray] = []

        self._live_frame: Optional[np.ndarray] = None
        self._test_frame: Optional[np.ndarray] = None

        self._det: Optional[camcal.CharucoDetection] = None
        self._found: bool = False

        # calibration results
        self._K: Optional[np.ndarray] = None
        self._dist: Optional[np.ndarray] = None
        self._rms: Optional[float] = None

        # accuracy test result
        self._avg_reproj_px: Optional[float] = None

        # Cache stats rows (PlaneFitting style)
        self._last_stats_rows: list[tuple[str, str]] | None = None

        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        self._build_controls()

        # FIXED INSTRUCTIONS (do NOT change dynamically)
        self.instructions_label.setText(
            "1) Press Start to open the camera\n"
            "2) Move the ChArUco board until FOUND\n"
            "3) Press SPACE to capture (only when FOUND)\n"
            "4) After 10 views: press Accuracy test"
        )

        self.set_viewport_background(active=False)

        # adopt intrinsics from state if already present (so button gating works)
        if getattr(self.state, "K_rgb", None) is not None:
            self._K = np.asarray(self.state.K_rgb, dtype=np.float64)
        if getattr(self.state, "dist_rgb", None) is not None:
            self._dist = np.asarray(self.state.dist_rgb, dtype=np.float64)

        self._update_buttons()
        self._update_panels()
        self.update_view()

    # ---------------- Controls ----------------

    def _build_controls(self) -> None:
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_redo = QPushButton("Redo")
        self.btn_test = QPushButton("Accuracy test")

        self.btn_start.clicked.connect(self.start_clicked)
        self.btn_stop.clicked.connect(self.stop_clicked)
        self.btn_redo.clicked.connect(self.redo_clicked)
        self.btn_test.clicked.connect(self.test_clicked)

        for b in (self.btn_start, self.btn_stop, self.btn_redo, self.btn_test):
            b.setFocusPolicy(Qt.NoFocus)
            b.setMinimumHeight(44)
            b.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            b.setMinimumWidth(0)

        # FlowLayout wrapper
        wrap = QWidget()
        wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        flow = FlowLayout(wrap, spacing=10)
        wrap.setLayout(flow)

        flow.addWidget(self.btn_start)
        flow.addWidget(self.btn_stop)
        flow.addWidget(self.btn_redo)
        flow.addWidget(self.btn_test)

        self.controls_content.addWidget(wrap)

    # ---------------- Stats panel (PlaneFitting style) ----------------

    def _update_panels(self) -> None:
        rows = self.stats_rows()
        if rows != self._last_stats_rows:
            self.set_stats_rows(rows)
            self._last_stats_rows = list(rows)

    def stats_rows(self) -> list[tuple[str, str]]:
        """
        Final spec:

        Before Accuracy test (i.e. not in test mode):
            ArUco markers: -, ChArUco corners: -, views: -/10

        After Accuracy test (test mode):
            ChArUco corners: N
            Avg. reprojection error: x.xxx px
        """
        if self._mode == "test":
            # Robust: re-detect on the frozen snapshot
            N = 0
            if self._test_frame is not None:
                det = camcal.detect_charuco(
                    self._test_frame,
                    board=self.board,
                    aruco_dict=self.aruco_dict,
                    detector_params=self.detector_params,
                )
                N = int(det.num_charuco)

            avg_str = "—" if self._avg_reproj_px is None else f"{self._avg_reproj_px:.3f} px"
            return [
                ("ChArUco corners", f"{N}/{self.MAX_CHARUCO_CORNERS}"),
                ("Avg. reprojection error", avg_str),
            ]

        # idle/live: show placeholders / live counts
        if self._det is None:
            aruco = "-"
            charuco = "-"
        else:
            aruco = str(int(self._det.num_aruco))
            charuco = str(int(self._det.num_charuco))

        return [
            ("ArUco markers", f"{aruco}/{self.MAX_ARUCO}" if aruco != "-" else "-"),
            ("ChArUco corners", f"{charuco}/{self.MAX_CHARUCO_CORNERS}" if charuco != "-" else "-"),
            ("Views", f"{len(self._views)}/{self.N_VIEWS}" if self._mode != "idle" else "-"),
        ]

    # ---------------- Template hooks ----------------

    def get_frame(self) -> Optional[np.ndarray]:
        if self._mode == "idle":
            return None

        if self._mode == "test":
            # stats should still reflect test-mode values
            self._update_panels()
            return self._test_frame

        # live
        if self.pipeline is None:
            self._update_panels()
            return self._live_frame

        frames = self.pipeline.poll_for_frames()
        if not frames:
            self._update_panels()
            return self._live_frame

        cf = frames.get_color_frame()
        if not cf:
            self._update_panels()
            return self._live_frame

        self._live_frame = self.color_frame_to_bgr(cf)

        self._det = camcal.detect_charuco(
            self._live_frame,
            board=self.board,
            aruco_dict=self.aruco_dict,
            detector_params=self.detector_params,
        )
        self._found = bool(self._det.num_charuco >= self.MIN_CHARUCO_LIVE_FOUND)

        self._update_panels()
        return self._live_frame

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        vis = frame_bgr.copy()

        # -------- TEST MODE: measured (cyan) vs projected (red) ----------
        if self._mode == "test":
            K = self._K if self._K is not None else getattr(self.state, "K_rgb", None)
            dist = self._dist if self._dist is not None else getattr(self.state, "dist_rgb", None)
            if K is None or dist is None:
                return vis

            rvec, tvec, det2, ok = camcal.estimate_charuco_pose(
                image=vis,
                board=self.board,
                aruco_dict=self.aruco_dict,
                K=K,
                dist=dist,
                detector_params=self.detector_params,
                min_charuco_corners=self.MIN_CHARUCO_LIVE_FOUND,
            )
            
            if (not ok) or det2.charuco_corners is None or det2.charuco_ids is None:
                vis = _draw_text_box(vis, ["Accuracy test failed: pose not found"], color=(0, 0, 255))
                return vis
            
            # --- SAVE ONCE ---
            if ok and det2.charuco_corners is not None and det2.charuco_ids is not None:
                if SAVE_CAMERA_CALIBRATION_TEST:
                    self._save_test_result(det2, rvec, tvec)

            if det2.aruco_ids is not None and len(det2.aruco_ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, det2.aruco_corners, det2.aruco_ids)

            # measured points (cyan circles)
            meas = det2.charuco_corners.reshape(-1, 2)
            for (u, v) in meas:
                cv2.circle(vis, (int(round(u)), int(round(v))), 5, (255, 255, 0), 2, cv2.LINE_AA)

            # projected points (red crosses)
            if hasattr(camcal, "_charuco_object_points"):
                obj_pts = camcal._charuco_object_points(self.board, det2.charuco_ids)
                proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
                proj = proj.reshape(-1, 2)
                for (u, v) in proj:
                    cv2.drawMarker(
                        vis,
                        (int(round(u)), int(round(v))),
                        (0, 0, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=14,
                        thickness=2,
                        line_type=cv2.LINE_AA,
                    )

            return vis

        # -------- LIVE MODE: show markers + charuco corners + FOUND/NOT FOUND ----------
        det = self._det
        if det is not None:
            if det.aruco_ids is not None and len(det.aruco_ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, det.aruco_corners, det.aruco_ids)
            if det.charuco_corners is not None and det.charuco_ids is not None and det.num_charuco > 0:
                try:
                    cv2.aruco.drawDetectedCornersCharuco(vis, det.charuco_corners, det.charuco_ids, (255, 255, 0))
                except Exception:
                    pts = det.charuco_corners.reshape(-1, 2)
                    for (u, v) in pts:
                        cv2.circle(vis, (int(round(u)), int(round(v))), 4, (255, 255, 0), 2)

        if self._found:
            vis = _draw_text_box(vis, ["FOUND (press SPACE)"], color=(0, 255, 0))
        else:
            vis = _draw_text_box(vis, ["NOT FOUND"], color=(0, 0, 255))

        return vis

    # ---------------- Actions ----------------

    def start_clicked(self) -> None:
        # if we were in test mode, going back to live
        self._test_frame = None
        self._avg_reproj_px = None
        self._mode = "idle"

        try:
            if self.pipeline is None:
                self.start_realsense(fps=self.FPS, color_size=(1920, 1080), depth_size=None, align_to=None)
            self.start_timer()
            self._mode = "live"
            self.set_viewport_background(active=True)
        except Exception as e:
            QMessageBox.critical(self, "Camera Error", f"Could not start camera.\n\n{e}")
            self._mode = "idle"
            self.stop_timer()
            self.stop_realsense()
            self.set_viewport_background(active=False)

        self._update_buttons()
        self._update_panels()
        self.setFocus()
        self.update_view()

    def stop_clicked(self) -> None:
        self.stop_timer()
        self.stop_realsense()
        self._mode = "idle"
        self._det = None
        self._found = False
        self._live_frame = None
        self.set_viewport_background(active=False)

        self._update_buttons()
        self._update_panels()
        self.setFocus()
        self.update_view()

    def redo_clicked(self) -> None:
        self._views.clear()
        self._avg_reproj_px = None
        self._test_frame = None
        self._mode = "idle"

        self._K = None
        self._dist = None
        self._rms = None

        # clear session outputs
        self.state.K_rgb = None
        self.state.dist_rgb = None

        self._det = None
        self._found = False
        self._live_frame = None

        self._last_stats_rows = None

        if callable(self.on_complete_changed):
            self.on_complete_changed()

        self._update_buttons()
        self._update_panels()
        self.setFocus()
        self.update_view()

    def test_clicked(self) -> None:
        global SAVE_CAMERA_CALIBRATION_TEST
        SAVE_CAMERA_CALIBRATION_TEST = True
        
        if self._K is None or self._dist is None:
            QMessageBox.information(self, "Accuracy test", "No intrinsics yet. Capture 10 images first.")
            return
        if len(self._views) < self.N_VIEWS:
            QMessageBox.information(self, "Accuracy test", f"Capture {self.N_VIEWS} images first.")
            return
        if self._live_frame is None:
            QMessageBox.information(self, "Accuracy test", "No live frame available yet.")
            return

        # 1) snapshot one frame
        self._mode = "test"
        self._test_frame = self._live_frame.copy()

        # 2) stop live updates (freeze)
        self.stop_timer()
        self.stop_realsense()
        self.set_viewport_background(active=False)

        # 3) compute reprojection error on this snapshot
        try:
            mean_px, _per_view, _stats = camcal.reprojection_error_charuco(
                test_images=[self._test_frame],
                board=self.board,
                aruco_dict=self.aruco_dict,
                K=self._K,
                dist=self._dist,
                detector_params=self.detector_params,
                min_charuco_corners=self.MIN_CHARUCO_LIVE_FOUND,
            )
            self._avg_reproj_px = float(mean_px)
        except Exception as e:
            QMessageBox.warning(self, "Accuracy test", f"Reprojection test failed:\n\n{e}")
            self._avg_reproj_px = None

        self._update_buttons()
        self._update_panels()
        self.setFocus()
        self.update_view()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self.capture_if_possible()
            event.accept()
            return
        super().keyPressEvent(event)

    def capture_if_possible(self) -> None:
        if self._mode != "live":
            return
        if not self._found:
            return
        if self._det is None or self._live_frame is None:
            return
        if int(self._det.num_charuco) < self.MIN_CHARUCO_CAPTURE:
            return
        if len(self._views) >= self.N_VIEWS:
            return

        self._views.append(self._live_frame.copy())

        # once we have 10 views -> calibrate intrinsics
        if len(self._views) >= self.N_VIEWS:
            ok = self._run_calibration_from_views()
            if ok:
                if callable(self.on_complete_changed):
                    self.on_complete_changed()
            else:
                QMessageBox.warning(self, "Calibration", "Calibration failed. Try again (Redo).")

        self._update_buttons()
        self._update_panels()
        self.update_view()

    def _run_calibration_from_views(self) -> bool:
        try:
            K, dist, rms, _stats = camcal.calibrate_charuco_intrinsics(
                calib_images=self._views,
                board=self.board,
                aruco_dict=self.aruco_dict,
                detector_params=self.detector_params,
                min_charuco_corners=self.MIN_CHARUCO_CAPTURE,
            )
        except Exception as e:
            QMessageBox.warning(self, "Calibration", f"Calibration error:\n\n{e}")
            return False

        self._K = np.asarray(K, dtype=np.float64)
        self._dist = np.asarray(dist, dtype=np.float64)
        self._rms = float(rms)

        self.state.K_rgb = self._K.copy()
        self.state.dist_rgb = self._dist.copy()

        self._avg_reproj_px = None
        self._last_stats_rows = None
        return True
    
    def _save_test_result(self, det2, rvec, tvec):
        global SAVE_CAMERA_CALIBRATION_TEST
    
        if not SAVE_CAMERA_CALIBRATION_TEST:
            return
    
        try:
            K = self._K
            dist = self._dist
    
            uv_meas = det2.charuco_corners.reshape(-1, 2)
            obj_pts = camcal._charuco_object_points(self.board, det2.charuco_ids)
    
            uv_proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            uv_proj = uv_proj.reshape(-1, 2)
    
            residuals_uv = uv_proj - uv_meas
            residual_norm = np.linalg.norm(residuals_uv, axis=1)
    
            mean_px = float(np.mean(residual_norm))
            N = int(len(residual_norm))
    
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"camera_calibration_test_{ts}.npz"
    
            np.savez(
                out_path,
                test_mean_reproj_px=mean_px,
                test_num_charuco=N,
                test_detected_uv=uv_meas,
                test_projected_uv=uv_proj,
                test_residuals_uv=residuals_uv,
                test_residual_norm_px=residual_norm,
                K_rgb=K,
                dist_rgb=dist,
            )
    
            SAVE_CAMERA_CALIBRATION_TEST = False
            print(f"[CameraCalibration] Saved test result -> {out_path}")
    
        except Exception as e:
            print(f"[CameraCalibration] Save failed: {e}")

    # ---------------- UI updates ----------------

    def _update_buttons(self) -> None:
        self.btn_start.setEnabled(self._mode in ("idle", "test"))
        self.btn_stop.setEnabled(self._mode == "live")
        self.btn_redo.setEnabled(True)

        have_intr = (self._K is not None and self._dist is not None)
        self.btn_test.setEnabled(self._mode == "live" and have_intr and len(self._views) >= self.N_VIEWS)

    # ---------------- Lifecycle (Wizard) ----------------

    def on_enter(self) -> None:
        pass

    def on_leave(self) -> None:
        super().on_leave()