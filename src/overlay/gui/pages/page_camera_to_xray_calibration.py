# overlay/gui/pages/page_camera_to_xray_calibration.py

from __future__ import annotations

import numpy as np
import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QGroupBox,
    QGridLayout,
    QSizePolicy,
    QPlainTextEdit,
)

from overlay.gui.state import SessionState
from overlay.calib import calibration_camera_to_xray as cam2x


# ============================================================
# Helpers (match style of PlaneFittingPage / CameraCalibrationPage)
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


def _format_3x4(T: np.ndarray) -> str:
    T = np.asarray(T, dtype=np.float64).reshape(3, 4)
    lines = []
    for r in range(3):
        lines.append(f"[{T[r,0]:+.6f}  {T[r,1]:+.6f}  {T[r,2]:+.6f}  {T[r,3]:+.6f}]")
    return "\n".join(lines)


def _project_cam_xyz_to_xray_uv(X_cam: np.ndarray, T_3x4: np.ndarray, Kx: np.ndarray) -> np.ndarray:
    """Project 3D points in camera frame -> X-ray pixels using [R|t] (cam->xray) and Kx."""
    X_cam = np.asarray(X_cam, dtype=np.float64).reshape(-1, 3)
    T_3x4 = np.asarray(T_3x4, dtype=np.float64).reshape(3, 4)
    Kx = np.asarray(Kx, dtype=np.float64).reshape(3, 3)

    R = T_3x4[:, :3]
    t = T_3x4[:, 3:4]  # (3,1)

    X_xray = (R @ X_cam.T) + t  # (3,N)
    Z = X_xray[2, :]
    valid = Z > 1e-9

    uv = np.full((X_cam.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        x = X_xray[0, valid] / Z[valid]
        y = X_xray[1, valid] / Z[valid]
        uv[valid, 0] = Kx[0, 0] * x + Kx[0, 2]
        uv[valid, 1] = Kx[1, 1] * y + Kx[1, 2]
    return uv


class CameraToXrayCalibrationPage(QWidget):
    """Step — Camera → X-ray calibration (PnP RANSAC)."""

    DISP_W = 960
    DISP_H = 540

    RIGHT_W = 260
    MARGIN = 20
    SPACING = 20

    def __init__(self, state: SessionState, on_complete_changed=None, parent=None):
        super().__init__(parent)
        self.state = state
        self.on_complete_changed = on_complete_changed

        self._build_ui()
        self.refresh()

    def _build_ui(self):
        # LEFT: fixed workspace image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setFixedSize(self.DISP_W, self.DISP_H)
        self.image_label.setStyleSheet("background-color: transparent; border-radius: 10px;")
        self.image_label.clear()

        left_container = QWidget()
        left_v = QVBoxLayout(left_container)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(0)
        left_v.addWidget(self.image_label, 0, Qt.AlignLeft | Qt.AlignTop)

        # Matrix block (below image) with ~1cm spacing
        self.lbl_T_title = QLabel("Estimated [R|t] (3×4) — camera → X-ray")
        self.lbl_T_title.setStyleSheet("font-weight: 600; color: #212529;")

        self.txt_T = QPlainTextEdit()
        self.txt_T.setReadOnly(True)
        self.txt_T.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.txt_T.setStyleSheet("font-family: Consolas, Menlo, monospace; color: #212529;")
        self.txt_T.setFixedHeight(110)
        self.txt_T.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.txt_T.setPlainText("–")

        left_v.addSpacing(38)  # ~1cm
        left_v.addWidget(self.lbl_T_title)
        left_v.addSpacing(10)
        left_v.addWidget(self.txt_T)
        left_v.addStretch(1)

        # RIGHT: controls + stats
        self.btn_run = QPushButton("Run PnP (RANSAC)")
        self.btn_run.clicked.connect(self.run_pnp)
        self.btn_run.setFixedHeight(42)

        controls = QGroupBox("Controls")
        cg = QGridLayout(controls)
        cg.setContentsMargins(8, 8, 8, 8)
        cg.setSpacing(8)
        cg.addWidget(self.btn_run, 0, 0)

        self.lbl_err = QLabel("Reprojection error (px): –")
        self.lbl_err.setWordWrap(True)

        self.lbl_missing = QLabel("")
        self.lbl_missing.setWordWrap(True)
        self.lbl_missing.setStyleSheet("color: #6c757d;")

        stats = QGroupBox("PnP Stats")
        sv = QVBoxLayout(stats)
        sv.setContentsMargins(8, 8, 8, 8)
        sv.setSpacing(8)
        sv.addWidget(self.lbl_err)
        sv.addWidget(self.lbl_missing)

        right_panel = QWidget()
        right_panel.setFixedWidth(self.RIGHT_W)
        right_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        right_v = QVBoxLayout(right_panel)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(12)
        right_v.setAlignment(Qt.AlignTop)
        right_v.addWidget(controls)
        right_v.addWidget(stats)
        right_v.addStretch(1)

        # MAIN
        main = QHBoxLayout(self)
        main.setContentsMargins(self.MARGIN, self.MARGIN, self.MARGIN, self.MARGIN)
        main.setSpacing(self.SPACING)
        main.addWidget(left_container, 0)
        main.addWidget(right_panel, 0)

    # ---------- rendering (cover+crop to 16:9 like other pages) ----------

    def _show_image(self, img_bgr: np.ndarray) -> None:
        target_w, target_h = self.DISP_W, self.DISP_H
        h, w = img_bgr.shape[:2]
        if h <= 0 or w <= 0:
            self.image_label.clear()
            return

        target_ratio = target_w / target_h
        src_ratio = w / h

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

    def _render_xray_overlay_bgr(self) -> np.ndarray | None:
        if getattr(self.state, "xray_image", None) is None:
            return None

        img = self.state.xray_image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # ROI points (same as marker selection ROI)
        uv_roi = getattr(self.state, "xray_points_uv", None)

        # measured + projected
        uv_meas = getattr(self.state, "pnp_uv_measured", None)
        uv_proj = getattr(self.state, "pnp_uv_projected", None)

        if uv_meas is None:
            uv_meas = uv_roi

        # green outline radius based on pick_radius_px
        pick_r = getattr(self.state, "pick_radius_px", None)
        roi_r = 6.0 if pick_r is None else max(3.0, 0.35 * float(pick_r))

        # 1) ROI outline (green)
        if uv_roi is not None:
            uv = np.asarray(uv_roi, dtype=np.float64).reshape(-1, 2)
            for (u, v) in uv:
                if not np.isfinite(u) or not np.isfinite(v):
                    continue
                cv2.circle(base, (int(round(u)), int(round(v))), int(round(roi_r)), (0, 255, 0), 2, cv2.LINE_AA)

        # 2) measured = red cross
        if uv_meas is not None:
            uv = np.asarray(uv_meas, dtype=np.float64).reshape(-1, 2)
            for (u, v) in uv:
                if not np.isfinite(u) or not np.isfinite(v):
                    continue
                uu, vv = int(round(u)), int(round(v))
                r = 6
                cv2.line(base, (uu - r, vv), (uu + r, vv), (0, 0, 255), 2, cv2.LINE_AA)
                cv2.line(base, (uu, vv - r), (uu, vv + r), (0, 0, 255), 2, cv2.LINE_AA)

        # 3) projected = turquoise cross (single color)
        if uv_proj is not None:
            uv = np.asarray(uv_proj, dtype=np.float64).reshape(-1, 2)
            for (u, v) in uv:
                if not np.isfinite(u) or not np.isfinite(v):
                    continue
                uu, vv = int(round(u)), int(round(v))
                r = 6
                cv2.line(base, (uu - r, vv), (uu + r, vv), (255, 255, 0), 2, cv2.LINE_AA)
                cv2.line(base, (uu, vv - r), (uu, vv + r), (255, 255, 0), 2, cv2.LINE_AA)

        return base

    # ---------- gating ----------

    def _missing(self) -> list[str]:
        m = []
        if getattr(self.state, "K_rgb", None) is None:
            m.append("K_rgb")
        if getattr(self.state, "K_xray", None) is None:
            m.append("K_xray")
        if getattr(self.state, "plane_model_c", None) is None:
            m.append("plane_model_c")
        if getattr(self.state, "cb_extremes_uv", None) is None:
            m.append("cb_extremes_uv (TL,TR,BL)")
        if getattr(self.state, "xray_points_uv", None) is None or len(self.state.xray_points_uv) == 0:
            m.append("xray_points_uv")
        if not bool(getattr(self.state, "xray_points_confirmed", False)):
            m.append("xray_points_confirmed")
        if getattr(self.state, "steps_per_edge", None) is None:
            m.append("steps_per_edge")
        return m

    # ---------- refresh ----------

    def refresh(self):
        miss = self._missing()
        self.btn_run.setEnabled(len(miss) == 0)
        self.lbl_missing.setText("" if len(miss) == 0 else ("Missing: " + ", ".join(miss)))

        overlay = self._render_xray_overlay_bgr()
        if overlay is None:
            self.image_label.setStyleSheet("background-color: transparent; border-radius: 10px;")
            self.image_label.clear()
        else:
            self.image_label.setStyleSheet("background-color: #202020; border-radius: 10px;")
            self._show_image(overlay)

        if getattr(self.state, "T_xray_from_cam_3x4", None) is not None:
            self.txt_T.setPlainText(_format_3x4(self.state.T_xray_from_cam_3x4))
        else:
            self.txt_T.setPlainText("–")

        if (
            getattr(self.state, "pnp_reproj_mean_px", None) is not None
            and getattr(self.state, "pnp_reproj_median_px", None) is not None
            and getattr(self.state, "pnp_reproj_max_px", None) is not None
        ):
            self.lbl_err.setText(
                "Reprojection error (px):\n"
                f"mean: {self.state.pnp_reproj_mean_px:.3f}\n"
                f"median: {self.state.pnp_reproj_median_px:.3f}\n"
                f"max: {self.state.pnp_reproj_max_px:.3f}"
            )
        else:
            self.lbl_err.setText("Reprojection error (px): –")

    # ---------- action ----------

    def run_pnp(self):
        miss = self._missing()
        if miss:
            QMessageBox.warning(self, "PnP", "Cannot run PnP.\n\nMissing: " + ", ".join(miss))
            return

        try:
            corners_uv = np.asarray(self.state.cb_extremes_uv, dtype=np.float64).reshape(3, 2)
            Krgb = np.asarray(self.state.K_rgb, dtype=np.float64)
            Kx = np.asarray(self.state.K_xray, dtype=np.float64)
            plane = np.asarray(self.state.plane_model_c, dtype=np.float64).reshape(4,)
            uv_meas = np.asarray(self.state.xray_points_uv, dtype=np.float64).reshape(-1, 2)
            s = int(self.state.steps_per_edge)

            corner_xyz_c = cam2x._intersect_corners_with_plane(
                corners_uv=corners_uv,
                rgb_intrinsics=Krgb,
                plane_model=plane,
            )

            xyz_c = cam2x._interpolate_marker_grid(
                corner_xyz=corner_xyz_c,
                steps_per_edge=s,
            )

            if xyz_c.shape[0] != uv_meas.shape[0]:
                QMessageBox.warning(
                    self,
                    "Point count mismatch",
                    f"2D points: {uv_meas.shape[0]}\n"
                    f"3D points from grid: {xyz_c.shape[0]} (steps_per_edge={s})\n\n"
                    "This step expects exactly (s+1)^2 correspondences.",
                )
                return

            pnp = cam2x._solve_xray_pnp(
                points_xyz_camera=xyz_c,
                points_uv_xray=uv_meas,
                xray_intrinsics=Kx,
                dist_coeffs=None,
                use_ransac=True,
                ransac_reproj_error_px=3.0,
                ransac_confidence=0.99,
                ransac_iterations=5000,
            )

            self.state.xray_points_xyz_c = xyz_c

            self.state.T_xray_from_cam_3x4 = np.asarray(pnp.T_3x4, dtype=np.float64)
            T4 = np.eye(4, dtype=np.float64)
            T4[:3, :4] = self.state.T_xray_from_cam_3x4
            self.state.T_xray_from_cam_4x4 = T4

            self.state.pnp_n_points = int(uv_meas.shape[0])
            self.state.pnp_inliers_idx = np.asarray(pnp.inlier_idx, dtype=np.int64)
            self.state.pnp_n_inliers = int(len(self.state.pnp_inliers_idx))

            self.state.pnp_reproj_errors_px = np.asarray(pnp.reproj_errors_px, dtype=np.float64)
            self.state.pnp_reproj_mean_px = float(pnp.reproj_mean_px)
            self.state.pnp_reproj_median_px = float(pnp.reproj_median_px)
            self.state.pnp_reproj_max_px = float(pnp.reproj_max_px)

            self.state.pnp_uv_measured = uv_meas
            self.state.pnp_uv_projected = _project_cam_xyz_to_xray_uv(
                X_cam=xyz_c,
                T_3x4=self.state.T_xray_from_cam_3x4,
                Kx=Kx,
            )

            self.refresh()
            if callable(self.on_complete_changed):
                self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(self, "PnP failed", str(e))
