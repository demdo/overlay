# overlay/gui/pages/page_camera_to_xray_calibration.py

from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import QPushButton, QMessageBox, QSizePolicy

from overlay.gui.state import SessionState
from overlay.gui.pages.templates.templ_static_image import StaticImagePage
from overlay.calib.calib_camera_to_xray import calibrate_camera_to_xray
from overlay.gui.widgets.widget_zoom_view import ZoomView


class CameraToXrayCalibrationPage(StaticImagePage):
    """
    Step — Camera → X-ray calibration (PnP)

    IMPORTANT (your constraint):
    - MUST NOT add any new SessionState fields dynamically.
    - SessionState fields used here (already defined in state.py):
        * K_xray
        * xray_image
        * xray_points_uv, xray_points_confirmed
        * xray_points_xyz_c
        * pnp_ransac_threshold_px
        * T_cx, T_xc
        * marker_radius_px
    - All PnP diagnostics/overlays live ONLY in this page as self._... locals.
    """

    def __init__(self, state: SessionState, on_complete_changed, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

        # ---------------- local (page-only) cache ----------------
        self._pnp_done: bool = False
        self._uv_projected: np.ndarray | None = None     # (N,2)
        self._inlier_idx: np.ndarray | None = None       # (M,)
        self._reproj_errors_px: np.ndarray | None = None # (N,)
        self._reproj_mean_px: float | None = None
        self._reproj_median_px: float | None = None
        self._reproj_p95_in_px: float | None = None
        self._n_points: int | None = None
        self._n_inliers: int | None = None
        self._R_cx: np.ndarray | None = None             # (3,3)
        self._t_cx: np.ndarray | None = None             # (3,)

        # ======================================================
        # LEFT: replace template image_label with zoomable view
        # ======================================================
        self.zoom_view = ZoomView(self.image_label)
        self.zoom_view.setGeometry(0, 0, self.image_label.width(), self.image_label.height())
        self.zoom_view.zoom_changed.connect(self._update_zoom_button)
        self.zoom_view.show()

        # ======================================================
        # RIGHT: controls
        # ======================================================
        self.btn_run = QPushButton("Run PnP (RANSAC)")
        self.btn_run.clicked.connect(self.on_run_pnp)

        self.btn_reset_zoom = QPushButton("Reset zoom")
        self.btn_reset_zoom.clicked.connect(self.on_reset_zoom)

        for b in (self.btn_run, self.btn_reset_zoom):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.controls_content.addWidget(b)

        # ======================================================
        # Instructions
        # ======================================================
        self.instructions_label.setText(
            "1) Ensure prerequisites are complete (intrinsics, plane, marker ROI)\n"
            "2) Click 'Run PnP (RANSAC)'\n"
            "3) Inspect measured (red) vs projected (cyan) points in the X-ray image\n"
            "Tip: Drag a SQUARE box with LMB to zoom, RMB/double-click to reset"
        )

        self.refresh()

    # ---------------- Completion ----------------

    def is_complete(self) -> bool:
        return self.state.T_cx is not None

    # ---------------- QWidget lifecycle ----------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "zoom_view"):
            self.zoom_view.setGeometry(0, 0, self.image_label.width(), self.image_label.height())

    def on_enter(self) -> None:
        self.refresh()

    # ---------------- Stats helpers ----------------

    def _default_stats_rows(self) -> list[tuple[str, str]]:
        """
        Stats layout shown before any solve has been run.
        All rows that are method-agnostic are present; values are "-".
        """
        return [
            ("Median",                  "-"),
            ("P95",                     "-"),
            ("Avg. reprojection error", "-"),
            ("", ""),
            ("R_cx",                    "-"),
            ("", ""),
            ("t_cx",                    "-"),
        ]

    def _result_stats_rows(self) -> list[tuple[str, str]]:
        """
        Stats layout shown after a successful solve.
        Only method-agnostic fields are included; R_cx and t_cx always shown.
        """
        med  = self._reproj_median_px
        mean = self._reproj_mean_px
        p95  = self._reproj_p95_in_px

        rows: list[tuple[str, str]] = [
            ("Median",                  "-" if med  is None else f"{float(med):.3f} px"),
            ("P95",                     "-" if p95  is None else f"{float(p95):.3f} px"),
            ("Avg. reprojection error", "-" if mean is None else f"{float(mean):.3f} px"),
        ]

        rows.append(("", ""))
        if self._R_cx is not None:
            R = self._R_cx
            R_lines = "\n".join(
                " ".join(f"{v:+.4f}" for v in row)
                for row in R
            )
            rows.append(("R_cx", "np.array (3×3)\n" + R_lines))
        else:
            rows.append(("R_cx", "-"))

        rows.append(("", ""))
        if self._t_cx is not None:
            t = self._t_cx.reshape(3)
            rows.append(("t_cx", f"[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]^T"))
        else:
            rows.append(("t_cx", "-"))

        return rows

    # ---------------- Refresh ----------------

    def refresh(self) -> None:
        img = self.state.xray_image

        # --------------------------------------------------
        # No image yet
        # --------------------------------------------------
        if img is None:
            self.set_viewport_background(active=False)

            self.zoom_view.set_image(None)
            self.zoom_view.clear_overlay_data()
            self.zoom_view.reset_zoom()
            self.zoom_view.setEnabled(False)

            self.set_stats_rows(self._default_stats_rows())

            self.btn_run.setEnabled(False)
            self._update_zoom_button()
            return

        # --------------------------------------------------
        # Image available
        # --------------------------------------------------
        self.set_viewport_background(active=True)
        self.zoom_view.set_image(img)

        marker_r = self.state.marker_radius_px
        pick_r = 0.6 * float(marker_r) if marker_r is not None else 20.0

        uv_roi  = self.state.xray_points_uv
        uv_meas = self.state.xray_points_uv

        pnp_done = bool(self._pnp_done and self._uv_projected is not None)

        # Outlier mask from cached inliers
        outlier_mask = None
        if pnp_done and self._inlier_idx is not None and uv_meas is not None:
            uv_meas_arr = np.asarray(uv_meas, dtype=np.float64).reshape(-1, 2)
            n = uv_meas_arr.shape[0]
            mask_in = np.zeros(n, dtype=bool)
            idx = np.asarray(self._inlier_idx, dtype=np.int64).reshape(-1)
            idx = idx[(idx >= 0) & (idx < n)]
            mask_in[idx] = True
            outlier_mask = ~mask_in

        if pnp_done:
            self.zoom_view.set_overlay_data(
                uv_roi=uv_roi,
                uv_measured=uv_meas,
                uv_projected=self._uv_projected,
                pick_radius_px=pick_r,
                marker_radius_px=1.2 * marker_r if marker_r is not None else None,
                show_residuals=True,
                outlier_mask=outlier_mask,
            )
            self.zoom_view.setEnabled(True)
        else:
            self.zoom_view.set_overlay_data(
                uv_roi=uv_roi,
                uv_measured=uv_meas,
                uv_projected=None,
                pick_radius_px=pick_r,
                marker_radius_px=1.2 * marker_r if marker_r is not None else None,
                show_residuals=False,
                outlier_mask=None,
            )
            self.zoom_view.reset_zoom()
            self.zoom_view.setEnabled(False)

        miss = self._missing()
        self.btn_run.setEnabled(len(miss) == 0)

        self.set_stats_rows(
            self._result_stats_rows() if pnp_done else self._default_stats_rows()
        )

        self._update_zoom_button()

    # ---------------- Zoom helpers ----------------

    def on_reset_zoom(self) -> None:
        self.zoom_view.reset_zoom()
        self._update_zoom_button()

    def _update_zoom_button(self) -> None:
        zoomed = self.zoom_view.has_zoom()
        self.btn_reset_zoom.setEnabled(bool(self._pnp_done and zoomed))

    # ---------------- Action ----------------

    def on_run_pnp(self) -> None:
        miss = self._missing()
        if miss:
            QMessageBox.warning(self, "PnP", "Cannot run PnP.\n\nMissing: " + ", ".join(miss))
            return

        try:
            Kx      = np.asarray(self.state.K_xray, dtype=np.float64)
            uv_meas = np.asarray(self.state.xray_points_uv, dtype=np.float64).reshape(-1, 2)
            xyz_c   = np.asarray(self.state.xray_points_xyz_c, dtype=np.float64).reshape(-1, 3)

            if xyz_c.shape[0] != uv_meas.shape[0]:
                QMessageBox.warning(
                    self,
                    "Point count mismatch",
                    f"2D points: {uv_meas.shape[0]}\n"
                    f"3D points: {xyz_c.shape[0]}\n\n"
                    "This step expects the same number of 2D and 3D correspondences.",
                )
                return

            ransac_thr = 1.5
            self.state.pnp_ransac_threshold_px = float(ransac_thr)

            pnp = calibrate_camera_to_xray(
                K_xray=Kx,
                points_xyz_camera=xyz_c,
                points_uv_xray=uv_meas,
                dist_coeffs=None,
                pose_method="iterative_ransac",
                ransac_reprojection_error_px=ransac_thr,
                ransac_confidence=0.99,
                ransac_iterations_count=5000,
            )

            # --------- persist ONLY allowed SessionState fields ----------
            self.state.T_cx = pnp.T_cx
            self.state.T_xc = pnp.T_xc

            # --------- local diagnostics (page-only) ----------
            self._pnp_done = True
            self._n_points = int(uv_meas.shape[0])
            self._R_cx     = np.asarray(pnp.rotation, dtype=np.float64)
            self._t_cx     = np.asarray(pnp.translation, dtype=np.float64).reshape(3)

            self._inlier_idx = np.asarray(pnp.inlier_idx, dtype=np.int64).reshape(-1)
            self._n_inliers  = int(self._inlier_idx.size)

            self._reproj_errors_px = np.asarray(pnp.reproj_errors_px, dtype=np.float64).reshape(-1)
            self._reproj_mean_px   = float(pnp.reproj_mean_px)
            self._reproj_median_px = float(pnp.reproj_median_px)

            # P95 on inliers
            self._reproj_p95_in_px = None
            if self._reproj_errors_px.size > 0 and self._inlier_idx.size > 0:
                idx = self._inlier_idx.copy()
                idx = idx[(idx >= 0) & (idx < self._reproj_errors_px.size)]
                if idx.size > 0:
                    e_in = self._reproj_errors_px[idx]
                    if e_in.size > 0:
                        self._reproj_p95_in_px = float(np.percentile(e_in, 95))

            self._uv_projected = np.asarray(pnp.uv_proj, dtype=np.float64).reshape(-1, 2)

            self.zoom_view.reset_zoom()
            self.refresh()
            self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(self, "PnP failed", str(e))

    # ---------------- Helpers ----------------

    def _missing(self) -> list[str]:
        m: list[str] = []

        if self.state.K_xray is None:
            m.append("K_xray")

        uv = self.state.xray_points_uv
        if uv is None or len(uv) == 0:
            m.append("xray_points_uv")
        if not bool(self.state.xray_points_confirmed):
            m.append("xray_points_confirmed")

        xyz = self.state.xray_points_xyz_c
        if xyz is None or len(xyz) == 0:
            m.append("xray_points_xyz_c")

        return m