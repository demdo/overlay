# overlay/gui/pages/page_camera_to_xray_calibration.py

from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import QPushButton, QMessageBox, QSizePolicy

from overlay.gui.state import SessionState
from overlay.gui.pages.templates.templ_static_image import StaticImagePage
from overlay.calib import calib_camera_to_xray as cam2x
from overlay.gui.widgets.widget_zoom_view import ZoomView


class CameraToXrayCalibrationPage(StaticImagePage):
    """
    Step — Camera → X-ray calibration (PnP)

    GUI wrapper around the same logic as the underlying calibration functions
    (no algorithmic differences intended).
    """

    def __init__(self, state: SessionState, on_complete_changed, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

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
        return getattr(self.state, "T_cx", None) is not None

    # ---------------- QWidget lifecycle ----------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "zoom_view"):
            self.zoom_view.setGeometry(0, 0, self.image_label.width(), self.image_label.height())

    def on_enter(self) -> None:
        self.refresh()

    # ---------------- Refresh ----------------

    def refresh(self) -> None:
        img = getattr(self.state, "xray_image", None)
    
        # --------------------------------------------------
        # No image yet
        # --------------------------------------------------
        if img is None:
            self.set_viewport_background(active=False)
    
            self.zoom_view.set_image(None)
            self.zoom_view.clear_overlay_data()
            self.zoom_view.reset_zoom()
            self.zoom_view.setEnabled(False)
    
            self.set_stats_rows([
                ("Inliers", "-"),
                ("RANSAC threshold", "-"),
                ("Median", "-"),
                ("P95", "-"),
                ("Avg. reprojection error", "-"),
            ])
    
            self.btn_run.setEnabled(False)
            self._update_zoom_button()
            return
    
        # --------------------------------------------------
        # Image available
        # --------------------------------------------------
        self.set_viewport_background(active=True)
    
        # Always show RAW X-ray in ZoomView (keeps base consistent)
        self.zoom_view.set_image(img)
    
        # Use true marker radius for visual overlays if available
        # (pick_radius_px is for interaction; marker_radius_px is the detected hole radius)
        pick_r = getattr(self.state, "pick_radius_px", None)
        marker_r = getattr(self.state, "marker_radius_px", None)
        if marker_r is None:
            marker_r = pick_r
    
        uv_roi = getattr(self.state, "xray_points_uv", None)
        uv_meas = getattr(self.state, "xray_points_uv", None)  # measured = marker selection
        uv_proj = getattr(self.state, "pnp_uv_projected", None)
    
        pnp_done = uv_proj is not None
    
        # --------------------------------------------------
        # NEW: Outlier mask from RANSAC inliers
        # --------------------------------------------------
        outlier_mask = None
        inlier_idx = getattr(self.state, "pnp_inliers_idx", None)
        if pnp_done and inlier_idx is not None and uv_meas is not None:
            uv_meas_arr = np.asarray(uv_meas, dtype=np.float64).reshape(-1, 2)
            n = uv_meas_arr.shape[0]
            mask_in = np.zeros(n, dtype=bool)
            idx = np.asarray(inlier_idx, dtype=np.int64).reshape(-1)
            idx = idx[(idx >= 0) & (idx < n)]  # safety clamp
            mask_in[idx] = True
            outlier_mask = ~mask_in
    
        if pnp_done:
            # After PnP: show full overlay + residual lines, zoom enabled
            self.zoom_view.set_overlay_data(
                uv_roi=uv_roi,
                uv_measured=uv_meas,
                uv_projected=uv_proj,
                pick_radius_px=pick_r,              # CROSS sizing (keep)
                marker_radius_px=1.2 * marker_r,    # CIRCLE sizing
                show_residuals=True,
                outlier_mask=outlier_mask,          
            )
            self.zoom_view.setEnabled(True)
        else:
            # Before PnP
            self.zoom_view.set_image(img)
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
    
        # Run button enable (prerequisites)
        miss = self._missing()
        self.btn_run.setEnabled(len(miss) == 0)
    
        # --------------------------------------------------
        # Stats
        # --------------------------------------------------
        rows: list[tuple[str, str]] = []
    
        n_points = getattr(self.state, "pnp_n_points", None)
        n_inliers = getattr(self.state, "pnp_n_inliers", None)
        did_ransac = (inlier_idx is not None) and (n_points is not None) and (n_inliers is not None)
    
        if did_ransac:
            rows.append(("Inliers", f"{int(n_inliers)} / {int(n_points)}"))
    
            thr = getattr(self.state, "pnp_ransac_threshold_px", None)
            rows.append(("RANSAC threshold", "-" if thr is None else f"{float(thr):.1f} px"))
    
            med = getattr(self.state, "pnp_reproj_median_px", None)
            rows.append(("Median", "-" if med is None else f"{float(med):.3f} px"))
    
            # P95 on inliers
            p95_txt = "-"
            errs = getattr(self.state, "pnp_reproj_errors_px", None)
            if errs is not None:
                e = np.asarray(errs, dtype=np.float64).reshape(-1)
                idx = np.asarray(inlier_idx, dtype=np.int64).reshape(-1)
                if idx.size > 0 and e.size > 0:
                    idx = idx[(idx >= 0) & (idx < e.size)]
                    if idx.size > 0:
                        e_in = e[idx]
                        if e_in.size > 0:
                            p95_txt = f"{float(np.percentile(e_in, 95)):.3f} px"
            rows.append(("P95", p95_txt))
        else:
            rows.append(("Inliers", "-"))
            rows.append(("RANSAC threshold", "-"))
            rows.append(("Median", "-"))
            rows.append(("P95", "-"))
    
        mean = getattr(self.state, "pnp_reproj_mean_px", None)
        rows.append(("Avg. reprojection error", "-" if mean is None else f"{float(mean):.3f} px"))
    
        self.set_stats_rows(rows)
        self._update_zoom_button()

    # ---------------- Zoom helpers ----------------

    def on_reset_zoom(self) -> None:
        self.zoom_view.reset_zoom()
        self._update_zoom_button()

    def _update_zoom_button(self) -> None:
        pnp_done = getattr(self.state, "pnp_uv_projected", None) is not None
        zoomed = self.zoom_view.has_zoom()
        self.btn_reset_zoom.setEnabled(bool(pnp_done and zoomed))

    # ---------------- Action ----------------

    def on_run_pnp(self) -> None:
        miss = self._missing()
        if miss:
            QMessageBox.warning(self, "PnP", "Cannot run PnP.\n\nMissing: " + ", ".join(miss))
            return

        try:
            Kx = np.asarray(self.state.K_xray, dtype=np.float64)
            uv_meas = np.asarray(self.state.xray_points_uv, dtype=np.float64).reshape(-1, 2)
            s = int(self.state.steps_per_edge)
            xyz_c = np.asarray(self.state.xray_points_xyz_c, dtype=np.float64).reshape(-1, 3)
            
            if xyz_c.shape[0] != uv_meas.shape[0]:
                QMessageBox.warning(
                    self,
                    "Point count mismatch",
                    f"2D points: {uv_meas.shape[0]}\n"
                    f"3D points from grid: {xyz_c.shape[0]} (steps_per_edge={s})\n\n"
                    "This step expects exactly (s+1)^2 correspondences.",
                )
                return

            ransac_thr = 6.0
            self.state.pnp_ransac_threshold_px = float(ransac_thr)

            pnp = cam2x._solve_xray_pnp(
                points_xyz_camera=xyz_c,
                points_uv_xray=uv_meas,
                xray_intrinsics=Kx,
                dist_coeffs=None,
                use_ransac=True,
                ransac_reproj_error_px=ransac_thr,
                ransac_confidence=0.99,
                ransac_iterations=5000,
            )

            # transforms
            T_cx = np.asarray(pnp.T_4x4, dtype=np.float64)   # camera -> xray
            T_xc = cam2x.invert_T(T_cx)                      # xray -> camera
            self.state.T_cx = T_cx
            self.state.T_xc = T_xc

            # stats
            self.state.pnp_n_points = int(uv_meas.shape[0])
            self.state.pnp_inliers_idx = np.asarray(pnp.inlier_idx, dtype=np.int64)
            self.state.pnp_n_inliers = int(len(self.state.pnp_inliers_idx))

            self.state.pnp_reproj_errors_px = np.asarray(pnp.reproj_errors_px, dtype=np.float64)
            self.state.pnp_reproj_mean_px = float(pnp.reproj_mean_px)
            self.state.pnp_reproj_median_px = float(pnp.reproj_median_px)
            self.state.pnp_reproj_max_px = float(pnp.reproj_max_px)

            # measured + projected (for overlay)
            self.state.pnp_uv_measured = uv_meas
            self.state.pnp_uv_projected = np.asarray(pnp.uv_proj, dtype=np.float64)

            # start unzoomed after successful run
            self.zoom_view.reset_zoom()

            self.refresh()
            self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(self, "PnP failed", str(e))

    # ---------------- Helpers ----------------

    def _missing(self) -> list[str]:
        m = []

        if getattr(self.state, "K_xray", None) is None:
            m.append("K_xray")

        uv = getattr(self.state, "xray_points_uv", None)
        if uv is None or len(uv) == 0:
            m.append("xray_points_uv")
        if not bool(getattr(self.state, "xray_points_confirmed", False)):
            m.append("xray_points_confirmed")

        if getattr(self.state, "steps_per_edge", None) is None:
            m.append("steps_per_edge")

        xyz = getattr(self.state, "xray_points_xyz_c", None)
        if xyz is None or len(xyz) == 0:
            m.append("xray_points_xyz_c")

        return m