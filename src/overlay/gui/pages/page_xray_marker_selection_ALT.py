# overlay/gui/pages/page_xray_marker_selection.py

from __future__ import annotations

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QPushButton, QFileDialog, QMessageBox, QSizePolicy
)

from overlay.gui.state import SessionState
from overlay.gui.pages.templates.templ_static_image import StaticImagePage
from overlay.gui.widgets.widget_xray_marker_selection import XrayMarkerSelectionWidget

from overlay.tools.xray_marker_selection import (
    detector_mask,
    sort_circles_grid,
    select_marker_roi_from_grid,
)
from overlay.tools.blob_detection import HoughCircleParams, detect_blobs_hough


class XrayMarkerSelectionPage(StaticImagePage):
    """
    Step — X-ray marker selection

    IMPORTANT (your constraint):
    - MUST NOT add any new SessionState fields dynamically.
    - SessionState fields used here (already defined in state.py):
        * xray_image, xray_image_path
        * xray_points_uv, xray_points_confirmed
        * xray_marker_overlay_bgr
        * marker_radius_px
    - Everything else stays LOCAL to this page:
        * circles_grid
        * pick_radius_px
    """

    def __init__(self, state: SessionState, on_complete_changed, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

        # ---------------- local (page-only) runtime ----------------
        self._circles_grid: np.ndarray | None = None     # (nrows, ncols, 3) with NaNs
        self._pick_radius_px: float | None = None        # interaction radius for selection

        # ======================================================
        # LEFT: replace template image_label with interactive widget
        # ======================================================
        self.marker_widget = XrayMarkerSelectionWidget(self.image_label)
        self.marker_widget.selection_proposed.connect(self.on_selection_proposed)
        self.marker_widget.selection_changed.connect(self._on_selection_changed)

        self.marker_widget.setGeometry(0, 0, self.image_label.width(), self.image_label.height())
        self.marker_widget.show()

        # ======================================================
        # RIGHT: controls
        # ======================================================
        self.btn_load = QPushButton("Load X-ray Image")
        self.btn_load.clicked.connect(self.load_xray_image)

        self.btn_detect = QPushButton("Detect markers")
        self.btn_detect.clicked.connect(self.on_detect_markers)

        self.btn_reset = QPushButton("Reset selection")
        self.btn_reset.clicked.connect(self.on_reset_selection)

        for b in (self.btn_load, self.btn_detect, self.btn_reset):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.controls_content.addWidget(b)

        # ======================================================
        # Instructions
        # ======================================================
        self.instructions_label.setText(
            "1) Load an X-ray image\n"
            "2) Click 'Detect markers'\n"
            "3) Select 3 anchor markers (LMB select, RMB undo, ESC cancel)\n"
            "4) Confirm selection"
        )

        self.refresh()

    # ------------------------------------------------------------------
    # Keep marker widget in sync with template viewport size
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.marker_widget.setGeometry(0, 0, self.image_label.width(), self.image_label.height())

    def _on_selection_changed(self):
        self.refresh()

    # ---------------- Completion ----------------

    def is_complete(self) -> bool:
        # use the dedicated state flag (no dynamic getattr needed)
        return self.state.has_xray_points_confirmed

    # ---------------- Refresh ----------------

    def refresh(self):
        img = self.state.xray_image

        # --------------------------------------------------
        # Case 1: no image loaded yet
        # --------------------------------------------------
        if img is None:
            self.marker_widget.set_image(None)
            self.marker_widget.set_grid(None)
            self.marker_widget.set_locked(False)

            self.set_stats_rows([
                ("Blobs detected", "-"),
                ("Blobs selected", "-"),
            ])

            self.btn_load.setEnabled(True)
            self.btn_detect.setEnabled(False)
            self.btn_reset.setEnabled(False)
            return

        # --------------------------------------------------
        # Case 2: image loaded
        # --------------------------------------------------
        self.marker_widget.set_image(img)

        if self._circles_grid is not None and self._pick_radius_px is not None:
            self.marker_widget.set_grid(self._circles_grid, float(self._pick_radius_px))
        else:
            self.marker_widget.set_grid(None)

        confirmed = bool(self.state.xray_points_confirmed)

        # Load: never again once image is loaded
        self.btn_load.setEnabled(False)

        # Detect: only once (unless you reload image)
        already_detected = self._circles_grid is not None
        self.btn_detect.setEnabled(not already_detected)

        # Reset: only if at least 1 anchor selected
        selected_cells = self.marker_widget.get_selected_cells()
        self.btn_reset.setEnabled(len(selected_cells) > 0)

        # Lock interaction after confirmation
        self.marker_widget.set_locked(confirmed)

        # Stats
        if self._circles_grid is None:
            detected_txt = "-"
            selected_txt = "- / 3"
        else:
            n_detected = int(np.isfinite(self._circles_grid[..., 0]).sum())
            n_selected = len(selected_cells)
            detected_txt = str(n_detected)
            selected_txt = f"{n_selected} / 3"

        self.set_stats_rows([
            ("Blobs detected", detected_txt),
            ("Blobs selected", selected_txt),
        ])

    # ---------------- Image loading ----------------

    def load_xray_image(self):
        dlg = QFileDialog(self.window(), "Select X-ray image")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.ima);;All Files (*)")
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)

        if dlg.exec() != QFileDialog.Accepted:
            return

        files = dlg.selectedFiles()
        path = files[0] if files else ""
        if not path:
            return

        try:
            if path.lower().endswith((".dcm", ".ima")):
                try:
                    import pydicom  # lazy import (keeps page import-safe)
                except Exception as e:
                    raise RuntimeError("pydicom is required to open .dcm/.ima files. Install with: pip install pydicom") from e

                ds = pydicom.dcmread(path)
                img = ds.pixel_array.astype(np.float32)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = img.astype(np.uint8)
            else:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError("Could not read image.")

            # --- store only EXISTING SessionState fields ---
            self.state.xray_image = img
            self.state.xray_image_path = path

            # reset dependent EXISTING state fields
            self.state.xray_points_uv = None
            self.state.xray_points_confirmed = False
            self.state.xray_marker_overlay_bgr = None
            self.state.marker_radius_px = None

            # reset local (page-only) detection products
            self._circles_grid = None
            self._pick_radius_px = None

            self.marker_widget.clear_selection()
            self.marker_widget.set_roi_cells([])
            self.marker_widget.set_locked(False)

            self.refresh()
            self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(self, "Failed to load X-ray image", str(e))

    # ---------------- Marker detection ----------------

    def on_detect_markers(self):
        if self.state.xray_image is None:
            return

        try:
            img = self.state.xray_image

            params = HoughCircleParams(
                min_radius=2,
                max_radius=7,
                dp=1.2,
                minDist=16,
                param1=120,
                param2=12,
                invert=True,
                median_ks=(3, 5),
            )

            use_clahe = True
            clahe_clip = 2.0
            clahe_tiles = (12, 12)
            row_tol_px = 13.0

            img_proc = img.copy()
            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tiles)
                img_proc = clahe.apply(img_proc)

            mask = detector_mask(img_proc)
            img_masked = img_proc.copy()
            img_masked[mask == 0] = 0

            circles_out = detect_blobs_hough(img_masked, params)
            if circles_out is None or len(circles_out) == 0:
                QMessageBox.warning(self, "Marker detection", "No circles detected.")
                return

            circles_out = np.asarray(circles_out, dtype=np.float32)
            circles_sorted = sort_circles_grid(circles_out, row_tol_px=row_tol_px)
            if circles_sorted is None or len(circles_sorted) == 0:
                QMessageBox.warning(self, "Marker detection", "No circles after sorting.")
                return

            r_med = float(np.median(circles_sorted[:, 2]))
            y_thresh_px = 2.5 * r_med

            c = circles_sorted.copy().astype(np.float32)
            current = c[:1].copy()
            y_ref = float(c[0, 1])
            rows = []

            for k in range(1, len(c)):
                pt = c[k]
                if abs(float(pt[1]) - y_ref) > y_thresh_px:
                    sort_idx = np.argsort(current[:, 0])
                    rows.append(current[sort_idx])
                    current = pt.reshape(1, -1)
                    y_ref = float(pt[1])
                else:
                    current = np.vstack([current, pt])
                    y_ref = float(np.mean(current[:, 1]))

            sort_idx = np.argsort(current[:, 0])
            rows.append(current[sort_idx])
            nrows = len(rows)

            center_row_idx = int(np.argmax([len(row) for row in rows]))
            ncols = len(rows[center_row_idx])

            circles_grid = np.full((nrows, ncols, 3), np.nan, dtype=np.float32)
            for i, row in enumerate(rows):
                n_points = len(row)
                left_pad = (ncols - n_points) // 2
                padded_row = np.full((ncols, 3), np.nan, dtype=np.float32)
                padded_row[left_pad:left_pad + n_points] = row
                circles_grid[i] = padded_row

            radii = circles_grid[..., 2]
            finite_r = radii[np.isfinite(radii)]

            # store TRUE detected radius in EXISTING state field
            if finite_r.size:
                self.state.marker_radius_px = float(np.median(finite_r))
            else:
                self.state.marker_radius_px = None

            # keep pick radius LOCAL (interaction only)
            if self.state.marker_radius_px is not None:
                pick_radius_px = 0.6 * float(self.state.marker_radius_px)
            else:
                pick_radius_px = 20.0

            self._circles_grid = circles_grid
            self._pick_radius_px = float(pick_radius_px)

            # reset selection/confirmation + overlay (EXISTING state fields)
            self.state.xray_points_uv = None
            self.state.xray_points_confirmed = False
            self.state.xray_marker_overlay_bgr = None

            self.marker_widget.clear_selection()
            self.marker_widget.set_roi_cells([])
            self.marker_widget.set_locked(False)

            self.refresh()
            self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(self, "Marker detection failed", str(e))

    # ---------------- Confirm flow ----------------

    def on_selection_proposed(self, payload):
        cells = payload.get("cells", None)
        if cells is None or len(cells) != 3:
            return

        ans = QMessageBox.question(
            self,
            "Confirm markers",
            "Do these 3 selected markers look correct?",
            QMessageBox.Yes | QMessageBox.No
        )

        if ans != QMessageBox.Yes:
            self.state.xray_points_uv = None
            self.state.xray_points_confirmed = False
            self.state.xray_marker_overlay_bgr = None
            self.marker_widget.clear_selection()
            self.marker_widget.set_roi_cells([])
            self.marker_widget.set_locked(False)
            self.refresh()
            self.on_complete_changed()
            return

        if self._circles_grid is None:
            return

        xy, roi_cells_set = select_marker_roi_from_grid(
            circles_grid=self._circles_grid,
            selected_cells=cells,
        )

        self.state.xray_points_uv = np.asarray(xy, dtype=float)
        self.state.xray_points_confirmed = True

        self.marker_widget.set_roi_cells(list(roi_cells_set))
        self.marker_widget.set_locked(True)

        # Cache FULL-SIZE ROI overlay
        self.state.xray_marker_overlay_bgr = self._render_roi_overlay_bgr()

        self.refresh()
        self.on_complete_changed()

    # ---------------- Reset ----------------

    def on_reset_selection(self):
        self.state.xray_points_uv = None
        self.state.xray_points_confirmed = False
        self.state.xray_marker_overlay_bgr = None

        self.marker_widget.clear_selection()
        self.marker_widget.set_roi_cells([])
        self.marker_widget.set_locked(False)

        self.refresh()
        self.on_complete_changed()

    # ---------------- Helpers ----------------

    def _render_roi_overlay_bgr(self) -> np.ndarray | None:
        """
        Render full-size X-ray image with only ROI overlay:
        - green circles
        - red crosses
        """
        img = self.state.xray_image
        uv = self.state.xray_points_uv

        if img is None or uv is None or len(uv) == 0:
            return None

        img8 = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

        # use local pick radius if available
        if self._pick_radius_px is None:
            roi_r = 6.0
        else:
            roi_r = max(3.0, 0.35 * float(self._pick_radius_px))

        cross_r = int(round(0.6 * roi_r))

        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)

        for (u, v) in uv:
            if not np.isfinite(u) or not np.isfinite(v):
                continue
            uu, vv = int(round(u)), int(round(v))

            cv2.circle(out, (uu, vv), int(round(roi_r)), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(out, (uu - cross_r, vv), (uu + cross_r, vv), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(out, (uu, vv - cross_r), (uu, vv + cross_r), (0, 0, 255), 2, cv2.LINE_AA)

        return out