# overlay/gui/pages/page_xray_marker_selection.py

import os
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QStackedWidget
)

from overlay.gui.state import SessionState
from overlay.gui.widgets.widget_xray_marker_selection import XrayMarkerSelectionWidget

from overlay.tools.xray_marker_selection import detector_mask_radial, sort_circles_grid
from overlay.tools.blob_detection import HoughCircleParams, detect_blobs_hough


class XrayMarkerSelectionPage(QWidget):
    """
    Step 3 — X-ray marker selection

    Flow:
      1) Load X-ray image
      2) Detect markers -> circles_grid + pick_radius_px
      3) Select 3 anchors (L-shape). On 3rd click:
         - Confirm Yes/No
         - Yes:
             * ROI = grid-aligned rectangle spanning the 3 selected cells
             * Highlight all ROI markers (no rectangle)
             * state.xray_points_uv = ALL ROI marker (u,v)
             * lock widget, disable detect/reset
             * allow Next (is_complete = confirmed)
         - No:
             * clear selection and pick again
    """

    def __init__(self, state: SessionState, on_complete_changed):
        super().__init__()
        self.state = state
        self.on_complete_changed = on_complete_changed

        self._last_render_key = None  # prevents wiping widget overlays

        root = QVBoxLayout(self)
        root.setSpacing(12)

        self.title = QLabel("Step 3 — X-ray marker selection")
        root.addWidget(self.title)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, stretch=1)

        # -------- View A (Load) --------
        view_load = QWidget()
        load_layout = QVBoxLayout(view_load)
        load_layout.setSpacing(10)

        self.load_info = QLabel(
            "Load an X-ray image.\n"
            "Then click 'Detect markers' and select 3 anchor markers (L-shape):\n"
            "• Left click = select\n"
            "• Right click = undo\n"
            "After the 3rd marker you will be asked to confirm.\n"
        )

        self.btn_load = QPushButton("Load X-ray image…")
        self.btn_load.clicked.connect(self.load_xray_image)

        load_layout.addWidget(self.load_info)
        load_layout.addWidget(self.btn_load)
        load_layout.addStretch(1)
        self.stack.addWidget(view_load)

        # -------- View B (Viewer) --------
        view_viewer = QWidget()
        viewer_layout = QVBoxLayout(view_viewer)
        viewer_layout.setSpacing(10)

        top_row = QHBoxLayout()
        self.path_label = QLabel("")
        self.path_label.setStyleSheet("color: #6c757d;")

        self.btn_reload = QPushButton("Load different X-ray image…")
        self.btn_reload.clicked.connect(self.load_xray_image)

        top_row.addWidget(self.path_label, stretch=1)
        top_row.addWidget(self.btn_reload)
        viewer_layout.addLayout(top_row)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #6c757d;")
        viewer_layout.addWidget(self.status_label)

        self.marker_widget = XrayMarkerSelectionWidget()
        self.marker_widget.selection_proposed.connect(self.on_selection_proposed)
        viewer_layout.addWidget(self.marker_widget, stretch=1)

        controls = QHBoxLayout()

        self.btn_detect = QPushButton("Detect markers")
        self.btn_detect.clicked.connect(self.on_detect_markers)

        self.btn_reset = QPushButton("Reset selection")
        self.btn_reset.clicked.connect(self.on_reset_selection)

        controls.addWidget(self.btn_detect)
        controls.addWidget(self.btn_reset)
        controls.addStretch(1)

        viewer_layout.addLayout(controls)
        self.stack.addWidget(view_viewer)

        self.refresh()

    # ---------------- Completion ----------------

    def is_complete(self) -> bool:
        return self.state.xray_points_confirmed

    # ---------------- UI helpers ----------------

    def _update_buttons(self):
        confirmed = self.state.xray_points_confirmed
        self.btn_detect.setEnabled(not confirmed)
        self.btn_reset.setEnabled(not confirmed)

    # ---------------- Refresh ----------------

    def refresh(self):
        if not self.state.has_xray_image:
            self.stack.setCurrentIndex(0)
            return

        self.stack.setCurrentIndex(1)
        self.path_label.setText(os.path.basename(self.state.xray_image_path or ""))

        # Only re-init widget if image/grid changed (prevents wiping overlays)
        render_key = (self.state.xray_image_path, bool(self.state.has_circles_grid))
        if render_key != self._last_render_key:
            self._last_render_key = render_key

            self.marker_widget.set_image(self.state.xray_image)
            if self.state.has_circles_grid:
                self.marker_widget.set_grid(self.state.circles_grid, float(self.state.pick_radius_px))
            else:
                self.marker_widget.set_grid(None)

        # Status
        if self.state.xray_points_confirmed:
            n = 0 if self.state.xray_points_uv is None else len(self.state.xray_points_uv)
            self.status_label.setText(f"Markers confirmed. {n} ROI markers selected. Ready.")
        else:
            if self.state.has_circles_grid:
                self.status_label.setText("Select 3 anchor markers (Left=select, Right=undo).")
            else:
                self.status_label.setText("Image loaded. Click 'Detect markers' before selecting.")

        self._update_buttons()

    # ---------------- Image loading ----------------

    def load_xray_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select X-ray image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        if not path:
            return

        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Could not read image (unsupported format or file error).")

            self.state.xray_image = img
            self.state.xray_image_path = path

            # Reset everything depending on this image
            self.state.circles_grid = None
            self.state.pick_radius_px = None
            self.state.xray_points_uv = None
            self.state.xray_points_confirmed = False

            self.marker_widget.clear_selection()
            self.marker_widget.set_locked(False)

            self._last_render_key = None
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

            # --- Hough params (tune if needed) ---
            params = HoughCircleParams(
                min_radius=3,
                max_radius=7,
                dp=1.2,
                minDist=26,
                param1=120,
                param2=8,
                invert=True,
                median_ks=5,
            )

            # --- preprocessing / mask params ---
            use_clahe = True
            clahe_clip = 2.0
            clahe_tiles = (12, 12)

            n_angles = 360
            smooth_sigma = 2.0
            r_min_frac = 0.20
            r_max_frac = 0.98
            shrink_px = 12

            row_tol_px = 13.0

            # CLAHE
            img_proc = img.copy()
            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tiles)
                img_proc = clahe.apply(img_proc)

            # Radial mask
            mask, _ = detector_mask_radial(
                img_proc,
                n_angles=n_angles,
                r_min_frac=r_min_frac,
                r_max_frac=r_max_frac,
                smooth_sigma=smooth_sigma,
                peak_prominence=0.0,
                shrink_px=shrink_px,
            )

            img_masked = img_proc.copy()
            img_masked[mask == 0] = 0

            # Detect circles
            circles_out = detect_blobs_hough(img_masked, params)
            if circles_out is None or len(circles_out) == 0:
                QMessageBox.warning(self, "Marker detection", "No circles detected.")
                return

            circles_out = np.asarray(circles_out, dtype=np.float32)

            # Sort into grid-like ordering
            circles_sorted = sort_circles_grid(circles_out, row_tol_px=row_tol_px)
            if circles_sorted is None or len(circles_sorted) < 10:
                QMessageBox.warning(self, "Marker detection", "Too few circles after sorting.")
                return

            # Group into rows
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
            ncols = int(len(rows[center_row_idx]))

            circles_grid = np.full((nrows, ncols, 3), np.nan, dtype=np.float32)
            for i, row in enumerate(rows):
                n_points = len(row)
                left_pad = (ncols - n_points) // 2
                padded_row = np.full((ncols, 3), np.nan, dtype=np.float32)
                padded_row[left_pad:left_pad + n_points] = row
                circles_grid[i] = padded_row

            # pick radius threshold
            radii = circles_grid[..., 2]
            finite_r = radii[np.isfinite(radii)]
            pick_radius_px = 0.6 * float(np.median(finite_r)) if finite_r.size else 20.0

            # Save to state
            self.state.circles_grid = circles_grid
            self.state.pick_radius_px = float(pick_radius_px)

            # Reset selection/confirmation
            self.state.xray_points_uv = None
            self.state.xray_points_confirmed = False

            self.marker_widget.clear_selection()
            self.marker_widget.set_locked(False)

            self._last_render_key = None
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
            # No -> clear and pick again
            self.state.xray_points_uv = None
            self.state.xray_points_confirmed = False
            self.marker_widget.clear_selection()
            self.marker_widget.set_locked(False)
            self.status_label.setText("Selection rejected. Please select 3 anchor markers again.")
            self._update_buttons()
            self.on_complete_changed()
            return

        # YES -> ROI rectangle in grid (min/max row/col)
        rs = [c[0] for c in cells]
        cs = [c[1] for c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)

        grid = self.state.circles_grid  # (R,C,3)
        roi_cells = []
        roi_uv = []

        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                x, y, _ = grid[rr, cc]
                if not np.isfinite(x):
                    continue
                roi_cells.append((rr, cc))
                roi_uv.append((float(x), float(y)))

        # Store ALL ROI markers
        self.state.xray_points_uv = np.asarray(roi_uv, dtype=float)
        self.state.xray_points_confirmed = True
        
        # ---------- DEBUG: verify pixel correctness ----------
        dbg = cv2.cvtColor(self.state.xray_image, cv2.COLOR_GRAY2BGR)
        for (u, v) in self.state.xray_points_uv:
            cv2.circle(
                dbg,
                (int(round(u)), int(round(v))),
                4,
                (0, 255, 255),
                2
            )
        cv2.imwrite("debug_roi_points.png", dbg)

        # Highlight ROI markers and lock UI
        self.marker_widget.set_roi_cells(roi_cells)
        self.marker_widget.set_locked(True)

        self.status_label.setText(f"Markers confirmed. {len(roi_uv)} ROI markers selected. Ready.")
        self._update_buttons()
        self.on_complete_changed()

    # ---------------- Reset ----------------

    def on_reset_selection(self):
        # Note: button is disabled when confirmed anyway
        self.state.xray_points_uv = None
        self.state.xray_points_confirmed = False

        self.marker_widget.clear_selection()
        self.marker_widget.set_locked(False)

        self.status_label.setText("Selection cleared. Select 3 anchor markers again.")
        self._update_buttons()
        self.on_complete_changed()
