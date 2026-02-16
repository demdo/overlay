# overlay/gui/pages/page_xray_marker_selection.py

from __future__ import annotations

import os
import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QGroupBox, QGridLayout, QSizePolicy
)

from overlay.gui.state import SessionState
from overlay.gui.widgets.widget_xray_marker_selection import XrayMarkerSelectionWidget

from overlay.tools.xray_marker_selection import detector_mask_radial, sort_circles_grid
from overlay.tools.blob_detection import HoughCircleParams, detect_blobs_hough


class XrayMarkerSelectionPage(QWidget):
    """
    Step — X-ray marker selection

    Goals:
      - Control bar RIGHT aligned like other pages
      - X-ray image size "like before": nice 960×540 workspace (max), NEVER clipped
      - Before loading: white workspace (no instructions)
      - NO logic changes (only layout + robust refresh gating)
    """

    # "nice size like before"
    MAX_W = 960
    MAX_H = 540

    RIGHT_W = 260
    MARGIN = 20
    SPACING = 20

    def __init__(self, state: SessionState, on_complete_changed):
        super().__init__()
        self.state = state
        self.on_complete_changed = on_complete_changed

        # ======================================================
        # LEFT: marker selection widget (key sizing behavior)
        # ======================================================
        self.marker_widget = XrayMarkerSelectionWidget()
        self.marker_widget.selection_proposed.connect(self.on_selection_proposed)

        # IMPORTANT FIX:
        # - allow shrinking -> prevents clipping
        # - cap growth -> keeps the "nice" size
        self.marker_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.marker_widget.setMaximumSize(self.MAX_W, self.MAX_H)
        self.marker_widget.resize(self.MAX_W, self.MAX_H)

        # white before image
        self.marker_widget.setStyleSheet("background-color: white; border-radius: 10px;")

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # anchor top-left
        left_layout.addWidget(self.marker_widget, 0, Qt.AlignLeft | Qt.AlignTop)
        left_layout.addStretch(1)

        # ======================================================
        # RIGHT: control bar (same geometry as other pages)
        # ======================================================
        self.btn_load = QPushButton("Load different X-ray image…")
        self.btn_load.clicked.connect(self.load_xray_image)

        self.btn_detect = QPushButton("Detect markers")
        self.btn_detect.clicked.connect(self.on_detect_markers)

        self.btn_reset = QPushButton("Reset selection")
        self.btn_reset.clicked.connect(self.on_reset_selection)

        for b in (self.btn_load, self.btn_detect, self.btn_reset):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        controls = QGroupBox("Controls")
        g = QGridLayout(controls)
        g.setContentsMargins(8, 8, 8, 8)
        g.setSpacing(8)
        g.addWidget(self.btn_load, 0, 0, 1, 2)
        g.addWidget(self.btn_detect, 1, 0)
        g.addWidget(self.btn_reset, 1, 1)
        g.setColumnStretch(0, 1)
        g.setColumnStretch(1, 1)

        self.path_label = QLabel("")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #6c757d;")

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #6c757d;")

        status_box = QGroupBox("Status")
        sv = QVBoxLayout(status_box)
        sv.setContentsMargins(8, 8, 8, 8)
        sv.setSpacing(8)
        sv.addWidget(self.path_label)
        sv.addWidget(self.status_label)

        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        right_layout.setSpacing(12)
        right_layout.addWidget(controls)
        right_layout.addWidget(status_box)
        right_layout.addStretch(1)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)
        right_panel.setFixedWidth(self.RIGHT_W)
        right_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        # ======================================================
        # MAIN LAYOUT (matches others)
        # ======================================================
        main = QHBoxLayout(self)
        main.setContentsMargins(self.MARGIN, self.MARGIN, self.MARGIN, self.MARGIN)
        main.setSpacing(self.SPACING)

        # left stretch=1 ensures control bar aligns like other pages
        main.addWidget(left_container, 1)
        main.addWidget(right_panel, 0)

        self.refresh()

    # ---------------- Completion ----------------

    def is_complete(self) -> bool:
        return bool(getattr(self.state, "xray_points_confirmed", False))

    # ---------------- Refresh ----------------

    def refresh(self):
        img = getattr(self.state, "xray_image", None)

        if img is None:
            # White empty workspace
            self.marker_widget.set_image(None)
            self.marker_widget.set_grid(None)
            self.marker_widget.set_locked(False)

            self.path_label.setText("")
            self.status_label.setText("")

            self.btn_detect.setEnabled(False)
            self.btn_reset.setEnabled(False)
            return

        # Show image
        self.marker_widget.set_image(img)

        circles_grid = getattr(self.state, "circles_grid", None)
        pick_r = getattr(self.state, "pick_radius_px", None)
        if circles_grid is not None and pick_r is not None:
            self.marker_widget.set_grid(circles_grid, float(pick_r))
        else:
            self.marker_widget.set_grid(None)

        # Status
        self.path_label.setText(os.path.basename(getattr(self.state, "xray_image_path", "") or ""))

        confirmed = bool(getattr(self.state, "xray_points_confirmed", False))
        if confirmed:
            n = 0 if getattr(self.state, "xray_points_uv", None) is None else len(self.state.xray_points_uv)
            self.status_label.setText(f"Markers confirmed. {n} ROI markers selected. Ready.")
            self.btn_detect.setEnabled(False)
            self.btn_reset.setEnabled(False)
            self.marker_widget.set_locked(True)
        else:
            if circles_grid is None:
                self.status_label.setText("Image loaded. Click 'Detect markers'.")
                self.btn_detect.setEnabled(True)
                self.btn_reset.setEnabled(False)
            else:
                self.status_label.setText("Select 3 anchor markers (Left=select, Right=undo).")
                self.btn_detect.setEnabled(True)
                self.btn_reset.setEnabled(True)
            self.marker_widget.set_locked(False)

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
                raise ValueError("Could not read image.")

            self.state.xray_image = img
            self.state.xray_image_path = path

            # reset dependent states (unchanged)
            self.state.circles_grid = None
            self.state.pick_radius_px = None
            self.state.xray_points_uv = None
            self.state.xray_points_confirmed = False

            self.marker_widget.clear_selection()
            self.marker_widget.set_locked(False)

            self.refresh()
            self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(self, "Failed to load X-ray image", str(e))

    # ---------------- Marker detection ----------------

    def on_detect_markers(self):
        if getattr(self.state, "xray_image", None) is None:
            return

        try:
            img = self.state.xray_image

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

            use_clahe = True
            clahe_clip = 2.0
            clahe_tiles = (12, 12)

            n_angles = 360
            smooth_sigma = 2.0
            r_min_frac = 0.20
            r_max_frac = 0.98
            shrink_px = 12

            row_tol_px = 13.0

            img_proc = img.copy()
            if use_clahe:
                clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tiles)
                img_proc = clahe.apply(img_proc)

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

            circles_out = detect_blobs_hough(img_masked, params)
            if circles_out is None or len(circles_out) == 0:
                QMessageBox.warning(self, "Marker detection", "No circles detected.")
                return

            circles_out = np.asarray(circles_out, dtype=np.float32)
            circles_sorted = sort_circles_grid(circles_out, row_tol_px=row_tol_px)
            if circles_sorted is None or len(circles_sorted) < 10:
                QMessageBox.warning(self, "Marker detection", "Too few circles after sorting.")
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
                    rows.append(current[np.argsort(current[:, 0])])
                    current = pt.reshape(1, -1)
                    y_ref = float(pt[1])
                else:
                    current = np.vstack([current, pt])
                    y_ref = float(np.mean(current[:, 1]))

            rows.append(current[np.argsort(current[:, 0])])

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

            radii = circles_grid[..., 2]
            finite_r = radii[np.isfinite(radii)]
            pick_radius_px = 0.6 * float(np.median(finite_r)) if finite_r.size else 20.0

            self.state.circles_grid = circles_grid
            self.state.pick_radius_px = float(pick_radius_px)

            # reset selection/confirmation (unchanged)
            self.state.xray_points_uv = None
            self.state.xray_points_confirmed = False

            self.marker_widget.clear_selection()
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
            self.marker_widget.clear_selection()
            self.marker_widget.set_locked(False)
            self.refresh()
            self.on_complete_changed()
            return

        rs = [c[0] for c in cells]
        cs = [c[1] for c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)

        grid = self.state.circles_grid
        roi_cells = []
        roi_uv = []

        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                x, y, _ = grid[rr, cc]
                if not np.isfinite(x):
                    continue
                roi_cells.append((rr, cc))
                roi_uv.append((float(x), float(y)))

        self.state.xray_points_uv = np.asarray(roi_uv, dtype=float)
        self.state.xray_points_confirmed = True

        self.marker_widget.set_roi_cells(roi_cells)
        self.marker_widget.set_locked(True)

        self.refresh()
        self.on_complete_changed()

    # ---------------- Reset ----------------

    def on_reset_selection(self):
        self.state.xray_points_uv = None
        self.state.xray_points_confirmed = False
        self.marker_widget.clear_selection()
        self.marker_widget.set_locked(False)
        self.refresh()
        self.on_complete_changed()
