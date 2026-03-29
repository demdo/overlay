# overlay/gui/pages/page_xray_marker_selection.py

from __future__ import annotations

import cv2
import numpy as np
import pydicom

from PySide6.QtWidgets import (
    QPushButton, QFileDialog, QMessageBox, QSizePolicy
)

from overlay.gui.state import SessionState
from overlay.gui.pages.templates.templ_static_image import StaticImagePage
from overlay.gui.widgets.widget_xray_marker_selection import XrayMarkerSelectionWidget

from overlay.tools.blob_detection import HoughCircleParams
from overlay.tools.xray_marker_selection import (
    run_xray_marker_detection,
    compute_roi_from_grid,
)


# ============================================================
# Display / coordinate helpers
# ============================================================

def _rot180_image(img: np.ndarray) -> np.ndarray:
    """
    Return a 180° rotated copy of the image for DISPLAY / UI only.

    Purpose
    -------
    This is only a viewing convenience so that the X-ray image appears in
    the orientation that is easier for manual marker selection.

    Important
    ---------
    The internal geometric state remains RAW:
    - state.xray_image stays raw
    - state.xray_points_uv stays raw
    - self._circles stays raw
    """
    return cv2.rotate(img, cv2.ROTATE_180)


def _uv_rot180_to_raw(uv_rot: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Map 2D points from the 180°-rotated display image back into RAW image coordinates.

    Parameters
    ----------
    uv_rot : (N, 2) array
        Pixel coordinates measured in the rotated display image.
    width, height : int
        Size of the RAW image.

    Returns
    -------
    uv_raw : (N, 2) array
        Equivalent pixel coordinates in the RAW image system.

    Notes
    -----
    For a 180° image rotation:
        u_raw = (width  - 1) - u_rot
        v_raw = (height - 1) - v_rot
    """
    uv_rot = np.asarray(uv_rot, dtype=np.float64).reshape(-1, 2)
    uv_raw = uv_rot.copy()
    uv_raw[:, 0] = (width - 1) - uv_rot[:, 0]
    uv_raw[:, 1] = (height - 1) - uv_rot[:, 1]
    return uv_raw


def _uv_raw_to_rot180(uv_raw: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Map 2D points from RAW image coordinates into the 180°-rotated display image.

    Parameters
    ----------
    uv_raw : (N, 2) array
        Pixel coordinates in the RAW image system.
    width, height : int
        Size of the RAW image.

    Returns
    -------
    uv_rot : (N, 2) array
        Equivalent pixel coordinates in the rotated display image.

    Notes
    -----
    For a 180° image rotation, the transform is its own inverse.
    """
    uv_raw = np.asarray(uv_raw, dtype=np.float64).reshape(-1, 2)
    uv_rot = uv_raw.copy()
    uv_rot[:, 0] = (width - 1) - uv_raw[:, 0]
    uv_rot[:, 1] = (height - 1) - uv_raw[:, 1]
    return uv_rot


class XrayMarkerSelectionPage(StaticImagePage):
    """
    Step — X-ray marker selection

    IMPORTANT:
    - MUST NOT add any new SessionState fields dynamically.
    - SessionState fields used here (already defined in state.py):
        * xray_image, xray_image_path
        * xray_points_uv, xray_points_confirmed
        * xray_marker_overlay_bgr
        * marker_radius_px
    - Everything else stays LOCAL to this page:
        * circles
        * pick_radius_px

    Coordinate convention in this page
    ----------------------------------
    - RAW image / RAW pixel coordinates are the internal source of truth.
    - The marker-selection widget shows a 180° rotated DISPLAY image.
    - Any detected / displayed points are converted between display and raw.
    """

    def __init__(self, state: SessionState, on_complete_changed, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

        # ---------------- local (page-only) runtime ----------------
        self._circles: np.ndarray | None = None       # (N,3), always RAW
        self._pick_radius_px: float | None = None     # interaction radius

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
        return self.state.has_xray_points_confirmed

    # ---------------- Refresh ----------------

    def refresh(self):
        img_raw = self.state.xray_image

        # --------------------------------------------------
        # Case 1: no image loaded yet
        # --------------------------------------------------
        if img_raw is None:
            self.marker_widget.set_image(None)
            self.marker_widget.set_circles(None)
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
        img_disp = _rot180_image(img_raw)
        self.marker_widget.set_image(img_disp)

        if self._circles is not None and self._pick_radius_px is not None:
            h, w = img_raw.shape[:2]
            circles_disp = np.asarray(self._circles, dtype=np.float64).copy()
            circles_disp[:, :2] = _uv_raw_to_rot180(circles_disp[:, :2], w, h)
            self.marker_widget.set_circles(circles_disp, float(self._pick_radius_px))
        else:
            self.marker_widget.set_circles(None)

        confirmed = bool(self.state.xray_points_confirmed)

        # Load: never again once image is loaded
        self.btn_load.setEnabled(False)

        # Detect: only once (unless you reload image)
        already_detected = self._circles is not None
        self.btn_detect.setEnabled(not already_detected)

        # Reset: only if at least 1 anchor selected
        selected_idx = self.marker_widget.get_selected_indices()
        self.btn_reset.setEnabled(len(selected_idx) > 0)

        # Lock interaction after confirmation
        self.marker_widget.set_locked(confirmed)

        # Stats
        if self._circles is None:
            detected_txt = "-"
            selected_txt = "- / 3"
        else:
            n_detected = int(len(self._circles))
            n_selected = len(selected_idx)
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
            # --------------------------------------------------
            # Load image
            # --------------------------------------------------
            if path.lower().endswith((".dcm", ".ima")):
                ds = pydicom.dcmread(path)
                img = ds.pixel_array.astype(np.float32)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = img.astype(np.uint8)
            else:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError("Could not read image.")

            # --------------------------------------------------
            # Store RAW image in EXISTING SessionState
            # --------------------------------------------------
            self.state.xray_image = img
            self.state.xray_image_path = path

            # reset dependent EXISTING state fields
            self.state.xray_points_uv = None
            self.state.xray_points_confirmed = False
            self.state.xray_marker_overlay_bgr = None
            self.state.marker_radius_px = None

            # --------------------------------------------------
            # Reset local (page-only)
            # --------------------------------------------------
            self._circles = None
            self._pick_radius_px = None

            self.marker_widget.clear_selection()
            self.marker_widget.set_roi_indices([])
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
            img_raw = self.state.xray_image
            h, w = img_raw.shape[:2]
            img_disp = _rot180_image(img_raw)

            params = HoughCircleParams(
                min_radius=2,
                max_radius=7,
                dp=1.2,
                minDist=8,
                param1=120,
                param2=9,
                invert=True,
                median_ks=(3, 5),
            )

            # Detection runs on the DISPLAY image because that is the
            # orientation in which the user selects anchors.
            res = run_xray_marker_detection(
                img_disp,
                hough_params=params,
                use_clahe=True,
                clahe_clip=2.0,
                clahe_tiles=(12, 12),
                use_mask=False,
            )

            if res.circles is None or len(res.circles) == 0:
                QMessageBox.warning(self, "Marker detection", "No circles detected.")
                return

            circles_rot = np.asarray(res.circles, dtype=np.float64).reshape(-1, 3)

            # Convert detected circle centers back to RAW coordinates.
            circles_raw = circles_rot.copy()
            circles_raw[:, :2] = _uv_rot180_to_raw(circles_rot[:, :2], w, h)

            radii = circles_raw[:, 2]
            finite_r = radii[np.isfinite(radii)]
            if finite_r.size:
                self.state.marker_radius_px = float(np.median(finite_r))
            else:
                self.state.marker_radius_px = None

            if self.state.marker_radius_px is not None:
                pick_radius_px = 0.6 * float(self.state.marker_radius_px)
            else:
                pick_radius_px = 20.0

            # Store circles in RAW coordinates internally.
            self._circles = circles_raw
            self._pick_radius_px = float(pick_radius_px)

            # reset selection/confirmation + overlay
            self.state.xray_points_uv = None
            self.state.xray_points_confirmed = False
            self.state.xray_marker_overlay_bgr = None

            self.marker_widget.clear_selection()
            self.marker_widget.set_roi_indices([])
            self.marker_widget.set_locked(False)

            self.refresh()
            self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(self, "Marker detection failed", str(e))

    # ---------------- Confirm flow ----------------

    def on_selection_proposed(self, selected_idx):
        if selected_idx is None or len(selected_idx) != 3:
            return

        selected_idx = [int(k) for k in selected_idx]

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
            self.marker_widget.set_roi_indices([])
            self.marker_widget.set_locked(False)
            self.refresh()
            self.on_complete_changed()
            return

        if self._circles is None or self._pick_radius_px is None:
            self.marker_widget.clear_selection()
            self.marker_widget.set_roi_indices([])
            self.marker_widget.set_locked(False)
            self.refresh()
            self.on_complete_changed()
            return

        margin_px = 1.1 * float(self._pick_radius_px)

        # self._circles are RAW, so roi_uv is RAW as well.
        roi_uv, roi_idx, _dbg = compute_roi_from_grid(
            circles=self._circles,
            anchor_idx=selected_idx,
            margin_px=margin_px,
            gate_tol_pitch=0.40,
            min_steps=2,
        )

        self.state.xray_points_uv = np.asarray(roi_uv, dtype=float)
        self.state.xray_points_confirmed = True
        
        # ============================================================
        # DEBUG SAVE: store correspondences
        # ============================================================
        
        xyz = self.state.xray_points_xyz_c
        uv = self.state.xray_points_uv
        
        if xyz is not None and uv is not None:
            xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
            uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        
            if xyz.shape[0] == uv.shape[0]:
                out_path = r"C:\Users\domin\Documents\Studium\Master\Masterarbeit\Projekt\Data\pose_debug.npz"
        
                np.savez(
                    out_path,
                    points_xyz=xyz,
                    points_uv=uv,
                )
        
                print("Saved correspondences to:", out_path)
            else:
                print("WARNING: xyz / uv mismatch:", xyz.shape, uv.shape)
        
        
        

        self.marker_widget.set_roi_indices(list(np.asarray(roi_idx, dtype=np.int64).tolist()))
        self.marker_widget.set_locked(True)

        self.state.xray_marker_overlay_bgr = self._render_roi_overlay_bgr()

        self.refresh()
        self.on_complete_changed()

    # ---------------- Reset ----------------

    def on_reset_selection(self):
        self.state.xray_points_uv = None
        self.state.xray_points_confirmed = False
        self.state.xray_marker_overlay_bgr = None

        self.marker_widget.clear_selection()
        self.marker_widget.set_roi_indices([])
        self.marker_widget.set_locked(False)

        self.refresh()
        self.on_complete_changed()

    # ---------------- Helpers ----------------

    def _render_roi_overlay_bgr(self) -> np.ndarray | None:
        """
        Render full-size RAW X-ray image with only RAW ROI overlay:
        - green circles
        - red crosses

        Notes
        -----
        This overlay is intentionally rendered in the RAW image coordinate
        system because the downstream calibration page also operates on the
        RAW X-ray image.
        """
        img = self.state.xray_image
        uv = self.state.xray_points_uv

        if img is None or uv is None or len(uv) == 0:
            return None

        img8 = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

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