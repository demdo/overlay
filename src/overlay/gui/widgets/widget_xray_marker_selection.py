# overlay/gui/widgets/widget_xray_marker_selection.py

from __future__ import annotations

from typing import List, Optional, Tuple, Set

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget


def _qpixmap_from_gray_u8(img_gray_u8: np.ndarray) -> QPixmap:
    """img_gray_u8: (H,W) uint8."""
    h, w = img_gray_u8.shape
    qimg = QImage(img_gray_u8.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


class XrayMarkerSelectionWidget(QWidget):
    """
    Embedded marker selection inside the GUI (no separate window).

    - Fit-to-view image display
    - Shows all markers from circles
    - Select 3 anchor markers:
        Left click: select
        Right click: undo last
    - After 3 anchors: emits selection_proposed to the page
    - Page confirms YES/NO
        YES: page calls set_roi_indices(...) and set_locked(True)
        NO:  page calls clear_selection()
    """

    selection_proposed = Signal(list)  # List[int] (3 anchor indices)
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        # Image
        self._img_gray_u8: Optional[np.ndarray] = None
        self._pix: Optional[QPixmap] = None
        self._img_w: int = 0
        self._img_h: int = 0

        # Fit-to-view transform (image -> widget)
        self._scale: float = 1.0
        self._off_x: int = 0
        self._off_y: int = 0

        # Circle picking
        self._circles: Optional[np.ndarray] = None
        self._prepared = None
        self._pick_radius_px: float = 20.0

        # Anchor selection overlays
        self._selected_idx: List[int] = []
        self._highlight_idx: Set[int] = set()
        self._hover_idx: Optional[int] = None

        # ROI highlighting after confirmation
        self._roi_idx: Set[int] = set()

        # Lock interaction after confirmation
        self._locked: bool = False

    # ---------------- Picking helpers ----------------

    def _prepare_nearest_index_data(self, circles: np.ndarray):
        if circles is None:
            return None

        c = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
        xy = c[:, :2]
        finite = np.isfinite(xy).all(axis=1)

        if not np.any(finite):
            return None

        idx = np.flatnonzero(finite).astype(np.int32)
        xy = xy[finite].astype(np.float32)
        return idx, xy

    def _nearest_index(self, prepared, x: float, y: float):
        if prepared is None:
            return None, float("inf")

        idx, xy = prepared
        q = np.array([float(x), float(y)], dtype=np.float32)
        d2 = np.sum((xy - q) ** 2, axis=1)
        k = int(np.argmin(d2))
        dist = float(np.sqrt(d2[k]))
        return int(idx[k]), dist

    # ---------------- Public API ----------------

    def set_image(self, img_gray_u8: np.ndarray):
        if img_gray_u8 is None:
            self._img_gray_u8 = None
            self._pix = None
            self._img_w = 0
            self._img_h = 0
            self.update()
            return

        if img_gray_u8.dtype != np.uint8:
            img_gray_u8 = np.clip(img_gray_u8, 0, 255).astype(np.uint8)

        self._img_gray_u8 = img_gray_u8
        self._pix = _qpixmap_from_gray_u8(img_gray_u8)
        self._img_h, self._img_w = img_gray_u8.shape[:2]
        self.update()

    def set_circles(self, circles: Optional[np.ndarray], pick_radius_px: Optional[float] = None):
        self._circles = circles
        if circles is None:
            self._prepared = None
        else:
            self._prepared = self._prepare_nearest_index_data(circles)

        if pick_radius_px is not None:
            self._pick_radius_px = float(pick_radius_px)

        self.update()

    def set_data(self, img_gray_u8: np.ndarray, circles: Optional[np.ndarray], pick_radius_px: Optional[float]):
        self.set_image(img_gray_u8)
        self.set_circles(circles, pick_radius_px)

    def clear_selection(self):
        self._selected_idx.clear()
        self._highlight_idx.clear()
        self._hover_idx = None
        self._roi_idx.clear()
        self._locked = False
        self.update()
        self.selection_changed.emit()

    def set_locked(self, locked: bool):
        self._locked = bool(locked)
        self.update()

    def set_roi_indices(self, indices):
        self._roi_idx = set(int(k) for k in indices) if indices is not None else set()
        self.update()

    # ---------------- Internal helpers ----------------

    def _widget_to_image_xy(self, xw: int, yw: int) -> Tuple[int, int]:
        xi = (xw - self._off_x) / (self._scale + 1e-12)
        yi = (yw - self._off_y) / (self._scale + 1e-12)
        return int(round(xi)), int(round(yi))

    def _is_inside_image_display(self, xw: int, yw: int) -> bool:
        if self._pix is None:
            return False
        disp_w = int(round(self._img_w * self._scale))
        disp_h = int(round(self._img_h * self._scale))
        return (self._off_x <= xw <= self._off_x + disp_w) and (self._off_y <= yw <= self._off_y + disp_h)

    # ---------------- Qt painting ----------------

    def paintEvent(self, event):
        if self._pix is None:
            return

        painter = QPainter(self)

        H, W = self._img_h, self._img_w
        vw, vh = self.width(), self.height()

        sx = vw / max(1, W)
        sy = vh / max(1, H)
        self._scale = min(sx, sy)

        disp_w = int(round(W * self._scale))
        disp_h = int(round(H * self._scale))
        self._off_x = (vw - disp_w) // 2
        self._off_y = (vh - disp_h) // 2

        scaled_pix = self._pix.scaled(disp_w, disp_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(self._off_x, self._off_y, scaled_pix)

        if self._circles is None:
            painter.end()
            return

        circles = np.asarray(self._circles, dtype=np.float64).reshape(-1, 3)

        r_values = circles[:, 2]
        finite_r = r_values[np.isfinite(r_values)]

        if finite_r.size > 0:
            r_med = float(np.median(finite_r))
        else:
            r_med = 5.0  # fallback

        rd_display = int(round(r_med * self._scale))
        rd_display = max(1, rd_display)

        # thickness proportional to image scaling
        t_normal = max(1, int(round(2 * self._scale)))
        t_anchor = max(1, int(round(4 * self._scale)))
        t_roi = max(1, int(round(3 * self._scale)))
        t_cross = max(1, int(round(1 * self._scale)))
        t_hover = max(1, int(round(2.0 * self._scale)))

        pen_normal = QPen(QColor(0, 255, 0), t_normal)
        pen_anchor = QPen(QColor(0, 255, 255), t_anchor)
        pen_roi = QPen(QColor(255, 165, 0), t_roi)
        pen_cross = QPen(QColor(255, 0, 0), t_cross)
        pen_hover = QPen(QColor(255, 255, 0), t_hover)

        for k in range(circles.shape[0]):
            x, y, r = circles[k]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            xd = int(round(self._off_x + float(x) * self._scale))
            yd = int(round(self._off_y + float(y) * self._scale))
            rd = rd_display

            if k in self._highlight_idx:
                pen = pen_anchor
            elif k in self._roi_idx:
                pen = pen_roi
            else:
                pen = pen_normal

            painter.setPen(pen)
            painter.drawEllipse(xd - rd, yd - rd, 2 * rd, 2 * rd)

            painter.setPen(pen_cross)
            cross = int(np.clip(0.35 * rd, 2, 6))
            painter.drawLine(xd - cross, yd, xd + cross, yd)
            painter.drawLine(xd, yd - cross, xd, yd + cross)

        if (not self._locked) and (self._hover_idx is not None):
            x, y, _ = circles[self._hover_idx]
            if np.isfinite(x):
                xd = int(round(self._off_x + float(x) * self._scale))
                yd = int(round(self._off_y + float(y) * self._scale))

                # Use pick radius (consistent with click threshold)
                rd = int(round(self._pick_radius_px * self._scale))
                rd = max(2, rd)

                painter.setPen(pen_hover)
                painter.drawEllipse(xd - rd, yd - rd, 2 * rd, 2 * rd)

        painter.end()

    # ---------------- Qt mouse interaction ----------------

    def mousePressEvent(self, event):
        if self._circles is None or self._prepared is None:
            return

        if self._locked:
            return  # no interaction after confirmation

        xw = int(event.position().x())
        yw = int(event.position().y())
        if not self._is_inside_image_display(xw, yw):
            return

        x, y = self._widget_to_image_xy(xw, yw)

        if event.button() == Qt.LeftButton:
            idx, dist = self._nearest_index(self._prepared, x, y)
            if idx is None or dist > self._pick_radius_px:
                return
            if idx in self._highlight_idx:
                return

            self._selected_idx.append(idx)
            self._highlight_idx.add(idx)
            self.update()
            self.selection_changed.emit()

            if len(self._selected_idx) == 3:
                self.selection_proposed.emit(list(self._selected_idx))

        elif event.button() == Qt.RightButton:
            if self._selected_idx:
                last = self._selected_idx.pop()
                self._highlight_idx.discard(last)
                self.update()
                self.selection_changed.emit()

    def mouseMoveEvent(self, event):
        if self._locked:
            return

        if self._circles is None or self._prepared is None:
            return

        xw = int(event.position().x())
        yw = int(event.position().y())

        if not self._is_inside_image_display(xw, yw):
            if self._hover_idx is not None:
                self._hover_idx = None
                self.update()
            return

        x, y = self._widget_to_image_xy(xw, yw)
        idx, dist = self._nearest_index(self._prepared, x, y)

        new_hover = idx if (idx is not None and dist <= self._pick_radius_px) else None
        if new_hover != self._hover_idx:
            self._hover_idx = new_hover
            self.update()

    def get_selected_indices(self):
        return list(self._selected_idx)