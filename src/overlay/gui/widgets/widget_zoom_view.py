# overlay/gui/widgets/widget_zoom_view.py
#
# Square drag-zoom view for X-ray images (e.g. 1024x1024)
# - Always keeps a SQUARE crop in image coordinates
# - Always displays the crop in a centered SQUARE viewport inside the widget
# - Overlays (ROI circles + measured/projected crosses + optional residual lines)
#   are drawn in IMAGE space and therefore SCALE WITH ZOOM.
#
# Interactions:
#   - LMB drag: draw a SQUARE box -> zoom into that region
#   - RMB click OR double-click: reset zoom
#
# Signals:
#   - zoom_changed(bool): emitted when zoom state changes (zoomed vs not zoomed)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QPainter,
    QImage,
    QPixmap,
    QColor,
    QPen,
    QBrush,
)
from PySide6.QtWidgets import QWidget


def _np_to_qimage(img: np.ndarray) -> QImage:
    """Convert uint8 grayscale or BGR uint8 numpy image to QImage (deep copy)."""
    if img is None:
        return QImage()

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        h, w = img.shape
        q = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        return q.copy()

    if img.ndim == 3 and img.shape[2] == 3:
        # assume BGR (OpenCV default) -> RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        q = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        return q.copy()

    raise ValueError(f"Unsupported image shape for QImage conversion: {img.shape}")


@dataclass
class _OverlayData:
    uv_roi: Optional[np.ndarray] = None         # (N,2)
    uv_measured: Optional[np.ndarray] = None    # (N,2)
    uv_projected: Optional[np.ndarray] = None   # (N,2)
    show_residuals: bool = False
    outlier_mask: Optional[np.ndarray] = None

    # kept for API compatibility (not used for cross length anymore)
    pick_radius_px: Optional[float] = None

    # true blob radius in IMAGE px (median detected)
    marker_radius_px: Optional[float] = None


class ZoomView(QWidget):
    zoom_changed = Signal(bool)  # True if currently zoomed (crop != base crop)

    def __init__(self, parent=None):
        super().__init__(parent)

        # stored as uint8 grayscale (for geometry/cropping)
        self._img_gray_u8: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None

        # Crops in IMAGE coords: QRectF(x, y, s, s) square
        self._base_crop: Optional[QRectF] = None
        self._crop: Optional[QRectF] = None

        # Selection rectangle in VIEW coords (inside square viewport)
        self._dragging = False
        self._sel_start: Optional[QPointF] = None
        self._sel_rect_view: Optional[QRectF] = None

        self._overlay = _OverlayData()

        # tuneables
        self._min_sel_px = 12.0  # min selection size in view pixels

        # visuals
        self._bg = QColor(12, 12, 12)
        self._viewport_bg = QColor(0, 0, 0)

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def set_image(self, img: Optional[np.ndarray]) -> None:
        """
        Set base image.
        Accepts grayscale or BGR.
        - Internally stores grayscale uint8 for geometry/cropping.
        - Displays the ORIGINAL image (color if provided).
        Resets zoom to centered square crop.
        """
        if img is None:
            self._img_gray_u8 = None
            self._pixmap = None
            self._base_crop = None
            self._crop = None
            self._sel_rect_view = None
            self._emit_zoom_changed()
            self.update()
            return

        arr = np.asarray(img)

        # --- build grayscale for crop geometry ---
        if arr.ndim == 3 and arr.shape[2] == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            gray = arr

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        self._img_gray_u8 = gray
        h, w = gray.shape[:2]

        # centered square base crop
        s = float(min(w, h))
        x0 = float((w - s) * 0.5)
        y0 = float((h - s) * 0.5)
        self._base_crop = QRectF(x0, y0, s, s)
        self._crop = QRectF(self._base_crop)

        # --- build DISPLAY pixmap from ORIGINAL (keep color if available) ---
        if arr.ndim == 3 and arr.shape[2] == 3:
            disp = arr
            if disp.dtype != np.uint8:
                disp = np.clip(disp, 0, 255).astype(np.uint8)
            qimg = _np_to_qimage(disp)   # BGR -> RGB inside helper
        else:
            qimg = _np_to_qimage(gray)

        self._pixmap = QPixmap.fromImage(qimg)

        self._sel_rect_view = None
        self._emit_zoom_changed()
        self.update()

    def set_overlay_data(
        self,
        uv_measured=None,
        uv_projected=None,
        uv_roi=None,
        pick_radius_px: Optional[float] = None,
        marker_radius_px: Optional[float] = None,
        show_residuals: bool = False,
        outlier_mask: Optional[np.ndarray] = None,
    ) -> None:
        self._overlay = _OverlayData(
            uv_roi=None if uv_roi is None else np.asarray(uv_roi, dtype=np.float64).reshape(-1, 2),
            uv_measured=None if uv_measured is None else np.asarray(uv_measured, dtype=np.float64).reshape(-1, 2),
            uv_projected=None if uv_projected is None else np.asarray(uv_projected, dtype=np.float64).reshape(-1, 2),
            show_residuals=bool(show_residuals),
            pick_radius_px=None if pick_radius_px is None else float(pick_radius_px),
            marker_radius_px=None if marker_radius_px is None else float(marker_radius_px),
            outlier_mask=None if outlier_mask is None else np.asarray(outlier_mask, dtype=bool),
        )
        self.update()

    def clear_overlay_data(self) -> None:
        self._overlay = _OverlayData()
        self.update()

    def reset_zoom(self) -> None:
        if self._base_crop is None:
            return
        self._crop = QRectF(self._base_crop)
        self._sel_rect_view = None
        self._emit_zoom_changed()
        self.update()

    def has_zoom(self) -> bool:
        if self._base_crop is None or self._crop is None:
            return False
        return (
            abs(self._crop.left() - self._base_crop.left()) > 1e-6
            or abs(self._crop.top() - self._base_crop.top()) > 1e-6
            or abs(self._crop.width() - self._base_crop.width()) > 1e-6
        )

    # ---------------------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------------------

    def _viewport_square(self) -> QRectF:
        """Centered square viewport in widget coordinates."""
        w = float(max(1, self.width()))
        h = float(max(1, self.height()))
        side = float(min(w, h))
        x = (w - side) * 0.5
        y = (h - side) * 0.5
        return QRectF(x, y, side, side)

    def _clamp_point_to_viewport(self, p: QPointF, vp: QRectF) -> QPointF:
        x = min(max(p.x(), vp.left()), vp.right())
        y = min(max(p.y(), vp.top()), vp.bottom())
        return QPointF(x, y)

    def _make_square_rect_from_two_points(self, a: QPointF, b: QPointF, vp: QRectF) -> QRectF:
        """Build a square QRectF inside vp with corner at a and extending toward b."""
        ax, ay = a.x(), a.y()
        bx, by = b.x(), b.y()

        dx = bx - ax
        dy = by - ay
        s = max(abs(dx), abs(dy))

        sx = 1.0 if dx >= 0 else -1.0
        sy = 1.0 if dy >= 0 else -1.0

        x0 = ax
        y0 = ay
        x1 = ax + sx * s
        y1 = ay + sy * s

        left = min(x0, x1)
        top = min(y0, y1)
        right = max(x0, x1)
        bottom = max(y0, y1)

        # shift into viewport if needed
        if left < vp.left():
            shift = vp.left() - left
            left += shift
            right += shift
        if right > vp.right():
            shift = right - vp.right()
            left -= shift
            right -= shift
        if top < vp.top():
            shift = vp.top() - top
            top += shift
            bottom += shift
        if bottom > vp.bottom():
            shift = bottom - vp.bottom()
            top -= shift
            bottom -= shift

        # final clamp
        left = min(max(left, vp.left()), vp.right())
        top = min(max(top, vp.top()), vp.bottom())
        side = min(vp.right() - left, vp.bottom() - top, right - left, bottom - top)
        return QRectF(left, top, side, side)

    def _view_to_image(self, x_view: float, y_view: float, vp: QRectF) -> tuple[float, float]:
        """Map a point from view coords (within vp) to image coords (within current crop)."""
        if self._crop is None:
            return 0.0, 0.0
        rel_x = (x_view - vp.left()) / max(1e-9, vp.width())
        rel_y = (y_view - vp.top()) / max(1e-9, vp.height())
        u = self._crop.left() + rel_x * self._crop.width()
        v = self._crop.top() + rel_y * self._crop.height()
        return float(u), float(v)

    def _image_to_view(self, u: float, v: float, vp: QRectF) -> Optional[QPointF]:
        """Map image coords to view coords. Returns None if outside current crop."""
        if self._crop is None:
            return None
        if u < self._crop.left() or u > self._crop.right() or v < self._crop.top() or v > self._crop.bottom():
            return None
        rel_x = (u - self._crop.left()) / max(1e-9, self._crop.width())
        rel_y = (v - self._crop.top()) / max(1e-9, self._crop.height())
        x = vp.left() + rel_x * vp.width()
        y = vp.top() + rel_y * vp.height()
        return QPointF(float(x), float(y))

    def _scale_view_per_image_px(self, vp: QRectF) -> float:
        """How many VIEW pixels correspond to 1 IMAGE pixel within current crop."""
        if self._crop is None:
            return 1.0
        return float(vp.width() / max(1e-9, self._crop.width()))

    # ---------------------------------------------------------------------
    # Painting
    # ---------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)

        # Smooth image scaling (pixmap). Overlay should look "pixel/crisp".
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        painter.fillRect(self.rect(), self._bg)

        vp = self._viewport_square()
        painter.fillRect(vp, self._viewport_bg)

        if self._pixmap is None or self._img_gray_u8 is None or self._crop is None:
            if self._sel_rect_view is not None:
                self._draw_selection(painter)
            return

        # draw cropped image into square viewport
        crop = self._crop
        src = QRectF(crop.left(), crop.top(), crop.width(), crop.height())
        painter.drawPixmap(vp, self._pixmap, src)

        # overlays: match marker selection look (NO antialias)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        self._draw_overlay_image(painter, vp)
        painter.restore()

        # selection rect
        if self._sel_rect_view is not None:
            self._draw_selection(painter)

    def _draw_selection(self, painter: QPainter) -> None:
        if self._sel_rect_view is None:
            return
        pen = QPen(QColor(255, 255, 255), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self._sel_rect_view)

        fill = QColor(255, 255, 255, 30)
        painter.fillRect(self._sel_rect_view, QBrush(fill))

    def _draw_overlay_image(self, painter: QPainter, vp: QRectF) -> None:
        ov = self._overlay
        if ov is None or self._crop is None:
            return
    
        scale = self._scale_view_per_image_px(vp)
    
        # ---- Circle radius (IMAGE px) ----
        # Use true blob radius if provided. Fallback to 6 px if missing.
        circle_r_img = float(max(1.0, ov.marker_radius_px)) if ov.marker_radius_px is not None else 6.0
    
        # Circle radius in VIEW px (for drawing)
        r_circle_view = float(circle_r_img * scale)
    
        # ---- Cross half-length in VIEW px (match marker selection widget) ----
        # cross = clip(0.35 * rd_display, 2, 6)
        cross_half_view = float(np.clip(0.35 * r_circle_view, 2.0, 6.0))
    
        # ---- Pen widths: constant in VIEW px (do not scale with zoom) ----
        green_roi_pen = QPen(QColor(0, 255, 0), 2)
        orange_roi_pen = QPen(QColor(255, 165, 0), 2)  # outlier ROI circles
        meas_pen = QPen(QColor(255, 0, 0), 1)          # thin = avoids "filled diamond"
        proj_pen = QPen(QColor(0, 255, 255), 1)        # thin
        res_pen = QPen(QColor(255, 255, 255), 1)
    
        for p in (green_roi_pen, orange_roi_pen, meas_pen, proj_pen, res_pen):
            p.setCapStyle(Qt.SquareCap)
            p.setJoinStyle(Qt.MiterJoin)
    
        # ------------------------------------------------------------
        # Residuals first (behind)
        # ------------------------------------------------------------
        if ov.show_residuals and ov.uv_measured is not None and ov.uv_projected is not None:
            painter.setPen(res_pen)
            n = min(len(ov.uv_measured), len(ov.uv_projected))
            for i in range(n):
                u1, v1 = ov.uv_measured[i]
                u2, v2 = ov.uv_projected[i]
                if not (np.isfinite(u1) and np.isfinite(v1) and np.isfinite(u2) and np.isfinite(v2)):
                    continue
                p1 = self._image_to_view(float(u1), float(v1), vp)
                p2 = self._image_to_view(float(u2), float(v2), vp)
                if p1 is None or p2 is None:
                    continue
    
                # snap to integer pixels for crisp look
                x1, y1 = float(int(round(p1.x()))), float(int(round(p1.y())))
                x2, y2 = float(int(round(p2.x()))), float(int(round(p2.y())))
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    
        # ------------------------------------------------------------
        # ROI circles (inlier/outlier coloring)
        # ------------------------------------------------------------
        if ov.uv_roi is not None:
            painter.setBrush(Qt.NoBrush)
            for i, (u, v) in enumerate(ov.uv_roi):
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                p = self._image_to_view(float(u), float(v), vp)
                if p is None:
                    continue
    
                x, y = float(int(round(p.x()))), float(int(round(p.y())))
    
                is_outlier = False
                if ov.outlier_mask is not None and i < len(ov.outlier_mask):
                    is_outlier = bool(ov.outlier_mask[i])
    
                painter.setPen(orange_roi_pen if is_outlier else green_roi_pen)
                painter.drawEllipse(QPointF(x, y), r_circle_view, r_circle_view)
    
        # ------------------------------------------------------------
        # measured crosses
        # ------------------------------------------------------------
        if ov.uv_measured is not None:
            painter.setPen(meas_pen)
            r = cross_half_view
            for u, v in ov.uv_measured:
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                p = self._image_to_view(float(u), float(v), vp)
                if p is None:
                    continue
                x, y = float(int(round(p.x()))), float(int(round(p.y())))
                painter.drawLine(QPointF(x - r, y), QPointF(x + r, y))
                painter.drawLine(QPointF(x, y - r), QPointF(x, y + r))
    
        # ------------------------------------------------------------
        # projected crosses
        # ------------------------------------------------------------
        if ov.uv_projected is not None:
            painter.setPen(proj_pen)
            r = cross_half_view
            for u, v in ov.uv_projected:
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                p = self._image_to_view(float(u), float(v), vp)
                if p is None:
                    continue
                x, y = float(int(round(p.x()))), float(int(round(p.y())))
                painter.drawLine(QPointF(x - r, y), QPointF(x + r, y))
                painter.drawLine(QPointF(x, y - r), QPointF(x, y + r))

    # ---------------------------------------------------------------------
    # Mouse interaction (square zoom)
    # ---------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if not self.isEnabled():
            super().mousePressEvent(event)
            return

        vp = self._viewport_square()

        if event.button() == Qt.RightButton:
            self.reset_zoom()
            return

        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        p = QPointF(event.position())
        if not vp.contains(p):
            super().mousePressEvent(event)
            return

        self._dragging = True
        self._sel_start = self._clamp_point_to_viewport(p, vp)
        self._sel_rect_view = QRectF(self._sel_start.x(), self._sel_start.y(), 0.0, 0.0)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if not self.isEnabled():
            super().mouseMoveEvent(event)
            return

        if not self._dragging or self._sel_start is None:
            super().mouseMoveEvent(event)
            return

        vp = self._viewport_square()
        p = self._clamp_point_to_viewport(QPointF(event.position()), vp)
        self._sel_rect_view = self._make_square_rect_from_two_points(self._sel_start, p, vp)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if not self.isEnabled():
            super().mouseReleaseEvent(event)
            return

        if event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return

        if not self._dragging:
            super().mouseReleaseEvent(event)
            return

        self._dragging = False
        vp = self._viewport_square()

        if self._sel_rect_view is not None and self._crop is not None:
            r = self._sel_rect_view
            if r.width() >= self._min_sel_px and r.height() >= self._min_sel_px:
                u0, v0 = self._view_to_image(r.left(), r.top(), vp)
                u1, v1 = self._view_to_image(r.right(), r.bottom(), vp)

                x = min(u0, u1)
                y = min(v0, v1)
                s = max(abs(u1 - u0), abs(v1 - v0))

                # clamp to current crop bounds
                x = max(self._crop.left(), x)
                y = max(self._crop.top(), y)
                max_s = min(self._crop.right() - x, self._crop.bottom() - y)
                s = min(s, max_s)

                if s > 1.0:
                    self._crop = QRectF(float(x), float(y), float(s), float(s))
                    self._emit_zoom_changed()

        self._sel_rect_view = None
        self._sel_start = None
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        if not self.isEnabled():
            super().mouseDoubleClickEvent(event)
            return
        if event.button() == Qt.LeftButton:
            self.reset_zoom()
            return
        super().mouseDoubleClickEvent(event)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _emit_zoom_changed(self) -> None:
        self.zoom_changed.emit(self.has_zoom())