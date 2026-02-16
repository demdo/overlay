# overlay/gui/widgets/widget_xray_marker_selection.py

from __future__ import annotations

from typing import List, Optional, Tuple, Set, Dict, Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget

from overlay.tools.xray_marker_selection import (
    prepare_nearest_cell_data,
    nearest_cell,
)

Cell = Tuple[int, int]


def _qpixmap_from_gray_u8(img_gray_u8: np.ndarray) -> QPixmap:
    """img_gray_u8: (H,W) uint8."""
    h, w = img_gray_u8.shape
    qimg = QImage(img_gray_u8.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


class XrayMarkerSelectionWidget(QWidget):
    """
    Embedded marker selection inside the GUI (no separate window).

    - Fit-to-view image display
    - Shows all markers from circles_grid
    - Select 3 anchor markers:
        Left click: select
        Right click: undo last
    - After 3 anchors: emits selection_proposed to the page
    - Page confirms YES/NO
        YES: page calls set_roi_cells(...) and set_locked(True)
        NO:  page calls clear_selection()
    """

    selection_proposed = Signal(object)  # {"uv": [(u,v),...], "cells": [(r,c),...]}

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

        # Grid picking
        self._circles_grid: Optional[np.ndarray] = None
        self._prepared = None
        self._pick_radius_px: float = 20.0

        # Anchor selection overlays
        self._selected_cells: List[Cell] = []
        self._highlight_cells: Set[Cell] = set()
        self._hover_cell: Optional[Cell] = None

        # ROI highlighting after confirmation
        self._roi_cells: Set[Cell] = set()

        # Lock interaction after confirmation
        self._locked: bool = False

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

    def set_grid(self, circles_grid: Optional[np.ndarray], pick_radius_px: Optional[float] = None):
        self._circles_grid = circles_grid
        if circles_grid is None:
            self._prepared = None
        else:
            self._prepared = prepare_nearest_cell_data(circles_grid)

        if pick_radius_px is not None:
            self._pick_radius_px = float(pick_radius_px)

        self.update()

    def set_data(self, img_gray_u8: np.ndarray, circles_grid: Optional[np.ndarray], pick_radius_px: Optional[float]):
        self.set_image(img_gray_u8)
        self.set_grid(circles_grid, pick_radius_px)

    def clear_selection(self):
        self._selected_cells.clear()
        self._highlight_cells.clear()
        self._hover_cell = None
        self._roi_cells.clear()
        self._locked = False
        self.update()

    def set_locked(self, locked: bool):
        self._locked = bool(locked)

    def set_roi_cells(self, cells: List[Cell]):
        self._roi_cells = set(cells) if cells is not None else set()
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

        if self._circles_grid is None:
            painter.end()
            return

        grid = self._circles_grid
        nrows, ncols, _ = grid.shape

        pen_normal = QPen(QColor(0, 255, 0), 2)
        pen_anchor = QPen(QColor(0, 255, 255), 4)
        pen_roi    = QPen(QColor(255, 165, 0), 3)
        pen_cross  = QPen(QColor(255, 0, 0), 1)
        pen_hover  = QPen(QColor(255, 255, 0), 2)

        for i in range(nrows):
            for j in range(ncols):
                x, y, r = grid[i, j]
                if not np.isfinite(x):
                    continue

                xd = int(round(self._off_x + float(x) * self._scale))
                yd = int(round(self._off_y + float(y) * self._scale))
                rd = int(round(float(r) * self._scale))

                cell = (i, j)
                if cell in self._highlight_cells:
                    pen = pen_anchor
                elif cell in self._roi_cells:
                    pen = pen_roi
                else:
                    pen = pen_normal

                painter.setPen(pen)
                painter.drawEllipse(xd - rd, yd - rd, 2 * rd, 2 * rd)

                painter.setPen(pen_cross)
                painter.drawLine(xd - 5, yd, xd + 5, yd)
                painter.drawLine(xd, yd - 5, xd, yd + 5)

        if (not self._locked) and (self._hover_cell is not None):
            i, j = self._hover_cell
            x, y, r = grid[i, j]
            if np.isfinite(x):
                xd = int(round(self._off_x + float(x) * self._scale))
                yd = int(round(self._off_y + float(y) * self._scale))
                rd = int(round(float(r) * self._scale))
                painter.setPen(pen_hover)
                painter.drawEllipse(xd - rd - 3, yd - rd - 3, 2 * rd + 6, 2 * rd + 6)

        painter.end()

    # ---------------- Qt mouse interaction ----------------

    def mousePressEvent(self, event):
        if self._circles_grid is None or self._prepared is None:
            return

        if self._locked:
            return  # no interaction after confirmation

        xw = int(event.position().x())
        yw = int(event.position().y())
        if not self._is_inside_image_display(xw, yw):
            return

        x, y = self._widget_to_image_xy(xw, yw)

        if event.button() == Qt.LeftButton:
            cell, dist = nearest_cell(self._prepared, x, y)
            if cell is None or dist > self._pick_radius_px:
                return
            if cell in self._highlight_cells:
                return

            self._selected_cells.append(cell)
            self._highlight_cells.add(cell)
            self.update()

            if len(self._selected_cells) == 3:
                anchors_uv = []
                for (ri, ci) in self._selected_cells:
                    px, py, _ = self._circles_grid[ri, ci]
                    anchors_uv.append((float(px), float(py)))

                payload: Dict[str, Any] = {
                    "uv": anchors_uv,
                    "cells": list(self._selected_cells),
                }
                self.selection_proposed.emit(payload)

        elif event.button() == Qt.RightButton:
            if self._selected_cells:
                last = self._selected_cells.pop()
                self._highlight_cells.discard(last)
                self.update()

    def mouseMoveEvent(self, event):
        if self._locked:
            return

        if self._circles_grid is None or self._prepared is None:
            return

        xw = int(event.position().x())
        yw = int(event.position().y())

        if not self._is_inside_image_display(xw, yw):
            if self._hover_cell is not None:
                self._hover_cell = None
                self.update()
            return

        x, y = self._widget_to_image_xy(xw, yw)
        cell, dist = nearest_cell(self._prepared, x, y)

        new_hover = cell if (cell is not None and dist <= self._pick_radius_px) else None
        if new_hover != self._hover_cell:
            self._hover_cell = new_hover
            self.update()
