# overlay/gui/pages/templates/page_base_image.py

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import cv2

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy, QPushButton, QGridLayout
)


class BaseImagePage(QWidget):
    """
    Layout (as in your sketch):

      LEFT COLUMN:
        [ Instructions ]   (same width as image)
        [ Image viewport ] (fixed target size, scales down if window too small)

      RIGHT COLUMN (aligned to IMAGE top, not to instructions):
        [ Controls ]
        [ Stats ]

    Exposed API for derived pages:
      - self.instructions_label : QLabel
      - self.controls_content   : QVBoxLayout (put buttons here)
      - self.image_label        : QLabel (display)
    """

    # Gap between LEFT block and RIGHT block
    GAP_PX = 9

    # IMPORTANT: gap between instructions and image should match GAP_PX
    LEFT_SPACING_PX = GAP_PX
    RIGHT_SPACING_PX = 10

    RIGHT_COL_W = 220

    # Target viewport size (derived classes can override)
    VIEWPORT_SIZE: Tuple[int, int] | None = None

    # Display rendering only
    # "cover" crops to fill, "fit" letterboxes (never crops)
    RENDER_MODE: str | None = None

    def __init__(self, parent=None):
        super().__init__(parent)

        self._last_viewport_px: Tuple[int, int] | None = None

        self._build_ui()
        self.set_viewport_background(active=False)
        self._update_viewport_size()

    # ---------------------------------------------------------------------
    # UI
    # ---------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Main row
        self._row = QHBoxLayout()
        self._row.setSpacing(self.GAP_PX)
        root.addLayout(self._row, stretch=1)
        
        self._nav_bar = QFrame()
        self._nav_bar.setStyleSheet("QFrame { background: transparent; }")

        nav = QHBoxLayout(self._nav_bar)
        nav.setContentsMargins(0, 0, 0, 0)
        nav.setSpacing(12)
        nav.addStretch(1)

        self.btn_back = QPushButton("Back")
        self.btn_back.setObjectName("SecondaryBtn")

        self.btn_next = QPushButton("Next")

        nav.addWidget(self.btn_back)
        nav.addWidget(self.btn_next)

        root.addWidget(self._nav_bar, stretch=0)

        # -------------------------
        # LEFT COLUMN (instructions + image)
        # -------------------------
        self._left_col = QVBoxLayout()
        self._left_col.setSpacing(self.LEFT_SPACING_PX)
        self._left_col.setAlignment(Qt.AlignTop)
        self._row.addLayout(self._left_col, stretch=1)
        
        # Image viewport
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image_label.setMinimumSize(320, 180)
        self.image_label.setStyleSheet("background: rgb(20,20,20); border-radius: 12px;")
        self._left_col.addWidget(self.image_label, stretch=0, alignment=Qt.AlignLeft | Qt.AlignTop)

        # Instructions box
        self.instructions_box, self.instructions_content = self._make_box(
            title="Instructions",
            bg="#f8f9fa",
            title_content_spacing=2,
        )
        
        self.instructions_label = QLabel()
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.instructions_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        self.instructions_content.addWidget(self.instructions_label, stretch=0)
        self._left_col.addWidget(
            self.instructions_box,
            stretch=0,
            alignment=Qt.AlignLeft | Qt.AlignTop
        )

        # -------------------------
        # RIGHT COLUMN (aligned to IMAGE top)
        # -------------------------
        self._right_col = QVBoxLayout()
        self._right_col.setSpacing(self.RIGHT_SPACING_PX)
        self._right_col.setAlignment(Qt.AlignTop)  # <-- NEU
        self._row.addLayout(self._right_col, stretch=0)
        
        # Container with fixed width
        self._right_container = QFrame()
        self._right_container.setFixedWidth(self.RIGHT_COL_W)
        self._right_container.setStyleSheet("QFrame { background: transparent; }")
        
        self._right_col.addWidget(self._right_container, stretch=0, alignment=Qt.AlignTop)

        right_inner = QVBoxLayout(self._right_container)
        right_inner.setContentsMargins(0, 0, 0, 0)
        right_inner.setSpacing(self.RIGHT_SPACING_PX)

        # Controls
        self.controls_box, self.controls_content = self._make_box(
            title="Controls",
            bg="#f1f3f5",
            title_content_spacing=2,
        )
        
        right_inner.addWidget(self.controls_box, stretch=0)

        # Stats
        self.stats_box, self.stats_content = self._make_box(
            title="Stats",
            bg="#f8f9fa",
            title_content_spacing=2,
        )
        #self.stats_box.setFixedWidth(self.RIGHT_COL_W)
        right_inner.addWidget(self.stats_box, stretch=0)
        right_inner.addStretch(1)

    def _make_box(self, *, title: str, bg: str,
              min_h: int | None = None,
              title_content_spacing: int = 4):
        box = QFrame()
    
        if min_h is not None:
            box.setMinimumHeight(min_h)
    
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        box.setStyleSheet(
            f"QFrame {{ background: {bg}; border-radius: 12px; }}"
        )
    
        v = QVBoxLayout(box)
        v.setContentsMargins(12, 8, 12, 10)
        v.setSpacing(title_content_spacing)
    
        lbl = QLabel(title)
        lbl.setStyleSheet(
            "font-size: 12px; font-weight: 600; color: #495057;"
        )
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    
        v.addWidget(lbl)
        content = QVBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(4)
        v.addLayout(content)
    
        return box, content

    
    def set_stats_rows(self, rows: list[tuple[str, str]]) -> None:
        # Clear old widgets
        while self.stats_content.count():
            item = self.stats_content.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
    
        for k, v in rows:
    
            # -----------------------------
            # CASE 1: Multi-line → MATRIX
            # -----------------------------
            if "\n" in v:
                lines = v.split("\n")
    
                # 1) Header row
                header = QLabel(f"{k}: {lines[0]}")
                header.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                header.setWordWrap(False)
                self.stats_content.addWidget(header, stretch=0)
    
                # 2) Matrix grid
                grid_widget = QWidget()
                grid = QGridLayout(grid_widget)
                grid.setContentsMargins(0, 0, 0, 0)
                grid.setHorizontalSpacing(12)  # Abstand zwischen Spalten
                grid.setVerticalSpacing(0)
    
                for r, line in enumerate(lines[1:]):
                    values = line.strip().split()
    
                    for c, val in enumerate(values):
                        lbl = QLabel(val)
                        lbl.setAlignment(Qt.AlignLeft)  # oder AlignRight
                        lbl.setWordWrap(False)
                        grid.addWidget(lbl, r, c)
    
                self.stats_content.addWidget(grid_widget, stretch=0)
    
            # -----------------------------
            # CASE 2: Normal row
            # -----------------------------
            else:
                lbl = QLabel(f"{k}: {v}")
                lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                lbl.setWordWrap(False)
                self.stats_content.addWidget(lbl, stretch=0)
    
        self.stats_box.adjustSize()
        self.stats_box.updateGeometry()


    # ---------------------------------------------------------------------
    # Resizing / alignment
    # ---------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_viewport_size()
       

    def _update_viewport_size(self) -> None:
        """Target VIEWPORT_SIZE exactly, scale down only if window too small."""
        target_w, target_h = self.VIEWPORT_SIZE

        cr = self.contentsRect()

        # Available width for the LEFT block
        avail_w = max(1, cr.width() - self.RIGHT_COL_W - self.GAP_PX)
        
        instr_h = self.instructions_box.sizeHint().height()
        nav_h = self._nav_bar.sizeHint().height()
        
        max_img_h = max(1, cr.height() - instr_h - nav_h - self.LEFT_SPACING_PX)

        scale = min(avail_w / target_w, max_img_h / target_h, 1.0)
        scale = max(scale, 0.2)

        w = int(round(target_w * scale))
        h = int(round(target_h * scale))

        if self._last_viewport_px == (w, h):
            return
        self._last_viewport_px = (w, h)

        self.image_label.setFixedSize(QSize(w, h))
        self.instructions_box.setFixedWidth(w)

        # refresh view
        self.update_view()

    # ---------------------------------------------------------------------
    # Helpers used by derived pages
    # ---------------------------------------------------------------------

    def set_viewport_background(self, *, active: bool) -> None:
        bg = "rgb(0,0,0)" if active else "rgb(20,20,20)"
        self.image_label.setStyleSheet(f"background: {bg}; border-radius: 12px;")

    def clear_view(self) -> None:
        self.image_label.clear()

    # ---------------------------------------------------------------------
    # Rendering (display only)
    # ---------------------------------------------------------------------

    def update_view(self) -> None:
        frame = self.get_frame()
        if frame is None:
            self.clear_view()
            return

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        frame = self.draw_overlay(frame)

        tw = max(1, self.image_label.width())
        th = max(1, self.image_label.height())

        if self.RENDER_MODE == "fit":
            disp = self._resize_fit(frame, tw, th)
        else:
            disp = self._resize_cover(frame, tw, th)

        self.image_label.setPixmap(self._to_pixmap(disp))

    @staticmethod
    def _to_pixmap(img_bgr: np.ndarray) -> QPixmap:
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    @staticmethod
    def _resize_cover(img: np.ndarray, tw: int, th: int) -> np.ndarray:
        h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            return np.zeros((th, tw, 3), dtype=np.uint8)

        s = max(tw / w, th / h)
        nw = int(round(w * s))
        nh = int(round(h * s))
        r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        x0 = max(0, (nw - tw) // 2)
        y0 = max(0, (nh - th) // 2)
        out = r[y0:y0 + th, x0:x0 + tw]
        if out.shape[0] != th or out.shape[1] != tw:
            out = cv2.resize(out, (tw, th), interpolation=cv2.INTER_AREA)
        return out

    @staticmethod
    def _resize_fit(img: np.ndarray, tw: int, th: int) -> np.ndarray:
        h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            return np.zeros((th, tw, 3), dtype=np.uint8)

        s = min(tw / w, th / h)
        nw = max(1, int(round(w * s)))
        nh = max(1, int(round(h * s)))
        r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        out = np.zeros((th, tw, 3), dtype=np.uint8)
        x0 = (tw - nw) // 2
        y0 = (th - nh) // 2
        out[y0:y0 + nh, x0:x0 + nw] = r
        return out

    # ---------------------------------------------------------------------
    # Hooks (override)
    # ---------------------------------------------------------------------

    def get_frame(self) -> Optional[np.ndarray]:
        return None

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        return frame_bgr
