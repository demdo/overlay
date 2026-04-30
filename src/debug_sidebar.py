from __future__ import annotations

import sys

import cv2
import numpy as np
import pyrealsense2 as rs

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# Helpers
# ============================================================

def _np_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    if img_bgr is None:
        return QPixmap()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape

    qimg = QImage(
        img_rgb.data,
        w,
        h,
        ch * w,
        QImage.Format_RGB888,
    )
    return QPixmap.fromImage(qimg.copy())


# ============================================================
# Professional sidebar — scaled 60%
# ============================================================

class _OverlayControlRail(QFrame):
    """
    Standalone debug version of the live-overlay sidebar.

    - Starts collapsed.
    - Can be opened/closed repeatedly.
    - Overall size is scaled to 60% of the previous version.
    """

    SCALE = 0.60

    HANDLE_W = int(round(34 * SCALE))
    PANEL_W = int(round(380 * SCALE))
    PANEL_H = int(round(330 * SCALE))
    GAP = int(round(10 * SCALE))

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._expanded = False

        self.setObjectName("overlayControlRail")
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_StyledBackground, True)

        # ---------------- Handle ----------------
        self.handle = QToolButton(self)
        self.handle.setObjectName("overlayRailHandle")
        self.handle.setText("‹")
        self.handle.setCursor(Qt.PointingHandCursor)
        self.handle.setToolTip("Show overlay controls")
        self.handle.clicked.connect(self._toggle)

        # ---------------- Panel ----------------
        self.panel = QFrame(self)
        self.panel.setObjectName("overlayRailPanel")
        self.panel.setFrameShape(QFrame.NoFrame)
        self.panel.setAttribute(Qt.WA_StyledBackground, True)

        # Compact title: no subtitle
        self.title_label = QLabel("Overlay Controls")
        self.title_label.setObjectName("overlayRailTitle")
        self.title_label.setFixedWidth(self._s(230))

        # ---------------- Alpha ----------------
        self.lbl_alpha = QLabel("Alpha")
        self.lbl_alpha.setObjectName("overlayRailLabel")

        self.lbl_alpha_value = QLabel("0.50")
        self.lbl_alpha_value.setObjectName("overlayRailValue")
        self.lbl_alpha_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        alpha_head = QHBoxLayout()
        alpha_head.setContentsMargins(0, 0, 0, 0)
        alpha_head.setSpacing(self._s(12))
        alpha_head.addWidget(self.lbl_alpha)
        alpha_head.addStretch(1)
        alpha_head.addWidget(self.lbl_alpha_value)

        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setObjectName("overlayRailSlider")
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(50)
        self.slider_alpha.setSingleStep(1)
        self.slider_alpha.setPageStep(10)
        self.slider_alpha.setTracking(True)
        self.slider_alpha.setMinimumHeight(self._s(28))

        # ---------------- d_x ----------------
        self.lbl_dx = QLabel("Plane depth")
        self.lbl_dx.setObjectName("overlayRailLabel")

        self.lbl_dx_value = QLabel("0.0")
        self.lbl_dx_value.setObjectName("overlayRailValue")
        self.lbl_dx_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        dx_head = QHBoxLayout()
        dx_head.setContentsMargins(0, 0, 0, 0)
        dx_head.setSpacing(self._s(12))
        dx_head.addWidget(self.lbl_dx)
        dx_head.addStretch(1)
        dx_head.addWidget(self.lbl_dx_value)

        self.slider_dx = QSlider(Qt.Horizontal)
        self.slider_dx.setObjectName("overlayRailSlider")
        self.slider_dx.setRange(-50, 50)
        self.slider_dx.setValue(0)
        self.slider_dx.setSingleStep(1)
        self.slider_dx.setPageStep(10)
        self.slider_dx.setTracking(True)
        self.slider_dx.setMinimumHeight(self._s(28))

        # ---------------- Tip ----------------
        self.chk_show_tip = QCheckBox("Show pointer tip")
        self.chk_show_tip.setObjectName("overlayRailCheck")
        self.chk_show_tip.setChecked(True)
        self.chk_show_tip.setMinimumHeight(self._s(26))

        # ---------------- Cards ----------------
        alpha_card = QFrame()
        alpha_card.setObjectName("overlayRailGroup")
        alpha_card.setAttribute(Qt.WA_StyledBackground, True)
        alpha_layout = QVBoxLayout(alpha_card)
        alpha_layout.setContentsMargins(self._s(18), self._s(14), self._s(18), self._s(14))
        alpha_layout.setSpacing(self._s(10))
        alpha_layout.addLayout(alpha_head)
        alpha_layout.addWidget(self.slider_alpha)

        dx_card = QFrame()
        dx_card.setObjectName("overlayRailGroup")
        dx_card.setAttribute(Qt.WA_StyledBackground, True)
        dx_layout = QVBoxLayout(dx_card)
        dx_layout.setContentsMargins(self._s(18), self._s(14), self._s(18), self._s(14))
        dx_layout.setSpacing(self._s(10))
        dx_layout.addLayout(dx_head)
        dx_layout.addWidget(self.slider_dx)

        tip_card = QFrame()
        tip_card.setObjectName("overlayRailGroup")
        tip_card.setAttribute(Qt.WA_StyledBackground, True)
        tip_layout = QVBoxLayout(tip_card)
        tip_layout.setContentsMargins(self._s(18), self._s(13), self._s(18), self._s(13))
        tip_layout.setSpacing(0)
        tip_layout.addWidget(self.chk_show_tip)

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(self._s(22), self._s(22), self._s(22), self._s(22))
        panel_layout.setSpacing(self._s(16))
        panel_layout.addWidget(self.title_label, 0, Qt.AlignLeft)
        panel_layout.addWidget(alpha_card)
        panel_layout.addWidget(dx_card)
        panel_layout.addWidget(tip_card)
        panel_layout.addStretch(1)

        self._apply_style()
        self._update_geometry()

    def _s(self, value: float) -> int:
        return max(1, int(round(float(value) * self.SCALE)))

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self.handle.setText("›" if self._expanded else "‹")
        self.handle.setToolTip("Hide overlay controls" if self._expanded else "Show overlay controls")
        self._update_geometry()

        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "_position_controls"):
            parent._position_controls()

    def _update_geometry(self) -> None:
        handle_h = self._s(86)
        handle_y = 0

        if self._expanded:
            total_w = self.PANEL_W + self.GAP + self.HANDLE_W
            total_h = self.PANEL_H
            self.setFixedSize(total_w, total_h)

            self.panel.show()
            self.panel.setGeometry(0, 0, self.PANEL_W, self.PANEL_H)
            self.handle.setGeometry(
                self.PANEL_W + self.GAP,
                handle_y,
                self.HANDLE_W,
                handle_h,
            )
        else:
            self.panel.hide()
            self.setFixedSize(self.HANDLE_W, handle_h)
            self.handle.setGeometry(0, 0, self.HANDLE_W, handle_h)

    def _apply_style(self) -> None:
        border_radius_panel = self._s(22)
        border_radius_handle = self._s(17)
        border_radius_group = self._s(16)
        checkbox_radius = self._s(5)

        font_title = self._s(22)
        font_label = self._s(14)
        font_handle = self._s(28)

        slider_groove_h = self._s(7)
        slider_handle = self._s(18)
        slider_handle_margin = -self._s(6)

        checkbox_size = self._s(18)

        self.setStyleSheet(
            f"""
            QFrame#overlayControlRail {{
                background: transparent;
                border: none;
            }}

            QFrame#overlayRailPanel {{
                background-color: rgba(18, 24, 38, 232);
                border: 1px solid rgba(255, 255, 255, 44);
                border-radius: {border_radius_panel}px;
            }}

            QToolButton#overlayRailHandle {{
                background-color: rgba(18, 24, 38, 226);
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: {border_radius_handle}px;
                color: #f8fafc;
                font-size: {font_handle}px;
                font-weight: 700;
                padding: 0px;
            }}

            QToolButton#overlayRailHandle:hover {{
                background-color: rgba(30, 41, 59, 242);
                border: 1px solid rgba(96, 165, 250, 150);
                color: #ffffff;
            }}

            QLabel#overlayRailTitle {{
                background: transparent;
                color: #f8fafc;
                font-size: {font_title}px;
                font-weight: 800;
                min-height: {self._s(28)}px;
            }}

            QFrame#overlayRailGroup {{
                background-color: rgba(255, 255, 255, 24);
                border: 1px solid rgba(255, 255, 255, 34);
                border-radius: {border_radius_group}px;
            }}

            QLabel#overlayRailLabel {{
                background: transparent;
                color: #e2e8f0;
                font-size: {font_label}px;
                font-weight: 700;
                min-height: {self._s(22)}px;
            }}

            QLabel#overlayRailValue {{
                background: transparent;
                color: #ffffff;
                font-size: {font_label}px;
                font-weight: 800;
                min-width: {self._s(86)}px;
                min-height: {self._s(22)}px;
            }}

            QCheckBox#overlayRailCheck {{
                background: transparent;
                color: #ffffff;
                font-size: {font_label}px;
                font-weight: 700;
                spacing: {self._s(10)}px;
            }}

            QCheckBox#overlayRailCheck::indicator {{
                width: {checkbox_size}px;
                height: {checkbox_size}px;
                border-radius: {checkbox_radius}px;
                border: 1px solid rgba(255, 255, 255, 95);
                background-color: rgba(15, 23, 42, 180);
            }}

            QCheckBox#overlayRailCheck::indicator:checked {{
                background-color: #3b82f6;
                border: 1px solid #93c5fd;
            }}

            QSlider#overlayRailSlider {{
                background: transparent;
                min-height: {self._s(28)}px;
            }}

            QSlider#overlayRailSlider::groove:horizontal {{
                height: {slider_groove_h}px;
                background-color: rgba(148, 163, 184, 80);
                border-radius: {max(1, slider_groove_h // 2)}px;
            }}

            QSlider#overlayRailSlider::sub-page:horizontal {{
                height: {slider_groove_h}px;
                background-color: #3b82f6;
                border-radius: {max(1, slider_groove_h // 2)}px;
            }}

            QSlider#overlayRailSlider::add-page:horizontal {{
                height: {slider_groove_h}px;
                background-color: rgba(148, 163, 184, 80);
                border-radius: {max(1, slider_groove_h // 2)}px;
            }}

            QSlider#overlayRailSlider::handle:horizontal {{
                width: {slider_handle}px;
                height: {slider_handle}px;
                margin: {slider_handle_margin}px 0px;
                border-radius: {max(1, slider_handle // 2)}px;
                background-color: #ffffff;
                border: {self._s(3)}px solid #3b82f6;
            }}

            QSlider#overlayRailSlider::handle:horizontal:hover {{
                border: {self._s(3)}px solid #60a5fa;
            }}
            """
        )


# ============================================================
# Debug page
# ============================================================

class DebugSidebarPage(QWidget):
    FPS = 30
    COLOR_SIZE = (1280, 720)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.pipeline: rs.pipeline | None = None
        self.config: rs.config | None = None
        self.timer: QTimer | None = None

        self._live_color: np.ndarray | None = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        self.setObjectName("debugSidebarPage")
        self.setFocusPolicy(Qt.StrongFocus)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: black;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.controls = _OverlayControlRail(self)

        self.setStyleSheet(
            """
            QWidget#debugSidebarPage {
                background: black;
            }
            """
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.image_label.setGeometry(self.rect())
        self._position_controls()
        self._update_image_label()

    def _position_controls(self) -> None:
        margin_top = 18
        margin_right = 18
        size = self.controls.size()

        x = self.width() - size.width() - margin_right - 8
        y = margin_top

        self.controls.setGeometry(x, y, size.width(), size.height())
        self.controls.raise_()

    def _connect_signals(self) -> None:
        self.controls.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        self.controls.slider_dx.valueChanged.connect(self._on_dx_changed)
        self.controls.chk_show_tip.toggled.connect(self._on_show_tip_toggled)

    def _start_realsense(self) -> None:
        if self.pipeline is not None:
            return

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            self.COLOR_SIZE[0],
            self.COLOR_SIZE[1],
            rs.format.bgr8,
            self.FPS,
        )

        pipeline.start(config)

        self.pipeline = pipeline
        self.config = config

    def _stop_realsense(self) -> None:
        if self.pipeline is None:
            return

        try:
            self.pipeline.stop()
        except Exception:
            pass

        self.pipeline = None
        self.config = None

    def _start_timer(self) -> None:
        if self.timer is None:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._on_timer)

        self.timer.start(int(round(1000.0 / float(self.FPS))))

    def _stop_timer(self) -> None:
        if self.timer is not None:
            self.timer.stop()

    def _on_timer(self) -> None:
        if self.pipeline is None:
            return

        try:
            frames = self.pipeline.poll_for_frames()
            if not frames:
                return

            cf = frames.get_color_frame()
            if not cf:
                return

            img = np.asanyarray(cf.get_data())
            if img is None or img.size == 0:
                return

            self._live_color = img
            self._update_image_label()

        except Exception as e:
            self._stop_timer()
            self._stop_realsense()
            print(f"[ERROR] Live update failed: {e}")

    def _update_image_label(self) -> None:
        if self._live_color is None:
            return

        pix = _np_to_qpixmap(self._live_color)
        if pix.isNull():
            return

        scaled = pix.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _on_alpha_changed(self, value: int) -> None:
        alpha = float(value) / 100.0
        self.controls.lbl_alpha_value.setText(f"{alpha:.2f}")

    def _on_dx_changed(self, value: int) -> None:
        self.controls.lbl_dx_value.setText(f"{float(value):.1f}")

    def _on_show_tip_toggled(self, checked: bool) -> None:
        _ = checked

    def start(self) -> None:
        try:
            self._start_realsense()
            self._start_timer()
            self.setFocus()

        except Exception as e:
            self._stop_timer()
            self._stop_realsense()
            print(f"[ERROR] Could not open RealSense camera: {e}")

    def stop(self) -> None:
        self._stop_timer()
        self._stop_realsense()

    def closeEvent(self, event) -> None:
        self.stop()
        super().closeEvent(event)


# ============================================================
# Spyder-compatible launcher
# ============================================================

def main() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    w = DebugSidebarPage()
    w.resize(1280, 720)
    w.setWindowTitle("Debug Sidebar")
    w.show()
    w.start()

    app.exec()


if __name__ == "__main__":
    main()
