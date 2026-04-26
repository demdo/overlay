from __future__ import annotations

import sys
import cv2
import numpy as np
import pyrealsense2 as rs

from PySide6.QtCore import Qt, QTimer, QRect
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
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
    bytes_per_line = ch * w

    qimg = QImage(
        img_rgb.data,
        w,
        h,
        bytes_per_line,
        QImage.Format_RGB888,
    )
    return QPixmap.fromImage(qimg.copy())


# ============================================================
# Right slide-out control rail
# ============================================================

class _OverlayControlRail(QFrame):
    """
    Professional right-side drawer:
    - collapsed: only slim handle visible
    - expanded: clean side panel opens to the left
    """

    HANDLE_W = 26
    PANEL_W = 320
    PANEL_H = 290
    RADIUS = 18
    HANDLE_RADIUS = 16

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._expanded = True

        self.setObjectName("overlayControlRail")
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.handle = QToolButton(self)
        self.handle.setObjectName("overlayRailHandle")
        self.handle.setText("❯")
        self.handle.setCursor(Qt.PointingHandCursor)
        self.handle.setToolTip("Hide controls")
        self.handle.clicked.connect(self._toggle)

        self.panel = QFrame(self)
        self.panel.setObjectName("overlayRailPanel")
        self.panel.setFrameShape(QFrame.NoFrame)

        self.title_label = QLabel("Overlay Controls")
        self.title_label.setObjectName("overlayRailTitle")

        self.subtitle_label = QLabel("Live adjustment")
        self.subtitle_label.setObjectName("overlayRailSubtitle")

        # ---------- Alpha ----------
        self.lbl_alpha = QLabel("Alpha")
        self.lbl_alpha.setObjectName("overlayRailLabel")

        self.lbl_alpha_value = QLabel("0.50")
        self.lbl_alpha_value.setObjectName("overlayRailValue")

        alpha_head = QHBoxLayout()
        alpha_head.setContentsMargins(0, 0, 0, 0)
        alpha_head.setSpacing(10)
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

        # ---------- d_x ----------
        self.lbl_dx = QLabel("d_x [mm]")
        self.lbl_dx.setObjectName("overlayRailLabel")

        self.lbl_dx_value = QLabel("0.0")
        self.lbl_dx_value.setObjectName("overlayRailValue")

        dx_head = QHBoxLayout()
        dx_head.setContentsMargins(0, 0, 0, 0)
        dx_head.setSpacing(10)
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

        # ---------- Tip ----------
        self.chk_show_tip = QCheckBox("Show tip")
        self.chk_show_tip.setObjectName("overlayRailCheck")
        self.chk_show_tip.setChecked(True)

        # ---------- Group cards ----------
        alpha_card = QFrame()
        alpha_card.setObjectName("overlayRailGroup")
        alpha_layout = QVBoxLayout(alpha_card)
        alpha_layout.setContentsMargins(16, 14, 16, 14)
        alpha_layout.setSpacing(12)
        alpha_layout.addLayout(alpha_head)
        alpha_layout.addWidget(self.slider_alpha)

        dx_card = QFrame()
        dx_card.setObjectName("overlayRailGroup")
        dx_layout = QVBoxLayout(dx_card)
        dx_layout.setContentsMargins(16, 14, 16, 14)
        dx_layout.setSpacing(12)
        dx_layout.addLayout(dx_head)
        dx_layout.addWidget(self.slider_dx)

        tip_card = QFrame()
        tip_card.setObjectName("overlayRailGroup")
        tip_layout = QVBoxLayout(tip_card)
        tip_layout.setContentsMargins(16, 14, 16, 14)
        tip_layout.setSpacing(10)
        tip_layout.addWidget(self.chk_show_tip)

        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(3)
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.subtitle_label)

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(18, 18, 18, 18)
        panel_layout.setSpacing(14)
        panel_layout.addLayout(header_layout)
        panel_layout.addWidget(alpha_card)
        panel_layout.addWidget(dx_card)
        panel_layout.addWidget(tip_card)
        panel_layout.addStretch(1)

        self._apply_style()
        self._update_geometry()

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self.handle.setText("❯" if self._expanded else "❮")
        self.handle.setToolTip("Hide controls" if self._expanded else "Show controls")
        self._update_geometry()

        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "_position_controls"):
            parent._position_controls()

    def _update_geometry(self) -> None:
        total_w = self.PANEL_W + self.HANDLE_W + 6 if self._expanded else self.HANDLE_W
        self.setFixedSize(total_w, self.PANEL_H)

        handle_margin_y = 14
        handle_h = self.PANEL_H - 2 * handle_margin_y

        if self._expanded:
            self.panel.show()
            self.panel.setGeometry(0, 0, self.PANEL_W, self.PANEL_H)
            self.handle.setGeometry(self.PANEL_W + 6, handle_margin_y, self.HANDLE_W, handle_h)
        else:
            self.panel.hide()
            self.handle.setGeometry(0, handle_margin_y, self.HANDLE_W, handle_h)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QFrame#overlayRailPanel {{
                background: rgba(248, 250, 252, 242);
                border: 1px solid rgba(15, 23, 42, 26);
                border-radius: {self.RADIUS}px;
            }}

            QToolButton#overlayRailHandle {{
                background: rgba(248, 250, 252, 236);
                border: 1px solid rgba(15, 23, 42, 24);
                border-radius: {self.HANDLE_RADIUS}px;
                color: #0f172a;
                font-size: 13px;
                font-weight: 700;
                padding: 0px;
                text-align: center;
            }}

            QToolButton#overlayRailHandle:hover {{
                background: rgba(255, 255, 255, 245);
            }}

            QLabel#overlayRailTitle {{
                color: #0f172a;
                font-size: 18px;
                font-weight: 700;
            }}

            QLabel#overlayRailSubtitle {{
                color: #64748b;
                font-size: 12px;
                font-weight: 500;
            }}

            QFrame#overlayRailGroup {{
                background: rgba(255, 255, 255, 228);
                border: 1px solid rgba(15, 23, 42, 16);
                border-radius: 14px;
            }}

            QLabel#overlayRailLabel {{
                color: #334155;
                font-size: 13px;
                font-weight: 600;
                min-height: 20px;
            }}

            QLabel#overlayRailValue {{
                color: #0f172a;
                font-size: 13px;
                font-weight: 700;
                min-width: 44px;
                min-height: 20px;
                qproperty-alignment: AlignRight | AlignVCenter;
            }}

            QCheckBox#overlayRailCheck {{
                color: #334155;
                font-size: 13px;
                font-weight: 600;
                spacing: 8px;
            }}

            QCheckBox#overlayRailCheck::indicator {{
                width: 16px;
                height: 16px;
            }}

            QSlider#overlayRailSlider {{
                min-height: 18px;
            }}

            QSlider#overlayRailSlider::groove:horizontal {{
                height: 6px;
                background: #dbe4ee;
                border-radius: 3px;
            }}

            QSlider#overlayRailSlider::sub-page:horizontal {{
                background: #60a5fa;
                border-radius: 3px;
            }}

            QSlider#overlayRailSlider::handle:horizontal {{
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
                background: #2563eb;
                border: 1px solid #1d4ed8;
            }}
            """
        )


# ============================================================
# Debug page
# ============================================================

class DebugSidebarPage(QWidget):
    FPS = 30
    COLOR_SIZE = (1920, 1080)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.pipeline: rs.pipeline | None = None
        self.config: rs.config | None = None
        self.timer: QTimer | None = None

        self._live_color: np.ndarray | None = None

        self._build_ui()
        self._connect_signals()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

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

        self.controls.setGeometry(QRect(x, y, size.width(), size.height()))
        self.controls.raise_()

    def _connect_signals(self) -> None:
        self.controls.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        self.controls.slider_dx.valueChanged.connect(self._on_dx_changed)
        self.controls.chk_show_tip.toggled.connect(self._on_show_tip_toggled)

    # --------------------------------------------------------
    # RealSense
    # --------------------------------------------------------

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
            QMessageBox.critical(
                self,
                "Debug Sidebar",
                f"Live update failed.\n\n{e}",
            )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    def _update_image_label(self) -> None:
        if self._live_color is None:
            return

        pix = _np_to_qpixmap(self._live_color)
        if pix.isNull():
            return

        scaled = pix.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    # --------------------------------------------------------
    # Dummy callbacks
    # --------------------------------------------------------

    def _on_alpha_changed(self, value: int) -> None:
        alpha = float(value) / 100.0
        self.controls.lbl_alpha_value.setText(f"{alpha:.2f}")

    def _on_dx_changed(self, value: int) -> None:
        self.controls.lbl_dx_value.setText(f"{float(value):.1f}")

    def _on_show_tip_toggled(self, checked: bool) -> None:
        _ = checked

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def start(self) -> None:
        try:
            self._start_realsense()
            self._start_timer()
            self.setFocus()

        except Exception as e:
            self._stop_timer()
            self._stop_realsense()
            QMessageBox.critical(
                self,
                "Camera",
                f"Could not open RealSense camera.\n\n{e}",
            )

    def stop(self) -> None:
        self._stop_timer()
        self._stop_realsense()

    def closeEvent(self, event) -> None:
        self.stop()
        super().closeEvent(event)


# ============================================================
# Standalone launcher
# ============================================================

def main() -> int:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    w = DebugSidebarPage()
    w.resize(1920, 1080)
    w.setWindowTitle("Debug Sidebar")
    w.show()
    w.start()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())