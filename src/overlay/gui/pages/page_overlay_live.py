from __future__ import annotations

import cv2
import numpy as np
import pyrealsense2 as rs

from PySide6.QtCore import Qt, QTimer, QRect
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
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

from overlay.gui.state import SessionState
from overlay.tools.homography import estimate_plane_induced_homography
from overlay.tools.warp import blend_xray_overlay


# ============================================================
# Helpers
# ============================================================

def _draw_point(
    img: np.ndarray,
    uv: np.ndarray,
    *,
    color=(0, 0, 255),
    radius=8,
    cross_size=20,
    thickness=2,
) -> np.ndarray:
    out = img.copy()

    if uv is None:
        return out

    uv = np.asarray(uv, dtype=np.float64).reshape(2)
    if not np.isfinite(uv).all():
        return out

    u, v = np.round(uv).astype(int)

    cv2.circle(out, (u, v), radius, color, thickness, cv2.LINE_AA)
    cv2.drawMarker(
        out,
        (u, v),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=cross_size,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )
    return out


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
# Professional sidebar — scaled 60%
# ============================================================

class _OverlayControlRail(QFrame):
    """
    Right-side live overlay controls.

    - Starts collapsed.
    - Can be opened/closed repeatedly.
    - Professional dark translucent sidebar.
    - Overall size scaled to 60%.
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
        alpha_layout.setContentsMargins(
            self._s(18),
            self._s(14),
            self._s(18),
            self._s(14),
        )
        alpha_layout.setSpacing(self._s(10))
        alpha_layout.addLayout(alpha_head)
        alpha_layout.addWidget(self.slider_alpha)

        dx_card = QFrame()
        dx_card.setObjectName("overlayRailGroup")
        dx_card.setAttribute(Qt.WA_StyledBackground, True)
        dx_layout = QVBoxLayout(dx_card)
        dx_layout.setContentsMargins(
            self._s(18),
            self._s(14),
            self._s(18),
            self._s(14),
        )
        dx_layout.setSpacing(self._s(10))
        dx_layout.addLayout(dx_head)
        dx_layout.addWidget(self.slider_dx)

        tip_card = QFrame()
        tip_card.setObjectName("overlayRailGroup")
        tip_card.setAttribute(Qt.WA_StyledBackground, True)
        tip_layout = QVBoxLayout(tip_card)
        tip_layout.setContentsMargins(
            self._s(18),
            self._s(13),
            self._s(18),
            self._s(13),
        )
        tip_layout.setSpacing(0)
        tip_layout.addWidget(self.chk_show_tip)

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(
            self._s(22),
            self._s(22),
            self._s(22),
            self._s(22),
        )
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
        self.handle.setToolTip(
            "Hide overlay controls" if self._expanded else "Show overlay controls"
        )
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
# Page
# ============================================================

class OverlayLivePage(QWidget):
    FPS = 30
    COLOR_SIZE = (1920, 1080)
    ALPHA_SLIDER_MAX = 100
    DX_OFFSET_MIN_MM = -50
    DX_OFFSET_MAX_MM = 50

    def __init__(self, state: SessionState, on_complete_changed=None, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

        self.pipeline: rs.pipeline | None = None
        self.config: rs.config | None = None
        self.timer: QTimer | None = None

        self._live_color: np.ndarray | None = None
        self._display_bgr: np.ndarray | None = None

        self._xray_gray_u8: np.ndarray | None = None
        self._overlay_cache = None

        self._alpha: float = 0.50
        self._show_tip: bool = True

        self._base_d_x_mm: float | None = None
        self._current_d_x_mm: float | None = None
        self._dx_offset_mm: int = 0

        self._H_xc_lookup: dict[int, np.ndarray] = {}

        self._build_ui()
        self._connect_signals()
        self._load_xray_from_state()
        self._initialize_dx()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self) -> None:
        self.setObjectName("overlayLivePage")
        self.setFocusPolicy(Qt.StrongFocus)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: black;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.controls = _OverlayControlRail(self)

        self.setStyleSheet(
            """
            QWidget#overlayLivePage {
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
        self.controls.slider_dx.valueChanged.connect(self._on_dx_offset_changed)
        self.controls.chk_show_tip.toggled.connect(self._on_show_tip_toggled)

    # --------------------------------------------------------
    # State / prerequisites
    # --------------------------------------------------------

    def _has_prerequisites(self) -> bool:
        return (
            self.state.K_rgb is not None
            and self.state.K_xray is not None
            and self.state.T_xc is not None
            and self.state.d_x is not None
            and self.state.tip_uv_c is not None
        )

    def _load_xray_from_state(self) -> None:
        if self.state.xray_image_anatomy is None:
            self._xray_gray_u8 = None
            return

        img = np.asarray(self.state.xray_image_anatomy)

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Keep stored image RAW. It is converted to the X-ray working image space
        # only immediately before warping, so H_xc remains unchanged.
        self._xray_gray_u8 = img

    def _initialize_dx(self) -> None:
        if self.state.d_x is None:
            self._base_d_x_mm = None
            self._current_d_x_mm = None
            self._dx_offset_mm = 0
            self._H_xc_lookup = {}
            self.controls.lbl_dx_value.setText("-")
            return

        self._base_d_x_mm = float(self.state.d_x)
        self._current_d_x_mm = float(self.state.d_x)
        self._dx_offset_mm = 0

        self.controls.slider_dx.blockSignals(True)
        self.controls.slider_dx.setRange(self.DX_OFFSET_MIN_MM, self.DX_OFFSET_MAX_MM)
        self.controls.slider_dx.setValue(0)
        self.controls.slider_dx.setSingleStep(1)
        self.controls.slider_dx.setPageStep(10)
        self.controls.slider_dx.blockSignals(False)

        self.controls.lbl_dx_value.setText(f"{self._current_d_x_mm:.1f} mm")
        self.controls.lbl_alpha_value.setText(f"{self._alpha:.2f}")

    # --------------------------------------------------------
    # Homography
    # --------------------------------------------------------

    def _precompute_H_xc_lookup(self) -> None:
        self._H_xc_lookup = {}

        if not self._has_prerequisites():
            return

        if self._base_d_x_mm is None:
            return

        T_xc = np.asarray(self.state.T_xc, dtype=np.float64)
        R_xc = T_xc[:3, :3]
        t_xc = T_xc[:3, 3]

        K_c = np.asarray(self.state.K_rgb, dtype=np.float64)
        K_x = np.asarray(self.state.K_xray, dtype=np.float64)

        for dx_offset_mm in range(self.DX_OFFSET_MIN_MM, self.DX_OFFSET_MAX_MM + 1):
            d_x_mm = float(self._base_d_x_mm + dx_offset_mm)

            H_xc = estimate_plane_induced_homography(
                K_c=K_c,
                R_xc=R_xc,
                t_xc=t_xc,
                K_x=K_x,
                d_x=d_x_mm,
            )
            self._H_xc_lookup[int(dx_offset_mm)] = H_xc

    def _get_current_H_xc(self) -> np.ndarray:
        if not self._H_xc_lookup:
            raise RuntimeError("H_xc lookup is empty.")

        if self._dx_offset_mm not in self._H_xc_lookup:
            raise RuntimeError(
                f"No precomputed H_xc for d_x offset {self._dx_offset_mm} mm."
            )

        return self._H_xc_lookup[self._dx_offset_mm]

    def _invalidate_overlay_cache(self) -> None:
        self._overlay_cache = None
        self.state.H_xc = None

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    def _blend_live_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._xray_gray_u8 is None:
            out = frame_bgr.copy()
        else:
            H_xc = self._get_current_H_xc()
            self.state.H_xc = H_xc

            if self._overlay_cache is None:
                # H_xc/K_xray are defined in XRAY_WORKING_FLIPPED_UV space.
                # Therefore, convert the loaded RAW X-ray image into the same
                # working image space before warping. Do not modify H_xc.
                xray_working_u8 = cv2.flip(self._xray_gray_u8, 1)

                out, cache = blend_xray_overlay(
                    camera_bgr=frame_bgr,
                    xray_gray_u8=xray_working_u8,
                    H_xc=H_xc,
                    alpha=self._alpha,
                )
                self._overlay_cache = cache
            else:
                out = self._overlay_cache.blend(frame_bgr, alpha=self._alpha)

        if self._show_tip:
            out = _draw_point(
                out,
                self.state.tip_uv_c,
                color=(0, 0, 255),
                radius=8,
                cross_size=20,
                thickness=2,
            )

        return out

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
            self._display_bgr = self._blend_live_frame(self._live_color)
            self._update_image_label()

        except Exception as e:
            self._stop_timer()
            self._stop_realsense()
            QMessageBox.critical(
                self,
                "Live Overlay",
                f"Live update failed.\n\n{e}",
            )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    def _update_image_label(self) -> None:
        if self._display_bgr is None:
            return

        pix = _np_to_qpixmap(self._display_bgr)
        if pix.isNull():
            return

        scaled = pix.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    # --------------------------------------------------------
    # Controls callbacks
    # --------------------------------------------------------

    def _on_alpha_changed(self, value: int) -> None:
        self._alpha = float(value) / float(self.ALPHA_SLIDER_MAX)
        self.controls.lbl_alpha_value.setText(f"{self._alpha:.2f}")

    def _on_dx_offset_changed(self, value: int) -> None:
        if self._base_d_x_mm is None:
            return

        self._dx_offset_mm = int(value)
        self._current_d_x_mm = float(self._base_d_x_mm + self._dx_offset_mm)

        self.controls.lbl_dx_value.setText(f"{self._current_d_x_mm:.1f} mm")

        self.state.d_x = float(self._current_d_x_mm)
        self.state.H_xc = self._H_xc_lookup.get(self._dx_offset_mm)

        self._overlay_cache = None

    def _on_show_tip_toggled(self, checked: bool) -> None:
        self._show_tip = bool(checked)

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def on_enter(self) -> None:
        self._load_xray_from_state()
        self._initialize_dx()

        if not self._has_prerequisites():
            QMessageBox.information(
                self,
                "Live Overlay",
                "Missing prerequisites from previous steps.",
            )
            return

        if self._xray_gray_u8 is None:
            QMessageBox.information(
                self,
                "Live Overlay",
                "No anatomical X-ray image available in state.",
            )
            return

        try:
            self._precompute_H_xc_lookup()
            self._invalidate_overlay_cache()

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

    def on_leave(self) -> None:
        self._stop_timer()
        self._stop_realsense()

    def closeEvent(self, event) -> None:
        self.on_leave()
        super().closeEvent(event)