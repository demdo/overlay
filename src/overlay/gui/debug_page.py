# tests/test_overlay_preview_page.py

from __future__ import annotations

import sys
import numpy as np
import cv2

from PySide6.QtWidgets import QApplication, QMainWindow

from overlay.gui.state import SessionState
from overlay.gui.pages.page_overlay_preview import OverlayPreviewPage


QSS = """
QMainWindow { background: #f3f4f6; }
QPushButton {
    background: #2f80ed;
    color: white;
    border: none;
    padding: 10px 14px;
    border-radius: 10px;
    font-weight: 600;
}
QPushButton:hover { background: #256ad1; }
QPushButton:disabled { background: #a6c8ff; color: #f8f9fa; }
"""


# ============================================================
# Dummy data
# ============================================================

def make_dummy_preview_rgb() -> np.ndarray:
    """
    Create a synthetic RGB image for preview.
    """
    h, w = 1080, 1920
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # background gradient
    for y in range(h):
        val = int(50 + 150 * (y / h))
        img[y, :] = (val, val, val)

    # draw some structure
    cv2.rectangle(img, (400, 200), (1500, 900), (200, 200, 200), 2)
    cv2.putText(img, "PREVIEW RGB", (700, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    return img


def make_dummy_state() -> SessionState:
    state = SessionState()

    # ----------------------------------------------------------
    # Intrinsics
    # ----------------------------------------------------------
    state.K_rgb = np.array(
        [
            [1400.0,    0.0, 960.0],
            [   0.0, 1400.0, 540.0],
            [   0.0,    0.0,   1.0],
        ],
        dtype=np.float64,
    )

    state.K_xray = np.array(
        [
            [1000.0,    0.0, 512.0],
            [   0.0, 1000.0, 512.0],
            [   0.0,    0.0,   1.0],
        ],
        dtype=np.float64,
    )

    # ----------------------------------------------------------
    # Transform X-ray -> Camera
    # ----------------------------------------------------------
    state.T_xc = np.eye(4, dtype=np.float64)

    # small translation so something happens
    state.T_xc[2, 3] = 500.0  # 500 mm

    # ----------------------------------------------------------
    # Plane distance
    # ----------------------------------------------------------
    state.d_x = 500.0  # mm

    # ----------------------------------------------------------
    # Preview image + tip
    # ----------------------------------------------------------
    state.preview_rgb = make_dummy_preview_rgb()

    state.tip_uv_c = np.array([960.0, 540.0], dtype=np.float64)
    state.tip_xyz_c = np.array([0.0, 0.0, 500.0], dtype=np.float64)

    return state


# ============================================================
# Main Window
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Test — Overlay Preview Page")
        self.resize(1400, 900)

        self.state = make_dummy_state()

        self.page = OverlayPreviewPage(
            state=self.state,
            on_complete_changed=self.on_complete_changed,
        )

        self.setCentralWidget(self.page)
        self.setStyleSheet(QSS)

        if hasattr(self.page, "on_enter"):
            self.page.on_enter()

    def on_complete_changed(self):
        print("---- on_complete_changed ----")
        print("H_xc:")
        print(self.state.H_xc)
        print("-----------------------------")

    def closeEvent(self, event):
        if hasattr(self.page, "on_leave"):
            self.page.on_leave()
        super().closeEvent(event)


# ============================================================
# Run
# ============================================================

def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()