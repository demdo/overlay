# debug.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
)

from overlay.gui.state import SessionState
from overlay.gui.pages.page_xray_marker_selection import XrayMarkerSelectionPage


QSS = """
QMainWindow { background: #f3f4f6; }
#ContentCard { background: white; border-radius: 14px; }
#ContentTitle { font-size: 18px; font-weight: 600; color: #212529; }
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
#SecondaryBtn { background: #dee2e6; color: #212529; }
#SecondaryBtn:hover { background: #ced4da; }
#SecondaryBtn:disabled { background: #e9ecef; color: #adb5bd; }
"""


class DebugMarkerSelectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Debug — X-ray Marker Selection")
        self.resize(1400, 900)

        self.state = SessionState()

        # Optional:
        # if SessionState / page logic ever checks these downstream,
        # you can prefill them here. For marker selection alone this is
        # usually not necessary, but harmless if you want it.
        #
        # self.state.xray_points_confirmed = False
        # self.state.xray_points_uv = None
        # self.state.xray_marker_overlay_bgr = None
        # self.state.marker_radius_px = None

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Debug — X-ray Marker Selection Only")
        title.setObjectName("ContentTitle")
        layout.addWidget(title)

        self.page = XrayMarkerSelectionPage(
            state=self.state,
            on_complete_changed=self.on_complete_changed,
        )
        layout.addWidget(self.page, stretch=1)

        self.setCentralWidget(root)
        self.setStyleSheet(QSS)

        if hasattr(self.page, "on_enter"):
            self.page.on_enter()

        self.on_complete_changed()

    def on_complete_changed(self):
        confirmed = bool(self.state.xray_points_confirmed)
        n_pts = 0 if self.state.xray_points_uv is None else int(len(self.state.xray_points_uv))

        print("=" * 60)
        print("MARKER SELECTION STATE")
        print("=" * 60)
        print("confirmed:", confirmed)
        print("num points:", n_pts)

        if confirmed and self.state.xray_points_uv is not None:
            print("xray_points_uv shape:", self.state.xray_points_uv.shape)

    def closeEvent(self, event):
        if hasattr(self.page, "on_leave"):
            self.page.on_leave()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = DebugMarkerSelectionWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()