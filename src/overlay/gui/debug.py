# debug.py
# -*- coding: utf-8 -*-
"""
Run ONLY the PlaneFittingPage (no wizard, no other pages).

Usage:
    python debug.py

Note:
- PlaneFittingPage currently requires state.K_rgb != None (guard in start_clicked),
  so we provide a dummy K_rgb here for debugging.
"""

from __future__ import annotations

import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow

from overlay.gui.state import SessionState
from overlay.gui.pages.page_plane_fitting import PlaneFittingPage


def main() -> None:
    app = QApplication(sys.argv)

    win = QMainWindow()
    win.setWindowTitle("DEBUG — Plane Fitting")

    state = SessionState()

    # Dummy intrinsics (replace with real calibrated K if you want)
    fx = fy = 1000.0
    cx, cy = 960.0, 540.0
    state.K_rgb = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    def on_complete_changed() -> None:
        pass

    page = PlaneFittingPage(state=state, on_complete_changed=on_complete_changed)

    win.setCentralWidget(page)
    win.resize(1400, 800)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()