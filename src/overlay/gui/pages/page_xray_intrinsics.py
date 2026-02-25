# overlay/gui/pages/page_xray_intrinsics.py

from __future__ import annotations

import json
import numpy as np

from PySide6.QtWidgets import (
    QPushButton,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
    QWidget,
)

from overlay.gui.state import SessionState
from overlay.gui.pages.templates.templ_static_image import StaticImagePage


def _hide_widget(w: QWidget | None) -> None:
    if w is None:
        return
    try:
        w.setVisible(False)
    except Exception:
        pass
    try:
        w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    except Exception:
        pass
    try:
        w.setMinimumSize(0, 0)
    except Exception:
        pass
    try:
        w.setMaximumSize(0, 0)
    except Exception:
        pass


def _hide_if_exists(obj, attr_name: str) -> None:
    _hide_widget(getattr(obj, attr_name, None))


class XrayIntrinsicsPage(StaticImagePage):
    """
    Step — Load X-ray intrinsics (K_xray)

    Special-case page:
      - NO image/viewport on the left
      - NO instructions box
      - stats only:
          K_xray: -                (before)
          K_xray: np.array (3×3)   (after load)
    """

    def __init__(self, state: SessionState, on_complete_changed, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

        # ======================================================
        # RIGHT: controls
        # ======================================================
        self.btn_load = QPushButton("Load X-ray intrinsics (JSON)")
        self.btn_load.clicked.connect(self.load_intrinsics)
        self.btn_load.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.controls_content.addWidget(self.btn_load)

        # ======================================================
        # SPECIAL-CASE UI: hide instructions panel completely
        # ======================================================
        _hide_if_exists(self, "instructions_box")
        _hide_if_exists(self, "instructions_group")
        _hide_if_exists(self, "instructions_panel")
        _hide_if_exists(self, "instructions_frame")
        _hide_if_exists(self, "instructions_label")

        # ======================================================
        # SPECIAL-CASE UI: hide the entire image/viewport area
        # ======================================================
        # Different templates name these differently; hide all common ones robustly.
        _hide_if_exists(self, "image_label")
        _hide_if_exists(self, "viewport")
        _hide_if_exists(self, "viewport_widget")
        _hide_if_exists(self, "viewport_frame")
        _hide_if_exists(self, "viewport_container")
        _hide_if_exists(self, "image_container")
        _hide_if_exists(self, "image_frame")
        _hide_if_exists(self, "left_panel")
        _hide_if_exists(self, "left_widget")
        _hide_if_exists(self, "workspace_widget")

        # Also make sure no image is set (even if the widgets are hidden)
        try:
            self.set_viewport_background(active=False)
        except Exception:
            pass
        try:
            self.set_image(None)
        except Exception:
            pass

        self.refresh()

    def is_complete(self) -> bool:
        return getattr(self.state, "K_xray", None) is not None

    def refresh(self) -> None:
        # Stats only (init-safe: do NOT call on_complete_changed here)
        if self.is_complete():
            self.set_stats_rows([
                ("K_xray", "np.array (3×3)"),
            ])
        else:
            self.set_stats_rows([
                ("K_xray", "-"),
            ])

        # Keep image/viewport off (even if template changes later)
        try:
            self.set_viewport_background(active=False)
        except Exception:
            pass
        try:
            self.set_image(None)
        except Exception:
            pass

    def load_intrinsics(self) -> None:
        dlg = QFileDialog(self.window(), "Select X-ray intrinsics JSON")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("JSON Files (*.json);;All Files (*)")
    
        # keep dialog style consistent + avoid native-dialog edge cases
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
    
        if dlg.exec() != QFileDialog.Accepted:
            return
    
        files = dlg.selectedFiles()
        path = files[0] if files else ""
        if not path:
            return
    
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
    
            K = np.array(data, dtype=float)
            if K.shape != (3, 3):
                raise ValueError("Intrinsics matrix must be 3×3.")
    
            self.state.K_xray = K
    
            self.refresh()
            # Trigger UI update ONLY after actual user action (prevents init crash)
            self.on_complete_changed()
    
        except Exception as e:
            QMessageBox.critical(self, "Failed to load X-ray intrinsics", str(e))
            
            