import json
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QStackedWidget
)

from overlay.gui.state import SessionState


def format_K(K: np.ndarray) -> str:
    return "\n".join(
        "[ " + "  ".join(f"{v:9.3f}" for v in row) + " ]"
        for row in K
    )


class XrayIntrinsicsPage(QWidget):
    def __init__(self, state: SessionState, on_complete_changed):
        super().__init__()
        self.state = state
        self.on_complete_changed = on_complete_changed

        root = QVBoxLayout(self)
        root.setSpacing(12)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, stretch=1)

        # ---------- View A: Load ----------
        view_load = QWidget()
        load_layout = QVBoxLayout(view_load)
        load_layout.setSpacing(10)

        self.load_info = QLabel("Load X-ray intrinsics as a 3×3 matrix JSON.")
        self.btn_load = QPushButton("Load X-ray intrinsics (JSON)")
        self.btn_load.clicked.connect(self.load_intrinsics)

        load_layout.addWidget(self.load_info)
        load_layout.addWidget(self.btn_load)
        load_layout.addStretch(1)
        self.stack.addWidget(view_load)

        # ---------- View B: Summary ----------
        view_summary = QWidget()
        sum_layout = QVBoxLayout(view_summary)
        sum_layout.setSpacing(10)

        self.summary_title = QLabel("X-ray intrinsics loaded:")
        self.summary_mat = QLabel("")
        self.summary_mat.setStyleSheet("font-family: Consolas, Menlo, Monaco, monospace;")

        self.btn_reload = QPushButton("Load different X-ray intrinsics…")
        self.btn_reload.clicked.connect(self.load_intrinsics)

        self.hint = QLabel("Click Next to continue.")
        self.hint.setStyleSheet("color: #6c757d;")

        sum_layout.addWidget(self.summary_title)
        sum_layout.addWidget(self.summary_mat)
        sum_layout.addWidget(self.btn_reload)
        sum_layout.addWidget(self.hint)
        sum_layout.addStretch(1)
        self.stack.addWidget(view_summary)

        self.refresh()

    def is_complete(self) -> bool:
        return self.state.K_xray is not None

    def refresh(self):
        if self.is_complete():
            self.summary_mat.setText(format_K(self.state.K_xray))
            self.stack.setCurrentIndex(1)
        else:
            self.stack.setCurrentIndex(0)

    def load_intrinsics(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select X-ray intrinsics JSON", "", "JSON Files (*.json)"
        )
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
            self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(self, "Failed to load X-ray intrinsics", str(e))
