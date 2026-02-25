# overlay/gui/main.py
#
# Wizard runner (Mode A):
#   1) X-ray intrinsics
#   2) Camera Calibration
#   3) Plane Fitting
#   4) X-ray Marker Selection
#   5) Camera -> X-ray Calibration (PnP)
#
# Uses the per-page template navigation buttons (btn_back / btn_next).
# Sidebar is display-only (wizard gating), no click navigation.

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QMessageBox,
    QPushButton,
)

from overlay.gui.state import SessionState
from overlay.gui.pages.page_xray_intrinsics import XrayIntrinsicsPage
from overlay.gui.pages.page_camera_calibration import CameraCalibrationPage
from overlay.gui.pages.page_plane_fitting import PlaneFittingPage
from overlay.gui.pages.page_xray_marker_selection import XrayMarkerSelectionPage
from overlay.gui.pages.page_camera_to_xray_calibration import CameraToXrayCalibrationPage


# Kept exactly like your current main (same QSS)
QSS = """
QMainWindow { background: #f3f4f6; }
#Sidebar { background: #e9ecef; }
#StepList { background: transparent; border: none; outline: none; }
#StepList::item { padding: 10px 12px; margin: 4px 8px; border-radius: 8px; color: #6c757d; }
#StepList::item:selected { background: #2f80ed; color: white; }
#ContentCard { background: white; border-radius: 14px; }
#ContentTitle { font-size: 18px; font-weight: 600; color: #212529; }
QPushButton { background: #2f80ed; color: white; border: none; padding: 10px 14px; border-radius: 10px; font-weight: 600; }
QPushButton:hover { background: #256ad1; }
QPushButton:disabled { background: #a6c8ff; color: #f8f9fa; }
#NavBar { background: transparent; }
#SecondaryBtn { background: #dee2e6; color: #212529; }
#SecondaryBtn:hover { background: #ced4da; }
#SecondaryBtn:disabled { background: #e9ecef; color: #adb5bd; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("DeCAF Calibration — Mode A (Wizard)")
        self.resize(1200, 700)

        self.state = SessionState()
        self.state.steps_per_edge = 10  # TODO: adjust to your marker grid

        # ORDER:
        # 0 = X-ray intrinsics
        # 1 = Camera Calibration
        # 2 = Plane fitting
        # 3 = X-ray marker selection
        # 4 = Camera -> X-ray calibration (PnP)
        self.current_step = 0

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        # ---------- Sidebar (wizard display only) ----------
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(260)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(12)

        logo_label = QLabel("DeCAF")
        logo_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #212529; margin-left: 8px;")
        sidebar_layout.addWidget(logo_label)

        steps_title = QLabel("Calibration (Mode A)")
        steps_title.setStyleSheet("color: #495057; font-weight: 600; margin-left: 8px;")
        sidebar_layout.addWidget(steps_title)

        self.step_list = QListWidget()
        self.step_list.setObjectName("StepList")
        self.step_list.setFocusPolicy(Qt.NoFocus)
        self.step_list.setDisabled(True)  # wizard navigation only
        sidebar_layout.addWidget(self.step_list, stretch=1)

        self.item_xray = QListWidgetItem("1) Load X-ray intrinsics")
        self.item_rgb = QListWidgetItem("2) Camera Calibration")
        self.item_plane = QListWidgetItem("3) Plane fitting")
        self.item_markers = QListWidgetItem("4) X-ray marker selection")
        self.item_cam2xray = QListWidgetItem("5) Camera → X-ray (PnP)")

        for it in (self.item_xray, self.item_rgb, self.item_plane, self.item_markers, self.item_cam2xray):
            self.step_list.addItem(it)

        settings_btn = QPushButton("Settings")
        settings_btn.setEnabled(False)
        sidebar_layout.addWidget(settings_btn)

        # ---------- Content card ----------
        content_card = QFrame()
        content_card.setObjectName("ContentCard")
        content_layout = QVBoxLayout(content_card)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(12)

        self.lbl_step_title = QLabel("")
        self.lbl_step_title.setObjectName("ContentTitle")
        content_layout.addWidget(self.lbl_step_title)

        self.pages = QStackedWidget()
        content_layout.addWidget(self.pages, stretch=1)

        # Pages
        self.page_xray = XrayIntrinsicsPage(self.state, self.on_complete_changed)
        self.page_cam_calib = CameraCalibrationPage(self.state, self.on_complete_changed)
        self.page_plane = PlaneFittingPage(self.state, self.on_complete_changed)
        self.page_markers = XrayMarkerSelectionPage(self.state, self.on_complete_changed)
        self.page_cam2xray = CameraToXrayCalibrationPage(self.state, self.on_complete_changed)

        self.pages.addWidget(self.page_xray)       # index 0
        self.pages.addWidget(self.page_cam_calib)  # index 1
        self.pages.addWidget(self.page_plane)      # index 2
        self.pages.addWidget(self.page_markers)    # index 3
        self.pages.addWidget(self.page_cam2xray)   # index 4

        # Wire template nav buttons to wizard logic
        self._wire_nav(self.page_xray, idx=0)
        self._wire_nav(self.page_cam_calib, idx=1)
        self._wire_nav(self.page_plane, idx=2)
        self._wire_nav(self.page_markers, idx=3)
        self._wire_nav(self.page_cam2xray, idx=4)

        root_layout.addWidget(sidebar)
        root_layout.addWidget(content_card, stretch=1)
        self.setCentralWidget(root)

        self.setStyleSheet(QSS)

        # Initial enter + UI state
        self.pages.setCurrentIndex(0)
        self._call_page_enter(0)
        self.update_ui()

    # ------------------------------------------------------------------
    # Wizard gating
    # ------------------------------------------------------------------

    def step_complete(self, idx: int) -> bool:
        if idx == 0:  # X-ray intrinsics
            return getattr(self.state, "K_xray", None) is not None
        if idx == 1:  # camera calibration
            return getattr(self.state, "K_rgb", None) is not None
        if idx == 2:  # plane fitting
            return (
                getattr(self.state, "plane_model_c", None) is not None
                and getattr(self.state, "plane_stats", None) is not None
            )
        if idx == 3:  # marker selection
            return bool(getattr(self.state, "xray_points_confirmed", False))
        if idx == 4:  # camera -> xray (PnP)
            return getattr(self.state, "T_cx", None) is not None
        return False

    def can_enter_step(self, idx: int) -> bool:
        # must have completed all previous steps
        for k in range(idx):
            if not self.step_complete(k):
                return False
        return True

    # ------------------------------------------------------------------
    # Navigation wiring (template Back/Next on each page)
    # ------------------------------------------------------------------

    def _wire_nav(self, page: QWidget, idx: int) -> None:
        if hasattr(page, "btn_back"):
            page.btn_back.clicked.connect(lambda: self.go_to(idx - 1))
        if hasattr(page, "btn_next"):
            page.btn_next.clicked.connect(lambda: self.go_to(idx + 1))

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_complete_changed(self):
        self.update_ui()

    # ------------------------------------------------------------------
    # UI update
    # ------------------------------------------------------------------

    def update_ui(self):
        # Switch stack
        self.pages.setCurrentIndex(self.current_step)

        # Sidebar selection (visual only)
        self.step_list.blockSignals(True)
        self.step_list.setCurrentRow(self.current_step)
        self.step_list.blockSignals(False)

        # Sidebar statuses (✅ / ⬜ / 🔒)
        def _set_item_text(item: QListWidgetItem, done: bool, locked: bool, base: str):
            prefix = "✅ " if done else ("🔒 " if locked else "⬜ ")
            item.setText(prefix + base)

        done = [self.step_complete(i) for i in range(self.pages.count())]
        locked = [(not self.can_enter_step(i)) and (not done[i]) for i in range(self.pages.count())]

        _set_item_text(self.item_xray, done[0], locked[0], "1) Load X-ray intrinsics")
        _set_item_text(self.item_rgb, done[1], locked[1], "2) Camera Calibration")
        _set_item_text(self.item_plane, done[2], locked[2], "3) Plane fitting")
        _set_item_text(self.item_markers, done[3], locked[3], "4) X-ray marker selection")
        _set_item_text(self.item_cam2xray, done[4], locked[4], "5) Camera → X-ray (PnP)")

        # Title
        titles = [
            "Step 1 — Load X-ray intrinsics",
            "Step 2 — Camera Calibration",
            "Step 3 — Plane fitting",
            "Step 4 — X-ray marker selection",
            "Step 5 — Camera → X-ray (PnP)",
        ]
        self.lbl_step_title.setText(titles[self.current_step])

        # Enable/disable template Back/Next on current page
        self._update_page_nav_buttons()

    def _update_page_nav_buttons(self) -> None:
        p = self.pages.currentWidget()
        last_idx = self.pages.count() - 1

        # Back: enabled if not first step
        if hasattr(p, "btn_back"):
            p.btn_back.setEnabled(self.current_step > 0)

        # Next: enabled only if current step complete AND next step is enterable
        if hasattr(p, "btn_next"):
            if self.current_step >= last_idx:
                p.btn_next.setEnabled(False)
            else:
                allow = self.step_complete(self.current_step) and self.can_enter_step(self.current_step + 1)
                p.btn_next.setEnabled(bool(allow))

    # ------------------------------------------------------------------
    # Step switching
    # ------------------------------------------------------------------

    def go_to(self, idx: int) -> None:
        idx = int(idx)
        idx = max(0, min(self.pages.count() - 1, idx))

        if idx == self.current_step:
            return

        # Wizard gating:
        # - allow going backwards always
        # - allow going forwards only if current is complete AND target is enterable
        if idx > self.current_step:
            if not self.step_complete(self.current_step):
                QMessageBox.information(self, "Step incomplete", "Please complete the current step first.")
                return
            if not self.can_enter_step(idx):
                QMessageBox.information(self, "Step locked", "Please complete the previous step(s) first.")
                return

        # Leave current page
        self._call_page_leave(self.current_step)

        self.current_step = idx

        # Enter new page
        self.pages.setCurrentIndex(self.current_step)
        self._call_page_enter(self.current_step)

        self.update_ui()

    def _call_page_enter(self, idx: int):
        p = self.pages.widget(idx)
        if p and hasattr(p, "on_enter"):
            p.on_enter()

    def _call_page_leave(self, idx: int):
        p = self.pages.widget(idx)
        if p and hasattr(p, "on_leave"):
            p.on_leave()

    def closeEvent(self, event):
        # Safety: stop any running timers/cameras on app close
        for i in range(self.pages.count()):
            self._call_page_leave(i)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()