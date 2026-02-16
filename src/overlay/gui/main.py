# overlay/gui/main.py

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QListWidget, QListWidgetItem,
    QStackedWidget, QHBoxLayout, QVBoxLayout, QFrame, QMessageBox, QPushButton
)

from overlay.gui.state import SessionState
from overlay.gui.pages.page_xray_intrinsics import XrayIntrinsicsPage
from overlay.gui.pages.page_camera_calibration import CameraCalibrationPage
from overlay.gui.pages.page_xray_marker_selection import XrayMarkerSelectionPage
from overlay.gui.pages.page_plane_fitting import PlaneFittingPage
from overlay.gui.pages.page_camera_to_xray_calibration import CameraToXrayCalibrationPage


QSS = """
QMainWindow {
    background: #f3f4f6;
}
#Sidebar {
    background: #e9ecef;
}
#StepList {
    background: transparent;
    border: none;
    outline: none;
}
#StepList::item {
    padding: 10px 12px;
    margin: 4px 8px;
    border-radius: 8px;
    color: #6c757d;
}
#StepList::item:selected {
    background: #2f80ed;
    color: white;
}
#ContentCard {
    background: white;
    border-radius: 14px;
}
#ContentTitle {
    font-size: 18px;
    font-weight: 600;
    color: #212529;
}
QPushButton {
    background: #2f80ed;
    color: white;
    border: none;
    padding: 10px 14px;
    border-radius: 10px;
    font-weight: 600;
}
QPushButton:hover {
    background: #256ad1;
}
QPushButton:disabled {
    background: #a6c8ff;
    color: #f8f9fa;
}
#NavBar {
    background: transparent;
}
#SecondaryBtn {
    background: #dee2e6;
    color: #212529;
}
#SecondaryBtn:hover {
    background: #ced4da;
}
#SecondaryBtn:disabled {
    background: #e9ecef;
    color: #adb5bd;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeCAF Calibration — Mode A (Wizard)")
        self.resize(1200, 700)

        self.state = SessionState()
        self.state.steps_per_edge = 10   # TODO: an dein Marker-Grid anpassen

        # NEW ORDER:
        # 0 = X-ray intrinsics
        # 1 = Camera Calibration
        # 2 = Plane fitting
        # 3 = X-ray marker selection
        # 4 = Camera to X-ray Calibration
        self.current_step = 0

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        # --- Sidebar ---
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
        self.step_list.setDisabled(True)  # IMPORTANT: wizard navigation only
        sidebar_layout.addWidget(self.step_list, stretch=1)

        # Steps (store items so we can update labels)
        self.item_xray = QListWidgetItem("1) Load X-ray intrinsics")
        self.item_rgb = QListWidgetItem("2) Camera Calibration")
        self.item_plane = QListWidgetItem("3) Plane fitting")
        self.item_markers = QListWidgetItem("4) X-ray marker selection")
        self.item_cam2xray = QListWidgetItem("5) Camera → X-ray (PnP)")
        
        self.step_list.addItem(self.item_xray)
        self.step_list.addItem(self.item_rgb)
        self.step_list.addItem(self.item_plane)
        self.step_list.addItem(self.item_markers)
        self.step_list.addItem(self.item_cam2xray)

        # Optional placeholder button (disabled)
        settings_btn = QPushButton("Settings")
        settings_btn.setEnabled(False)
        sidebar_layout.addWidget(settings_btn)

        # --- Content card ---
        content_card = QFrame()
        content_card.setObjectName("ContentCard")
        content_layout = QVBoxLayout(content_card)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(12)

        title = QLabel("Step Workspace")
        title.setObjectName("ContentTitle")
        content_layout.addWidget(title)

        # Pages stack
        self.pages = QStackedWidget()
        content_layout.addWidget(self.pages, stretch=1)

        self.page_xray = XrayIntrinsicsPage(self.state, self.on_complete_changed)
        self.page_cam_calib = CameraCalibrationPage(self.state, self.on_complete_changed)
        self.page_plane = PlaneFittingPage(self.state, self.on_complete_changed)
        self.page_markers = XrayMarkerSelectionPage(self.state, self.on_complete_changed)
        self.page_cam2xray = CameraToXrayCalibrationPage(self.state, self.on_complete_changed)

        self.pages.addWidget(self.page_xray)            # index 0
        self.pages.addWidget(self.page_cam_calib)       # index 1
        self.pages.addWidget(self.page_plane)           # index 2
        self.pages.addWidget(self.page_markers)         # index 3
        self.pages.addWidget(self.page_cam2xray)        # index 4

        # Bottom navigation bar
        nav = QFrame()
        nav.setObjectName("NavBar")
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(12)

        nav_layout.addStretch(1)

        self.btn_back = QPushButton("Back")
        self.btn_back.setObjectName("SecondaryBtn")
        self.btn_next = QPushButton("Next")

        self.btn_back.clicked.connect(self.go_back)
        self.btn_next.clicked.connect(self.go_next)

        self.btn_back.setFocusPolicy(Qt.NoFocus)
        self.btn_next.setFocusPolicy(Qt.NoFocus)

        self.btn_back.setAutoDefault(False)
        self.btn_back.setDefault(False)
        self.btn_next.setAutoDefault(False)
        self.btn_next.setDefault(False)

        nav_layout.addWidget(self.btn_back)
        nav_layout.addWidget(self.btn_next)

        content_layout.addWidget(nav)

        # Assemble
        root_layout.addWidget(sidebar)
        root_layout.addWidget(content_card, stretch=1)
        self.setCentralWidget(root)

        self.setStyleSheet(QSS)
        self.update_ui()

    # -------- Wizard logic --------

    def step_complete(self, idx: int) -> bool:
        if idx == 0:   # Step 1: X-ray intrinsics
            return self.state.K_xray is not None
        if idx == 1:   # Step 2: Camera Calibration
            return self.state.K_rgb is not None
        if idx == 2:   # Step 3: plane fitting
            return (
                getattr(self.state, "plane_model_c", None) is not None
                and getattr(self.state, "plane_stats", None) is not None
            )
        if idx == 3:   # Step 4: marker selection (minimal gate)
            return self.state.xray_image is not None
        if idx == 4:   # Step 5: Camera -> X-ray (PnP)
            return getattr(self.state, "T_xray_from_cam_4x4", None) is not None
        return False

    def can_enter_step(self, idx: int) -> bool:
        if idx == 1 and not self.step_complete(0):  # Step 2 requires Step 1
            return False
        if idx == 2 and not self.step_complete(1):  # Step 3 requires Step 2
            return False
        if idx == 3 and not self.step_complete(2):  # Step 4 requires Step 3
            return False
        if idx == 4 and not self.step_complete(3):  # Step 5 requires Step 4
            return False
        return True

    def on_complete_changed(self):
        self.update_ui()

    def update_ui(self):
        # ------------------------------------------------------------
        # 1) Ensure pages exist / refresh current data
        # ------------------------------------------------------------
        # Refresh all pages (they should internally decide what to do if inputs missing)
        if hasattr(self, "page_xray") and hasattr(self.page_xray, "refresh"):
            self.page_xray.refresh()
    
        if hasattr(self, "page_rgb") and hasattr(self.page_rgb, "refresh"):
            self.page_rgb.refresh()
    
        if hasattr(self, "page_plane") and hasattr(self.page_plane, "refresh"):
            self.page_plane.refresh()
    
        if hasattr(self, "page_markers") and hasattr(self.page_markers, "refresh"):
            self.page_markers.refresh()
    
        if hasattr(self, "page_cam2xray") and hasattr(self.page_cam2xray, "refresh"):
            self.page_cam2xray.refresh()
    
        # ------------------------------------------------------------
        # 2) Update stacked widget to current step
        # ------------------------------------------------------------
        if hasattr(self, "pages"):
            self.pages.setCurrentIndex(self.current_step)
    
        # ------------------------------------------------------------
        # 3) Update sidebar item labels (✅ / ⬜ / 🔒)
        # ------------------------------------------------------------
        def _set_item_text(item, done: bool, locked: bool, base: str):
            if item is None:
                return
            prefix = "✅ " if done else ("🔒 " if locked else "⬜ ")
            item.setText(prefix + base)
    
        # Determine completion status
        xray_done = self.step_complete(0)
        rgb_done = self.step_complete(1)
        plane_done = self.step_complete(2)
        markers_done = self.step_complete(3)
        cam2xray_done = self.step_complete(4)
    
        # Determine locking status (locked if cannot enter AND not already done)
        xray_locked = (not self.can_enter_step(0)) and (not xray_done)
        rgb_locked = (not self.can_enter_step(1)) and (not rgb_done)
        plane_locked = (not self.can_enter_step(2)) and (not plane_done)
        markers_locked = (not self.can_enter_step(3)) and (not markers_done)
        cam2xray_locked = (not self.can_enter_step(4)) and (not cam2xray_done)
    
        _set_item_text(self.item_xray, xray_done, xray_locked, "1) Load X-ray intrinsics")
        _set_item_text(self.item_rgb, rgb_done, rgb_locked, "2) Camera Calibration")
        _set_item_text(self.item_plane, plane_done, plane_locked, "3) Plane fitting")
        _set_item_text(self.item_markers, markers_done, markers_locked, "4) X-ray marker selection")
        _set_item_text(self.item_cam2xray, cam2xray_done, cam2xray_locked, "5) Camera → X-ray (PnP)")
    
        # Keep sidebar selection synced (without re-triggering signals too aggressively)
        if hasattr(self, "step_list"):
            if self.step_list.currentRow() != self.current_step:
                self.step_list.blockSignals(True)
                self.step_list.setCurrentRow(self.current_step)
                self.step_list.blockSignals(False)
    
        # ------------------------------------------------------------
        # 4) Prev/Next button enabled state
        # ------------------------------------------------------------
        # Prev enabled if not first step
        if hasattr(self, "btn_prev"):
            self.btn_prev.setEnabled(self.current_step > 0)
    
        # Next enabled if there is a next step AND current step is complete
        if hasattr(self, "btn_next") and hasattr(self, "pages"):
            last_idx = self.pages.count() - 1
            if self.current_step < last_idx:
                self.btn_next.setEnabled(self.step_complete(self.current_step))
            else:
                self.btn_next.setEnabled(False)
    
        # ------------------------------------------------------------
        # 5) Optional: status label / title (if you have one)
        # ------------------------------------------------------------
        if hasattr(self, "lbl_step_title"):
            titles = [
                "Step 1 — Load X-ray intrinsics",
                "Step 2 — Camera Calibration",
                "Step 3 — Plane fitting",
                "Step 4 — X-ray marker selection",
                "Step 5 — Camera → X-ray (PnP)",
            ]
            if 0 <= self.current_step < len(titles):
                self.lbl_step_title.setText(titles[self.current_step])


    def go_next(self):
        if not self.step_complete(self.current_step):
            QMessageBox.information(self, "Step incomplete", "Please complete the current step first.")
            return

        next_step = self.current_step + 1
        if next_step >= self.pages.count():
            QMessageBox.information(self, "End", "No further steps implemented yet.")
            return

        if not self.can_enter_step(next_step):
            QMessageBox.information(self, "Step locked", "Please complete the previous step first.")
            return

        self.current_step = next_step
        self.update_ui()

    def go_back(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_ui()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
