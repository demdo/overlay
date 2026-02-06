# overlay/gui/main.py

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QListWidget, QListWidgetItem,
    QStackedWidget, QHBoxLayout, QVBoxLayout, QFrame, QMessageBox, QPushButton
)

from overlay.gui.state import SessionState
from overlay.gui.pages.page_rgb_intrinsics import RgbIntrinsicsPage
from overlay.gui.pages.page_xray_intrinsics import XrayIntrinsicsPage
from overlay.gui.pages.page_xray_marker_selection import XrayMarkerSelectionPage
from overlay.gui.pages.page_plane_fitting import PlaneFittingPage


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
        self.current_step = 0  # 0=RGB, 1=X-ray, 2=Markers, 3=Plane

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
        self.item_rgb = QListWidgetItem("1) Load RGB intrinsics")
        self.item_xray = QListWidgetItem("2) Load X-ray intrinsics")
        self.item_markers = QListWidgetItem("3) X-ray marker selection")
        self.item_plane = QListWidgetItem("4) Plane fitting")
        self.step_list.addItem(self.item_rgb)
        self.step_list.addItem(self.item_xray)
        self.step_list.addItem(self.item_markers)
        self.step_list.addItem(self.item_plane)

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

        self.page_rgb = RgbIntrinsicsPage(self.state, self.on_complete_changed)
        self.page_xray = XrayIntrinsicsPage(self.state, self.on_complete_changed)
        self.page_markers = XrayMarkerSelectionPage(self.state, self.on_complete_changed)
        self.page_plane = PlaneFittingPage(self.state, self.on_complete_changed)  # NEW

        self.pages.addWidget(self.page_rgb)      # index 0
        self.pages.addWidget(self.page_xray)     # index 1
        self.pages.addWidget(self.page_markers)  # index 2
        self.pages.addWidget(self.page_plane)    # index 3

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
        if idx == 0:
            return self.state.K_rgb is not None
        if idx == 1:
            return self.state.K_xray is not None
        if idx == 2:
            # marker selection page sets xray_image
            return self.state.xray_image is not None
        if idx == 3:
            # plane fitting page sets plane_model + stats (and pts3d)
            return getattr(self.state, "plane_model_c", None) is not None and getattr(self.state, "plane_stats", None) is not None
        return False

    def can_enter_step(self, idx: int) -> bool:
        # Step 2 requires Step 1
        if idx == 1 and not self.step_complete(0):
            return False
        # Step 3 requires Step 2
        if idx == 2 and not self.step_complete(1):
            return False
        # Step 4 requires Step 3
        if idx == 3 and not self.step_complete(2):
            return False
        return True

    def on_complete_changed(self):
        # Called by pages after successful load / reload
        self.update_ui()

    def update_ui(self):
        # Enforce gating if current step becomes invalid
        if not self.can_enter_step(self.current_step):
            self.current_step = 0

        # Show current page
        self.pages.setCurrentIndex(self.current_step)

        # Refresh pages so they show Load vs Summary correctly
        self.page_rgb.refresh()
        self.page_xray.refresh()
        self.page_markers.refresh()
        # Plane page may not have refresh() (safe call)
        if hasattr(self.page_plane, "refresh"):
            self.page_plane.refresh()

        # Back enabled only if not first step
        self.btn_back.setEnabled(self.current_step > 0)

        # Next enabled only if current step complete AND next step (if any) is allowed
        if self.current_step == 0:
            self.btn_next.setEnabled(self.step_complete(0))
        elif self.current_step == 1:
            self.btn_next.setEnabled(self.step_complete(1))
        elif self.current_step == 2:
            self.btn_next.setEnabled(self.step_complete(2))
        elif self.current_step == 3:
            self.btn_next.setEnabled(False)
        else:
            self.btn_next.setEnabled(False)

        # Sidebar: highlight current step
        self.step_list.setCurrentRow(self.current_step)

        # Sidebar: show done/locked via text
        rgb_done = self.step_complete(0)
        xray_done = self.step_complete(1)
        markers_done = self.step_complete(2)
        plane_done = self.step_complete(3)

        xray_locked = not self.can_enter_step(1)
        markers_locked = not self.can_enter_step(2) and not markers_done
        plane_locked = not self.can_enter_step(3) and not plane_done

        self.item_rgb.setText(("✅ " if rgb_done else "⬜ ") + "1) Load RGB intrinsics")
        self.item_xray.setText(("🔒 " if xray_locked else ("✅ " if xray_done else "⬜ ")) + "2) Load X-ray intrinsics")
        self.item_markers.setText(("🔒 " if markers_locked else ("✅ " if markers_done else "⬜ ")) + "3) X-ray marker selection")
        self.item_plane.setText(("🔒 " if plane_locked else ("✅ " if plane_done else "⬜ ")) + "4) Plane fitting")

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
