# overlay/gui/pages/page_zoom_dummy.py

from __future__ import annotations

import numpy as np
import cv2

from PySide6.QtWidgets import QSizePolicy, QPushButton

from overlay.gui.pages.templates.templ_static_image import StaticImagePage
from overlay.gui.widgets.widget_zoom_view import ZoomView


def _make_dummy_image(w=1024, h=1024) -> np.ndarray:
    """
    Creates a synthetic 1024x1024 uint8 BGR image
    with a dense ROI cluster to simulate your PnP case.
    """

    # --- soft radial gradient (like X-ray feel) ---
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w // 2, h // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    r_norm = np.clip(1.0 - r / (0.8 * (w // 2)), 0, 1)
    gray = (r_norm * 180 + 40).astype(np.uint8)

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # --- light grid background ---
    for yy in range(0, h, 64):
        cv2.line(img, (0, yy), (w, yy), (50, 50, 50), 1, cv2.LINE_AA)
    for xx in range(0, w, 64):
        cv2.line(img, (xx, 0), (xx, h), (50, 50, 50), 1, cv2.LINE_AA)

    # --- dense ROI cluster in center ---
    cluster_half = 10
    spacing = 20

    for j in range(-cluster_half, cluster_half + 1):
        for i in range(-cluster_half, cluster_half + 1):
            u = cx + i * spacing
            v = cy + j * spacing

            # measured (red)
            cv2.drawMarker(
                img,
                (u, v),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=2,
            )

            # projected (cyan) slightly shifted
            cv2.drawMarker(
                img,
                (u + 3, v + 2),
                (255, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=2,
            )

    return img


class ZoomDummyPage(StaticImagePage):
    """
    Dummy page to test interactive zoom inside StaticImagePage template.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Replace template image_label with zoomable view
        self.zoom_view = ZoomView(self.image_label)
        self.zoom_view.setGeometry(0, 0, self.image_label.width(), self.image_label.height())
        self.zoom_view.show()

        # Right controls (optional)
        self.btn_reset_zoom = QPushButton("Reset zoom (RMB also works)")
        self.btn_reset_zoom.clicked.connect(self.zoom_view.reset_zoom)
        self.btn_reset_zoom.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.controls_content.addWidget(self.btn_reset_zoom)

        # Instructions
        self.instructions_label.setText(
            "Drag with LEFT mouse button to zoom into a rectangle.\n"
            "Right-click or double-click to reset.\n"
            "This is a dummy page to validate zoom behavior."
        )

        # Stats
        self.set_stats_rows([
            ("Zoom active", "No"),
            ("Hint", "Drag to zoom"),
        ])

        # Set dummy image
        self._img = _make_dummy_image()
        self.set_viewport_background(active=True)
        self.zoom_view.set_image(self._img)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.zoom_view.setGeometry(0, 0, self.image_label.width(), self.image_label.height())

    def on_enter(self) -> None:
        # keep stats in sync
        self.set_stats_rows([
            ("Zoom active", "Yes" if self.zoom_view.has_zoom() else "No"),
            ("Hint", "RMB / double-click resets"),
        ])