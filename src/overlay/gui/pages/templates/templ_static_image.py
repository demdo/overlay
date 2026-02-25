# overlay/gui/pages/templates/page_static_image.py

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from overlay.gui.pages.templates.templ_base_image import BaseImagePage


class StaticImagePage(BaseImagePage):
    """
    Static image page template (no live video, no timers).

    Layout comes from BaseImagePage:
      LEFT:
        [ Image viewport ]
        [ Instructions ]   (under image)

      RIGHT:
        [ Controls ]
        [ Stats ]

    Intended use:
      - call set_image(...) once after loading/computation
      - call request_redraw() after overlay/state changes
      - optionally override draw_overlay()
    """

    # X-ray is typically 1024x1024 (square). BaseImagePage scales down as needed.
    VIEWPORT_SIZE: Tuple[int, int] = (512, 512)

    # Never crop calibration images.
    RENDER_MODE: str = "fit"

    def __init__(self, parent=None):
        self._frame: Optional[np.ndarray] = None  # grayscale (H,W) or BGR (H,W,3)
        super().__init__(parent)
        self.set_viewport_background(active=False)
        self.clear_view()

    # ------------------------------------------------------------------
    # Convenience API (static pages use this instead of timers)
    # ------------------------------------------------------------------

    def set_image(self, img: Optional[np.ndarray]) -> None:
        """Set the image shown in the viewport (grayscale or BGR)."""
        self._frame = img
        self.request_redraw()

    def clear_image(self) -> None:
        """Remove the current image from the viewport."""
        self._frame = None
        self.request_redraw()

    def request_redraw(self) -> None:
        """Call after any changes that should be reflected in the viewport."""
        self.update_view()

    # ------------------------------------------------------------------
    # BaseImagePage hook
    # ------------------------------------------------------------------

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the stored static frame for BaseImagePage.update_view()."""
        return self._frame
