# -*- coding: utf-8 -*-
"""
debug_xray_axes.py

Laedt ein X-Ray Bild und zeichnet das Bildkoordinatensystem ein:

    Ursprung  = Bildmitte
    x-Achse   = u-Richtung (horizontal, nach rechts)
    y-Achse   = v-Richtung (vertikal, nach unten)

Nur anzeigen, kein Speichern, kein uv_xray noetig.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog


AXIS_LENGTH_PX   = 150
FONT_SCALE       = 1.6
FONT_THICKNESS   = 2
LINE_THICKNESS   = 3
ARROW_TIP_LENGTH = 0.12

COLOR_X      = (0,   80, 255)   # Rot    -> x (u)
COLOR_Y      = (0,  200,  50)   # Gruen  -> y (v)
COLOR_ORIGIN = (255, 200,   0)  # Cyan   -> Ursprung


def _qt() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _pick_image(title: str) -> Path | None:
    _qt()
    p, _ = QFileDialog.getOpenFileName(
        None, title, "",
        "Bilder (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
    )
    return Path(p) if p else None


def _draw_axes(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    h, w = out.shape[:2]
    ox, oy = w // 2, h // 2

    # x-Achse -> u (nach rechts)
    x_end = (ox + AXIS_LENGTH_PX, oy)
    cv2.arrowedLine(out, (ox, oy), x_end,
                    COLOR_X, LINE_THICKNESS, cv2.LINE_AA,
                    tipLength=ARROW_TIP_LENGTH)
    cv2.putText(out, "x  (u, rechts)", (x_end[0] + 10, x_end[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                COLOR_X, FONT_THICKNESS, cv2.LINE_AA)

    # y-Achse -> v (nach unten)
    y_end = (ox, oy + AXIS_LENGTH_PX)
    cv2.arrowedLine(out, (ox, oy), y_end,
                    COLOR_Y, LINE_THICKNESS, cv2.LINE_AA,
                    tipLength=ARROW_TIP_LENGTH)
    cv2.putText(out, "y  (v, unten)", (y_end[0] + 10, y_end[1]),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                COLOR_Y, FONT_THICKNESS, cv2.LINE_AA)

    # Ursprung
    cv2.circle(out, (ox, oy), 10, COLOR_ORIGIN, -1, cv2.LINE_AA)
    cv2.putText(out, f"O ({ox}, {oy})", (ox + 14, oy - 14),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.65,
                COLOR_ORIGIN, FONT_THICKNESS, cv2.LINE_AA)

    # Bildgroesse oben links
    cv2.putText(out, f"{w} x {h} px", (20, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)

    return out


def main() -> None:
    _qt()

    p_img = _pick_image("X-Ray Bild laden")
    if p_img is None:
        print("[ABBRUCH] Kein Bild gewaehlt.")
        return

    img = cv2.imread(str(p_img), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] Bild konnte nicht geladen werden: {p_img}")
        return
    print(f"[OK] {p_img.name}  shape={img.shape}")

    annotated = _draw_axes(img)

    win = "X-Ray KOS  (beliebige Taste zum Schliessen)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    h, w = annotated.shape[:2]
    cv2.resizeWindow(win, min(w, 1400), min(h, 900))
    cv2.imshow(win, annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()