# -*- coding: utf-8 -*-
"""
debug_xray_click_principal_point.py

Lädt:
- ein X-ray Bild (.bmp)
- eine NPZ mit K_xray

Zeigt:
- Principal Point aus K_xray
- Pixelkoordinaten per Mausklick

Bedienung:
- Linksklick  -> Punkt setzen und (u,v) anzeigen
- r          -> Klickpunkte zurücksetzen
- q / ESC    -> beenden
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog


# ============================================================
# Qt helpers
# ============================================================

def _qt() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _pick_file(title: str, filt: str) -> Path | None:
    _qt()
    path, _ = QFileDialog.getOpenFileName(None, title, "", filt)
    return Path(path) if path else None


# ============================================================
# IO
# ============================================================

def load_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not load image: {path}")
    return img


def load_kx(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as z:
        for key in ("K", "Kx", "K_xray"):
            if key in z.files:
                K = np.asarray(z[key], dtype=np.float64)
                if K.shape != (3, 3):
                    raise RuntimeError(f"{key} has invalid shape: {K.shape}")
                return K
    raise RuntimeError("No key 'K', 'Kx', or 'K_xray' found in NPZ.")


# ============================================================
# Drawing
# ============================================================

def draw_cross(
    img: np.ndarray,
    pt: tuple[int, int],
    color: tuple[int, int, int],
    size: int = 12,
    thickness: int = 2,
) -> None:
    x, y = pt
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def draw_principal_point(img: np.ndarray, K: np.ndarray) -> None:
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    p = (int(round(cx)), int(round(cy)))
    draw_cross(img, p, (0, 255, 255), size=14, thickness=2)
    cv2.circle(img, p, 8, (0, 255, 255), 2, cv2.LINE_AA)

    txt = f"PP_xray = ({cx:.2f}, {cy:.2f})"
    cv2.putText(
        img,
        txt,
        (p[0] + 15, p[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def redraw_canvas(
    base_img: np.ndarray,
    Kx: np.ndarray,
    clicked_points: list[tuple[int, int]],
) -> np.ndarray:
    vis = base_img.copy()

    draw_principal_point(vis, Kx)

    for i, (u, v) in enumerate(clicked_points):
        draw_cross(vis, (u, v), (0, 0, 255), size=10, thickness=2)
        cv2.circle(vis, (u, v), 6, (0, 0, 255), 2, cv2.LINE_AA)

        label = f"P{i}: ({u}, {v})"
        cv2.putText(
            vis,
            label,
            (u + 12, v - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    info1 = "Left click: add point"
    info2 = "r: reset   q/ESC: quit"
    cv2.putText(vis, info1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, info2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


# ============================================================
# Main
# ============================================================

def main() -> None:
    img_path = _pick_file("Select X-ray BMP", "Bitmap image (*.bmp);;All files (*)")
    if img_path is None:
        print("No X-ray image selected.")
        return

    kx_path = _pick_file("Select K_xray NPZ", "NPZ (*.npz)")
    if kx_path is None:
        print("No K_xray NPZ selected.")
        return

    img = load_image_bgr(img_path)
    Kx = load_kx(kx_path)

    cx = float(Kx[0, 2])
    cy = float(Kx[1, 2])

    print("\n" + "=" * 70)
    print("K_xray")
    print("=" * 70)
    print(np.array2string(Kx, precision=6, suppress_small=False))
    print(f"Principal point: cx={cx:.6f}, cy={cy:.6f}")

    clicked_points: list[tuple[int, int]] = []
    win = "X-ray click debug"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    state = {"vis": redraw_canvas(img, Kx, clicked_points)}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((int(x), int(y)))
            print(f"Clicked pixel: u={x}, v={y}")
            state["vis"] = redraw_canvas(img, Kx, clicked_points)

    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, state["vis"])
        k = cv2.waitKey(20) & 0xFF

        if k in (27, ord("q")):
            break
        elif k == ord("r"):
            clicked_points.clear()
            state["vis"] = redraw_canvas(img, Kx, clicked_points)
            print("Reset clicked points.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()