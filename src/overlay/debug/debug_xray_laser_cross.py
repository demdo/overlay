# -*- coding: utf-8 -*-
"""
debug_laser_cross.py

Select X-ray image via Qt file dialog.
Click the laser-cross and print coordinates.
Optionally compare with principal point from K_x.
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog


def load_kx(npz_path: str):
    data = np.load(npz_path)
    for key in ("K", "Kx", "K_xray"):
        if key in data:
            return data[key]
    raise KeyError("No K found in npz.")


def main():
    # --- Qt file dialog ---
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    image_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select X-ray image",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
    )

    if not image_path:
        print("No image selected.")
        return

    kx_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select K_x (optional)",
        "",
        "NPZ files (*.npz)"
    )

    # --- load image ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Image not found!")

    H, W = img.shape

    # --- load intrinsics ---
    cx = cy = None
    if kx_path:
        K = load_kx(kx_path)
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        print("\n=== K_x ===")
        print(K)
        print(f"Principal point: ({cx:.2f}, {cy:.2f})")

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def draw():
        tmp = vis.copy()

        # image center
        center = (W // 2, H // 2)
        cv2.drawMarker(tmp, center, (0, 165, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(tmp, f"center {center}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

        # principal point
        if cx is not None:
            pp = (int(cx), int(cy))
            cv2.drawMarker(tmp, pp, (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(tmp, f"pp ({cx:.1f}, {cy:.1f})", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

        return tmp

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("\n=== CLICK ===")
            print(f"u = {x}, v = {y}")

            print("\nrelative to image center:")
            print(f"du = {x - W//2:+d}")
            print(f"dv = {y - H//2:+d}")

            if cx is not None:
                print("\nrelative to principal point:")
                print(f"du = {x - cx:+.2f}")
                print(f"dv = {y - cy:+.2f}")

            # draw clicked point
            cv2.drawMarker(vis, (x, y), (0,255,0), cv2.MARKER_CROSS, 20, 2)

    cv2.namedWindow("debug_laser_cross", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("debug_laser_cross", on_click)

    while True:
        cv2.imshow("debug_laser_cross", draw())
        key = cv2.waitKey(20)

        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()