"""
Interactive CheckerboardDetector test.

This script opens a RealSense RGB stream, runs the C++ HydraMarker
CheckerboardDetector on every frame, and visualizes the current detection result.

The goal of this test is NOT marker decoding yet. It only verifies that the
CheckerboardDetector produces the data required for the next pipeline stage:

    1. image-space checkerboard/grid corners
    2. local grid indices for each corner
    3. valid cells formed by four neighboring corners

Controls:
    t   Toggle visualization mode
    ESC Exit

Visualization modes:
    0: corners only
       Shows only detected corner points.

    1: cells only
       Shows valid cells formed from four neighboring corners.
       The magenta point marks the cell center, which will later be sampled
       by the DotDetector.

    2: corners + cells
       Shows both the cell structure and the local corner indices.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QFileDialog

import hydramarker_cpp


def select_file(title: str, file_filter: str) -> Path | None:
    """
    Open a Qt file dialog and return the selected path.

    The field file is required because the CheckerboardDetector is constructed
    from a MarkerField. At this stage, the field is mainly used to define the
    expected marker/grid configuration.
    """
    app = QApplication.instance() or QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        file_filter,
    )

    if not path:
        return None

    return Path(path)


def draw_corners(vis: np.ndarray, det, draw_indices: bool = False) -> None:
    """
    Draw detected grid corners.

    Parameters
    ----------
    vis:
        Image used for visualization.
    det:
        CheckerboardDetection returned by the C++ detector.
    draw_indices:
        If True, draw the local grid index (i, j) next to each assigned corner.
        This is useful for debugging the grid reconstruction, but should be
        disabled in "corners only" mode to keep the view clean.
    """
    for corner in det.corners:
        u, v = corner.uv

        cv2.circle(
            vis,
            (int(round(u)), int(round(v))),
            4,
            (0, 255, 0),
            -1,
            lineType=cv2.LINE_AA,
        )

        if draw_indices and corner.i >= 0 and corner.j >= 0:
            cv2.putText(
                vis,
                f"{corner.i},{corner.j}",
                (int(round(u)) + 6, int(round(v)) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )


def draw_cells(vis: np.ndarray, det, draw_indices: bool = True) -> None:
    """
    Draw valid checkerboard cells.

    A cell is defined by four neighboring corners:

        0: (i,   j)
        1: (i+1, j)
        2: (i,   j+1)
        3: (i+1, j+1)

    These cells are the direct input for the DotDetector. The DotDetector will
    later sample the image around each cell center to decide whether this cell
    contains a dot, no dot, or is uncertain/invalid.
    """
    for cell in det.cells:
        corner_ids = cell.corner_indices

        pts = []
        for idx in corner_ids:
            corner = det.corners[idx]
            u, v = corner.uv
            pts.append((int(round(u)), int(round(v))))

        p00 = pts[0]
        p10 = pts[1]
        p01 = pts[2]
        p11 = pts[3]

        polygon = np.array([p00, p10, p11, p01], dtype=np.int32)

        cv2.polylines(
            vis,
            [polygon],
            isClosed=True,
            color=(255, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        center_u, center_v = cell.center_uv

        cv2.circle(
            vis,
            (int(round(center_u)), int(round(center_v))),
            3,
            (255, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
        )

        if draw_indices:
            cv2.putText(
                vis,
                f"{cell.i},{cell.j}",
                (int(round(center_u)) + 5, int(round(center_v)) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )


def draw_status(vis: np.ndarray, mode_name: str, det) -> None:
    """
    Draw a small status overlay with the current mode and detection statistics.
    """
    text = f"mode: {mode_name} | press t to toggle | ESC to quit"

    if det is not None:
        text += f" | corners: {len(det.corners)} | cells: {len(det.cells)}"

    cv2.putText(
        vis,
        text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    """
    Start the RealSense stream and run the detector continuously.
    """
    field_path = select_file(
        "Select HydraMarker field file",
        "HydraMarker Field (*.field);;All Files (*)",
    )

    if field_path is None:
        print("No field file selected.")
        return

    field = hydramarker_cpp.MarkerField(str(field_path))
    detector = hydramarker_cpp.CheckerboardDetector(field)

    pipe = rs.pipeline()
    cfg = rs.config()

    cfg.enable_stream(
        rs.stream.color,
        1920,
        1080,
        rs.format.bgr8,
        30,
    )

    pipe.start(cfg)

    cv2.namedWindow("det", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "det",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    mode = 0
    mode_names = {
        0: "corners",
        1: "cells",
        2: "corners + cells",
    }

    try:
        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            vis = img.copy()

            det = detector.detect(img)

            if det is not None:
                if mode == 0:
                    draw_corners(vis, det, draw_indices=False)

                elif mode == 1:
                    draw_cells(vis, det, draw_indices=True)

                elif mode == 2:
                    draw_cells(vis, det, draw_indices=True)
                    draw_corners(vis, det, draw_indices=True)

            draw_status(vis, mode_names[mode], det)

            cv2.imshow("det", vis)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key == ord("t"):
                mode = (mode + 1) % 3

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()