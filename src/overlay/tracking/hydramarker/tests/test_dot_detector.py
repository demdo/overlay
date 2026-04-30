"""
Interactive DotDetector test.

Controls:
    t       Toggle visualization mode
    SPACE   Save current visualization as PNG
    p       Pause / unpause live update
    ESC     Exit

Visualization modes:
    0: corners only
    1: dots only
    2: corners + dots
"""

import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QFileDialog

import hydramarker_cpp


def select_file(title: str, file_filter: str) -> Path | None:
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


def draw_corners(vis: np.ndarray, checker_det, draw_indices: bool = False) -> None:
    for corner in checker_det.corners:
        u, v = corner.uv
        p = (int(round(u)), int(round(v)))

        cv2.circle(
            vis,
            p,
            4,
            (0, 255, 0),
            -1,
            lineType=cv2.LINE_AA,
        )

        if draw_indices and corner.i >= 0 and corner.j >= 0:
            cv2.putText(
                vis,
                f"{corner.i},{corner.j}",
                (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )


def draw_dots(vis: np.ndarray, dot_det, draw_indices: bool = True) -> None:
    for cell in dot_det.cells:
        u, v = cell.center_uv
        p = (int(round(u)), int(round(v)))

        if cell.value == 1:
            label = "1"
            color = (0, 255, 0)
        elif cell.value == 0:
            label = "0"
            color = (0, 0, 255)
        else:
            label = "?"
            color = (0, 255, 255)

        cv2.circle(
            vis,
            p,
            8,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            vis,
            label,
            (p[0] - 5, p[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        if draw_indices:
            cv2.putText(
                vis,
                f"{cell.i},{cell.j}",
                (p[0] + 10, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )


def get_stats_text(mode_name: str, checker_det, dot_det, paused: bool) -> list[str]:
    lines = []

    lines.append(f"mode: {mode_name}")
    lines.append("t: toggle | SPACE: save PNG | p: pause | ESC: quit")

    if paused:
        lines.append("PAUSED")

    if checker_det is not None:
        lines.append(f"corners: {len(checker_det.corners)}")
        lines.append(f"cells:   {len(checker_det.cells)}")

    if dot_det is not None:
        n_dot = sum(1 for c in dot_det.cells if c.value == 1)
        n_empty = sum(1 for c in dot_det.cells if c.value == 0)
        n_invalid = sum(1 for c in dot_det.cells if c.value < 0)

        lines.append(f"dots:    {n_dot}")
        lines.append(f"empty:   {n_empty}")
        lines.append(f"invalid: {n_invalid}")
        lines.append(f"grid:    {dot_det.cols} x {dot_det.rows}")
        lines.append(f"origin:  {dot_det.origin_i}, {dot_det.origin_j}")

    lines.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return lines


def draw_info_panel(vis: np.ndarray, lines: list[str]) -> None:
    x0 = 25
    y0 = 25
    line_h = 28
    pad = 14

    width = 520
    height = pad * 2 + line_h * len(lines)

    overlay = vis.copy()

    cv2.rectangle(
        overlay,
        (x0, y0),
        (x0 + width, y0 + height),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    y = y0 + pad + 20

    for line in lines:
        color = (0, 255, 255)

        if line == "PAUSED":
            color = (0, 0, 255)

        cv2.putText(
            vis,
            line,
            (x0 + pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        y += line_h


def render_visualization(
    img: np.ndarray,
    mode: int,
    mode_name: str,
    checker_det,
    dot_det,
    paused: bool,
) -> np.ndarray:
    vis = img.copy()

    if checker_det is not None:
        if mode == 0:
            draw_corners(vis, checker_det, draw_indices=False)

        elif mode == 1:
            if dot_det is not None:
                draw_dots(vis, dot_det, draw_indices=True)

        elif mode == 2:
            if dot_det is not None:
                draw_dots(vis, dot_det, draw_indices=True)
            draw_corners(vis, checker_det, draw_indices=False)

    lines = get_stats_text(mode_name, checker_det, dot_det, paused)
    draw_info_panel(vis, lines)

    return vis


def save_png(vis: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = output_dir / f"dot_detector_{timestamp}.png"

    cv2.imwrite(str(path), vis)
    print(f"Saved PNG: {path}")


def main() -> None:
    field_path = select_file(
        "Select HydraMarker field file",
        "HydraMarker Field (*.field);;All Files (*)",
    )

    if field_path is None:
        print("No field file selected.")
        return

    field = hydramarker_cpp.MarkerField(str(field_path))

    checker_detector = hydramarker_cpp.CheckerboardDetector(field)
    dot_detector = hydramarker_cpp.DotDetector()

    output_dir = Path(__file__).resolve().parent / "dot_detector_snapshots"

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

    cv2.namedWindow("dot detector", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "dot detector",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    mode = 0
    mode_names = {
        0: "corners",
        1: "dots",
        2: "corners + dots",
    }

    paused = False

    last_img = None
    last_checker_det = None
    last_dot_det = None
    last_vis = None

    try:
        while True:
            if not paused:
                frames = pipe.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                img = np.asanyarray(color_frame.get_data())

                checker_det = checker_detector.detect(img)
                dot_det = None

                if checker_det is not None:
                    dot_det = dot_detector.detect(img, checker_det)

                last_img = img.copy()
                last_checker_det = checker_det
                last_dot_det = dot_det

            if last_img is None:
                continue

            last_vis = render_visualization(
                last_img,
                mode,
                mode_names[mode],
                last_checker_det,
                last_dot_det,
                paused,
            )

            cv2.imshow("dot detector", last_vis)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key == ord("t"):
                mode = (mode + 1) % 3

            elif key == ord("p"):
                paused = not paused

            elif key == 32:
                if last_vis is not None:
                    save_png(last_vis, output_dir)

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()