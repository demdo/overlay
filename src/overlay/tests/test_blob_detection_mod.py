# test_blob_detection.py
#
# Window 1: ZOOM + MEASURE (pixel grid ALWAYS visible in every zoom view)
# Window 2: GRID + BLOBS + PRINT ROI ARRAYS (pixel grid ALWAYS visible too)
#
# Notes:
# - A TRUE pixel-boundary grid is only physically resolvable when a source pixel
#   spans >= ~2 display pixels. For smaller zoom, we draw a visible "fallback" grid
#   in display-space so you ALWAYS see a grid.
# - We DO NOT change the background to white; we draw grid lines as an overlay.
#
# Controls (Window 1):
#   LMB drag      : square zoom
#   RMB / dblLMB  : reset zoom
#   m             : toggle measure mode (ONLY when zoomed)
#   LMB click x2  : (in measure) pick (0,0) then (0,10)
#   ESC           : quit
#
# Controls (Window 2):
#   LMB drag      : square zoom
#   RMB / dblLMB  : reset zoom (to full image)
#   g             : toggle ideal grid crosses (green)
#   b             : run blob detection
#   o             : toggle blob center crosses (red)
#   t             : print ROI 11x11 measured + ideal arrays (u,v) in px
#   ESC           : quit (prints metrics)

from __future__ import annotations

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

try:
    import pydicom
    _HAS_PYDICOM = True
except ModuleNotFoundError:
    _HAS_PYDICOM = False


# -----------------------------
# Grid constants
# -----------------------------
PITCH_MM = 2.54
GRID_N = 11
N_PITCHES_REF = 10
KNOWN_DIST_MM = PITCH_MM * N_PITCHES_REF  # 25.4 mm


# -----------------------------
# Overlay imports (your project)
# -----------------------------
try:
    from overlay.tools.blob_detection import HoughCircleParams, detect_blobs_hough
    from overlay.tools.xray_marker_selection import (
        detector_mask,
        sort_circles_grid,
        prepare_nearest_cell_data,
        nearest_cell,
        extract_xy_from_cells,
    )
    _HAS_OVERLAY = True
except Exception as e:
    _HAS_OVERLAY = False
    _OVERLAY_IMPORT_ERR = e


# ============================================================
# Explorer dialog + image loading
# ============================================================

def _pick_file_dialog() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    filetypes = [
        ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.ima"),
        ("All Files", "*.*"),
    ]
    path = filedialog.askopenfilename(title="Select X-ray image", filetypes=filetypes)
    root.destroy()
    return path or ""


def load_xray_image() -> tuple[np.ndarray, str]:
    path = _pick_file_dialog()
    if not path:
        raise RuntimeError("No file selected.")
    path = os.path.abspath(path)

    if path.lower().endswith((".dcm", ".ima")):
        if not _HAS_PYDICOM:
            raise ModuleNotFoundError("pydicom required: pip install pydicom")
        ds = pydicom.dcmread(path, force=True)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not read image.")

    if img.ndim != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img_f = img.astype(np.float32)
        img_f = cv2.normalize(img_f, None, 0, 255, cv2.NORM_MINMAX)
        img = img_f.astype(np.uint8)

    return img, path


# ============================================================
# Drawing helpers
# ============================================================

def _draw_cross(img_bgr: np.ndarray, x: int, y: int, size: int = 6, color=(0, 255, 0), thick: int = 2):
    cv2.line(img_bgr, (x - size, y), (x + size, y), color, thick, cv2.LINE_AA)
    cv2.line(img_bgr, (x, y - size), (x, y + size), color, thick, cv2.LINE_AA)


def draw_always_visible_pixel_grid(vis_bgr: np.ndarray, pixel_size: float) -> np.ndarray:
    """
    Draw a grid in DISPLAY space so that a grid is ALWAYS visible.
    - If pixel_size >= 2: draw "true" pixel grid (boundaries of source pixels).
    - If pixel_size < 2: draw a fallback grid (still visible), because true pixel boundaries
      are subpixel and cannot be shown reliably.

    pixel_size = display_px_per_image_px (e.g., disp_w / roi_w)
    """
    h, w = vis_bgr.shape[:2]

    # Display-step in pixels
    if pixel_size >= 2.0:
        step = int(round(pixel_size))   # approximately boundary every source pixel
        alpha = 0.35
        thick = 1
    else:
        # Fallback: ALWAYS show some grid even at low zoom
        # (true pixel boundaries are not resolvable here)
        step = 6
        alpha = 0.20
        thick = 1

    step = max(1, step)

    overlay = vis_bgr.copy()
    col = (0, 0, 0)  # black

    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h - 1), col, thick, cv2.LINE_8)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w - 1, y), col, thick, cv2.LINE_8)

    return cv2.addWeighted(overlay, float(alpha), vis_bgr, 1.0 - float(alpha), 0.0)


def build_ideal_grid_11x11_from_two_points(
    u0v0: tuple[float, float], u1v1: tuple[float, float]
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    u0v0 = (0,0) in pixel coords
    u1v1 = (0,10) in pixel coords (10 pitches to the right)

    Returns:
      grid_11 (11,11,2) in ORIGINAL pixel coords
      pitch_px
      ex, ey unit vectors
    """
    p0 = np.array(u0v0, dtype=np.float32)
    p1 = np.array(u1v1, dtype=np.float32)

    vec = p1 - p0
    dist_px = float(np.linalg.norm(vec))
    if dist_px < 1e-6:
        raise ValueError("Reference points too close.")

    pitch_px = dist_px / float(N_PITCHES_REF)
    ex = vec / dist_px
    ey = np.array([-ex[1], ex[0]], dtype=np.float32)
    if ey[1] < 0:
        ey = -ey

    grid = np.zeros((GRID_N, GRID_N, 2), dtype=np.float32)
    for i in range(GRID_N):
        for j in range(GRID_N):
            grid[i, j] = p0 + j * pitch_px * ex + i * pitch_px * ey

    return grid, pitch_px, ex, ey


def show_blob_error_histogram(
    err_px: np.ndarray,
    total_points: int | None = None,
    bins: int = 40,
    title: str = r"Blob detection: distribution of $e$",
):
    """
    Histogram for blob errors (euclidean pixel error), styled like your plane-fit histogram.

    Changes vs previous version:
    - Legend n is shown as x/y (e.g. 121/121) to see ratio immediately.
      If total_points is None, uses y=x (i.e. N/N).
    - NO dashed P95 marker line
    - NO P95 annotation text in plot
    - Bars with left edge >= P95 are colored red exactly like plane-fit (#d62728)
    """

    err_px = np.asarray(err_px, dtype=float).ravel()
    err_px = err_px[np.isfinite(err_px)]
    if err_px.size == 0:
        print("show_blob_error_histogram: no finite errors.")
        return

    # --- stats ---
    N = int(err_px.size)
    total = int(total_points) if total_points is not None else N

    e_mean = float(np.mean(err_px))
    e_med  = float(np.median(err_px))
    e_rms  = float(np.sqrt(np.mean(err_px ** 2)))
    e_p95  = float(np.percentile(err_px, 95))

    # x-range: show some tail beyond P95
    xmax = max(e_p95 * 1.8, 1e-6)

    # --- histogram ---
    counts, edges = np.histogram(err_px, bins=bins, range=(0.0, xmax))
    widths = np.diff(edges)
    lefts = edges[:-1]

    # --- LaTeX-like style (same as your plane-fit) ---
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
    })

    fig, ax = plt.subplots()

    # draw bars first (single bar call), then recolor per-bin like plane-fit
    bars = ax.bar(
        lefts,
        counts,
        width=widths,
        align="edge",
        alpha=0.85,
        edgecolor="none",
    )

    # color bins right of P95 red (exact same logic + color as plane-fit)
    for left, bar in zip(lefts, bars):
        if left >= e_p95:
            bar.set_color("#d62728")  # red (upper 5%)
        else:
            bar.set_color("#1f77b4")  # blue

    # labels
    ax.set_title(title)
    ax.set_xlabel(r"Blob error $e\;[\mathrm{px}]$")
    ax.set_ylabel(r"Count")
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.2)

    # stats-only legend (upper-right) with n = x/y
    handles = [ax.plot([], [], " ")[0] for _ in range(5)]
    labels = [
        rf"$n={N}$",
        rf"$\bar{{e}}={e_mean:.3f}\,\mathrm{{px}}$",
        rf"$\tilde{{e}}={e_med:.3f}\,\mathrm{{px}}$",
        rf"$\mathrm{{RMS}}={e_rms:.3f}\,\mathrm{{px}}$",
        rf"$P_{{95}}={e_p95:.3f}\,\mathrm{{px}}$",
    ]
    ax.legend(handles, labels, loc="upper right", frameon=True, borderpad=0.8)

    fig.tight_layout()
    plt.show(block=True)


# ============================================================
# Window 1: Zoom + Measure
#   - Grid overlay ALWAYS visible (drawn in display space)
# ============================================================

class ZoomMeasureWindow:
    def __init__(self, img_gray: np.ndarray, win_name="X-ray (ZOOM)"):
        self.win = win_name
        self.img = img_gray
        self.h, self.w = img_gray.shape[:2]

        self.base_roi = (0, 0, self.w, self.h)
        self.roi = self.base_roi

        # fixed display size: full image size (keeps constant while zooming)
        self.disp_w = self.w
        self.disp_h = self.h

        # zoom selection
        self.dragging = False
        self.start = None
        self.sel = None

        # measure mode (ONLY allowed when zoomed)
        self.measure_mode = False
        self.measure_pts_disp = []

        # result
        self.u0v0 = None
        self.u1v1 = None

        print("\n=== Window 1 (ZOOM) ===")
        print("LMB drag      : square zoom")
        print("RMB / dblLMB  : reset zoom")
        print("m             : toggle measure (ONLY when zoomed)")
        print("Measure clicks: P0=(0,0), P1=(0,10)")
        print("ESC           : quit\n")

    def is_zoomed(self) -> bool:
        return self.roi != self.base_roi

    def _disp_to_img(self, xd: int, yd: int) -> tuple[float, float]:
        rx, ry, rw, rh = self.roi
        sx = rw / float(self.disp_w)
        sy = rh / float(self.disp_h)
        return float(rx + xd * sx), float(ry + yd * sy)

    def _render(self) -> np.ndarray:
        rx, ry, rw, rh = self.roi
        roi_gray = self.img[ry:ry + rh, rx:rx + rw]

        # Nearest neighbor keeps crisp pixel blocks
        vis_gray = cv2.resize(roi_gray, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST)
        vis = cv2.cvtColor(vis_gray, cv2.COLOR_GRAY2BGR)

        # grid overlay ALWAYS visible
        pixel_size = self.disp_w / float(rw)
        vis = draw_always_visible_pixel_grid(vis, pixel_size=pixel_size)

        # selection rectangle (only in zoom mode)
        if self.sel is not None and not self.measure_mode:
            x0, y0, x1, y1 = self.sel
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 255), 2, cv2.LINE_AA)

        # measure overlay
        if self.measure_mode:
            if len(self.measure_pts_disp) >= 1:
                x0, y0 = self.measure_pts_disp[0]
                cv2.circle(vis, (x0, y0), 6, (255, 255, 255), 2, cv2.LINE_AA)
            if len(self.measure_pts_disp) == 2:
                x0, y0 = self.measure_pts_disp[0]
                x1, y1 = self.measure_pts_disp[1]
                cv2.circle(vis, (x1, y1), 6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(vis, (x0, y0), (x1, y1), (255, 255, 255), 2, cv2.LINE_AA)

        return vis

    def toggle_measure(self):
        if not self.is_zoomed():
            print("[measure] Not allowed in full view. Zoom in first.")
            return
        self.measure_mode = not self.measure_mode
        self.measure_pts_disp.clear()
        print(f"[measure] {'ON' if self.measure_mode else 'OFF'}")

    def reset_zoom(self):
        self.roi = self.base_roi
        self.dragging = False
        self.start = None
        self.sel = None
        if self.measure_mode:
            self.measure_mode = False
            self.measure_pts_disp.clear()
            print("[zoom] reset -> measure OFF")

    def _mouse(self, event, x, y, flags, param):
        x = max(0, min(self.disp_w - 1, int(x)))
        y = max(0, min(self.disp_h - 1, int(y)))

        # reset
        if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_LBUTTONDBLCLK:
            self.reset_zoom()
            return

        # measure mode
        if self.measure_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.measure_pts_disp.append((x, y))
                if len(self.measure_pts_disp) > 2:
                    self.measure_pts_disp = [self.measure_pts_disp[-1]]

                if len(self.measure_pts_disp) == 2:
                    (x0d, y0d), (x1d, y1d) = self.measure_pts_disp
                    u0, v0 = self._disp_to_img(x0d, y0d)
                    u1, v1 = self._disp_to_img(x1d, y1d)

                    dist_px = float(np.hypot(u1 - u0, v1 - v0))
                    px_per_mm = dist_px / float(KNOWN_DIST_MM)
                    mm_per_px = float(KNOWN_DIST_MM) / dist_px

                    self.u0v0 = (u0, v0)
                    self.u1v1 = (u1, v1)

                    print(f"[measure] dist_px = {dist_px:.3f} px  for  {KNOWN_DIST_MM:.2f} mm")
                    print(f"[measure] px/mm    = {px_per_mm:.6f}")
                    print(f"[measure] mm/px    = {mm_per_px:.9f}")
                    print("[measure] points:")
                    print(f"          P0 (0,0)  = ({u0:.3f}, {v0:.3f})")
                    print(f"          P1 (0,10) = ({u1:.3f}, {v1:.3f})")
                    print("[measure] Opening Window 2...\n")
            return

        # zoom drag
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.sel = (x, y, x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.dragging and self.start is not None:
            x0, y0 = self.start
            dx = x - x0
            dy = y - y0
            side = max(abs(dx), abs(dy))
            sx = 1 if dx >= 0 else -1
            sy = 1 if dy >= 0 else -1
            x1 = x0 + sx * side
            y1 = y0 + sy * side

            xa, xb = sorted([x0, x1])
            ya, yb = sorted([y0, y1])

            xa = max(0, min(self.disp_w - 1, xa))
            xb = max(0, min(self.disp_w - 1, xb))
            ya = max(0, min(self.disp_h - 1, ya))
            yb = max(0, min(self.disp_h - 1, yb))

            side2 = min(xb - xa, yb - ya)
            xb = xa + side2
            yb = ya + side2
            self.sel = (int(xa), int(ya), int(xb), int(yb))
            return

        if event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            if self.sel is None:
                return

            x0d, y0d, x1d, y1d = self.sel
            if (x1d - x0d) < 10:
                self.sel = None
                return

            u0, v0 = self._disp_to_img(x0d, y0d)
            u1, v1 = self._disp_to_img(x1d, y1d)

            new_x = int(np.floor(min(u0, u1)))
            new_y = int(np.floor(min(v0, v1)))
            new_w = int(np.ceil(abs(u1 - u0)))
            new_h = int(np.ceil(abs(v1 - v0)))

            side = max(1, min(new_w, new_h))
            new_x = max(0, min(self.w - 1, new_x))
            new_y = max(0, min(self.h - 1, new_y))
            side = min(side, self.w - new_x, self.h - new_y)

            self.roi = (new_x, new_y, int(side), int(side))
            self.sel = None
            self.start = None
            print(f"[zoom] ROI = (x={self.roi[0]}, y={self.roi[1]}, w={self.roi[2]}, h={self.roi[3]})")

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.disp_w, self.disp_h)
        cv2.setMouseCallback(self.win, self._mouse)

        while True:
            cv2.imshow(self.win, self._render())

            if self.u0v0 is not None and self.u1v1 is not None:
                break

            key = cv2.waitKey(15) & 0xFF
            if key == 27:
                cv2.destroyWindow(self.win)
                return None
            if key == ord("m"):
                self.toggle_measure()

        cv2.destroyWindow(self.win)
        return self.u0v0, self.u1v1, self.roi


# ============================================================
# Window 2: Grid + Blobs + ROI metrics + print arrays
#   - Grid overlay ALWAYS visible (display-space)
# ============================================================

class GridBlobsWindow:
    def __init__(
        self,
        img_gray: np.ndarray,
        initial_roi: tuple[int, int, int, int],
        grid_uv_11x11: np.ndarray,
        p0_uv: tuple[float, float],
        p1_uv: tuple[float, float],
    ):
        if img_gray.ndim != 2 or img_gray.dtype != np.uint8:
            raise ValueError("Expected grayscale uint8 image.")

        self.win = "X-ray (GRID + BLOBS)"
        self.img = img_gray
        self.h, self.w = img_gray.shape

        # Zoom state
        self.base_roi = (0, 0, self.w, self.h)
        self.roi = initial_roi

        # Fixed display size
        self.disp_w = self.w
        self.disp_h = self.h

        # Interaction state
        self.dragging = False
        self.start = None
        self.sel = None

        # Geometry refs
        self.p0_uv = np.array(p0_uv, dtype=np.float32)
        self.p1_uv = np.array(p1_uv, dtype=np.float32)

        # Ideal 11x11 (for overlay display)
        self.grid_uv_11x11 = np.asarray(grid_uv_11x11, dtype=np.float32)  # (11,11,2)

        # Toggles
        self.show_grid = True
        self.show_blobs = True

        # Detection results
        self.circles_grid = None         # (R,C,3)

        # Ideal grid in same index system as circles_grid
        self.ideal_grid_full = None      # (R,C,3)

        # ROI (exact 11x11 by index)
        self.roi_cells = None
        self.roi_cells_sorted = None
        self.xy_meas_roi = None
        self.xy_ideal_roi = None

        # Reshaped for printing
        self.meas_11x11_uv = None
        self.ideal_11x11_uv = None

        print("\n=== Window 2 (GRID + BLOBS) ===")
        print("LMB drag      : square zoom")
        print("RMB / dblLMB  : reset zoom (to full image)")
        print("g             : toggle IDEAL 11x11 grid crosses (green)")
        print("b             : run blob detection (same as Page Marker Selection)")
        print("o             : toggle BLOB center crosses (red)")
        print("t             : print ROI 11x11 measured + ideal arrays to terminal")
        print("ESC           : quit (prints metrics)\n")

        if not _HAS_OVERLAY:
            print("[warn] Overlay imports failed; blob detection won't run.")
            print(f"       {_OVERLAY_IMPORT_ERR}\n")

    # ---------- coords ----------
    def _disp_to_img(self, xd: int, yd: int) -> tuple[float, float]:
        rx, ry, rw, rh = self.roi
        sx = rw / float(self.disp_w)
        sy = rh / float(self.disp_h)
        return float(rx + xd * sx), float(ry + yd * sy)

    def _img_to_disp_many(self, uv: np.ndarray) -> np.ndarray:
        rx, ry, rw, rh = self.roi
        uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        xd = (uv[:, 0] - rx) * (self.disp_w / float(rw))
        yd = (uv[:, 1] - ry) * (self.disp_h / float(rh))
        return np.stack([xd, yd], axis=1)

    # ---------- drawing ----------
    def _render(self) -> np.ndarray:
        rx, ry, rw, rh = self.roi
        roi_gray = self.img[ry:ry + rh, rx:rx + rw]
    
        # Nearest-neighbour keeps pixel blocks crisp
        vis_gray = cv2.resize(
            roi_gray,
            (self.disp_w, self.disp_h),
            interpolation=cv2.INTER_NEAREST
        )
        vis = cv2.cvtColor(vis_gray, cv2.COLOR_GRAY2BGR)
    
        # ---------------------------------------------------------
        # ALWAYS-visible pixel grid overlay (display-space)
        # ---------------------------------------------------------
        pixel_size = self.disp_w / float(rw)
        vis = draw_always_visible_pixel_grid(vis, pixel_size=pixel_size)
    
        # ---------------------------------------------------------
        # Ideal 11x11 crosses (green)
        # ---------------------------------------------------------
        if self.show_grid and self.grid_uv_11x11 is not None:
            pts_disp = self._img_to_disp_many(self.grid_uv_11x11)
            for xd, yd in pts_disp:
                _draw_cross(
                    vis,
                    int(round(xd)),
                    int(round(yd)),
                    size=8,
                    color=(0, 255, 0),
                    thick=2,
                )
    
        # ---------------------------------------------------------
        # Detected blob centers (red)
        # (NEW: uses self.detected_uv instead of circles_grid)
        # ---------------------------------------------------------
        if self.show_blobs and getattr(self, "detected_uv", None) is not None:
            if len(self.detected_uv) > 0:
                uv_disp = self._img_to_disp_many(self.detected_uv)
                for xd, yd in uv_disp:
                    _draw_cross(
                        vis,
                        int(round(xd)),
                        int(round(yd)),
                        size=7,
                        color=(0, 0, 255),
                        thick=2,
                    )
    
        # ---------------------------------------------------------
        # Zoom selection rectangle
        # ---------------------------------------------------------
        if self.sel is not None:
            x0, y0, x1, y1 = self.sel
            cv2.rectangle(
                vis,
                (x0, y0),
                (x1, y1),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    
        return vis

    # ---------- ideal grid full ----------
    def _build_ideal_grid_full(self, circles_grid: np.ndarray, anchor_cell: tuple[int, int]) -> tuple[np.ndarray, float]:
        p0 = self.p0_uv.astype(np.float32)
        p1 = self.p1_uv.astype(np.float32)

        vec = p1 - p0
        dist_px = float(np.linalg.norm(vec))
        if dist_px < 1e-6:
            raise ValueError("Measured reference points too close.")

        pitch_px = dist_px / float(N_PITCHES_REF)
        ex = vec / dist_px
        ey = np.array([-ex[1], ex[0]], dtype=np.float32)
        if ey[1] < 0:
            ey = -ey

        R, C, _ = circles_grid.shape
        ideal = np.full((R, C, 3), np.nan, dtype=np.float32)

        i0, j0 = anchor_cell
        for i in range(R):
            for j in range(C):
                pt = p0 + (j - j0) * pitch_px * ex + (i - i0) * pitch_px * ey
                ideal[i, j, 0] = float(pt[0])
                ideal[i, j, 1] = float(pt[1])
                ideal[i, j, 2] = np.nan

        return ideal, pitch_px

    # ---------- ROI extraction ----------
    def _build_roi_cells_11x11(self, anchor_cell: tuple[int, int], shape_rc: tuple[int, int]) -> set[tuple[int, int]]:
        i0, j0 = anchor_cell
        R, C = shape_rc
        cells = set()
        for di in range(GRID_N):
            for dj in range(GRID_N):
                i = i0 + di
                j = j0 + dj
                if 0 <= i < R and 0 <= j < C:
                    cells.add((i, j))
        return cells

    # ---------- blob detection ----------
    def run_blob_detection_same_as_page(self):
        """
        Run blob detection (same preprocessing + Hough params as your Page),
        but DO NOT build a padded circles_grid for accuracy (left_pad can shift rows).
    
        Instead, we:
          1) Detect all blobs (centers)
          2) Build the IDEAL 11x11 grid from the measured p0/p1
          3) Assign each ideal grid node to the nearest detected blob (unique matching + gating)
          4) Store:
               - self.xy_meas_roi   (121,2) with NaNs for missing
               - self.xy_ideal_roi  (121,2)
               - self.meas_11x11_uv / self.ideal_11x11_uv for your existing 't' printing
          5) For red overlay, we store self.detected_uv = (N,2) all detected centers.
    
        Your existing outputs (prints, 't' table, metrics on exit) keep working.
        """
        if not _HAS_OVERLAY:
            print("[blobs] Cannot run. overlay tools not importable.")
            return
    
        print("[blobs] Running SAME pipeline as Page Xray Marker Selection...")
    
        # ------------------------------------------------------------------
        # 1) Same Hough params + preprocessing choices (match Page)
        # ------------------------------------------------------------------
        params = HoughCircleParams(
            min_radius=2,
            max_radius=7,
            dp=1.2,
            minDist=16,
            param1=120,
            param2=12,
            invert=True,
            median_ks=(3, 5),
        )
        use_clahe = True
        clahe_clip = 2.0
        clahe_tiles = (12, 12)
    
        img_proc = self.img.copy()
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tiles)
            img_proc = clahe.apply(img_proc)
    
        mask = detector_mask(img_proc)
        img_masked = img_proc.copy()
        img_masked[mask == 0] = 0
    
        circles_out = detect_blobs_hough(img_masked, params)
        if circles_out is None or len(circles_out) == 0:
            print("[blobs] No circles detected.")
            self.circles_grid = None
            self.detected_uv = None
            self.xy_meas_roi = None
            self.xy_ideal_roi = None
            self.meas_11x11_uv = None
            self.ideal_11x11_uv = None
            return
    
        circles_out = np.asarray(circles_out, dtype=np.float32)
        meas_uv = circles_out[:, :2].astype(np.float32, copy=False)  # (N,2)
    
        # For drawing red crosses (all detections)
        self.detected_uv = meas_uv.copy()
    
        print(f"[blobs] detected (raw) = {meas_uv.shape[0]}")
    
        # ------------------------------------------------------------------
        # 2) Build IDEAL 11x11 from measured p0/p1 (same as green overlay)
        # ------------------------------------------------------------------
        grid_11, pitch_px, ex, ey = build_ideal_grid_11x11_from_two_points(
            (float(self.p0_uv[0]), float(self.p0_uv[1])),
            (float(self.p1_uv[0]), float(self.p1_uv[1])),
        )
        ideal_uv = grid_11.reshape(-1, 2).astype(np.float32)  # (121,2) row-major
    
        # ------------------------------------------------------------------
        # 3) Assign each ideal node to nearest detected blob (unique + gating)
        # ------------------------------------------------------------------
        # Distance matrix: (121, N)
        dx = ideal_uv[:, None, 0] - meas_uv[None, :, 0]
        dy = ideal_uv[:, None, 1] - meas_uv[None, :, 1]
        d2 = dx * dx + dy * dy
    
        # Gate: if nearest is further than ~0.65 pitch, treat as missing
        gate2 = float((0.65 * pitch_px) ** 2)
    
        xy_meas = np.full_like(ideal_uv, np.nan, dtype=np.float32)  # (121,2)
        used = np.zeros((meas_uv.shape[0],), dtype=bool)
    
        # Fill in order of "most confident" ideals first
        best_d2 = np.min(d2, axis=1)
        order = np.argsort(best_d2)
    
        for k in order:
            if best_d2[k] > gate2:
                continue
    
            # prefer nearest unused detection; fall back to next nearest
            cand = np.argsort(d2[k])
            chosen = None
            for j in cand:
                j = int(j)
                if not used[j] and float(d2[k, j]) <= gate2:
                    chosen = j
                    break
    
            if chosen is None:
                continue
    
            xy_meas[k] = meas_uv[chosen]
            used[chosen] = True
    
        n_matched = int(np.isfinite(xy_meas[:, 0]).sum())
        print(f"[ideal] pitch_px={pitch_px:.3f}  gate={np.sqrt(gate2):.2f}px")
        print(f"[match] matched = {n_matched}/121")
    
        # ------------------------------------------------------------------
        # 4) Store for your existing table/metrics
        # ------------------------------------------------------------------
        self.xy_meas_roi = xy_meas
        self.xy_ideal_roi = ideal_uv.copy()
    
        self.meas_11x11_uv = self.xy_meas_roi.reshape(GRID_N, GRID_N, 2)
        self.ideal_11x11_uv = self.xy_ideal_roi.reshape(GRID_N, GRID_N, 2)
    
        # Keep show_blobs behavior
        self.show_blobs = True
    
        # ------------------------------------------------------------------
        # 5) Optional: also keep a "circles_grid" for compatibility (not used for accuracy)
        #     We set it to None so your old padded-grid code can't accidentally be used.
        # ------------------------------------------------------------------
        self.circles_grid = None

    def print_roi_11x11_arrays(self, decimals: int = 1):
        if self.meas_11x11_uv is None or self.ideal_11x11_uv is None:
            print("[print] ROI arrays not available. Run blob detection ('b') first.")
            return

        meas = np.asarray(self.meas_11x11_uv, dtype=np.float32)
        ideal = np.asarray(self.ideal_11x11_uv, dtype=np.float32)
        fmt = f"{{:.{decimals}f}}"

        print("\n==================== ROI 11x11 FULL TABLE ====================")
        print("Each cell: (i,j)  meas(u,v)  ideal(u,v)  err_px")
        for i in range(GRID_N):
            print(f"\n--- row {i:02d} ---")
            for j in range(GRID_N):
                mu, mv = float(meas[i, j, 0]), float(meas[i, j, 1])
                iu, iv = float(ideal[i, j, 0]), float(ideal[i, j, 1])
                if np.isfinite(mu) and np.isfinite(mv) and np.isfinite(iu) and np.isfinite(iv):
                    err = float(np.hypot(mu - iu, mv - iv))
                    err_s = fmt.format(err)
                else:
                    err_s = "nan"
                print(
                    f"({i:02d},{j:02d})  "
                    f"meas=({fmt.format(mu)},{fmt.format(mv)})  "
                    f"ideal=({fmt.format(iu)},{fmt.format(iv)})  "
                    f"err={err_s}"
                )
        print("===============================================================\n")

    def _print_metrics_on_exit(self):
        if self.xy_meas_roi is None or self.xy_ideal_roi is None:
            print("[metrics] ROI data not available. Run blob detection ('b') first.")
            return

        meas = np.asarray(self.xy_meas_roi, dtype=np.float32)
        ideal = np.asarray(self.xy_ideal_roi, dtype=np.float32)

        valid = (
            np.isfinite(meas[:, 0]) & np.isfinite(meas[:, 1]) &
            np.isfinite(ideal[:, 0]) & np.isfinite(ideal[:, 1])
        )
        n_valid = int(valid.sum())
        if n_valid == 0:
            print("[metrics] No valid ROI points.")
            return

        err = np.linalg.norm(meas[valid] - ideal[valid], axis=1).astype(np.float64)
        show_blob_error_histogram(err, bins=35, title=r"Blob detection: distribution of $e$")

        rms = float(np.sqrt(np.mean(err**2)))
        mean = float(np.mean(err))
        median = float(np.median(err))
        p95 = float(np.percentile(err, 95))

        print("\n========== BLOB ACCURACY METRICS (px) ==========")
        print(f"ROI points (total) : {meas.shape[0]}  (requested 121)")
        print(f"ROI points (valid) : {n_valid}")
        print(f"RMS                : {rms:.4f} px")
        print(f"Mean               : {mean:.4f} px")
        print(f"Median             : {median:.4f} px")
        print(f"P95                : {p95:.4f} px")
        print("================================================\n")

    def toggle_grid(self):
        self.show_grid = not self.show_grid
        print(f"[grid] {'ON' if self.show_grid else 'OFF'}")

    def toggle_blobs(self):
        if self.circles_grid is None:
            print("[blobs] No blobs yet. Press 'b' first.")
            return
        self.show_blobs = not self.show_blobs
        print(f"[blobs] overlay {'ON' if self.show_blobs else 'OFF'}")

    def reset_zoom(self):
        self.roi = self.base_roi
        self.dragging = False
        self.start = None
        self.sel = None
        print("[zoom] reset to full view")

    def _mouse(self, event, x, y, flags, param):
        x = max(0, min(self.disp_w - 1, int(x)))
        y = max(0, min(self.disp_h - 1, int(y)))

        if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_LBUTTONDBLCLK:
            self.reset_zoom()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.sel = (x, y, x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.dragging and self.start is not None:
            x0, y0 = self.start
            dx = x - x0
            dy = y - y0
            side = max(abs(dx), abs(dy))
            sx = 1 if dx >= 0 else -1
            sy = 1 if dy >= 0 else -1
            x1 = x0 + sx * side
            y1 = y0 + sy * side

            xa, xb = sorted([x0, x1])
            ya, yb = sorted([y0, y1])

            xa = max(0, min(self.disp_w - 1, xa))
            xb = max(0, min(self.disp_w - 1, xb))
            ya = max(0, min(self.disp_h - 1, ya))
            yb = max(0, min(self.disp_h - 1, yb))

            side2 = min(xb - xa, yb - ya)
            xb = xa + side2
            yb = ya + side2

            self.sel = (int(xa), int(ya), int(xb), int(yb))
            return

        if event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            if self.sel is None:
                return

            x0d, y0d, x1d, y1d = self.sel
            if (x1d - x0d) < 10:
                self.sel = None
                return

            u0, v0 = self._disp_to_img(x0d, y0d)
            u1, v1 = self._disp_to_img(x1d, y1d)

            new_x = int(np.floor(min(u0, u1)))
            new_y = int(np.floor(min(v0, v1)))
            new_w = int(np.ceil(abs(u1 - u0)))
            new_h = int(np.ceil(abs(v1 - v0)))

            side = max(1, min(new_w, new_h))
            new_x = max(0, min(self.w - 1, new_x))
            new_y = max(0, min(self.h - 1, new_y))
            side = min(side, self.w - new_x, self.h - new_y)

            self.roi = (new_x, new_y, int(side), int(side))
            self.sel = None
            self.start = None
            print(f"[zoom] ROI = (x={self.roi[0]}, y={self.roi[1]}, w={self.roi[2]}, h={self.roi[3]})")

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.disp_w, self.disp_h)
        cv2.setMouseCallback(self.win, self._mouse)
    
        while True:
            cv2.imshow(self.win, self._render())
            key = cv2.waitKey(15) & 0xFF
    
            if key == 27:
                self._print_metrics_on_exit()
                break
            elif key == ord("g"):
                self.toggle_grid()
            elif key == ord("b"):
                self.run_blob_detection_same_as_page()
            elif key == ord("o"):
                self.toggle_blobs()
            elif key == ord("t"):
                self.print_roi_11x11_arrays(decimals=1)
    
        cv2.destroyWindow(self.win)


# ============================================================
# Main
# ============================================================

def main():
    img, path = load_xray_image()
    print(f"Loaded: {path} | shape={img.shape} dtype={img.dtype}")

    # Window 1: zoom + measure
    w1 = ZoomMeasureWindow(img)
    res = w1.run()
    if res is None:
        return

    (u0, v0), (u1, v1), roi_after = res

    # ideal 11x11 for display
    grid_11, pitch_px, ex, ey = build_ideal_grid_11x11_from_two_points((u0, v0), (u1, v1))
    print(f"[ideal11] pitch_px={pitch_px:.6f}")

    # Window 2: grid + blobs + ROI metrics + prints
    w2 = GridBlobsWindow(
        img_gray=img,
        initial_roi=roi_after,
        grid_uv_11x11=grid_11,
        p0_uv=(u0, v0),
        p1_uv=(u1, v1),
    )
    w2.run()


if __name__ == "__main__":
    main()