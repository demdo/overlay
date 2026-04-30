# -*- coding: utf-8 -*-
"""
debug_marker_selection.py
=========================

RAW-only Debug-Skript:

- X-ray wird per Qt-Dateidialog gewählt
- Marker-Detection läuft direkt auf dem RAW-Bild
- 3 Anchor-Punkte werden direkt auf dem RAW-Bild gewählt
- ROI wird direkt in RAW-Koordinaten berechnet
- points_uv werden direkt in RAW als NPZ gespeichert

Steuerung:
  USE_FLIPPING = True   -> current/default behavior: flipped X-ray/source ordering
  USE_FLIPPING = False  -> unflipped camera-view ordering

  Linksklick  = Anchor wählen
  R           = Reset
  Q / ESC     = Beenden
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2
import pydicom

from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.blob_detection import HoughCircleParams
from overlay.tools.xray_marker_selection import (
    run_xray_marker_detection,
    compute_roi_from_grid,
)


# ── Konfiguration ─────────────────────────────────────────────────────────────

USE_FLIPPING = False  # default: True

OUT_UV_PATH_DEFAULT = (
    r"C:\Users\domin\Documents\Studium\Master\Masterarbeit\Projekt\Data\uv_debug_raw.npz"
)

HOUGH_PARAMS = HoughCircleParams(
    min_radius=2,
    max_radius=7,
    dp=1.2,
    minDist=8,
    param1=120,
    param2=7,
    invert=True,
    median_ks=(3, 5),
)

N_LABEL = 11

DISPLAY_MAX_W = 1024
DISPLAY_MAX_H = 1024


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_open_file_qt() -> str | None:
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    path, _ = QFileDialog.getOpenFileName(
        None,
        "X-ray Bild wählen",
        "",
        "X-ray / DICOM (*.dcm *.ima *.png *.jpg *.jpeg *.tif *.tiff *.bmp);;Alle Dateien (*)",
    )

    if created_app:
        app.quit()

    path = path.strip()
    return path if path else None


def pick_save_file_qt(default_path: str) -> str | None:
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    path, _ = QFileDialog.getSaveFileName(
        None,
        "NPZ speichern unter",
        default_path,
        "NumPy NPZ (*.npz)",
    )

    if created_app:
        app.quit()

    path = path.strip()
    if not path:
        return None
    if not path.lower().endswith(".npz"):
        path += ".npz"
    return path


def load_xray(path: str) -> np.ndarray:
    if path.lower().endswith((".dcm", ".ima")):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img.astype(np.uint8)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {path}")
    return img


def nearest_circle_index(circles: np.ndarray, u_click: float, v_click: float) -> int | None:
    if circles is None or len(circles) == 0:
        return None

    xy = circles[:, :2].astype(np.float64)
    finite = np.isfinite(xy).all(axis=1)
    if not np.any(finite):
        return None

    xyf = xy[finite]
    idx_map = np.flatnonzero(finite)

    d2 = (xyf[:, 0] - float(u_click)) ** 2 + (xyf[:, 1] - float(v_click)) ** 2
    return int(idx_map[np.argmin(d2)])


def point_label_sets(dbg: dict, max_n: int = 11) -> list[tuple[int, str, tuple[int, int, int]]]:
    """
    Liefert Punktlabels für:
    - erste Zeile (gi == 0): R0..R10
    - erste Spalte (gj == 0): C0..C10
    """
    gi = np.asarray(dbg["grid_i"])
    gj = np.asarray(dbg["grid_j"])

    labels: list[tuple[int, str, tuple[int, int, int]]] = []

    row0 = np.where(gi == 0)[0]
    row0 = row0[np.argsort(gj[row0])]
    for idx in row0[:max_n]:
        labels.append((int(idx), f"R{int(gj[idx])}", (0, 220, 0)))

    col0 = np.where(gj == 0)[0]
    col0 = col0[np.argsort(gi[col0])]
    for idx in col0[:max_n]:
        labels.append((int(idx), f"C{int(gi[idx])}", (0, 0, 220)))

    return labels


def print_roi_debug(title: str, roi_uv: np.ndarray, dbg: dict) -> None:
    N = roi_uv.shape[0]
    nu = int(dbg["nu"])
    nv = int(dbg["nv"])
    expected = (nu + 1) * (nv + 1)

    print("\n" + "═" * 72)
    print(title)
    print(f"  Punkte gefunden: {N}  (erwartet: {(nu + 1)}*{(nv + 1)} = {expected})")
    print(f"  USE_FLIPPING = {USE_FLIPPING}")
    print(f"  nu={nu}  nv={nv}  pitch={dbg['pitch']:.2f}px")
    print(f"  Lu={dbg['Lu']:.1f}px  Lv={dbg['Lv']:.1f}px")
    print(f"  tol_px={dbg['tol_px']:.2f}  gate_tol_pitch={dbg['gate_tol_pitch']}")

    gi = np.asarray(dbg["grid_i"])
    gj = np.asarray(dbg["grid_j"])

    row0 = np.where(gi == 0)[0]
    row0 = row0[np.argsort(gj[row0])]
    col0 = np.where(gj == 0)[0]
    col0 = col0[np.argsort(gi[col0])]

    print("\nErste Zeile:")
    for k in row0[:11]:
        print(
            f"  [{k:3d}] i={gi[k]:2d} j={gj[k]:2d}  "
            f"uv=({roi_uv[k,0]:8.2f}, {roi_uv[k,1]:8.2f})"
        )

    print("\nErste Spalte:")
    for k in col0[:11]:
        print(
            f"  [{k:3d}] i={gi[k]:2d} j={gj[k]:2d}  "
            f"uv=({roi_uv[k,0]:8.2f}, {roi_uv[k,1]:8.2f})"
        )

    if len(row0) >= 2:
        du = roi_uv[row0[1], 0] - roi_uv[row0[0], 0]
        dv = roi_uv[row0[1], 1] - roi_uv[row0[0], 1]
        print(f"\nDelta erste Zeile   [0]->[1]: Δu={du:+.2f}  Δv={dv:+.2f}")

    if len(col0) >= 2:
        du = roi_uv[col0[1], 0] - roi_uv[col0[0], 0]
        dv = roi_uv[col0[1], 1] - roi_uv[col0[0], 1]
        print(f"Delta erste Spalte  [0]->[1]: Δu={du:+.2f}  Δv={dv:+.2f}")


def draw_cross(
    img: np.ndarray,
    u: float,
    v: float,
    color: tuple[int, int, int],
    size: int = 7,
    thickness: int = 2,
) -> None:
    uu = int(round(u))
    vv = int(round(v))
    cv2.line(img, (uu - size, vv), (uu + size, vv), color, thickness, cv2.LINE_AA)
    cv2.line(img, (uu, vv - size), (uu, vv + size), color, thickness, cv2.LINE_AA)


def draw_state(
    img_gray: np.ndarray,
    circles: np.ndarray,
    anchors: list[int],
    roi_uv: np.ndarray | None,
    dbg: dict | None,
    *,
    title: str,
) -> np.ndarray:
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # alle detektierten Kreise
    for (x, y, r) in circles:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(
                vis,
                (int(round(x)), int(round(y))),
                max(2, int(round(r))),
                (0, 255, 0),
                1,
            )

    # Anchors
    anchor_labels = ["TL", "TR", "BL"]

    for k, idx in enumerate(anchors):
        x, y, r = circles[idx]
        label = anchor_labels[k] if k < 3 else f"A{k}"

        cv2.circle(
            vis,
            (int(round(x)), int(round(y))),
            max(4, int(round(r)) + 2),
            (0, 140, 255),
            2,
        )

        cv2.putText(
            vis,
            label,
            (int(round(x)) + 6, int(round(y)) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )

    # ROI
    if roi_uv is not None and dbg is not None:
        gi = np.asarray(dbg["grid_i"])
        gj = np.asarray(dbg["grid_j"])

        for k, (u, v) in enumerate(roi_uv):
            if gi[k] == 0:
                color = (0, 255, 0)      # erste Zeile
            elif gj[k] == 0:
                color = (0, 0, 255)      # erste Spalte
            else:
                color = (255, 200, 0)    # Rest
            cv2.circle(vis, (int(round(u)), int(round(v))), 4, color, -1)

        for idx, label, color in point_label_sets(dbg, max_n=N_LABEL):
            u, v = roi_uv[idx]
            draw_cross(vis, u, v, color=color, size=8, thickness=1)
            cv2.putText(
                vis,
                label,
                (int(round(u)) + 4, int(round(v)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    h = vis.shape[0]
    cv2.putText(
        vis,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"USE_FLIPPING={USE_FLIPPING} | Anchors: {len(anchors)}/3  |  LMB=select  R=reset  Q=quit",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return vis


def make_display_transform(
    img_shape: tuple[int, int],
    max_w: int,
    max_h: int,
) -> tuple[float, int, int, int, int]:
    """
    Returns:
        scale, disp_w, disp_h, off_x, off_y
    """
    h, w = img_shape[:2]
    scale = min(max_w / float(w), max_h / float(h))
    if scale <= 0:
        scale = 1.0

    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))

    off_x = max(0, (max_w - disp_w) // 2)
    off_y = max(0, (max_h - disp_h) // 2)

    return scale, disp_w, disp_h, off_x, off_y


def render_for_display(
    img_bgr: np.ndarray,
    max_w: int,
    max_h: int,
) -> tuple[np.ndarray, float, int, int]:
    """
    Resizes while preserving aspect ratio and centers the image on a black canvas.
    Returns:
        canvas, scale, off_x, off_y
    """
    scale, disp_w, disp_h, off_x, off_y = make_display_transform(img_bgr.shape, max_w, max_h)

    if abs(scale - 1.0) < 1e-12:
        resized = img_bgr.copy()
    else:
        resized = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    canvas[off_y:off_y + disp_h, off_x:off_x + disp_w] = resized
    return canvas, scale, off_x, off_y


def map_display_to_raw(
    x_disp: int,
    y_disp: int,
    raw_shape: tuple[int, int],
    scale: float,
    off_x: int,
    off_y: int,
) -> tuple[float, float] | None:
    h, w = raw_shape[:2]

    x_rel = x_disp - off_x
    y_rel = y_disp - off_y

    disp_w = int(round(w * scale))
    disp_h = int(round(h * scale))

    if x_rel < 0 or y_rel < 0 or x_rel >= disp_w or y_rel >= disp_h:
        return None

    u = x_rel / scale
    v = y_rel / scale
    return float(u), float(v)


def main():
    xray_path = pick_open_file_qt()
    if not xray_path:
        print("[INFO] Kein Bild gewählt. Abbruch.")
        return

    out_uv_path = pick_save_file_qt(OUT_UV_PATH_DEFAULT)
    if not out_uv_path:
        print("[INFO] Kein Speicherpfad gewählt. Abbruch.")
        return

    print("\n" + "=" * 72)
    print("DEBUG MARKER SELECTION")
    print("=" * 72)
    print(f"USE_FLIPPING = {USE_FLIPPING}")

    print(f"\nLade Bild: {xray_path}")
    img_raw = load_xray(xray_path)

    print("Führe Marker-Detection auf RAW image durch...")
    res = run_xray_marker_detection(
        img_raw,
        hough_params=HOUGH_PARAMS,
        use_clahe=True,
        clahe_clip=2.0,
        clahe_tiles=(12, 12),
        use_mask=False,
    )

    if res.circles is None or len(res.circles) == 0:
        print("[ERR] Keine Kreise detektiert.")
        sys.exit(1)

    circles_raw = np.asarray(res.circles, dtype=np.float64).reshape(-1, 3)
    print(f"Detektiert: {len(circles_raw)} Kreise")

    radii = circles_raw[:, 2]
    finite_r = radii[np.isfinite(radii)]
    pick_r = 0.6 * float(np.median(finite_r)) if finite_r.size else 20.0

    win_title = f"debug_marker_selection - RAW - {Path(xray_path).name}"

    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    anchors: list[int] = []
    roi_uv_raw: np.ndarray | None = None
    dbg: dict | None = None

    display_state = {
        "scale": 1.0,
        "off_x": 0,
        "off_y": 0,
    }

    def on_click(event, x, y, flags, param):
        nonlocal anchors, roi_uv_raw, dbg

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(anchors) >= 3:
            return

        mapped = map_display_to_raw(
            x_disp=x,
            y_disp=y,
            raw_shape=img_raw.shape,
            scale=float(display_state["scale"]),
            off_x=int(display_state["off_x"]),
            off_y=int(display_state["off_y"]),
        )
        if mapped is None:
            return

        u_raw, v_raw = mapped

        nearest = nearest_circle_index(circles_raw, u_raw, v_raw)
        if nearest is None:
            return

        d = np.linalg.norm(
            circles_raw[nearest, :2] - np.array([u_raw, v_raw], dtype=np.float64)
        )
        if d > pick_r * 3.0:
            return
        if nearest in anchors:
            return

        anchors.append(nearest)
        print(
            f"Anchor {len(anchors)}: idx={nearest}  "
            f"uv_raw=({circles_raw[nearest,0]:.1f}, {circles_raw[nearest,1]:.1f})"
        )

        if len(anchors) == 3:
            try:
                roi_uv_final_, roi_idx, dbg_ = compute_roi_from_grid(
                    circles=circles_raw,
                    anchor_idx=anchors,
                    margin_px=1.1 * pick_r,
                    gate_tol_pitch=0.40,
                    min_steps=2,
                )

                roi_uv_final = np.asarray(roi_uv_final_, dtype=np.float64).reshape(-1, 2)

                if USE_FLIPPING:
                    # current behavior: flipped / X-ray-source ordering
                    roi_uv_raw = roi_uv_final
                    dbg = dict(dbg_)
                    points_uv_to_save = roi_uv_raw
                    save_mode = "WITH_FLIPPING"

                else:
                    # unflipped / camera-view ordering from marker-selection debug output
                    roi_uv_raw = np.asarray(dbg_["debug_uv_raw"], dtype=np.float64).reshape(-1, 2)

                    dbg = dict(dbg_)
                    dbg["grid_i"] = dbg_["grid_i_raw"]
                    dbg["grid_j"] = dbg_["grid_j_raw"]

                    points_uv_to_save = roi_uv_raw
                    save_mode = "WITHOUT_FLIPPING"

                np.savez(
                    out_uv_path,
                    points_uv=points_uv_to_save.astype(np.float64),
                    points_uv_display=roi_uv_raw.astype(np.float64),
                    use_flipping=np.array(USE_FLIPPING, dtype=bool),
                    save_mode=np.array(save_mode),
                    nu=np.array(int(dbg["nu"]), dtype=np.int32),
                    nv=np.array(int(dbg["nv"]), dtype=np.int32),
                )

                print(f"[OK] saved RAW uv -> {out_uv_path}")
                print(f"[OK] save_mode = {save_mode}")

                print_roi_debug(f"ROI DEBUG — {save_mode}", roi_uv_raw, dbg)

            except Exception as e:
                print(f"[ERR] compute_roi_from_grid: {e}")
                roi_uv_raw = None
                dbg = None

    cv2.setMouseCallback(win_title, on_click)

    print("\nLinksklick = 3 Anchors auf RAW image wählen | R = Reset | Q/ESC = Quit\n")

    while True:
        vis = draw_state(
            img_raw,
            circles_raw,
            anchors,
            roi_uv_raw,
            dbg,
            title="RAW view",
        )

        vis_disp, scale, off_x, off_y = render_for_display(
            vis,
            max_w=DISPLAY_MAX_W,
            max_h=DISPLAY_MAX_H,
        )

        display_state["scale"] = scale
        display_state["off_x"] = off_x
        display_state["off_y"] = off_y

        cv2.imshow(win_title, vis_disp)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("r"), ord("R")):
            anchors = []
            roi_uv_raw = None
            dbg = None
            print("→ Reset.\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()