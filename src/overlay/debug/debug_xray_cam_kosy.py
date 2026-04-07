# -*- coding: utf-8 -*-
"""
debug_project_to_xray.py

Lädt einen Snapshot + uv_xray + X-Ray BMP und projiziert
die 3D-Punkte (points_xyz_camera) mit BEIDEN IPPE-Lösungen
ins X-Ray Bild. Zeigt beide Lösungen nebeneinander.

Auswahl-Reihenfolge:
  1. Snapshot NPZ
  2. uv_xray NPZ
  3. X-Ray BMP

Steuerung:
  Q / ESC — beenden
  S       — Screenshot speichern
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog


def _qt() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _pick_one(title: str, filt: str = "NumPy NPZ (*.npz)") -> Path | None:
    _qt()
    p, _ = QFileDialog.getOpenFileName(None, title, "", filt)
    return Path(p) if p else None


def _load_snap(path: Path) -> dict:
    with np.load(str(path), allow_pickle=False) as z:
        return {
            "name":    path.stem,
            "xyz_mm":  np.asarray(z["points_xyz_camera"], dtype=np.float64) * 1000.0,
            "K_xray":  np.asarray(z["K_xray"], dtype=np.float64),
        }


def _load_uv(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as z:
        for k in z.files:
            arr = np.asarray(z[k], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr
    raise ValueError(f"Keine (N,2)-Array in {path.name} gefunden.")


def _ippe_both(xyz_mm: np.ndarray, uv: np.ndarray, K: np.ndarray) -> list[dict]:
    dist = np.zeros((5, 1), dtype=np.float64)
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        xyz_mm.reshape(-1, 1, 3).astype(np.float64),
        uv.reshape(-1, 1, 2).astype(np.float64),
        K.astype(np.float64), dist,
        flags=cv2.SOLVEPNP_IPPE,
    )
    solutions = []
    for rvec, tvec in zip(rvecs, tvecs):
        rv = rvec.reshape(3)
        tv = tvec.reshape(3)
        R, _ = cv2.Rodrigues(rv.reshape(3, 1))
        pts, _ = cv2.projectPoints(
            xyz_mm.reshape(-1, 1, 3).astype(np.float64),
            rv.reshape(3, 1), tv.reshape(3, 1),
            K.astype(np.float64), dist,
        )
        err = float(np.mean(np.linalg.norm(pts.reshape(-1, 2) - uv, axis=1)))
        solutions.append({
            "rvec":   rv,
            "tvec":   tv,
            "R":      R,
            "reproj": err,
            "pts_proj": pts.reshape(-1, 2),  # projizierte uv_xray-Punkte (Verifikation)
        })
    return solutions


def _project_xyz_to_xray(xyz_mm: np.ndarray, sol: dict, K: np.ndarray) -> np.ndarray:
    """Projiziert xyz_mm mit gegebener IPPE-Lösung ins X-Ray Bild."""
    dist = np.zeros((5, 1), dtype=np.float64)
    pts, _ = cv2.projectPoints(
        xyz_mm.reshape(-1, 1, 3).astype(np.float64),
        sol["rvec"].reshape(3, 1),
        sol["tvec"].reshape(3, 1),
        K.astype(np.float64),
        dist,
    )
    return pts.reshape(-1, 2)


def _draw_projection(xray_bgr: np.ndarray, pts: np.ndarray,
                     uv_obs: np.ndarray, label: str,
                     color_proj: tuple, color_obs: tuple) -> np.ndarray:
    """
    Zeichnet auf eine Kopie des X-Ray Bildes:
      - projizierte Punkte (xyz_mm → X-Ray)  als Kreise
      - beobachtete uv_xray Punkte           als Kreuze
    """
    vis = xray_bgr.copy()
    H, W = vis.shape[:2]

    # beobachtete uv_xray Punkte (Kreuze, weiß)
    for (u, v) in uv_obs:
        u, v = int(round(u)), int(round(v))
        if 0 <= u < W and 0 <= v < H:
            cv2.drawMarker(vis, (u, v), color_obs, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

    # projizierte Punkte (Kreise, farbig)
    n_in = 0
    for (u, v) in pts:
        u, v = int(round(u)), int(round(v))
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(vis, (u, v), 6, color_proj, -1, cv2.LINE_AA)
            n_in += 1

    # Label
    cv2.putText(vis, label, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(vis, label, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Punkte im Bild: {n_in}/{len(pts)}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, f"Punkte im Bild: {n_in}/{len(pts)}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    return vis


def main() -> None:
    _qt()

    # --- Laden ---
    p_snap = _pick_one("Snapshot NPZ")
    if p_snap is None:
        return
    snap = _load_snap(p_snap)
    print(f"[OK] Snapshot: {p_snap.name}  xyz_mm={snap['xyz_mm'].shape}")

    p_uv = _pick_one("uv_xray NPZ")
    if p_uv is None:
        return
    uv = _load_uv(p_uv)
    print(f"[OK] uv_xray: {p_uv.name}  shape={uv.shape}")

    p_xray = _pick_one("X-Ray Bild (BMP/PNG/...)",
                        "Bilder (*.bmp *.png *.jpg *.jpeg *.tiff *.tif);;Alle (*)")
    if p_xray is None:
        return
    xray_raw = cv2.imread(str(p_xray), cv2.IMREAD_UNCHANGED)
    if xray_raw is None:
        print(f"[ERROR] Bild konnte nicht geladen werden: {p_xray}")
        return

    # Zu BGR konvertieren falls nötig
    if xray_raw.ndim == 2:
        xray_bgr = cv2.cvtColor(xray_raw, cv2.COLOR_GRAY2BGR)
    elif xray_raw.shape[2] == 4:
        xray_bgr = cv2.cvtColor(xray_raw, cv2.COLOR_BGRA2BGR)
    else:
        xray_bgr = xray_raw.copy()

    H, W = xray_bgr.shape[:2]
    print(f"[OK] X-Ray: {p_xray.name}  ({W}x{H})")

    # --- IPPE ---
    K_xray = snap["K_xray"]
    solutions = _ippe_both(snap["xyz_mm"], uv, K_xray)

    for i, sol in enumerate(solutions):
        tv = sol["tvec"]
        print(f"\nLösung {i+1}:")
        print(f"  tvec [mm]: tx={tv[0]:+9.2f}  ty={tv[1]:+9.2f}  tz={tv[2]:+9.2f}")
        print(f"  R[2,2]={sol['R'][2,2]:+.4f}   reproj={sol['reproj']:.4f} px")
        print(f"  tz>0={'Y' if tv[2]>0 else 'N'}  R[2,2]<0={'Y' if sol['R'][2,2]<0 else 'N'}  ty<0={'Y' if tv[1]<0 else 'N'}  tx>0={'Y' if tv[0]>0 else 'N'}")

    # --- Projektion ---
    # Farben: Lösung 1 = Cyan, Lösung 2 = Orange; beobachtet = Weiß
    vis1 = _draw_projection(
        xray_bgr,
        pts=_project_xyz_to_xray(snap["xyz_mm"], solutions[0], K_xray),
        uv_obs=uv,
        label=f"Loesung 1  (tx={solutions[0]['tvec'][0]:+.1f}  reproj={solutions[0]['reproj']:.3f}px)",
        color_proj=(255, 220, 0),   # Cyan
        color_obs=(255, 255, 255),
    )
    vis2 = _draw_projection(
        xray_bgr,
        pts=_project_xyz_to_xray(snap["xyz_mm"], solutions[1], K_xray),
        uv_obs=uv,
        label=f"Loesung 2  (tx={solutions[1]['tvec'][0]:+.1f}  reproj={solutions[1]['reproj']:.3f}px)",
        color_proj=(0, 140, 255),   # Orange
        color_obs=(255, 255, 255),
    )

    # Nebeneinander
    divider = np.full((H, 6, 3), (0, 255, 0), dtype=np.uint8)
    combined = np.hstack([vis1, divider, vis2])

    win = "IPPE Projektion — L1 (cyan) | L2 (orange)   Q=quit  S=save"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    disp_w = min(2 * W + 6, 2400)
    disp_h = int(disp_w * H / (2 * W + 6))
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.imshow(win, combined)

    print("\n[INFO] Q/ESC = beenden   S = speichern")
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (ord('q'), ord('Q'), 27):
            break
        if k in (ord('s'), ord('S')):
            out = p_xray.parent / (p_xray.stem + "_ippe_projection.png")
            cv2.imwrite(str(out), combined)
            print(f"[SAVE] {out}")
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()