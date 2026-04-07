# -*- coding: utf-8 -*-
"""
debug_xray_kosy.py

Kamera-Pan-Test: Kamera wird von links nach rechts geschwenkt,
Board bleibt fest -> gemeinsames uv_xray fuer alle drei Positionen.

Auswahl-Reihenfolge der Datei-Dialoge:
  1. Snapshot NPZ — Kamera schaut LINKS
  2. Snapshot NPZ — Kamera schaut MITTE
  3. Snapshot NPZ — Kamera schaut RECHTS
  4. uv_xray NPZ  — gemeinsam fuer alle drei

Gibt fuer jede Position BEIDE IPPE-Loesungen aus, keine Auswahl.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog


def _qt() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _pick_one(title: str) -> Path | None:
    _qt()
    p, _ = QFileDialog.getOpenFileName(None, title, "", "NumPy NPZ (*.npz)")
    return Path(p) if p else None


def _load_snap(path: Path) -> dict:
    with np.load(str(path), allow_pickle=False) as z:
        if "points_xyz_camera_filt" in z.files:
            xyz_key = "points_xyz_camera_filt"
        elif "points_xyz_camera" in z.files:
            xyz_key = "points_xyz_camera"
        else:
            raise KeyError(f"Kein xyz-Key gefunden. Keys: {list(z.files)}")
        return {
            "name":    path.stem,
            "xyz_mm":  np.asarray(z[xyz_key], dtype=np.float64) * 1000.0,
            "K_xray":  np.asarray(z["K_xray"], dtype=np.float64),
            "xyz_key": xyz_key,
        }


def _load_uv(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as z:
        if "uv_xray" in z.files:
            arr = np.asarray(z["uv_xray"], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr
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
        solutions.append({"rvec": rv, "tvec": tv, "R": R, "reproj": err})
    return solutions


def main() -> None:
    _qt()

    positions = [
        ("links",  "Kamera schaut LINKS"),
        ("mitte",  "Kamera schaut MITTE"),
        ("rechts", "Kamera schaut RECHTS"),
    ]

    # Snapshots laden
    snaps: list[tuple[str, dict]] = []
    for key, desc in positions:
        p = _pick_one(f"Snapshot NPZ — {desc}")
        if p is None:
            print(f"[ABBRUCH] Kein Snapshot fuer '{key}' gewaehlt.")
            return
        try:
            snap = _load_snap(p)
            snaps.append((key, snap))
            print(f"[OK] {key}: {p.name}  xyz_mm={snap['xyz_mm'].shape}  key='{snap['xyz_key']}'")
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")
            return

    # Gemeinsames uv_xray laden
    p_uv = _pick_one("uv_xray NPZ — gemeinsam fuer alle drei Positionen")
    if p_uv is None:
        print("[ABBRUCH] Kein uv_xray gewaehlt.")
        return
    try:
        uv = _load_uv(p_uv)
        print(f"[OK] uv_xray: {p_uv.name}  shape={uv.shape}")
    except Exception as e:
        print(f"[ERROR] {p_uv.name}: {e}")
        return

    K_xray = snaps[0][1]["K_xray"]

    sep = "=" * 70
    for key, snap in snaps:
        xyz_mm = snap["xyz_mm"]

        if xyz_mm.shape[0] != uv.shape[0]:
            print(f"\n[SKIP] {key}: xyz={xyz_mm.shape[0]} Punkte, uv={uv.shape[0]} Punkte.")
            continue

        solutions = _ippe_both(xyz_mm, uv, K_xray)

        print(f"\n{sep}")
        print(f"  [KAMERA {key.upper()}]  {snap['name']}")
        print(sep)
        for s_idx, sol in enumerate(solutions):
            tv = sol["tvec"]
            rv = sol["rvec"]
            print(f"  Loesung {s_idx + 1}:")
            print(f"    tvec [mm]: tx={tv[0]:+9.2f}  ty={tv[1]:+9.2f}  tz={tv[2]:+9.2f}")
            print(f"    rvec [rad]: [{rv[0]:+.4f}  {rv[1]:+.4f}  {rv[2]:+.4f}]")
            print(f"    R[2,2]={sol['R'][2,2]:+.4f}   reproj={sol['reproj']:.4f} px")
            print(
                f"    tz>0={'Y' if tv[2]>0 else 'N'}"
                f"  R[2,2]<0={'Y' if sol['R'][2,2]<0 else 'N'}"
                f"  ty<0={'Y' if tv[1]<0 else 'N'}"
                f"  tx>0={'Y' if tv[0]>0 else 'N'}"
            )

    print(f"\n{sep}")


if __name__ == "__main__":
    main()