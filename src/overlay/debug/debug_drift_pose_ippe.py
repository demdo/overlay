# -*- coding: utf-8 -*-
"""
debug_drift_pose_ippe.py

Gibt fuer alle Snapshots beide IPPE-Kandidaten-Posen aus,
basierend auf den Kalman-gefilterten 3D-Punkten (points_xyz_camera_filt).

IPPE wird direkt via cv2.solvePnPGeneric aufgerufen —
ohne Kandidaten-Selektion, ohne iterative refinement.

Benoetigte Snapshot-Keys:
    points_xyz_camera_filt  (N,3)  [m]
    K_xray                  (3,3)

uv_xray aus separater NPZ-Datei.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2
from PySide6.QtWidgets import QApplication, QFileDialog


# ============================================================
# Qt helpers
# ============================================================

def _qt() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _pick_npz_files(title: str) -> list[Path]:
    _qt()
    paths, _ = QFileDialog.getOpenFileNames(None, title, "", "NumPy NPZ (*.npz)")
    return sorted([Path(p) for p in paths if p], key=lambda p: p.name.lower())


# ============================================================
# Load helpers
# ============================================================

def _load_snapshots(paths: list[Path]) -> list[dict]:
    snaps: list[dict] = []
    for p in paths:
        try:
            with np.load(str(p), allow_pickle=False) as z:
                keys = list(z.files)

                missing = [k for k in ("points_xyz_camera_filt", "K_xray") if k not in keys]
                if missing:
                    print(f"[SKIP] {p.name}: missing keys {missing}")
                    continue

                xyz_m  = np.asarray(z["points_xyz_camera_filt"], dtype=np.float64)
                K_xray = np.asarray(z["K_xray"],                 dtype=np.float64)

                if xyz_m.ndim != 2 or xyz_m.shape[1] != 3:
                    print(f"[SKIP] {p.name}: points_xyz_camera_filt shape {xyz_m.shape}")
                    continue
                if K_xray.shape != (3, 3):
                    print(f"[SKIP] {p.name}: K_xray shape {K_xray.shape}")
                    continue

                snaps.append({
                    "name":   p.stem,
                    "xyz_mm": xyz_m * 1000.0,
                    "K_xray": K_xray,
                })

        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")
    return snaps


def _first_uv_array_from_npz(path: Path) -> np.ndarray | None:
    try:
        with np.load(str(path), allow_pickle=False) as z:
            if "uv_xray" in z.files:
                arr = np.asarray(z["uv_xray"], dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    return arr
            for k in z.files:
                arr = np.asarray(z[k], dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    return arr
    except Exception:
        pass
    return None


def _load_uv_xray(paths: list[Path], n_snapshots: int) -> list[np.ndarray | None]:
    result: list[np.ndarray | None] = [None] * n_snapshots
    if not paths:
        return result

    if len(paths) == 1:
        p = paths[0]
        try:
            with np.load(str(p), allow_pickle=False) as z:
                keys = list(z.files)

                if "uv_xray" in keys:
                    arr = np.asarray(z["uv_xray"], dtype=np.float64)
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        for i in range(n_snapshots):
                            result[i] = arr
                        return result
                    if arr.ndim == 3 and arr.shape[2] == 2:
                        for i in range(min(n_snapshots, arr.shape[0])):
                            result[i] = arr[i]
                        return result

                snap_keys = sorted(k for k in keys if k.startswith("snap_"))
                if snap_keys:
                    for i, k in enumerate(snap_keys[:n_snapshots]):
                        arr = np.asarray(z[k], dtype=np.float64)
                        if arr.ndim == 2 and arr.shape[1] == 2:
                            result[i] = arr
                    return result

                for k in keys:
                    arr = np.asarray(z[k], dtype=np.float64)
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        for i in range(n_snapshots):
                            result[i] = arr
                        return result

        except Exception as e:
            print(f"[WARN] {p.name}: {e}")
        return result

    for i, p in enumerate(paths[:n_snapshots]):
        arr = _first_uv_array_from_npz(p)
        if arr is not None:
            result[i] = arr
        else:
            print(f"[SKIP uv_xray] {p.name}: no (N,2) array found")
    return result


# ============================================================
# IPPE: beide Kandidaten direkt
# ============================================================

def _reproj_mean(
    xyz_mm:  np.ndarray,
    uv:      np.ndarray,
    K:       np.ndarray,
    rvec:    np.ndarray,
    tvec:    np.ndarray,
) -> float:
    dist = np.zeros((5, 1), dtype=np.float64)
    uv_proj, _ = cv2.projectPoints(
        xyz_mm.reshape(-1, 3),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        K,
        dist,
    )
    return float(np.mean(np.linalg.norm(uv - uv_proj.reshape(-1, 2), axis=1)))


def _ippe_candidates(
    xyz_mm: np.ndarray,
    uv:     np.ndarray,
    K:      np.ndarray,
) -> list[dict]:
    """
    Ruft cv2.solvePnPGeneric mit SOLVEPNP_IPPE auf und gibt beide
    Kandidaten als Liste von dicts zurueck.
    """
    dist = np.zeros((5, 1), dtype=np.float64)

    success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        objectPoints=xyz_mm.reshape(-1, 3).astype(np.float64),
        imagePoints=uv.reshape(-1, 2).astype(np.float64),
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not success or rvecs is None or len(rvecs) == 0:
        raise RuntimeError("SOLVEPNP_IPPE failed.")

    candidates = []
    for idx, (rv_raw, tv_raw) in enumerate(zip(rvecs, tvecs)):
        rv = np.asarray(rv_raw, dtype=np.float64).reshape(3)
        tv = np.asarray(tv_raw, dtype=np.float64).reshape(3)
        R, _ = cv2.Rodrigues(rv.reshape(3, 1))

        reproj = _reproj_mean(xyz_mm, uv, K, rv, tv)

        candidates.append({
            "idx":        idx,
            "rvec":       rv,
            "tvec":       tv,
            "R":          R,
            "reproj_mean": reproj,
        })

    return candidates


# ============================================================
# Output
# ============================================================

def _print_candidate(c: dict) -> None:
    tv = c["tvec"]
    rv = c["rvec"]
    print(f"  Kandidat {c['idx']}:")
    print(f"    tvec [mm]    : tx={tv[0]:+9.2f}  ty={tv[1]:+9.2f}  tz={tv[2]:+9.2f}")
    print(f"    rvec [rad]   : [{rv[0]:+.4f}  {rv[1]:+.4f}  {rv[2]:+.4f}]")
    print(f"    R[2,2]       : {c['R'][2,2]:+.4f}")
    print(f"    reproj [px]  : mean={c['reproj_mean']:.4f}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    snap_paths = _pick_npz_files("Snapshot NPZ-Dateien auswaehlen")
    if not snap_paths:
        print("[INFO] Keine Snapshots ausgewaehlt.")
        return

    snaps = _load_snapshots(snap_paths)
    if not snaps:
        print("[ERROR] Keine gueltigen Snapshots.")
        return

    uv_paths = _pick_npz_files("uv_xray NPZ auswaehlen")
    uv_list  = _load_uv_xray(uv_paths, len(snaps))

    for snap, uv in zip(snaps, uv_list):
        name   = snap["name"]
        xyz_mm = snap["xyz_mm"]
        K_xray = snap["K_xray"]

        print("\n" + "=" * 70)
        print(f"  {name}  [filtered]")
        print("=" * 70)

        if uv is None:
            print("  [SKIP] kein uv_xray gefunden.")
            continue

        if xyz_mm.shape[0] != uv.shape[0]:
            print(f"  [SKIP] mismatch: xyz={xyz_mm.shape[0]} Punkte, uv={uv.shape[0]} Punkte.")
            continue

        try:
            candidates = _ippe_candidates(xyz_mm, uv, K_xray)
            for c in candidates:
                _print_candidate(c)
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n[INFO] Fertig.")


if __name__ == "__main__":
    main()