# -*- coding: utf-8 -*-
"""
debug_pose_hom_comp.py

Vergleicht den Einfluss von Kalman-Filterung auf die Pose T_cx
fuer drei Datensaetze: single, raw, filtered.

Alle 3 Datensaetze liegen in derselben Snapshot-NPZ-Datei:
    points_xyz_camera_single  (121, 3)  [m]  -> single
    points_xyz_camera         (121, 3)  [m]  -> raw
    points_xyz_camera_filt    (121, 3)  [m]  -> filtered
    K_xray                    (3,  3)

uv_xray-Koordinaten kommen aus einer separaten NPZ-Datei.

Pipeline pro Snapshot pro Datensatz-Typ:
    1. T_bc  <- nominales 11x11 Board (2.54 mm pitch)
                rigid SVD fit: board_xyz -> points_xyz_camera_*
    2. T_bx  <- board_xy (aus board_xyz[:, :2]) + uv_xray
                DLT Homography + Decompose
    3. T_cx  =  T_bx @ inv(T_bc)

Plots (Figure, nicht gespeichert):
    Oben:   Std-Deviation von tx/ty/tz und dR als Balkendiagramm
            (single / raw / filtered nebeneinander)
    Unten:  Frame-to-frame Sprung |T_cx,i - T_cx,i-1| und dR,i vs i-1
            als Linienplot, alle 3 Typen ueberlagert
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.homography import (
    estimate_homography_dlt,
    decompose_homography_to_pose,
)


# ============================================================
# Konstanten
# ============================================================

_XYZ_KEYS: dict[str, str] = {
    "single":   "points_xyz_camera_single",
    "raw":      "points_xyz_camera",
    "filtered": "points_xyz_camera_filt",
}

_DATASET_TYPES = ("single", "raw", "filtered")

_DATASET_COLORS = {
    "single":   "#4C8BB5",
    "raw":      "#E07B3F",
    "filtered": "#5BAB6E",
}
_DATASET_MARKERS = {
    "single":   "o",
    "raw":      "s",
    "filtered": "^",
}


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
# Load: Snapshots
# ============================================================

def _load_snapshots(paths: list[Path]) -> list[dict]:
    snaps: list[dict] = []
    for p in paths:
        try:
            with np.load(str(p), allow_pickle=False) as z:
                keys = list(z.files)

                if "K_xray" not in keys:
                    print(f"[SKIP] {p.name}: missing key 'K_xray'")
                    continue
                K_xray = np.asarray(z["K_xray"], dtype=np.float64)
                if K_xray.shape != (3, 3):
                    print(f"[SKIP] {p.name}: K_xray shape {K_xray.shape}")
                    continue

                xyz_per_type: dict[str, np.ndarray] = {}
                for dtype, key in _XYZ_KEYS.items():
                    if key not in keys:
                        print(f"[WARN] {p.name}: missing key '{key}' (type '{dtype}')")
                        continue
                    arr = np.asarray(z[key], dtype=np.float64)
                    if arr.ndim != 2 or arr.shape[1] != 3:
                        print(f"[WARN] {p.name}: '{key}' shape {arr.shape}")
                        continue
                    xyz_per_type[dtype] = arr * 1000.0  # m -> mm

                if not xyz_per_type:
                    print(f"[SKIP] {p.name}: no valid xyz arrays found")
                    continue

                snaps.append({
                    "path":         p,
                    "name":         p.stem,
                    "K_xray":       K_xray,
                    "xyz_per_type": xyz_per_type,
                })
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")
    return snaps


# ============================================================
# Load: uv_xray
# ============================================================

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
# Nominales Board
# ============================================================

def _build_planar_board_xyz(
    *,
    nu: int         = 10,
    nv: int         = 10,
    pitch_mm: float = 2.54,
) -> np.ndarray:
    ncols = nu + 1
    nrows = nv + 1
    jj, ii = np.meshgrid(
        np.arange(ncols, dtype=np.float64),
        np.arange(nrows, dtype=np.float64),
        indexing="xy",
    )
    return np.column_stack([
        (jj * pitch_mm).ravel(),
        (ii * pitch_mm).ravel(),
        np.zeros(ncols * nrows, dtype=np.float64),
    ])


# ============================================================
# SE(3) rigid fit
# ============================================================

def _fit_rigid_transform_svd(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> np.ndarray:
    src = np.asarray(src_xyz, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst_xyz, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: {src.shape} vs {dst.shape}")

    sc = np.mean(src, axis=0)
    dc = np.mean(dst, axis=0)
    H  = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = dc.reshape(3, 1) - R @ sc.reshape(3, 1)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


# ============================================================
# T_bx via DLT Homography
# ============================================================

def _compose_T_bx(
    board_xyz: np.ndarray,
    uv_xray:   np.ndarray,
    K_xray:    np.ndarray,
) -> np.ndarray:
    board_xy      = board_xyz[:, :2].copy()
    H_bx          = estimate_homography_dlt(uv_xray, board_xy)
    R_bx, t_bx, _ = decompose_homography_to_pose(H_bx, K_xray)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_bx
    T[:3, 3]  = np.asarray(t_bx, dtype=np.float64).reshape(3)
    return T


# ============================================================
# SE(3) invert
# ============================================================

def _invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3:4]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3:] = -R.T @ t
    return Ti


# ============================================================
# Metriken
# ============================================================

def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos_theta = np.clip(0.5 * (np.trace(R_a.T @ R_b) - 1.0), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _frame_to_frame(T_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Gibt (dt_mm, dR_deg) jeweils shape (S-1,) zurueck:
        dt_mm[i]  = |t_{i+1} - t_i|
        dR_deg[i] = angle(R_{i+1}, R_i)
    """
    Ts    = np.asarray(T_list, dtype=np.float64)
    t_all = Ts[:, :3, 3]
    R_all = Ts[:, :3, :3]

    dt_mm  = np.linalg.norm(np.diff(t_all, axis=0), axis=1)
    dR_deg = np.array(
        [_rotation_angle_deg(R_all[i], R_all[i + 1]) for i in range(len(R_all) - 1)],
        dtype=np.float64,
    )
    return dt_mm, dR_deg


def _std_metrics(T_list: list[np.ndarray]) -> dict[str, float]:
    """
    Gibt std von tx, ty, tz [mm] und mittleren Rotationswinkel-std [deg] zurueck.
    dR_std: std der paarweisen Winkel gegenueber dem Gesamtmittel-R
    (approximiert als std des Rotationswinkels aller R_i gegen R_mean).
    """
    Ts    = np.asarray(T_list, dtype=np.float64)
    t_all = Ts[:, :3, 3]
    R_all = Ts[:, :3, :3]

    # Approximiertes R_mean via SVD des gestapelten R
    R_stack = R_all.reshape(-1, 3)
    U, _, Vt = np.linalg.svd(R_stack.T @ R_stack)  # nicht exakt, aber robust
    # Einfachere Approximation: R des ersten als Referenz, dann std der Winkel
    R_ref   = R_all[0]
    angles  = np.array([_rotation_angle_deg(R_ref, R) for R in R_all])

    return {
        "std_tx": float(np.std(t_all[:, 0])),
        "std_ty": float(np.std(t_all[:, 1])),
        "std_tz": float(np.std(t_all[:, 2])),
        "std_dR": float(np.std(angles)),
    }


# ============================================================
# Console summary
# ============================================================

def _print_summary(dtype: str, names: list[str], T_list: list[np.ndarray]) -> None:
    Ts    = np.asarray(T_list, dtype=np.float64)
    t_all = Ts[:, :3, 3]

    dt_mm, dR_deg = _frame_to_frame(T_list)
    m = _std_metrics(T_list)

    print("\n" + "=" * 70)
    print(f"T_cx  [{dtype.upper()}]")
    print("=" * 70)
    print(f"\nTranslation std [mm]:  tx={m['std_tx']:.3f}  ty={m['std_ty']:.3f}  tz={m['std_tz']:.3f}")
    print(f"Rotation    std [deg]: dR={m['std_dR']:.4f}")
    print(f"\nFrame-to-frame |dt|:  mean={np.mean(dt_mm):.3f} mm   max={np.max(dt_mm):.3f} mm")
    print(f"Frame-to-frame  dR:   mean={np.mean(dR_deg):.4f} deg  max={np.max(dR_deg):.4f} deg")

    print("\nPer snapshot:")
    for name, t in zip(names, t_all):
        print(f"  {name}: tx={t[0]:+9.3f}  ty={t[1]:+9.3f}  tz={t[2]:+9.3f}")


# ============================================================
# Plot
# ============================================================

def _show_plots(results: dict[str, dict]) -> None:
    """
    Figure mit 4 Subplots:
        [0,0] Std tx/ty/tz als gruppiertes Balkendiagramm
        [0,1] Std dR als gruppiertes Balkendiagramm
        [1,0] Frame-to-frame |dt| als Linienplot
        [1,1] Frame-to-frame dR als Linienplot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "T_cx Stabilitaet: single vs raw vs filtered\n"
        "(nominales Board + rigid SVD fuer T_bc,  DLT Homography fuer T_bx)",
        fontsize=12,
        fontweight="bold",
    )

    # --- Std-Balken ---
    std_labels  = ["tx", "ty", "tz"]
    bar_width   = 0.22
    n_types     = len(_DATASET_TYPES)
    x_std       = np.arange(len(std_labels))

    ax_t = axes[0, 0]
    ax_r = axes[0, 1]

    for i, dtype in enumerate(_DATASET_TYPES):
        T_list = results.get(dtype, {}).get("T_list", [])
        if not T_list:
            continue
        m     = _std_metrics(T_list)
        color = _DATASET_COLORS[dtype]
        offset = (i - (n_types - 1) / 2) * bar_width

        ax_t.bar(
            x_std + offset,
            [m["std_tx"], m["std_ty"], m["std_tz"]],
            width=bar_width,
            color=color,
            label=dtype,
            alpha=0.85,
        )
        ax_r.bar(
            [0 + offset],
            [m["std_dR"]],
            width=bar_width,
            color=color,
            label=dtype,
            alpha=0.85,
        )

    ax_t.set_xticks(x_std)
    ax_t.set_xticklabels(std_labels)
    ax_t.set_ylabel("std [mm]")
    ax_t.set_title("Streuung Translation (std ueber alle Snapshots)")
    ax_t.legend(fontsize=9)
    ax_t.grid(axis="y", alpha=0.4, linestyle="--")

    ax_r.set_xticks([0])
    ax_r.set_xticklabels(["dR"])
    ax_r.set_ylabel("std [deg]")
    ax_r.set_title("Streuung Rotation (std ueber alle Snapshots)")
    ax_r.legend(fontsize=9)
    ax_r.grid(axis="y", alpha=0.4, linestyle="--")

    # --- Frame-to-frame Linienplots ---
    ax_dt = axes[1, 0]
    ax_dR = axes[1, 1]

    for dtype in _DATASET_TYPES:
        T_list = results.get(dtype, {}).get("T_list", [])
        if len(T_list) < 2:
            continue
        dt_mm, dR_deg = _frame_to_frame(T_list)

        # x-Achse: Uebergang i -> i+1, beschriftet als "i→i+1"
        xs     = np.arange(1, len(dt_mm) + 1)
        color  = _DATASET_COLORS[dtype]
        marker = _DATASET_MARKERS[dtype]

        ax_dt.plot(xs, dt_mm,  marker=marker, color=color,
                   label=dtype, linewidth=1.8, markersize=6)
        ax_dR.plot(xs, dR_deg, marker=marker, color=color,
                   label=dtype, linewidth=1.8, markersize=6)

    max_steps = max(
        (len(v.get("T_list", [])) - 1 for v in results.values() if len(v.get("T_list", [])) > 1),
        default=1,
    )
    xs_ref = np.arange(1, max_steps + 1)

    for ax, ylabel, title in zip(
        [ax_dt, ax_dR],
        ["|dt| [mm]", "dR [deg]"],
        [
            "Frame-to-frame Translationssprung  |t_i - t_{i-1}|",
            "Frame-to-frame Rotationssprung  dR(R_i, R_{i-1})",
        ],
    ):
        ax.set_xticks(xs_ref)
        ax.set_xticklabels([f"{i-1}→{i}" for i in xs_ref], fontsize=8, rotation=30)
        ax.set_xlabel("Snapshot-Uebergang")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.4, linestyle="--")

    plt.tight_layout()
    plt.show()


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

    uv_paths = _pick_npz_files("uv_xray NPZ-Datei auswaehlen")
    uv_list  = _load_uv_xray(uv_paths, len(snaps))

    board_xyz = _build_planar_board_xyz(nu=10, nv=10, pitch_mm=2.54)
    n_board   = board_xyz.shape[0]
    print(f"\n[INFO] Nominales Board: {n_board} Punkte (11x11, pitch=2.54 mm)")

    results: dict[str, dict] = {
        dt: {"names": [], "T_list": []} for dt in _DATASET_TYPES
    }

    for snap, uv in zip(snaps, uv_list):
        name   = snap["name"]
        K_xray = snap["K_xray"]

        if uv is None:
            print(f"[SKIP] {name}: kein uv_xray gefunden.")
            continue
        if uv.shape[0] != n_board:
            print(f"[SKIP] {name}: uv hat {uv.shape[0]} Punkte, erwartet {n_board}.")
            continue

        for dtype in _DATASET_TYPES:
            xyz_mm = snap["xyz_per_type"].get(dtype)
            if xyz_mm is None:
                continue
            if xyz_mm.shape[0] != n_board:
                print(f"[SKIP] {name} [{dtype}]: xyz hat {xyz_mm.shape[0]} Punkte.")
                continue

            try:
                T_bc = _fit_rigid_transform_svd(board_xyz, xyz_mm)
                T_cb = _invert_T(T_bc)
                T_bx = _compose_T_bx(board_xyz, uv, K_xray)
                T_cx = T_bx @ T_cb

                results[dtype]["names"].append(name)
                results[dtype]["T_list"].append(T_cx)

            except Exception as e:
                print(f"[ERROR] {name} [{dtype}]: {e}")

    any_result = False
    for dtype in _DATASET_TYPES:
        if results[dtype]["T_list"]:
            _print_summary(dtype, results[dtype]["names"], results[dtype]["T_list"])
            any_result = True

    if not any_result:
        print("[ERROR] Keine gueltigen Posen berechnet.")
        return

    _show_plots(results)
    print("\n[INFO] Fertig.")


if __name__ == "__main__":
    main()