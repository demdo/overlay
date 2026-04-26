# -*- coding: utf-8 -*-
"""
debug_pose_hom.py

Berechnet T_cx ueber denselben Homography-basierten Ablauf wie zuvor,
aber jetzt konsistent mit dem NEUEN kanonischen Board-KOS.

Zusatz:
- Laedt aus der UV-NPZ jetzt BEIDE Varianten:
    * uv_raw
    * uv_final
- Berechnet und printet die Pose fuer beide Varianten getrennt.

Neues Prinzip
-------------
- Das nominale Board kommt NICHT mehr aus einer lokalen Hilfsfunktion,
  sondern aus build_board_xyz_canonical(...).
- UV-Korrespondenzen muessen bereits in derselben Board-Reihenfolge
  gespeichert sein wie board_xyz:
      erste Zeile TL -> TR
      ...
      letzte Zeile BL -> BR

Pipeline
--------
1. T_bc  <- rigid SVD fit: canonical board_xyz -> points_xyz_camera_filt
2. T_bx  <- board_xy (aus canonical board_xyz[:, :2]) + uv_xray
            DLT Homography + Decompose
3. T_cx  =  T_bx @ inv(T_bc)

Erwartete NPZ-Inhalte
---------------------
XYZ-NPZ:
    - points_xyz_camera_filt   (N, 3) in [m]
    - K_xray                   (3, 3)

UV-NPZ:
    - uv_raw    (N, 2)
    - uv_final  (N, 2)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.homography import (
    estimate_homography_dlt,
    decompose_homography_to_pose,
    build_board_xyz_canonical,
)


# ============================================================
# Qt helpers
# ============================================================

def _qt() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _pick_npz_file(title: str) -> Path | None:
    _qt()
    path, _ = QFileDialog.getOpenFileName(None, title, "", "NumPy NPZ (*.npz)")
    path = path.strip()
    return Path(path) if path else None


# ============================================================
# Loading
# ============================================================

def _load_xyz_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as z:
        keys = list(z.files)

        if "points_xyz_camera_filt" not in keys:
            raise KeyError(
                "Missing key 'points_xyz_camera_filt' in XYZ NPZ.\n"
                f"Available keys: {keys}"
            )

        xyz = np.asarray(z["points_xyz_camera_filt"], dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(
                f"'points_xyz_camera_filt' has invalid shape {xyz.shape}, expected (N,3)."
            )

        print("[INFO] Using XYZ key: 'points_xyz_camera_filt'")

        if "K_xray" not in keys:
            raise KeyError(
                "Missing key 'K_xray' in XYZ NPZ.\n"
                f"Available keys: {keys}"
            )

        K_xray = np.asarray(z["K_xray"], dtype=np.float64)
        if K_xray.shape != (3, 3):
            raise ValueError(f"'K_xray' has invalid shape {K_xray.shape}, expected (3,3).")

        xyz_mm = xyz * 1000.0  # m -> mm
        return xyz_mm, K_xray


def _load_uv_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as z:
        keys = list(z.files)

        # --------------------------------------------------------
        # Case 1: new format with both variants
        # --------------------------------------------------------
        if "uv_raw" in keys and "uv_final" in keys:
            uv_raw = np.asarray(z["uv_raw"], dtype=np.float64)
            uv_final = np.asarray(z["uv_final"], dtype=np.float64)

            if uv_raw.ndim != 2 or uv_raw.shape[1] != 2:
                raise ValueError(f"'uv_raw' has invalid shape {uv_raw.shape}, expected (N,2).")
            if uv_final.ndim != 2 or uv_final.shape[1] != 2:
                raise ValueError(f"'uv_final' has invalid shape {uv_final.shape}, expected (N,2).")
            if uv_raw.shape != uv_final.shape:
                raise ValueError("uv_raw and uv_final must have the same shape.")

            print("[INFO] Using UV keys: 'uv_raw' and 'uv_final'")
            return uv_raw, uv_final

        # --------------------------------------------------------
        # Case 2: legacy/debug format with only one UV array
        # --------------------------------------------------------
        if "points_uv" in keys:
            uv = np.asarray(z["points_uv"], dtype=np.float64)

            if uv.ndim != 2 or uv.shape[1] != 2:
                raise ValueError(f"'points_uv' has invalid shape {uv.shape}, expected (N,2).")

            print("[INFO] Using legacy UV key: 'points_uv' for BOTH uv_raw and uv_final")
            return uv.copy(), uv.copy()

        raise KeyError(
            "UV NPZ must contain either:\n"
            "  - 'uv_raw' AND 'uv_final'\n"
            "or\n"
            "  - legacy key 'points_uv'\n"
            f"Available keys: {keys}"
        )


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

    H = (src - sc).T @ (dst - dc)
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


def _invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3:4]

    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3:] = -R.T @ t
    return Ti


# ============================================================
# T_bx via DLT Homography
# ============================================================

def _compose_T_bx(
    board_xyz: np.ndarray,
    uv_xray: np.ndarray,
    K_xray: np.ndarray,
) -> np.ndarray:
    board_xy = np.asarray(board_xyz[:, :2], dtype=np.float64).copy()

    H_bx = estimate_homography_dlt(uv_xray, board_xy)
    R_bx, t_bx, _ = decompose_homography_to_pose(H_bx, K_xray)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_bx
    T[:3, 3] = np.asarray(t_bx, dtype=np.float64).reshape(3)
    return T


# ============================================================
# Printing
# ============================================================

def _print_matrix(name: str, M: np.ndarray) -> None:
    print(f"\n{name} =")
    for row in np.asarray(M, dtype=np.float64):
        print("[" + " ".join(f"{v:11.6f}" for v in row) + "]")


def _print_pose_result(
    *,
    label: str,
    T_bc: np.ndarray,
    T_cb: np.ndarray,
    T_bx: np.ndarray,
    T_cx: np.ndarray,
) -> None:
    print("\n" + "=" * 70)
    print(f"RESULTS FOR {label}")
    print("=" * 70)

    _print_matrix("T_bc", T_bc)
    _print_matrix("T_cb", T_cb)
    _print_matrix("T_bx", T_bx)
    _print_matrix("T_cx", T_cx)

    t = T_cx[:3, 3]
    print("\nTranslation T_cx [mm]:")
    print(f"tx = {t[0]:+.6f}")
    print(f"ty = {t[1]:+.6f}")
    print(f"tz = {t[2]:+.6f}")
    
    print("\n--- T_xc (xray -> camera) ---")
    _print_matrix("T_xc", _invert_T(T_cx))
    
    t_xc = _invert_T(T_cx)[:3, 3]
    print("\nt_xc [mm]:")
    print(f"tx = {t_xc[0]:+.6f}")
    print(f"ty = {t_xc[1]:+.6f}")
    print(f"tz = {t_xc[2]:+.6f}")
    
    
def dump_points_txt(path, board_xyz, xyz_cam):
    with open(path, "w", encoding="utf-8") as f:
        f.write("IDX | Xb [mm]   Yb [mm]   Zb [mm] || Xc [mm]   Yc [mm]   Zc [mm]\n")
        f.write("-" * 90 + "\n")

        for i in range(len(board_xyz)):
            xb, yb, zb = board_xyz[i]
            xc, yc, zc = xyz_cam[i]
            f.write(
                f"{i:3d} | "
                f"{xb:9.3f} {yb:9.3f} {zb:9.3f} || "
                f"{xc:9.3f} {yc:9.3f} {zc:9.3f}\n"
            )

        f.write("\n")
        f.write("BOARD XYZ stats:\n")
        f.write(f"  min = {np.min(board_xyz, axis=0)}\n")
        f.write(f"  max = {np.max(board_xyz, axis=0)}\n")
        f.write(f"  mean= {np.mean(board_xyz, axis=0)}\n")

        f.write("\n")
        f.write("CAMERA XYZ stats:\n")
        f.write(f"  min = {np.min(xyz_cam, axis=0)}\n")
        f.write(f"  max = {np.max(xyz_cam, axis=0)}\n")
        f.write(f"  mean= {np.mean(xyz_cam, axis=0)}\n")


# ============================================================
# Main
# ============================================================

def main() -> None:
    xyz_path = _pick_npz_file("XYZ NPZ auswaehlen")
    if xyz_path is None:
        print("[INFO] Keine XYZ-NPZ ausgewaehlt.")
        return

    uv_path = _pick_npz_file("UV NPZ auswaehlen")
    if uv_path is None:
        print("[INFO] Keine UV-NPZ ausgewaehlt.")
        return

    print("\n" + "=" * 70)
    print("SELECT FILES")
    print("=" * 70)
    print(f"XYZ: {xyz_path}")
    print(f"UV : {uv_path}")

    xyz_mm, K_xray = _load_xyz_npz(xyz_path)
    uv_raw, uv_final = _load_uv_npz(uv_path)

    # Kanonisches Board-KOS
    board_xyz = build_board_xyz_canonical(
        nu=10,
        nv=10,
        pitch_mm=2.54,
    )
    n_board = board_xyz.shape[0]

    if xyz_mm.shape[0] != n_board:
        raise ValueError(f"XYZ has {xyz_mm.shape[0]} points, expected {n_board}.")
    if uv_raw.shape[0] != n_board or uv_final.shape[0] != n_board:
        raise ValueError(
            f"UV mismatch: uv_raw={uv_raw.shape[0]}, uv_final={uv_final.shape[0]}, expected {n_board}."
        )

    print("\n" + "=" * 70)
    print("INPUT INFO")
    print("=" * 70)
    print(f"Nominal board points : {n_board}")
    print(f"XYZ shape            : {xyz_mm.shape}")
    print(f"uv_raw shape         : {uv_raw.shape}")
    print(f"uv_final shape       : {uv_final.shape}")
    _print_matrix("K_xray", K_xray)

    # T_bc / T_cb
    T_bc = _fit_rigid_transform_svd(board_xyz, xyz_mm)
    T_cb = _invert_T(T_bc)

    # --------------------------------------------------------
    # RAW
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("POSE WITH uv_raw")
    print("=" * 70)

    T_bx_raw = _compose_T_bx(board_xyz, uv_raw, K_xray)
    T_cx_raw = T_bx_raw @ T_cb
    T_xc_raw = _invert_T(T_cx_raw)
    t_xc_raw = T_xc_raw[:3, 3]

    _print_pose_result(
        label="uv_raw",
        T_bc=T_bc,
        T_cb=T_cb,
        T_bx=T_bx_raw,
        T_cx=T_cx_raw,
    )

    # --------------------------------------------------------
    # FLIPPED
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("POSE WITH uv_final (flipped)")
    print("=" * 70)

    T_bx_final = _compose_T_bx(board_xyz, uv_final, K_xray)
    T_cx_final = T_bx_final @ T_cb
    T_xc_final = _invert_T(T_cx_final)
    t_xc_final = T_xc_final[:3, 3]

    _print_pose_result(
        label="uv_final (flipped)",
        T_bc=T_bc,
        T_cb=T_cb,
        T_bx=T_bx_final,
        T_cx=T_cx_final,
    )

    print("\n[INFO] Fertig.")
    
    dump_points_txt("debug_points_board_camera.txt", board_xyz, xyz_mm)


if __name__ == "__main__":
    main()