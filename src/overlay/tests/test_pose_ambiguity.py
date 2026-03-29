# overlay/debug/test_camera_to_xray_board_flip_test.py

from __future__ import annotations

from pathlib import Path
import sys
from itertools import product

from PySide6.QtWidgets import QApplication, QFileDialog

import cv2
import numpy as np

from overlay.calib.calib_camera_to_xray_handeye import (
    compose_camera_to_xray,
    pose_vectors_to_board_to_xray,
)
from overlay.tracking.transforms import (
    as_transform,
    make_transform,
)


# ============================================================
# Config
# ============================================================

K_KEYS = ("Kx", "K_xray", "K")

BOARD_ROWS = 11
BOARD_COLS = 11
BOARD_PITCH_MM = 2.54
CENTER_BOARD_AT_ORIGIN = False

PRINT_TRANSFORMS = False   # für Flip-Test besser erstmal kompakt
PRINT_R_ONLY = True

# Nur Einzel-Flips:
TEST_FLIPS = [
    ("none", (1.0, 1.0, 1.0)),
    ("flip_x", (-1.0, 1.0, 1.0)),
    ("flip_y", (1.0, -1.0, 1.0)),
    ("flip_z", (1.0, 1.0, -1.0)),
]

# Falls du ALLE Kombinationen testen willst, stattdessen:
# TEST_FLIPS = []
# for sx, sy, sz in product([1.0, -1.0], repeat=3):
#     name = f"sx_{int(sx):+d}_sy_{int(sy):+d}_sz_{int(sz):+d}"
#     TEST_FLIPS.append((name, (sx, sy, sz)))


# ============================================================
# File dialogs
# ============================================================

def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def select_K_file() -> Path:
    _ensure_app()

    dlg = QFileDialog()
    dlg.setWindowTitle("Select X-ray intrinsics file")
    dlg.setFileMode(QFileDialog.ExistingFile)
    dlg.setNameFilter("NumPy files (*.npz)")
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)

    if dlg.exec() != QFileDialog.Accepted:
        raise RuntimeError("No K file selected.")

    return Path(dlg.selectedFiles()[0])


def select_pose_files() -> list[Path]:
    _ensure_app()

    dlg = QFileDialog()
    dlg.setWindowTitle("Select pose_debug files")
    dlg.setFileMode(QFileDialog.ExistingFiles)
    dlg.setNameFilter("NumPy files (*.npz)")
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)

    if dlg.exec() != QFileDialog.Accepted:
        raise RuntimeError("No pose files selected.")

    files = [Path(p) for p in dlg.selectedFiles()]
    if len(files) < 1:
        raise ValueError("Please select at least 1 pose file.")
    return files


# ============================================================
# Load helpers
# ============================================================

def load_K(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as npz:
        for key in K_KEYS:
            if key in npz.files:
                K = np.asarray(npz[key], dtype=np.float64)
                break
        else:
            raise KeyError(
                f"{path.name} must contain one of {K_KEYS}. "
                f"Found keys: {list(npz.files)}"
            )

    if K.shape != (3, 3):
        raise ValueError(f"{path.name}: K has shape {K.shape}, expected (3,3)")

    return K


def load_pose_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as npz:
        required = ("points_xyz", "points_uv")
        missing = [k for k in required if k not in npz.files]
        if missing:
            raise KeyError(
                f"{path.name} missing keys: {missing}. "
                f"Found keys: {list(npz.files)}"
            )

        points_xyz = np.asarray(npz["points_xyz"], dtype=np.float64)
        points_uv = np.asarray(npz["points_uv"], dtype=np.float64)

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"{path.name}: points_xyz shape {points_xyz.shape}, expected (N,3)")
    if points_uv.ndim != 2 or points_uv.shape[1] != 2:
        raise ValueError(f"{path.name}: points_uv shape {points_uv.shape}, expected (N,2)")
    if points_xyz.shape[0] != points_uv.shape[0]:
        raise ValueError(
            f"{path.name}: point count mismatch xyz={points_xyz.shape[0]} uv={points_uv.shape[0]}"
        )

    return points_xyz, points_uv


# ============================================================
# Board geometry
# ============================================================

def make_board_points() -> np.ndarray:
    xs_mm = np.arange(BOARD_COLS, dtype=np.float64) * BOARD_PITCH_MM
    ys_mm = np.arange(BOARD_ROWS, dtype=np.float64) * BOARD_PITCH_MM

    if CENTER_BOARD_AT_ORIGIN:
        xs_mm = xs_mm - xs_mm.mean()
        ys_mm = ys_mm - ys_mm.mean()

    XX, YY = np.meshgrid(xs_mm, ys_mm, indexing="xy")
    ZZ = np.zeros_like(XX)

    pts_mm = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    pts_m = pts_mm / 1000.0
    return pts_m


def apply_board_flip(board_points_xyz: np.ndarray, signs: tuple[float, float, float]) -> np.ndarray:
    S = np.asarray(signs, dtype=np.float64).reshape(1, 3)
    return np.asarray(board_points_xyz, dtype=np.float64) * S


# ============================================================
# Board -> camera from 3D-3D rigid fit
# ============================================================

def fit_rigid_transform_point_to_point(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> np.ndarray:
    A = np.asarray(source_xyz, dtype=np.float64)
    B = np.asarray(target_xyz, dtype=np.float64)

    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"Expected (N,3), got {A.shape}")
    if A.shape[0] < 3:
        raise ValueError("Need at least 3 points.")

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return make_transform(R, t)


# ============================================================
# Planar PnP: return both solutions
# ============================================================

def solve_board_to_xray_two_solutions(
    board_points_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    Kx: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float | None]]:
    obj = np.asarray(board_points_xyz, dtype=np.float64).reshape(-1, 3)
    uv = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)

    ok, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
        objectPoints=obj,
        imagePoints=uv,
        cameraMatrix=Kx,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not ok:
        raise RuntimeError("cv2.solvePnPGeneric(..., SOLVEPNP_IPPE) failed.")

    solutions: list[tuple[np.ndarray, np.ndarray, np.ndarray, float | None]] = []

    for i in range(len(rvecs)):
        rvec = np.asarray(rvecs[i], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvecs[i], dtype=np.float64).reshape(3, 1)
        T_bx = pose_vectors_to_board_to_xray(rvec, tvec)

        err_i = None
        if reproj is not None and len(reproj) > i:
            err_i = float(np.asarray(reproj[i]).reshape(-1)[0])

        solutions.append((T_bx, rvec, tvec, err_i))

    return solutions


# ============================================================
# Printing
# ============================================================

def print_transform(name: str, T: np.ndarray) -> None:
    T = as_transform(T, name)
    print(f"\n{name} =")
    print(T)

    if PRINT_TRANSFORMS:
        print(f"\nR({name}) =")
        print(T[:3, :3])
        print(f"\nt({name}) =")
        print(T[:3, 3:4])


def summarize_solution(
    view_name: str,
    flip_name: str,
    sol_idx: int,
    reproj_err: float | None,
    T_bc: np.ndarray,
    T_bx: np.ndarray,
    T_cx: np.ndarray,
) -> None:
    R_bc = T_bc[:3, :3]
    R_bx = T_bx[:3, :3]
    R_cx = T_cx[:3, :3]

    r22 = float(R_cx[2, 2])
    neg_flag = "YES" if r22 < 0.0 else "NO"

    print(f"{view_name:25s} | {flip_name:10s} | sol {sol_idx} | "
          f"reproj={reproj_err if reproj_err is not None else np.nan:9.6f} | "
          f"R_cx[2,2]={r22:+.6f} | negative? {neg_flag}")

    if PRINT_R_ONLY:
        print("  R_bc[2,:] =", np.array2string(R_bc[2, :], precision=6, suppress_small=True))
        print("  R_bx[2,:] =", np.array2string(R_bx[2, :], precision=6, suppress_small=True))
        print("  R_cx[2,:] =", np.array2string(R_cx[2, :], precision=6, suppress_small=True))


# ============================================================
# Main
# ============================================================

def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    print("=" * 72)
    print("SELECT FILES")
    print("=" * 72)

    pose_files = select_pose_files()
    K_file = select_K_file()

    Kx = load_K(K_file)
    board_points_base = make_board_points()

    print("\n" + "=" * 72)
    print("K")
    print("=" * 72)
    print(Kx)
    print(f"\nfx   = {Kx[0,0]:.6f}")
    print(f"fy   = {Kx[1,1]:.6f}")
    print(f"skew = {Kx[0,1]:.6f}")
    print(f"cx   = {Kx[0,2]:.6f}")
    print(f"cy   = {Kx[1,2]:.6f}")

    print("\n" + "=" * 72)
    print("BOARD FLIP TEST")
    print("=" * 72)
    print("Goal: find cases where R_cx[2,2] becomes negative.\n")

    for i, path in enumerate(pose_files, start=1):
        points_xyz_cam, points_uv = load_pose_pair(path)

        if points_xyz_cam.shape[0] != board_points_base.shape[0]:
            raise ValueError(
                f"{path.name}: got {points_xyz_cam.shape[0]} points, "
                f"expected {board_points_base.shape[0]} for {BOARD_ROWS}x{BOARD_COLS}."
            )

        print("\n" + "-" * 72)
        print(f"VIEW {i}: {path.name}")
        print("-" * 72)

        for flip_name, signs in TEST_FLIPS:
            board_points = apply_board_flip(board_points_base, signs)

            T_bc = fit_rigid_transform_point_to_point(
                source_xyz=board_points,
                target_xyz=points_xyz_cam,
            )

            solutions = solve_board_to_xray_two_solutions(
                board_points_xyz=board_points,
                image_points_uv=points_uv,
                Kx=Kx,
            )

            for k, (T_bx, rvec, tvec, reproj_err) in enumerate(solutions, start=1):
                T_cx_direct = compose_camera_to_xray(T_bc=T_bc, T_bx=T_bx)

                summarize_solution(
                    view_name=path.name,
                    flip_name=flip_name,
                    sol_idx=k,
                    reproj_err=reproj_err,
                    T_bc=T_bc,
                    T_bx=T_bx,
                    T_cx=T_cx_direct,
                )

                if PRINT_TRANSFORMS:
                    print_transform(f"T_bc_{flip_name}", T_bc)
                    print_transform(f"T_bx_{flip_name}_sol{k}", T_bx)
                    print_transform(f"T_cx_{flip_name}_sol{k}", T_cx_direct)


if __name__ == "__main__":
    main()