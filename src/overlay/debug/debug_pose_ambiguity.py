from __future__ import annotations

from PySide6.QtWidgets import QApplication, QFileDialog

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[2]   # .../src

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np

from tracking.pose_solvers import solve_pose


# ============================================================
# Config
# ============================================================

K_KEYS = ("Kx", "K_xray", "K")

POSE_METHOD = "iterative_ransac"
DIST_COEFFS = None

REFINE_WITH_ITERATIVE = True
RANSAC_REPROJECTION_ERROR_PX = 1.5
RANSAC_CONFIDENCE = 0.99
RANSAC_ITERATIONS_COUNT = 5000

PRINT_ROTATION_MATRIX = True


# ============================================================
# File dialogs
# ============================================================

def select_pose_files() -> list[Path]:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dlg = QFileDialog()
    dlg.setWindowTitle("Select pose_debug files")
    dlg.setFileMode(QFileDialog.ExistingFiles)
    dlg.setNameFilter("NumPy files (*.npz)")
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)

    if dlg.exec() != QFileDialog.Accepted:
        raise RuntimeError("No pose files selected.")

    files = dlg.selectedFiles()
    paths = [Path(f) for f in files]

    if len(paths) < 1:
        raise ValueError("Select at least 1 pose file.")

    return paths


def select_K_file() -> Path:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dlg = QFileDialog()
    dlg.setWindowTitle("Select Kx file")
    dlg.setFileMode(QFileDialog.ExistingFile)
    dlg.setNameFilter("NumPy files (*.npz)")
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)

    if dlg.exec() != QFileDialog.Accepted:
        raise RuntimeError("No K file selected.")

    files = dlg.selectedFiles()
    return Path(files[0])


# ============================================================
# Load helpers
# ============================================================

def load_K(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"K file not found: {path}")

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
    if not path.exists():
        raise FileNotFoundError(f"Pose file not found: {path}")

    with np.load(str(path), allow_pickle=False) as npz:
        required = ("points_xyz", "points_uv")
        missing = [k for k in required if k not in npz.files]
        if missing:
            raise KeyError(
                f"{path.name} missing keys: {missing}. "
                f"Found keys: {list(npz.files)}"
            )

        xyz = np.asarray(npz["points_xyz"], dtype=np.float64)
        uv = np.asarray(npz["points_uv"], dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"{path.name}: points_xyz has shape {xyz.shape}, expected (N,3)")
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"{path.name}: points_uv has shape {uv.shape}, expected (N,2)")
    if xyz.shape[0] != uv.shape[0]:
        raise ValueError(
            f"{path.name}: point count mismatch "
            f"(points_xyz={xyz.shape[0]}, points_uv={uv.shape[0]})"
        )

    return xyz, uv


def concat_pose_pairs(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz_list: list[np.ndarray] = []
    uv_list: list[np.ndarray] = []
    sizes: list[int] = []

    for path in paths:
        xyz, uv = load_pose_pair(path)
        xyz_list.append(xyz)
        uv_list.append(uv)
        sizes.append(int(xyz.shape[0]))

    xyz_all = np.vstack(xyz_list)
    uv_all = np.vstack(uv_list)
    pose_sizes = np.asarray(sizes, dtype=np.int32)

    return xyz_all, uv_all, pose_sizes


# ============================================================
# Geometry helpers
# ============================================================

def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ t)
    return T_inv


def fit_plane_svd(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    center = P.mean(axis=0)

    X = P - center
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    normal = Vt[-1, :]
    normal = normal / np.linalg.norm(normal)

    return center, normal


def nonplanarity_report(points_xyz: np.ndarray) -> tuple[np.ndarray, float]:
    P = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    center = P.mean(axis=0)
    X = P - center
    _, S, _ = np.linalg.svd(X, full_matrices=False)

    if len(S) < 3:
        ratio = 0.0
    else:
        ratio = float(S[-1] / S[0]) if S[0] > 0 else 0.0

    return S, ratio


# ============================================================
# Printing
# ============================================================

def print_pose_result(
    *,
    result_name: str,
    rvec: np.ndarray,
    tvec: np.ndarray,
    reproj_mean_px: float,
    reproj_median_px: float,
    reproj_max_px: float,
    inlier_count: int,
    total_count: int,
) -> None:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    R = rodrigues_to_R(rvec)
    T_cx = make_transform(R, tvec)
    T_xc = invert_transform(T_cx)

    print("\n" + "=" * 72)
    print(result_name)
    print("=" * 72)

    print("rvec =")
    print(rvec)

    print("\ntvec =")
    print(tvec)

    if PRINT_ROTATION_MATRIX:
        print("\nR_cx =")
        print(R)

    print("\nT_cx =")
    print(T_cx)

    print("\nT_xc =")
    print(T_xc)

    print(f"\ninliers        = {inlier_count} / {total_count}")
    print(f"reproj mean    = {reproj_mean_px:.6f} px")
    print(f"reproj median  = {reproj_median_px:.6f} px")
    print(f"reproj max     = {reproj_max_px:.6f} px")


# ============================================================
# Main
# ============================================================

def main() -> None:
    np.set_printoptions(precision=8, suppress=True)

    print("=" * 72)
    print("SELECT INPUT FILES")
    print("=" * 72)

    pose_files = select_pose_files()
    K_file = select_K_file()

    print("\nSelected pose files:")
    for p in pose_files:
        print("  ", p)

    print("\nSelected K file:")
    print("  ", K_file)

    Kx = load_K(K_file)
    xyz_all, uv_all, pose_sizes = concat_pose_pairs(pose_files)

    print("\n" + "=" * 72)
    print("LOAD INPUT")
    print("=" * 72)
    print(f"K file         : {K_file}")
    print(f"K shape        : {Kx.shape}")
    print(f"# pose files    : {len(pose_files)}")
    print(f"pose sizes      : {pose_sizes.tolist()}")
    print(f"xyz_all shape   : {xyz_all.shape}")
    print(f"uv_all shape    : {uv_all.shape}")

    S, ratio = nonplanarity_report(xyz_all)
    print(f"\nSVD singular values of combined 3D set: {S}")
    print(f"non-planarity ratio S_min / S_max     : {ratio:.8f}")

    for i, path in enumerate(pose_files, start=1):
        xyz_i, uv_i = load_pose_pair(path)
        center_i, normal_i = fit_plane_svd(xyz_i)

        print("\n" + "-" * 72)
        print(f"Pose {i}: {path.name}")
        print(f"  xyz shape   : {xyz_i.shape}")
        print(f"  uv shape    : {uv_i.shape}")
        print(f"  center_xyz  : {center_i}")
        print(f"  normal_xyz  : {normal_i}")

    print("\n" + "=" * 72)
    print("SOLVE PER-POSE PNP")
    print("=" * 72)

    for i, path in enumerate(pose_files, start=1):
        xyz_i, uv_i = load_pose_pair(path)

        result_i = solve_pose(
            object_points_xyz=xyz_i,
            image_points_uv=uv_i,
            K=Kx,
            dist_coeffs=DIST_COEFFS,
            pose_method=POSE_METHOD,
            refine_with_iterative=REFINE_WITH_ITERATIVE,
            ransac_reprojection_error_px=RANSAC_REPROJECTION_ERROR_PX,
            ransac_confidence=RANSAC_CONFIDENCE,
            ransac_iterations_count=RANSAC_ITERATIONS_COUNT,
        )

        print_pose_result(
            result_name=f"POSE {i}: {path.name}",
            rvec=result_i.rvec,
            tvec=result_i.tvec,
            reproj_mean_px=result_i.reproj_mean_px,
            reproj_median_px=result_i.reproj_median_px,
            reproj_max_px=result_i.reproj_max_px,
            inlier_count=int(len(result_i.inlier_idx)),
            total_count=int(xyz_i.shape[0]),
        )

    if len(pose_files) >= 2:
        print("\n" + "=" * 72)
        print("SOLVE JOINT PNP")
        print("=" * 72)

        result = solve_pose(
            object_points_xyz=xyz_all,
            image_points_uv=uv_all,
            K=Kx,
            dist_coeffs=DIST_COEFFS,
            pose_method=POSE_METHOD,
            refine_with_iterative=REFINE_WITH_ITERATIVE,
            ransac_reprojection_error_px=RANSAC_REPROJECTION_ERROR_PX,
            ransac_confidence=RANSAC_CONFIDENCE,
            ransac_iterations_count=RANSAC_ITERATIONS_COUNT,
        )

        print_pose_result(
            result_name=f"JOINT {POSE_METHOD.upper()}",
            rvec=result.rvec,
            tvec=result.tvec,
            reproj_mean_px=result.reproj_mean_px,
            reproj_median_px=result.reproj_median_px,
            reproj_max_px=result.reproj_max_px,
            inlier_count=int(len(result.inlier_idx)),
            total_count=int(xyz_all.shape[0]),
        )


if __name__ == "__main__":
    main()