from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.calib.calib_xray_intrinsics import estimate_intrinsics_from_homographies


# ============================================================
# Config
# ============================================================

ENFORCE_ZERO_SKEW = True
IMAGE_SIZE = (1024, 1024)   # (width, height)


# ============================================================
# Qt helpers
# ============================================================

def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _select_folder(title: str) -> Path:
    _get_app()
    folder = QFileDialog.getExistingDirectory(
        None,
        title,
        "",
        QFileDialog.DontUseNativeDialog,
    )
    if not folder:
        raise RuntimeError("No folder selected.")
    return Path(folder)


def _select_save_file(title: str, default_dir: Path) -> Path:
    _get_app()
    path, _ = QFileDialog.getSaveFileName(
        None,
        title,
        str(default_dir / "xray_intrinsics_init_refinement_fixed_pp.npz"),
        "NPZ files (*.npz);;All Files (*)",
        options=QFileDialog.DontUseNativeDialog,
    )
    if not path:
        raise RuntimeError("No output file selected.")
    return Path(path)


# ============================================================
# Load helpers
# ============================================================

def _extract_view_number_from_name(name: str) -> int:
    m = re.search(r"VIEW(\d+)", name)
    if m is None:
        raise ValueError(f"Could not extract VIEW number from '{name}'")
    return int(m.group(1))


def load_homography_from_npz(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    preferred_keys = ["H", "homography", "H_x"]

    for key in preferred_keys:
        if key in data:
            H = np.asarray(data[key], dtype=np.float64)
            if H.shape != (3, 3):
                raise ValueError(
                    f"{npz_path.name}: key '{key}' has shape {H.shape}, expected (3,3)"
                )
            return H

    for key in data.files:
        arr = np.asarray(data[key], dtype=np.float64)
        if arr.shape == (3, 3):
            print(f"[INFO] {npz_path.name}: using fallback key '{key}'")
            return arr

    raise KeyError(
        f"{npz_path.name}: no homography found. "
        f"Expected one of {preferred_keys} or any array with shape (3,3)."
    )


def load_corr_from_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)

    if "XY" not in data:
        raise KeyError(f"{npz_path.name}: missing key 'XY'")
    if "uv" not in data:
        raise KeyError(f"{npz_path.name}: missing key 'uv'")

    XY = np.asarray(data["XY"], dtype=np.float64)
    uv = np.asarray(data["uv"], dtype=np.float64)

    if XY.ndim != 2 or XY.shape[1] != 2:
        raise ValueError(f"{npz_path.name}: XY must have shape (N,2), got {XY.shape}")
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"{npz_path.name}: uv must have shape (N,2), got {uv.shape}")
    if len(XY) != len(uv):
        raise ValueError(
            f"{npz_path.name}: XY/uv count mismatch: {len(XY)} vs {len(uv)}"
        )

    object_points = np.column_stack([
        XY,
        np.zeros((XY.shape[0], 1), dtype=np.float64),
    ])

    return object_points, uv


def _stem_from_h_file(path: Path) -> str:
    name = path.name
    if name.endswith("_H.npz"):
        return name[:-len("_H.npz")]
    return path.stem


def _stem_from_corr_file(path: Path) -> str:
    name = path.name
    if name.endswith("_corr.npz"):
        return name[:-len("_corr.npz")]
    return path.stem


def find_matching_pairs(H_folder: Path, corr_folder: Path) -> list[tuple[str, Path, Path]]:
    h_files = sorted(
        H_folder.glob("*_H.npz"),
        key=lambda p: _extract_view_number_from_name(p.name),
    )
    if not h_files:
        raise FileNotFoundError(f"No '*_H.npz' files found in H folder: {H_folder}")

    corr_files = sorted(
        corr_folder.glob("*_corr.npz"),
        key=lambda p: _extract_view_number_from_name(p.name),
    )
    if not corr_files:
        raise FileNotFoundError(f"No '*_corr.npz' files found in corr folder: {corr_folder}")

    corr_map: dict[str, Path] = {}
    for corr_path in corr_files:
        stem = _stem_from_corr_file(corr_path)
        corr_map[stem] = corr_path

    print("\nH stems (numerically sorted):")
    for h_path in h_files:
        print(f"  {h_path.name} -> {_stem_from_h_file(h_path)}")

    print("\ncorr stems (numerically sorted):")
    for corr_path in corr_files:
        print(f"  {corr_path.name} -> {_stem_from_corr_file(corr_path)}")

    pairs: list[tuple[str, Path, Path]] = []

    for h_path in h_files:
        stem = _stem_from_h_file(h_path)
        corr_path = corr_map.get(stem, None)
        if corr_path is None:
            print(f"[SKIP] {h_path.name}: no matching corr file found for stem '{stem}'")
            continue
        pairs.append((stem, h_path, corr_path))

    if not pairs:
        raise RuntimeError("No matching H/corr pairs found.")

    return pairs


def load_all_views(
    H_folder: Path,
    corr_folder: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    pairs = find_matching_pairs(H_folder, corr_folder)

    H_list: list[np.ndarray] = []
    object_points_per_view: list[np.ndarray] = []
    image_points_per_view: list[np.ndarray] = []
    used_names: list[str] = []

    for stem, h_path, corr_path in pairs:
        try:
            H = load_homography_from_npz(h_path)
            obj_pts, img_pts = load_corr_from_npz(corr_path)

            H_list.append(H)
            object_points_per_view.append(obj_pts)
            image_points_per_view.append(img_pts)
            used_names.append(stem)

            print(f"[OK] matched {h_path.name}  <->  {corr_path.name}")
        except Exception as e:
            print(f"[SKIP] {stem}: {e}")

    min_views = 2 if ENFORCE_ZERO_SKEW else 3
    if len(H_list) < min_views:
        raise RuntimeError(
            f"Only {len(H_list)} valid matched views found. "
            f"At least {min_views} are required."
        )

    return H_list, object_points_per_view, image_points_per_view, used_names


# ============================================================
# Print helpers
# ============================================================

def print_intrinsics(
    label: str,
    K: np.ndarray,
    rms: float | None = None,
    dist: np.ndarray | None = None,
) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(K)
    print(f"fx = {K[0,0]:.6f}")
    print(f"fy = {K[1,1]:.6f}")
    print(f"cx = {K[0,2]:.6f}")
    print(f"cy = {K[1,2]:.6f}")
    if rms is not None:
        print(f"RMS reprojection error = {rms:.6f}")
    if dist is not None and dist.size > 0:
        flat = np.asarray(dist, dtype=np.float64).reshape(-1)
        print(f"dist coeffs = {flat}")


def print_comparison(label_a: str, K_a: np.ndarray, label_b: str, K_b: np.ndarray) -> None:
    dK = K_b - K_a

    print(f"\nComparison: {label_b} - {label_a}")
    print("-" * (13 + len(label_b) + len(label_a)))
    print(dK)

    print("\nAbsolute deltas")
    print(f"Δfx = {dK[0,0]:+.6f}")
    print(f"Δfy = {dK[1,1]:+.6f}")
    print(f"Δcx = {dK[0,2]:+.6f}")
    print(f"Δcy = {dK[1,2]:+.6f}")

    print("\nRelative deltas wrt first")
    print(f"Δfx/fx = {dK[0,0] / K_a[0,0] * 100:+.6f} %")
    print(f"Δfy/fy = {dK[1,1] / K_a[1,1] * 100:+.6f} %")
    print(f"Δcx/cx = {dK[0,2] / K_a[0,2] * 100:+.6f} %")
    print(f"Δcy/cy = {dK[1,2] / K_a[1,2] * 100:+.6f} %")


# ============================================================
# Save helpers
# ============================================================

def save_individual_intrinsics_npzs(
    out_file: Path,
    K_zhang: np.ndarray,
    K_refined: np.ndarray,
    K_refined_fixed_pp: np.ndarray,
) -> None:
    base_dir = out_file.parent
    stem = out_file.stem

    out_zhang = base_dir / f"{stem}_K_zhang.npz"
    out_refined = base_dir / f"{stem}_K_refined.npz"
    out_refined_fixed_pp = base_dir / f"{stem}_K_refined_fixed_pp.npz"

    np.savez(out_zhang, K_xray=np.asarray(K_zhang, dtype=np.float64))
    np.savez(out_refined, K_xray=np.asarray(K_refined, dtype=np.float64))
    np.savez(out_refined_fixed_pp, K_xray=np.asarray(K_refined_fixed_pp, dtype=np.float64))

    print("\nSaved individual K files:")
    print(f"  {out_zhang}")
    print(f"  {out_refined}")
    print(f"  {out_refined_fixed_pp}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Select folder containing H files...")
    H_folder = _select_folder("Select folder with H NPZ files")

    print("Select folder containing corr files...")
    corr_folder = _select_folder("Select folder with corr NPZ files")

    print(f"\nLoading H files from:\n  {H_folder}")
    print(f"Loading corr files from:\n  {corr_folder}")

    H_list, object_points_per_view, image_points_per_view, used_names = load_all_views(
        H_folder,
        corr_folder,
    )

    print(f"\nUsing {len(H_list)} matched view pairs:")
    for i, name in enumerate(used_names, start=1):
        print(f"  {i:2d}: {name}")

    # --------------------------------------------------------
    # 1) Zhang only
    # --------------------------------------------------------
    result_zhang = estimate_intrinsics_from_homographies(
        H_list,
        enforce_zero_skew=ENFORCE_ZERO_SKEW,
        global_optimization=False,
    )
    K_zhang = result_zhang.K
    print_intrinsics("1) Zhang only", K_zhang)

    # --------------------------------------------------------
    # 2) Zhang + refinement (no radial, free PP)
    # --------------------------------------------------------
    result_refined = estimate_intrinsics_from_homographies(
        H_list,
        enforce_zero_skew=ENFORCE_ZERO_SKEW,
        global_optimization=True,
        image_size=IMAGE_SIZE,
        object_points_per_view=object_points_per_view,
        image_points_per_view=image_points_per_view,
        radial_model="none",
        fix_principal_point=False,
        principal_point_mode="init",
    )
    K_refined = result_refined.K
    print_intrinsics(
        "2) Zhang + global refinement (no radial, free PP)",
        K_refined,
        rms=result_refined.rms_reproj_error,
        dist=result_refined.dist_coeffs,
    )

    # --------------------------------------------------------
    # 3) Zhang + refinement (no radial, fixed PP = Zhang init)
    # --------------------------------------------------------
    result_refined_fixed_pp = estimate_intrinsics_from_homographies(
        H_list,
        enforce_zero_skew=ENFORCE_ZERO_SKEW,
        global_optimization=True,
        image_size=IMAGE_SIZE,
        object_points_per_view=object_points_per_view,
        image_points_per_view=image_points_per_view,
        radial_model="none",
        fix_principal_point=True,
        principal_point_mode="init",
    )
    K_refined_fixed_pp = result_refined_fixed_pp.K
    print_intrinsics(
        "3) Zhang + global refinement (no radial, fixed PP = Zhang)",
        K_refined_fixed_pp,
        rms=result_refined_fixed_pp.rms_reproj_error,
        dist=result_refined_fixed_pp.dist_coeffs,
    )

    # --------------------------------------------------------
    # Comparisons
    # --------------------------------------------------------
    print_comparison("Zhang", K_zhang, "Refined (free PP)", K_refined)
    print_comparison("Zhang", K_zhang, "Refined (fixed PP)", K_refined_fixed_pp)
    print_comparison("Refined (free PP)", K_refined, "Refined (fixed PP)", K_refined_fixed_pp)

    # --------------------------------------------------------
    # Save combined results
    # --------------------------------------------------------
    print("\nSelect output file location...")
    out_file = _select_save_file("Save intrinsics results as NPZ", H_folder)

    np.savez(
        out_file,
        K_zhang=K_zhang,
        K_refined=K_refined,
        K_refined_fixed_pp=K_refined_fixed_pp,
        rms_refined=np.nan if result_refined.rms_reproj_error is None else result_refined.rms_reproj_error,
        rms_refined_fixed_pp=np.nan if result_refined_fixed_pp.rms_reproj_error is None else result_refined_fixed_pp.rms_reproj_error,
        dist_coeffs=np.array([]) if result_refined.dist_coeffs is None else result_refined.dist_coeffs,
        dist_coeffs_fixed_pp=np.array([]) if result_refined_fixed_pp.dist_coeffs is None else result_refined_fixed_pp.dist_coeffs,
        used_view_names=np.array(used_names, dtype=object),
        enforce_zero_skew=np.array(ENFORCE_ZERO_SKEW),
        image_size=np.array(IMAGE_SIZE, dtype=np.int32),
        H_folder=np.array(str(H_folder)),
        corr_folder=np.array(str(corr_folder)),
    )

    print(f"\nSaved combined results to:\n  {out_file}")

    # --------------------------------------------------------
    # Save separate K files with unified key: K_xray
    # --------------------------------------------------------
    save_individual_intrinsics_npzs(
        out_file=out_file,
        K_zhang=K_zhang,
        K_refined=K_refined,
        K_refined_fixed_pp=K_refined_fixed_pp,
    )


if __name__ == "__main__":
    main()