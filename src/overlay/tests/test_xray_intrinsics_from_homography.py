# test_xray_intrinsics_from_homographies.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.calib.calib_xray_intrinsics import estimate_intrinsics_from_homographies


# ============================================================
# Config
# ============================================================

ENFORCE_ZERO_SKEW = False


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
        str(default_dir / "xray_intrinsics_Kx.npz"),
        "NPZ files (*.npz);;All Files (*)",
        options=QFileDialog.DontUseNativeDialog,
    )
    if not path:
        raise RuntimeError("No output file selected.")
    return Path(path)


# ============================================================
# Helpers
# ============================================================

def load_homography_from_npz(npz_path: Path) -> np.ndarray:
    """
    Load one homography from an .npz file.
    Supported keys:
    - "H"
    - "homography"
    - "H_x"
    - otherwise: first (3,3) array found
    """
    data = np.load(npz_path)
    preferred_keys = ["H", "homography", "H_x"]
    for key in preferred_keys:
        if key in data:
            H = np.asarray(data[key], dtype=np.float64)
            if H.shape != (3, 3):
                raise ValueError(f"{npz_path.name}: key '{key}' has shape {H.shape}, expected (3,3)")
            return H
    # fallback: first 3x3 array
    for key in data.files:
        arr = np.asarray(data[key], dtype=np.float64)
        if arr.shape == (3, 3):
            print(f"[INFO] {npz_path.name}: using fallback key '{key}'")
            return arr
    raise KeyError(
        f"{npz_path.name}: no homography found. "
        f"Expected one of {preferred_keys} or any array with shape (3,3)."
    )


def load_all_homographies(folder: Path) -> tuple[list[np.ndarray], list[Path]]:
    """
    Load all .npz homographies from a folder.
    """
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {folder}")

    H_list: list[np.ndarray] = []
    used_files: list[Path] = []

    for f in files:
        try:
            H = load_homography_from_npz(f)
            H_list.append(H)
            used_files.append(f)
            print(f"[OK] loaded {f.name}")
        except Exception as e:
            print(f"[SKIP] {f.name}: {e}")

    if len(H_list) < 3:
        raise RuntimeError(
            f"Only {len(H_list)} valid homographies found. "
            f"At least 3 are required."
        )

    return H_list, used_files


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Select folder containing homography .npz files...")
    H_folder = _select_folder("Select folder with homography NPZ files")
    print(f"Loading homographies from:\n  {H_folder}")

    H_list, used_files = load_all_homographies(H_folder)

    print(f"\nUsing {len(H_list)} homographies:")
    for f in used_files:
        print(f"  - {f.name}")

    result = estimate_intrinsics_from_homographies(
        H_list,
        enforce_zero_skew=ENFORCE_ZERO_SKEW,
    )

    Kx = result.K

    print("\nEstimated Kx:")
    print(Kx)
    print(f"\n  fx={Kx[0,0]:.2f}  fy={Kx[1,1]:.2f}  cx={Kx[0,2]:.2f}  cy={Kx[1,2]:.2f}")

    print("\nSelect output file location...")
    out_file = _select_save_file("Save Kx as NPZ", H_folder)

    np.savez(out_file, K=Kx)
    print(f"\nSaved Kx to:\n  {out_file}")


if __name__ == "__main__":
    main()