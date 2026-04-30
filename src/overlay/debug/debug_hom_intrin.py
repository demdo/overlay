from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.calib.calib_xray_intrinsics import estimate_intrinsics_from_homographies


# ============================================================
# Config
# ============================================================

ENFORCE_ZERO_SKEW = True
IMAGE_SIZE = (1024, 1024)   # (width, height)

LASER_CROSS_UV = (473.0, 424.0)

INITIAL_FOCAL_LENGTH_PX = 4650.0

PRINT_H_MATRICES = False
PRINT_POINT_SAMPLES = False

EXCLUDE_VIEWS = set()


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


def _save_intrinsics_dialog(default_name: str) -> Path | None:
    _get_app()
    path, _ = QFileDialog.getSaveFileName(
        None,
        "Save X-ray intrinsics NPZ",
        default_name,
        "NPZ files (*.npz);;All files (*.*)",
    )
    if not path:
        return None
    p = Path(path)
    if p.suffix.lower() != ".npz":
        p = p.with_suffix(".npz")
    return p


# ============================================================
# Load helpers
# ============================================================

def _extract_view_number_from_name(name: str) -> int:
    m = re.search(r"VIEW(\d+)", name)
    if m is None:
        raise ValueError(f"Could not extract VIEW number from '{name}'")
    return int(m.group(1))


def _stem_from_h_file(path: Path) -> str:
    name = path.name
    if name.endswith("__H_XRAY_UV_TRANSFORM.npz"):
        return name[:-len("__H_XRAY_UV_TRANSFORM.npz")]
    if name.endswith("_H.npz"):
        return name[:-len("_H.npz")]
    return path.stem


def _stem_from_corr_file(path: Path) -> str:
    name = path.name
    if name.endswith("__corr_XRAY_UV_TRANSFORM.npz"):
        return name[:-len("__corr_XRAY_UV_TRANSFORM.npz")]
    if name.endswith("_corr.npz"):
        return name[:-len("_corr.npz")]
    return path.stem


def load_homography_from_npz(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    for key in ["H", "homography", "H_x"]:
        if key in data:
            H = np.asarray(data[key], dtype=np.float64)
            if H.shape != (3, 3):
                raise ValueError(f"{npz_path.name}: key '{key}' has shape {H.shape}")
            return H

    for key in data.files:
        arr = np.asarray(data[key], dtype=np.float64)
        if arr.shape == (3, 3):
            print(f"[INFO] {npz_path.name}: using fallback key '{key}'")
            return arr

    raise KeyError(f"{npz_path.name}: no homography found.")


def load_corr_from_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        raise ValueError(f"{npz_path.name}: XY/uv mismatch: {len(XY)} vs {len(uv)}")

    object_points_xyz = np.column_stack(
        [XY, np.zeros((XY.shape[0], 1), dtype=np.float64)]
    )

    return XY, uv, object_points_xyz


def find_matching_pairs(H_folder: Path, corr_folder: Path) -> list[tuple[str, Path, Path]]:
    h_files = sorted(
        list(H_folder.glob("*_H.npz")) + list(H_folder.glob("*__H_XRAY_UV_TRANSFORM.npz")),
        key=lambda p: _extract_view_number_from_name(p.name),
    )
    if not h_files:
        raise FileNotFoundError(f"No H files found in H folder: {H_folder}")

    corr_files = sorted(
        list(corr_folder.glob("*_corr.npz")) + list(corr_folder.glob("*__corr_XRAY_UV_TRANSFORM.npz")),
        key=lambda p: _extract_view_number_from_name(p.name),
    )
    if not corr_files:
        raise FileNotFoundError(f"No corr files found in corr folder: {corr_folder}")

    corr_map = {_stem_from_corr_file(p): p for p in corr_files}

    pairs = []
    for h_path in h_files:
        stem = _stem_from_h_file(h_path)
        corr_path = corr_map.get(stem)
        if corr_path is None:
            print(f"[SKIP] {h_path.name}: no matching corr file for stem '{stem}'")
            continue
        pairs.append((stem, h_path, corr_path))

    if not pairs:
        raise RuntimeError("No matching H/corr pairs found.")

    return pairs


# ============================================================
# Homography helpers
# ============================================================

def normalize_H(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > 1e-12:
        return H / H[2, 2]
    return H / (np.linalg.norm(H) + 1e-12)


def apply_homography(H: np.ndarray, XY: np.ndarray) -> np.ndarray:
    XY = np.asarray(XY, dtype=np.float64).reshape(-1, 2)
    Xh = np.hstack([XY, np.ones((XY.shape[0], 1), dtype=np.float64)])
    xh = (H @ Xh.T).T
    return xh[:, :2] / xh[:, 2:3]


def compute_homography_reprojection_stats(
    H: np.ndarray,
    XY: np.ndarray,
    uv_meas: np.ndarray,
) -> dict:
    uv_pred = apply_homography(H, XY)
    err = np.linalg.norm(uv_meas - uv_pred, axis=1)
    return {
        "uv_pred": uv_pred,
        "errors_px": err,
        "mean_px": float(np.mean(err)),
        "median_px": float(np.median(err)),
        "max_px": float(np.max(err)),
        "p95_px": float(np.percentile(err, 95)),
    }


def compute_polygon_area(points_uv: np.ndarray) -> float:
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    if hull.shape[0] < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def principal_point_valid(K: np.ndarray, image_size: tuple[int, int]) -> bool:
    w, h = image_size
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return (0.0 <= cx < float(w)) and (0.0 <= cy < float(h))


# ============================================================
# View analysis
# ============================================================

def analyze_single_view(stem: str, H_path: Path, corr_path: Path) -> dict:
    H = load_homography_from_npz(H_path)
    XY, uv, obj_xyz = load_corr_from_npz(corr_path)

    Hn = normalize_H(H)
    reproj = compute_homography_reprojection_stats(Hn, XY, uv)

    return {
        "stem": stem,
        "H_path": H_path,
        "corr_path": corr_path,
        "H": H,
        "Hn": Hn,
        "XY": XY,
        "uv": uv,
        "obj_xyz": obj_xyz,
        "n_points": int(len(XY)),
        "cond_H": float(np.linalg.cond(Hn)),
        "uv_area": compute_polygon_area(uv),
        "xy_area": compute_polygon_area(XY),
        **reproj,
    }


def print_view_summary(v: dict) -> None:
    print(f"\n=== {v['stem']} ===")
    print(f"n_points          = {v['n_points']}")
    print(f"cond(H)           = {v['cond_H']:.6e}")
    print(f"XY hull area      = {v['xy_area']:.6f}")
    print(f"uv hull area      = {v['uv_area']:.6f}")
    print(f"H reproj mean px  = {v['mean_px']:.6f}")
    print(f"H reproj median   = {v['median_px']:.6f}")
    print(f"H reproj p95 px   = {v['p95_px']:.6f}")
    print(f"H reproj max px   = {v['max_px']:.6f}")


# ============================================================
# Zhang analysis
# ============================================================

def print_intrinsics_summary(label: str, K: np.ndarray, image_size: tuple[int, int]) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(K)
    print(f"fx = {K[0,0]:.6f}")
    print(f"fy = {K[1,1]:.6f}")
    print(f"cx = {K[0,2]:.6f}")
    print(f"cy = {K[1,2]:.6f}")
    print(f"principal point valid = {principal_point_valid(K, image_size)}")


def run_zhang_analysis(
    views: list[dict],
    image_size: tuple[int, int],
    enforce_zero_skew: bool = True,
) -> np.ndarray | None:
    H_list = [v["H"] for v in views]

    try:
        result = estimate_intrinsics_from_homographies(
            H_list,
            enforce_zero_skew=enforce_zero_skew,
            global_optimization=False,
        )
        K = result.K
        print_intrinsics_summary("Zhang (all views)", K, image_size)
        return K
    except Exception as e:
        print("\n[FAIL] Zhang (all views)")
        print(f"Reason: {e}")
        return None


def run_subset_rankings(views: list[dict]) -> None:
    print("\n============================================================")
    print("PER-VIEW RANKINGS")
    print("============================================================")

    by_mean = sorted(views, key=lambda v: v["mean_px"], reverse=True)
    by_max = sorted(views, key=lambda v: v["max_px"], reverse=True)
    by_cond = sorted(views, key=lambda v: v["cond_H"], reverse=True)

    print("\nWorst views by H reprojection mean:")
    for v in by_mean[:10]:
        print(f"  {v['stem']}: mean={v['mean_px']:.6f}, max={v['max_px']:.6f}")

    print("\nWorst views by H reprojection max:")
    for v in by_max[:10]:
        print(f"  {v['stem']}: max={v['max_px']:.6f}, mean={v['mean_px']:.6f}")

    print("\nWorst views by cond(H):")
    for v in by_cond[:10]:
        print(f"  {v['stem']}: cond(H)={v['cond_H']:.6e}")


# ============================================================
# BA-only center PP, optimize equal focal length
# ============================================================

def _as_opencv_object_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"object points must have shape (N,3), got {pts.shape}")
    return pts.astype(np.float32)


def _as_opencv_image_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"image points must have shape (N,2), got {pts.shape}")
    return pts.astype(np.float32)


def save_intrinsics_npz(
    out_path: Path,
    *,
    K: np.ndarray,
    dist: np.ndarray,
    rms: float,
    image_size: tuple[int, int],
    initial_f_px: float,
) -> None:
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)

    np.savez(
        out_path,
        # same compatibility keys as before
        K=K,
        K_x=K,
        K_xray=K,

        dist=dist,
        dist_x=dist,
        dist_xray=dist,

        # useful metadata
        rms=np.array(float(rms), dtype=np.float64),
        image_size=np.asarray(image_size, dtype=np.int32),
        width=np.array(image_size[0], dtype=np.int32),
        height=np.array(image_size[1], dtype=np.int32),

        method=np.array("BA_CENTER_PP_EQUAL_F", dtype="<U64"),
        uv_space=np.array("XRAY_WORKING_FLIPPED_UV", dtype="<U64"),
        uv_transform=np.array("horizontal_flip", dtype="<U32"),

        initial_f_px=np.array(float(initial_f_px), dtype=np.float64),
        fixed_principal_point=np.array(True, dtype=bool),
        cx_fixed=np.array(float(image_size[0]) / 2.0, dtype=np.float64),
        cy_fixed=np.array(float(image_size[1]) / 2.0, dtype=np.float64),
        equal_focal_lengths=np.array(True, dtype=bool),
        distortion_fixed_zero=np.array(True, dtype=bool),
    )

    print(f"\n[OK] saved intrinsics -> {out_path}")


def run_ba_center_pp_equal_f(
    views: list[dict],
    image_size: tuple[int, int],
    *,
    initial_f_px: float = 4650.0,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    print("\n============================================================")
    print("BA-ONLY: FIXED CENTER PP, OPTIMIZE EQUAL FOCAL LENGTH")
    print("============================================================")
    print(f"initial fx = fy = {initial_f_px:.6f}")
    print("principal point fixed at image center")
    print("fx = fy enforced via CALIB_FIX_ASPECT_RATIO")
    print("distortion fixed to zero")

    object_points_cv = [_as_opencv_object_points(v["obj_xyz"]) for v in views]
    image_points_cv = [_as_opencv_image_points(v["uv"]) for v in views]

    w, h = image_size
    cx_init = w / 2.0
    cy_init = h / 2.0

    K_init = np.array(
        [
            [initial_f_px, 0.0, cx_init],
            [0.0, initial_f_px, cy_init],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    dist_init = np.zeros((8, 1), dtype=np.float64)

    print("\nInitial K =")
    print(K_init)

    flags = 0
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    flags |= cv2.CALIB_FIX_ASPECT_RATIO

    flags |= cv2.CALIB_ZERO_TANGENT_DIST
    flags |= cv2.CALIB_FIX_K1
    flags |= cv2.CALIB_FIX_K2
    flags |= cv2.CALIB_FIX_K3
    flags |= cv2.CALIB_FIX_K4
    flags |= cv2.CALIB_FIX_K5
    flags |= cv2.CALIB_FIX_K6
    flags |= cv2.CALIB_FIX_S1_S2_S3_S4
    flags |= cv2.CALIB_FIX_TAUX_TAUY

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
        500,
        1e-12,
    )

    try:
        rms, K_ba, dist_ba, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=object_points_cv,
            imagePoints=image_points_cv,
            imageSize=image_size,
            cameraMatrix=K_init,
            distCoeffs=dist_init,
            flags=flags,
            criteria=criteria,
        )
    except Exception as e:
        print("\n[FAIL] BA center-PP equal-f calibration failed")
        print(f"Reason: {e}")
        return None

    print("\nBA center-PP equal-f result")
    print("---------------------------")
    print(K_ba)
    print(f"fx = {K_ba[0,0]:.6f}")
    print(f"fy = {K_ba[1,1]:.6f}")
    print(f"fx - fy = {K_ba[0,0] - K_ba[1,1]:+.12f}")
    print(f"cx = {K_ba[0,2]:.6f}")
    print(f"cy = {K_ba[1,2]:.6f}")
    print(f"principal point valid = {principal_point_valid(K_ba, image_size)}")
    print(f"RMS reprojection error = {float(rms):.6f}")
    print(f"dist coeffs = {np.asarray(dist_ba, dtype=np.float64).reshape(-1)}")

    print("\nOptimized focal change:")
    print(f"df = {K_ba[0,0] - initial_f_px:+.6f} px")

    default_name = "K_xray_ba_center_pp_equal_f.npz"
    out_path = _save_intrinsics_dialog(default_name)

    if out_path is None:
        print("[INFO] Save cancelled.")
    else:
        save_intrinsics_npz(
            out_path,
            K=K_ba,
            dist=dist_ba,
            rms=float(rms),
            image_size=image_size,
            initial_f_px=initial_f_px,
        )

    return K_ba, dist_ba, float(rms)


# ============================================================
# Pair filtering
# ============================================================

def filter_pairs(
    pairs: list[tuple[str, Path, Path]],
    exclude_views: set[str],
) -> list[tuple[str, Path, Path]]:
    if not exclude_views:
        return pairs

    filtered = []
    removed = []

    for stem, h_path, corr_path in pairs:
        if any(view_name in stem for view_name in exclude_views):
            removed.append(stem)
        else:
            filtered.append((stem, h_path, corr_path))

    print("\n============================================================")
    print("PAIR FILTER")
    print("============================================================")
    print(f"Configured excluded views: {sorted(exclude_views)}")

    if removed:
        print("Removed pairs:")
        for stem in removed:
            print(f"  {stem}")
    else:
        print("No matching excluded views were present.")

    return filtered


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Select folder containing H files...")
    H_folder = _select_folder("Select folder with H NPZ files")

    print("Select folder containing corr files...")
    corr_folder = _select_folder("Select folder with corr NPZ files")

    print(f"\nH folder:\n  {H_folder}")
    print(f"corr folder:\n  {corr_folder}")

    print("\n============================================================")
    print("CONFIG")
    print("============================================================")
    print(f"IMAGE_SIZE              = {IMAGE_SIZE}")
    print(f"LASER_CROSS_UV          = {LASER_CROSS_UV}")
    print(f"INITIAL_FOCAL_LENGTH_PX = {INITIAL_FOCAL_LENGTH_PX}")
    print(f"ENFORCE_ZERO_SKEW       = {ENFORCE_ZERO_SKEW}")
    print(f"EXCLUDE_VIEWS           = {sorted(EXCLUDE_VIEWS)}")

    pairs = find_matching_pairs(H_folder, corr_folder)
    pairs = filter_pairs(pairs, EXCLUDE_VIEWS)

    views: list[dict] = []

    for stem, h_path, corr_path in pairs:
        try:
            v = analyze_single_view(stem, h_path, corr_path)
            views.append(v)
            print_view_summary(v)
        except Exception as e:
            print(f"\n[SKIP] {stem}: {e}")

    min_views = 2 if ENFORCE_ZERO_SKEW else 3
    if len(views) < min_views:
        raise RuntimeError(f"Only {len(views)} valid views remain. Need at least {min_views}.")

    print("\n============================================================")
    print("GLOBAL SUMMARY")
    print("============================================================")
    print(f"Number of valid matched views = {len(views)}")
    print(f"Mean of per-view H reproj mean = {np.mean([v['mean_px'] for v in views]):.6f}")
    print(f"Median of per-view H reproj mean = {np.median([v['mean_px'] for v in views]):.6f}")
    print(f"Worst per-view H reproj mean = {np.max([v['mean_px'] for v in views]):.6f}")
    print(f"Worst per-view H reproj max = {np.max([v['max_px'] for v in views]):.6f}")

    run_subset_rankings(views)

    _ = run_zhang_analysis(
        views,
        IMAGE_SIZE,
        enforce_zero_skew=ENFORCE_ZERO_SKEW,
    )

    _ = run_ba_center_pp_equal_f(
        views,
        IMAGE_SIZE,
        initial_f_px=INITIAL_FOCAL_LENGTH_PX,
    )


if __name__ == "__main__":
    main()