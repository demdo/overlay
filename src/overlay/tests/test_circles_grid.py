# -*- coding: utf-8 -*-
"""
test_circles_grid.py

Variant A ONLY (cKDTree + KMeans), but with robust (i,j) assignment:

Instead of hard ij = round(u,v) + collision-drop,
we do a 4-candidate assignment per point (floor/ceil combos) and then
a global greedy matching by smallest residual (cell unique).

Writes ONLY:
- __circles_grid_A.txt
- __diag_A.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import cv2

from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

Cell = Tuple[int, int]


# ============================================================
# I/O
# ============================================================

def load_circles(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        z = np.load(path)
        arr = z["circles"] if "circles" in z else z["arr_0"]
    arr = np.asarray(arr, float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"circles must be (N,2) or (N,3), got {arr.shape}")
    if arr.shape[1] == 2:
        arr = np.hstack([arr, np.full((len(arr), 1), np.nan, float)])
    return arr.astype(float)


def _load_xray_gray(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in (".dcm", ".ima"):
        try:
            import pydicom
        except Exception as e:
            raise RuntimeError("pydicom required for .dcm/.ima. Install: pip install pydicom") from e

        ds = pydicom.dcmread(str(p))
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img.astype(np.uint8)

    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Could not read image.")
    return img


# ============================================================
# Homography helpers (affine rectification)
# ============================================================

def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _to_h(xy: np.ndarray) -> np.ndarray:
    return np.hstack([xy, np.ones((len(xy), 1), float)])


def _from_h(xyh: np.ndarray) -> np.ndarray:
    xyh = xyh / (xyh[:, 2:3] + 1e-12)
    return xyh[:, :2]


def estimate_vanishing_point(lines: np.ndarray) -> np.ndarray:
    L = lines.astype(float).copy()
    L /= (np.linalg.norm(L[:, :2], axis=1, keepdims=True) + 1e-12)
    _, _, Vt = np.linalg.svd(L)
    v = Vt[-1]
    return v / (v[2] + 1e-12)


def affine_rectification_H(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    l_inf = np.cross(v1, v2)
    l_inf = l_inf / (l_inf[2] + 1e-12)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [l_inf[0], l_inf[1], l_inf[2]],
        ],
        dtype=float,
    )


def estimate_affine_rectification_from_points(
    xy: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    *,
    k_nn: int = 8,
    angle_thresh_deg: float = 15.0,
) -> np.ndarray:
    tree = cKDTree(xy)
    _d, idx = tree.query(xy, k=min(k_nn + 1, len(xy)))

    ang_thr = np.deg2rad(angle_thresh_deg)

    lines1 = []
    lines2 = []

    for i in range(len(xy)):
        pi = xy[i]
        for jpos in range(1, idx.shape[1]):
            j = int(idx[i, jpos])
            pj = xy[j]
            v = pj - pi
            nv = float(np.linalg.norm(v))
            if nv < 1e-6:
                continue
            d = v / nv

            a1 = np.arccos(np.clip(abs(float(d @ u1)), 0.0, 1.0))
            a2 = np.arccos(np.clip(abs(float(d @ u2)), 0.0, 1.0))

            if a1 < ang_thr and a1 <= a2:
                lines1.append(np.cross(np.r_[pi, 1.0], np.r_[pj, 1.0]))
            elif a2 < ang_thr:
                lines2.append(np.cross(np.r_[pi, 1.0], np.r_[pj, 1.0]))

    lines1 = np.asarray(lines1, float)
    lines2 = np.asarray(lines2, float)
    if len(lines1) < 50 or len(lines2) < 50:
        raise RuntimeError(f"Not enough line evidence: n1={len(lines1)}, n2={len(lines2)}")

    v1 = estimate_vanishing_point(lines1)
    v2 = estimate_vanishing_point(lines2)
    return affine_rectification_H(v1, v2)


# ============================================================
# Grid Fit (Variant A) with 4-candidate assignment
# ============================================================

@dataclass
class GridFitResult:
    origin: np.ndarray
    a: np.ndarray
    b: np.ndarray
    uv_float: np.ndarray            # float lattice coords (u,v)
    ij_used: np.ndarray             # int (i,j) used for LS (N x 2, NaN where unused)
    residual_fit: np.ndarray        # per point residual (best candidate if assigned else NaN)
    mapping: Dict[Cell, int]        # (i,j)->k (unique cells)
    H_aff: Optional[np.ndarray] = None


def _compute_uv(xy_fit: np.ndarray, origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    M = np.stack([a, b], axis=1)  # 2x2
    Minv = np.linalg.inv(M)
    return (xy_fit - origin) @ Minv.T


def _candidate_assign_greedy(
    xy_fit: np.ndarray,
    uv: np.ndarray,
    origin: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_frac_gate: float = 0.80,
) -> tuple[Dict[Cell, int], np.ndarray]:
    """
    Build candidates for each point:
      (floor/ceil u) x (floor/ceil v) => 4 cells
    Compute residual for each candidate and greedily take smallest residual
    while enforcing cell-uniqueness and point-uniqueness.

    max_frac_gate: reject points that are insanely far from integer lattice
                   (norm of fractional part > gate). Keep large (0.8) to be permissive.
    """
    N = len(xy_fit)

    # fractional distance to nearest integer (diagnostic / gate)
    frac = np.abs(uv - np.rint(uv))
    frac_norm = np.linalg.norm(frac, axis=1)

    cand_list: List[tuple[float, int, int, int]] = []  # (res, k, i, j)

    for k in range(N):
        if not np.isfinite(frac_norm[k]) or frac_norm[k] > max_frac_gate:
            continue

        u, v = float(uv[k, 0]), float(uv[k, 1])
        u0 = int(np.floor(u))
        v0 = int(np.floor(v))
        u1 = u0 + 1
        v1 = v0 + 1

        for ii in (u0, u1):
            for jj in (v0, v1):
                pred = origin + ii * a + jj * b
                res = float(np.linalg.norm(xy_fit[k] - pred))
                cand_list.append((res, k, ii, jj))

    cand_list.sort(key=lambda t: t[0])

    used_point = np.zeros(N, dtype=bool)
    used_cell: set[Cell] = set()
    mapping: Dict[Cell, int] = {}
    best_res = np.full(N, np.nan, float)

    for res, k, i, j in cand_list:
        if used_point[k]:
            continue
        cell = (int(i), int(j))
        if cell in used_cell:
            continue
        used_point[k] = True
        used_cell.add(cell)
        mapping[cell] = int(k)
        best_res[k] = float(res)

    return mapping, best_res


def fit_grid_from_circles(
    circles: np.ndarray,
    *,
    use_affine_rectification: bool = True,
    k_nn: int = 8,
    short_quantile: float = 0.25,
    axis_swap_make_i_vertical: bool = True,
    refine_iters: int = 4,
) -> GridFitResult:
    xy_img = circles[:, :2].astype(float)
    N = len(xy_img)
    if N < 50:
        raise ValueError("Need at least ~50 circles.")

    # ---- (1) short neighbor vectors in IMAGE space -> directions
    tree = cKDTree(xy_img)
    d, idx = tree.query(xy_img, k=min(k_nn + 1, N))

    vecs = []
    dists = []
    for i in range(N):
        for jpos in range(1, idx.shape[1]):
            j = int(idx[i, jpos])
            v = xy_img[j] - xy_img[i]
            dist = float(np.linalg.norm(v))
            if dist > 1e-6:
                vecs.append(v)
                dists.append(dist)
    vecs = np.asarray(vecs, float)
    dists = np.asarray(dists, float)

    d_thr = float(np.quantile(dists, short_quantile))
    vecs_s = vecs[dists <= d_thr]
    dirs = vecs_s / (np.linalg.norm(vecs_s, axis=1, keepdims=True) + 1e-12)
    dirs[dirs[:, 0] < 0] *= -1.0

    km = KMeans(n_clusters=2, n_init=30, random_state=0)
    km.fit(dirs)
    u1 = _unit(km.cluster_centers_[0])
    u2 = _unit(km.cluster_centers_[1])

    # ---- (2) affine rectification
    H_aff = None
    xy_fit = xy_img
    if use_affine_rectification:
        H_aff = estimate_affine_rectification_from_points(
            xy_img, u1, u2, k_nn=k_nn, angle_thresh_deg=15.0
        )
        xy_fit = _from_h(_to_h(xy_img) @ H_aff.T)

    # ---- (3) estimate pitch/steps in FIT space
    tree2 = cKDTree(xy_fit)
    d2, idx2 = tree2.query(xy_fit, k=min(k_nn + 1, N))

    vecs2 = []
    dists2 = []
    for i in range(N):
        for jpos in range(1, idx2.shape[1]):
            j = int(idx2[i, jpos])
            v = xy_fit[j] - xy_fit[i]
            dist = float(np.linalg.norm(v))
            if dist > 1e-6:
                vecs2.append(v)
                dists2.append(dist)
    vecs2 = np.asarray(vecs2, float)
    dists2 = np.asarray(dists2, float)

    d_thr2 = float(np.quantile(dists2, short_quantile))
    vecs_s2 = vecs2[dists2 <= d_thr2]
    pitch2 = float(np.median(dists2[dists2 <= d_thr2]))

    dirs2 = vecs_s2 / (np.linalg.norm(vecs_s2, axis=1, keepdims=True) + 1e-12)
    dirs2[dirs2[:, 0] < 0] *= -1.0
    km2 = KMeans(n_clusters=2, n_init=30, random_state=0)
    km2.fit(dirs2)
    uu1 = _unit(km2.cluster_centers_[0])
    uu2 = _unit(km2.cluster_centers_[1])

    proj1 = np.abs(vecs_s2 @ uu1)
    proj2 = np.abs(vecs_s2 @ uu2)

    def _pick_step(proj: np.ndarray, pitch: float) -> float:
        good = proj[(proj > 0.6 * pitch) & (proj < 1.4 * pitch)]
        return float(np.median(good)) if len(good) else float(pitch)

    s1 = _pick_step(proj1, pitch2)
    s2 = _pick_step(proj2, pitch2)

    a = s1 * uu1
    b = s2 * uu2

    if axis_swap_make_i_vertical:
        if abs(a[1]) < abs(b[1]):
            a, b = b, a

    # ---- init origin (robust enough): median of points
    origin = np.median(xy_fit, axis=0)

    # ---- iterate: assign (4-candidate) -> LS -> repeat
    ij_used = np.full((N, 2), np.nan, float)
    best_res = np.full(N, np.nan, float)
    mapping: Dict[Cell, int] = {}

    for _ in range(refine_iters):
        uv = _compute_uv(xy_fit, origin, a, b)
        mapping, best_res = _candidate_assign_greedy(xy_fit, uv, origin, a, b, max_frac_gate=0.80)

        # build ij for LS from mapping (assigned points only)
        ij_used[:] = np.nan
        ks = list(mapping.values())
        if len(ks) < 50:
            break

        # inverse mapping for points -> cell
        point_to_cell = {k: cell for cell, k in mapping.items()}
        for k in ks:
            i, j = point_to_cell[k]
            ij_used[k, 0] = float(i)
            ij_used[k, 1] = float(j)

        # LS update (only assigned points)
        mask = np.isfinite(ij_used[:, 0]) & np.isfinite(ij_used[:, 1])
        A = np.column_stack([np.ones(mask.sum()), ij_used[mask, 0], ij_used[mask, 1]])
        cx, *_ = np.linalg.lstsq(A, xy_fit[mask, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(A, xy_fit[mask, 1], rcond=None)

        origin = np.array([cx[0], cy[0]], float)
        a = np.array([cx[1], cy[1]], float)
        b = np.array([cx[2], cy[2]], float)

    uv = _compute_uv(xy_fit, origin, a, b)
    mapping, best_res = _candidate_assign_greedy(xy_fit, uv, origin, a, b, max_frac_gate=0.80)

    print(f"[fit] pitch2={pitch2:.3f} | |a|={np.linalg.norm(a):.3f} | |b|={np.linalg.norm(b):.3f} | use_aff_rect={use_affine_rectification}")

    return GridFitResult(
        origin=origin,
        a=a,
        b=b,
        uv_float=uv,
        ij_used=ij_used,
        residual_fit=best_res,
        mapping=mapping,
        H_aff=H_aff,
    )


def shift_indices_nonnegative(fit: GridFitResult) -> Tuple[int, int]:
    cells = np.array(list(fit.mapping.keys()), int)
    di = -int(cells[:, 0].min()) if int(cells[:, 0].min()) < 0 else 0
    dj = -int(cells[:, 1].min()) if int(cells[:, 1].min()) < 0 else 0
    if di != 0 or dj != 0:
        fit.mapping = {(i + di, j + dj): k for (i, j), k in fit.mapping.items()}
        fit.uv_float = fit.uv_float + np.array([di, dj], float)
        # ij_used stays diagnostic only; no need to shift
    return di, dj


def build_circles_grid(circles: np.ndarray, fit: GridFitResult):
    cells = np.array(list(fit.mapping.keys()), int)
    i_min, j_min = cells.min(axis=0)
    i_max, j_max = cells.max(axis=0)

    nrows = int(i_max - i_min + 1)
    ncols = int(j_max - j_min + 1)

    circles_grid = np.full((nrows, ncols, 3), np.nan, float)
    for (i, j), k in fit.mapping.items():
        ii = int(i - i_min)
        jj = int(j - j_min)
        circles_grid[ii, jj] = circles[k]
    return circles_grid, int(i_min), int(j_min)


# ============================================================
# ONLY needed TXT
# ============================================================

def write_circles_grid_txt(out_path: Path, grid: np.ndarray) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        nr, nc = grid.shape[:2]
        for i in range(nr):
            for j in range(nc):
                x, y, r = grid[i, j]
                ok = int(np.isfinite(x) and np.isfinite(y))
                f.write(f"{i:4d} {j:4d} {x:10.2f} {y:10.2f} {r:8.3f} {ok:d}\n")


def write_diag_txt(out_path: Path, fit: GridFitResult) -> None:
    uv = fit.uv_float
    frac = np.abs(uv - np.rint(uv))
    frac_norm = np.linalg.norm(frac, axis=1)
    frac_norm = frac_norm[np.isfinite(frac_norm)]

    res = fit.residual_fit
    res_f = res[np.isfinite(res)]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("=== Variant A diag ===\n")
        f.write(f"mapped_cells: {len(fit.mapping)}\n")
        f.write(f"|a|={np.linalg.norm(fit.a):.4f}, |b|={np.linalg.norm(fit.b):.4f}\n")
        if len(frac_norm):
            f.write(f"frac_norm median: {np.median(frac_norm):.4f}\n")
            f.write(f"frac_norm p95:    {np.quantile(frac_norm, 0.95):.4f}\n")
            f.write(f"frac_norm max:    {np.max(frac_norm):.4f}\n")
        if len(res_f):
            f.write(f"res_fit median: {np.median(res_f):.4f}\n")
            f.write(f"res_fit p95:    {np.quantile(res_f, 0.95):.4f}\n")
            f.write(f"res_fit max:    {np.max(res_f):.4f}\n")


# ============================================================
# Visualization
# ============================================================

def show_grid_block_on_xray(
    xray_gray: np.ndarray,
    grid: np.ndarray,
    i0: int, j0: int,
    i1: int, j1: int,
    *,
    marker_size: int = 12,
    thickness: int = 2,
    title: str = "Variant A - block",
) -> None:
    vis = cv2.cvtColor(xray_gray, cv2.COLOR_GRAY2BGR)

    drawn = 0
    H, W = xray_gray.shape[:2]

    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                x, y, _ = grid[i, j]
                if np.isfinite(x) and np.isfinite(y):
                    cx = int(round(float(x)))
                    cy = int(round(float(y)))
                    if 0 <= cx < W and 0 <= cy < H:
                        cv2.drawMarker(
                            vis,
                            (cx, cy),
                            (0, 255, 0),
                            markerType=cv2.MARKER_CROSS,
                            markerSize=marker_size,
                            thickness=thickness,
                        )
                        drawn += 1

    print(f"[A] drawn in block: {drawn}")
    cv2.imshow(title, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Main
# ============================================================

def main():
    from PySide6.QtWidgets import QApplication, QFileDialog

    app = QApplication.instance() or QApplication([])

    circles_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select circles file",
        "",
        "NumPy (*.npy *.npz);;All Files (*)",
    )
    if not circles_path:
        return

    xray_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select X-ray image",
        "",
        "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.dcm *.ima);;All Files (*)",
    )
    if not xray_path:
        return

    circles_path = Path(circles_path)
    out_base = circles_path.with_suffix("")

    circles = load_circles(circles_path)
    xray_gray = _load_xray_gray(xray_path)

    fitA = fit_grid_from_circles(
        circles,
        use_affine_rectification=True,
        k_nn=8,
        short_quantile=0.25,
        axis_swap_make_i_vertical=True,
        refine_iters=4,
    )

    di, dj = shift_indices_nonnegative(fitA)

    cells = np.array(list(fitA.mapping.keys()), int)
    print("[A] SHIFTED i range:", int(cells[:, 0].min()), "..", int(cells[:, 0].max()))
    print("[A] SHIFTED j range:", int(cells[:, 1].min()), "..", int(cells[:, 1].max()))
    print("[A] Applied (di,dj):", di, dj)
    print("[A] Unique mapped cells:", len(cells))

    circles_grid, i_min, j_min = build_circles_grid(circles, fitA)
    print("[A] grid_ij0:", (i_min, j_min), "grid shape:", circles_grid.shape)

    # only needed txt
    grid_txt = out_base.with_name(out_base.name + "__circles_grid_A.txt")
    diag_txt = out_base.with_name(out_base.name + "__diag_A.txt")
    write_circles_grid_txt(grid_txt, circles_grid)
    write_diag_txt(diag_txt, fitA)
    print("[A] wrote:", grid_txt.name)
    print("[A] wrote:", diag_txt.name)

    # show block
    show_grid_block_on_xray(
        xray_gray,
        circles_grid,
        10, 10, 20, 20,
        marker_size=12,
        thickness=2,
        title="Variant A (4-candidate assign) - block (10..20)",
    )


if __name__ == "__main__":
    main()