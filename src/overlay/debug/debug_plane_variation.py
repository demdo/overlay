# -*- coding: utf-8 -*-
"""
debug_plane_consistency.py
==========================

Vergleicht drei Ebenen-Schätzungsstufen über mehrere plane_snapshot_*.npz:

    plane_abcd_single    — einzelner ransac_plane_open3d-Aufruf (echtes Raw)
    plane_abcd_raw       — fit_plane_stable (n_stable_runs RANSAC gemittelt)
    plane_abcd_filtered  — PlaneKalmanFilter output

Ausgabe: Konsolenzusammenfassung + Abbildung (kein Speichern).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PySide6.QtWidgets import QApplication, QFileDialog


# ============================================================
# Qt file picker
# ============================================================

def pick_files() -> list[Path]:
    app = QApplication.instance() or QApplication(sys.argv)
    files, _ = QFileDialog.getOpenFileNames(
        None,
        "Plane-fitting snapshots auswählen",
        "",
        "NumPy archives (*.npz)",
    )
    return [Path(f) for f in sorted(files)]


# ============================================================
# Helpers
# ============================================================

def _normalise(plane: np.ndarray) -> np.ndarray:
    """||n|| = 1, sign: nz < 0 (normal toward camera)."""
    plane = np.asarray(plane, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(plane[:3])
    if norm < 1e-12:
        raise ValueError("Near-zero normal.")
    plane = plane / norm
    if plane[2] > 0:
        plane = -plane
    return plane


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def _angles_vs_ref(normals: np.ndarray) -> np.ndarray:
    """Angle of each normal vs the first one (run 0)."""
    ref = normals[0]
    return np.array([_angle_deg(ref, n) for n in normals], dtype=np.float64)


def _z_intercept_mm(plane: np.ndarray) -> float:
    c, d = float(plane[2]), float(plane[3])
    return float((-d / c) * 1000.0) if abs(c) > 1e-12 else np.nan


def _centroid_mm(xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return np.full(3, np.nan)
    return pts.mean(axis=0) * 1000.0


# ============================================================
# Loading
# ============================================================

class Snapshot:
    def __init__(
        self,
        name: str,
        run_index: int,
        plane_single: np.ndarray | None,
        plane_raw: np.ndarray,
        plane_filtered: np.ndarray,
        xyz_single: np.ndarray,
        xyz_raw: np.ndarray,
        xyz_filtered: np.ndarray,
        inlier_ratio: float,
    ):
        self.name        = name
        self.run_index   = run_index
        self.inlier_ratio = inlier_ratio

        # normals
        self.normal_single   = plane_single[:3].copy() if plane_single is not None else None
        self.normal_raw      = plane_raw[:3].copy()
        self.normal_filt     = plane_filtered[:3].copy()

        # z-intercepts
        self.zint_single_mm  = _z_intercept_mm(plane_single) if plane_single is not None else np.nan
        self.zint_raw_mm     = _z_intercept_mm(plane_raw)
        self.zint_filt_mm    = _z_intercept_mm(plane_filtered)

        # centroids
        self.centroid_single_mm  = _centroid_mm(xyz_single)
        self.centroid_raw_mm     = _centroid_mm(xyz_raw)
        self.centroid_filt_mm    = _centroid_mm(xyz_filtered)


def load_snapshots(files: list[Path]) -> list[Snapshot]:
    snaps: list[Snapshot] = []

    for path in files:
        try:
            with np.load(str(path), allow_pickle=False) as z:
                keys = set(z.files)

                # plane_abcd_raw and plane_abcd_filtered are required
                missing = {"plane_abcd_raw", "plane_abcd_filtered"} - keys
                if missing:
                    print(f"[SKIP] {path.name}: missing {missing}")
                    continue

                plane_raw  = _normalise(z["plane_abcd_raw"])
                plane_filt = _normalise(z["plane_abcd_filtered"])
                plane_single = _normalise(z["plane_abcd_single"]) \
                               if "plane_abcd_single" in keys else None

                xyz_raw    = np.asarray(z["points_xyz_camera"],        dtype=np.float64) \
                             if "points_xyz_camera"        in keys else np.empty((0, 3))
                xyz_filt   = np.asarray(z["points_xyz_camera_filt"],   dtype=np.float64) \
                             if "points_xyz_camera_filt"   in keys else xyz_raw.copy()
                xyz_single = np.asarray(z["points_xyz_camera_single"], dtype=np.float64) \
                             if "points_xyz_camera_single" in keys else np.empty((0, 3))

                inlier_ratio = float(np.asarray(z["inlier_ratio"]).ravel()[0]) \
                               if "inlier_ratio" in keys else np.nan
                run_index    = int(np.asarray(z["run_index"]).ravel()[0]) \
                               if "run_index"    in keys else len(snaps)

                snaps.append(Snapshot(
                    name=path.stem,
                    run_index=run_index,
                    plane_single=plane_single,
                    plane_raw=plane_raw,
                    plane_filtered=plane_filt,
                    xyz_single=xyz_single,
                    xyz_raw=xyz_raw,
                    xyz_filtered=xyz_filt,
                    inlier_ratio=inlier_ratio,
                ))

        except Exception as e:
            print(f"[SKIP] {path.name}: {e}")

    snaps.sort(key=lambda s: s.run_index)
    return snaps


# ============================================================
# Metrics
# ============================================================

def compute_series(snaps: list[Snapshot]) -> dict:
    has_single = any(s.normal_single is not None for s in snaps)

    normals_raw  = np.array([s.normal_raw  for s in snaps])
    normals_filt = np.array([s.normal_filt for s in snaps])

    out = dict(
        run_index    = np.array([s.run_index    for s in snaps]),
        angles_raw   = _angles_vs_ref(normals_raw),
        angles_filt  = _angles_vs_ref(normals_filt),
        zint_raw_mm  = np.array([s.zint_raw_mm   for s in snaps]),
        zint_filt_mm = np.array([s.zint_filt_mm  for s in snaps]),
        cent_raw_mm  = np.array([s.centroid_raw_mm  for s in snaps]),
        cent_filt_mm = np.array([s.centroid_filt_mm for s in snaps]),
        inliers      = np.array([s.inlier_ratio      for s in snaps]),
        has_single   = has_single,
    )

    if has_single:
        normals_single = np.array([
            s.normal_single if s.normal_single is not None else s.normal_raw
            for s in snaps
        ])
        out["angles_single"]   = _angles_vs_ref(normals_single)
        out["zint_single_mm"]  = np.array([s.zint_single_mm  for s in snaps])
        out["cent_single_mm"]  = np.array([s.centroid_single_mm for s in snaps])

    return out


# ============================================================
# Console summary
# ============================================================

def _row(label: str, arr: np.ndarray, unit: str = "") -> str:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return f"  {label:<32s}: n/a"
    return (
        f"  {label:<32s}: "
        f"mean={np.mean(v):8.4f}{unit}  "
        f"std={np.std(v):7.4f}{unit}  "
        f"max={np.max(v):8.4f}{unit}  "
        f"range={np.max(v)-np.min(v):7.4f}{unit}"
    )


def print_summary(snaps: list[Snapshot], s: dict) -> None:
    n = len(snaps)
    has_single = s["has_single"]

    std_single = np.nanstd(s["angles_single"]) if has_single else None
    std_raw    = np.nanstd(s["angles_raw"])
    std_filt   = np.nanstd(s["angles_filt"])

    print("\n" + "=" * 74)
    print(f"SINGLE vs RAW vs FILTERED — {n} runs")
    print("=" * 74)

    print("\n── Normal angle vs. run 0 ───────────────────────────────────────────")
    if has_single:
        print(_row("single (1 RANSAC)", s["angles_single"], "°"))
    print(_row("raw    (fit_plane_stable)", s["angles_raw"],  "°"))
    print(_row("filtered (Kalman)",        s["angles_filt"], "°"))

    if has_single and std_single and std_single > 1e-9:
        print(f"\n  {'std reduction raw→filt':<32s}: "
              f"{(1.0 - std_filt / std_raw) * 100:.1f}%")
        print(f"  {'std reduction single→filt':<32s}: "
              f"{(1.0 - std_filt / std_single) * 100:.1f}%")
        print(f"  {'std reduction single→raw':<32s}: "
              f"{(1.0 - std_raw / std_single) * 100:.1f}%")
    else:
        red = (1.0 - std_filt / std_raw) * 100.0 if std_raw > 1e-9 else 0.0
        print(f"  {'std reduction raw→filt':<32s}: {red:.1f}%")

    print("\n── z-intercept (−d/nz) ──────────────────────────────────────────────")
    if has_single:
        print(_row("single", s["zint_single_mm"], " mm"))
    print(_row("raw",      s["zint_raw_mm"],  " mm"))
    print(_row("filtered", s["zint_filt_mm"], " mm"))

    print("\n── Grid centroid cz ─────────────────────────────────────────────────")
    for arr, label in [
        (s["cent_single_mm"][:, 2] if has_single else None, "single"),
        (s["cent_raw_mm"][:, 2],  "raw"),
        (s["cent_filt_mm"][:, 2], "filtered"),
    ]:
        if arr is None:
            continue
        print(f"  {label:<32s}: "
              f"mean={np.nanmean(arr):8.3f} mm  "
              f"std={np.nanstd(arr):6.3f} mm  "
              f"range={np.nanmax(arr)-np.nanmin(arr):.3f} mm")

    print("\n── RANSAC inlier ratio ──────────────────────────────────────────────")
    print(_row("inlier ratio", s["inliers"]))

    print("\n── Per run ──────────────────────────────────────────────────────────")
    if has_single:
        print(f"  {'run':>3}  {'single':>8}  {'raw':>8}  {'stable':>8}  "
              f"{'zint_s':>8}  {'zint_r':>8}  {'zint_f':>8}  {'inliers':>7}")
        for i, snap in enumerate(snaps):
            ang_s = s["angles_single"][i]
            ang_r = s["angles_raw"][i]
            ang_f = s["angles_filt"][i]
            print(
                f"  {snap.run_index:>3d}  "
                f"{ang_s:>8.4f}°  {ang_r:>8.4f}°  {ang_f:>8.4f}°  "
                f"{snap.zint_single_mm:>8.2f}  "
                f"{snap.zint_raw_mm:>8.2f}  "
                f"{snap.zint_filt_mm:>8.2f}  "
                f"{snap.inlier_ratio:>7.4f}"
            )
    else:
        print(f"  {'run':>3}  {'raw':>10}  {'filtered':>10}  "
              f"{'Δangle':>7}  {'zint_raw':>9}  {'zint_filt':>9}  {'inliers':>7}")
        for i, snap in enumerate(snaps):
            ar = s["angles_raw"][i]
            af = s["angles_filt"][i]
            print(
                f"  {snap.run_index:>3d}  {ar:>10.4f}°  {af:>10.4f}°  "
                f"{af-ar:>+7.4f}°  "
                f"{snap.zint_raw_mm:>9.2f}  {snap.zint_filt_mm:>9.2f}  "
                f"{snap.inlier_ratio:>7.4f}"
            )
    print()


# ============================================================
# Plotting
# ============================================================

COL_SINGLE = "#55A868"   # green
COL_RAW    = "#4C72B0"   # blue
COL_FILT   = "#DD8452"   # orange


def plot_metrics(snaps: list[Snapshot], s: dict) -> None:
    idx        = s["run_index"]
    has_single = s["has_single"]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Single RANSAC  vs  fit_plane_stable  vs  Kalman — {len(snaps)} runs",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    # ── (0,0:2)  Normal angle vs run 0 ───────────────────────────────────
    ax_ang = fig.add_subplot(gs[0, :2])
    if has_single:
        ax_ang.plot(idx, s["angles_single"], "^-",  color=COL_SINGLE, lw=1.5,
                    ms=5, label="single RANSAC")
    ax_ang.plot(idx, s["angles_raw"],    "o-",  color=COL_RAW,    lw=1.5,
                ms=5, label="fit_plane_stable")
    ax_ang.plot(idx, s["angles_filt"],   "s--", color=COL_FILT,   lw=1.5,
                ms=5, label="Kalman filtered")
    ax_ang.set_title("Normal angle vs. run 0", fontsize=10)
    ax_ang.set_xlabel("Run index")
    ax_ang.set_ylabel("deg")
    ax_ang.legend(fontsize=9)
    ax_ang.grid(True, alpha=0.3)

    # ── (0,2)  Violin: angle distribution ────────────────────────────────
    ax_vio = fig.add_subplot(gs[0, 2])
    if has_single:
        vdata  = [s["angles_single"], s["angles_raw"], s["angles_filt"]]
        vlabels = ["single", "stable", "Kalman"]
        vcols   = [COL_SINGLE, COL_RAW, COL_FILT]
    else:
        vdata   = [s["angles_raw"], s["angles_filt"]]
        vlabels = ["stable", "Kalman"]
        vcols   = [COL_RAW, COL_FILT]

    parts = ax_vio.violinplot(vdata, showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], vcols):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        parts[part].set_color("black")
        parts[part].set_linewidth(0.8)
    ax_vio.set_xticks(range(1, len(vlabels) + 1))
    ax_vio.set_xticklabels(vlabels, fontsize=8)
    ax_vio.set_title("Angle distribution", fontsize=10)
    ax_vio.set_ylabel("deg")
    ax_vio.grid(True, alpha=0.3, axis="y")

    # ── (1,0:2)  z-intercept ─────────────────────────────────────────────
    ax_z = fig.add_subplot(gs[1, :2])
    if has_single:
        ax_z.plot(idx, s["zint_single_mm"], "^-",  color=COL_SINGLE, lw=1.5,
                  ms=5, label="single RANSAC")
    ax_z.plot(idx, s["zint_raw_mm"],    "o-",  color=COL_RAW,    lw=1.5,
              ms=5, label="fit_plane_stable")
    ax_z.plot(idx, s["zint_filt_mm"],   "s--", color=COL_FILT,   lw=1.5,
              ms=5, label="Kalman filtered")
    ax_z.set_title("Plane z-intercept (−d / nz)", fontsize=10)
    ax_z.set_xlabel("Run index")
    ax_z.set_ylabel("mm")
    ax_z.legend(fontsize=9)
    ax_z.grid(True, alpha=0.3)

    # ── (1,2)  Violin: z-intercept distribution ───────────────────────────
    ax_zvio = fig.add_subplot(gs[1, 2])
    if has_single:
        zdata   = [s["zint_single_mm"], s["zint_raw_mm"], s["zint_filt_mm"]]
    else:
        zdata   = [s["zint_raw_mm"], s["zint_filt_mm"]]

    parts2 = ax_zvio.violinplot(zdata, showmedians=True, showextrema=True)
    for pc, col in zip(parts2["bodies"], vcols):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        parts2[part].set_color("black")
        parts2[part].set_linewidth(0.8)
    ax_zvio.set_xticks(range(1, len(vlabels) + 1))
    ax_zvio.set_xticklabels(vlabels, fontsize=8)
    ax_zvio.set_title("z-intercept distribution", fontsize=10)
    ax_zvio.set_ylabel("mm")
    ax_zvio.grid(True, alpha=0.3, axis="y")

    # ── (2,0:2)  Grid centroid cz ─────────────────────────────────────────
    ax_cz = fig.add_subplot(gs[2, :2])
    if has_single:
        ax_cz.plot(idx, s["cent_single_mm"][:, 2], "^-",  color=COL_SINGLE,
                   lw=1.5, ms=5, label="cz single")
    ax_cz.plot(idx, s["cent_raw_mm"][:, 2],  "o-",  color=COL_RAW,
               lw=1.5, ms=5, label="cz stable")
    ax_cz.plot(idx, s["cent_filt_mm"][:, 2], "s--", color=COL_FILT,
               lw=1.5, ms=5, label="cz Kalman")
    ax_cz.set_title("Grid centroid cz  (most sensitive to normal error)", fontsize=10)
    ax_cz.set_xlabel("Run index")
    ax_cz.set_ylabel("mm")
    ax_cz.legend(fontsize=9)
    ax_cz.grid(True, alpha=0.3)

    # ── (2,2)  RANSAC inlier ratio ────────────────────────────────────────
    ax_inl = fig.add_subplot(gs[2, 2])
    ax_inl.plot(idx, s["inliers"], "o-", color=COL_RAW, lw=1.5, ms=5)
    ax_inl.set_title("RANSAC inlier ratio", fontsize=10)
    ax_inl.set_xlabel("Run index")
    ax_inl.set_ylabel("ratio")
    lo = max(0.0, np.nanmin(s["inliers"]) - 0.05)
    ax_inl.set_ylim(lo, 1.02)
    ax_inl.grid(True, alpha=0.3)

    plt.show()


# ============================================================
# Main
# ============================================================

def main() -> None:
    files = pick_files()
    if not files:
        print("No files selected.")
        return

    snaps = load_snapshots(files)
    if not snaps:
        print("No valid snapshots loaded.")
        return

    series = compute_series(snaps)
    print_summary(snaps, series)
    plot_metrics(snaps, series)


if __name__ == "__main__":
    main()