# -*- coding: utf-8 -*-
"""
test_camera_to_xray_calibration.py

Explorer -> load the .npz saved by CameraToXrayCalibrationPage (cam2x_last_run.npz)
and plot a histogram of reprojection errors e [px], styled like test_blob_detection.

Expected npz keys (as saved in your Page):
- uv_measured            (N,2)
- uv_projected           (N,2)
- inliers_idx            (K,)  optional but recommended
- ransac_threshold_px    (float) optional

Notes:
- NO LaTeX dependency (MiKTeX problems avoided). We use Matplotlib mathtext only.
- Layout uses subplots_adjust (no tight_layout) to avoid "squeezed plot + huge white space".
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


# ============================================================
# Explorer dialog
# ============================================================

def _pick_npz_dialog() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    filetypes = [
        ("NumPy archives", "*.npz"),
        ("All Files", "*.*"),
    ]
    path = filedialog.askopenfilename(title="Select cam2x .npz (PnP debug export)", filetypes=filetypes)
    root.destroy()
    return path or ""


# ============================================================
# Plot
# ============================================================

def show_pnp_error_histogram(
    uv_measured: np.ndarray,
    uv_projected: np.ndarray,
    inliers_idx: np.ndarray | None = None,
    ransac_threshold_px: float | None = None,
    bins: int = 35,
    title: str = r"Camera-to-X-ray calibration: distribution of $e$",
):
    """
    Histogram for reprojection error e = ||u_meas - u_proj||_2 in pixels.

    Style aligned with show_plane_fit_histogram:
    - LaTeX everywhere (usetex=True)
    - blue bins up to P95, red bins from P95 onwards (same red as plane fitting)
    - NO dashed P95 line / NO P95 label in plot (P95 stays in legend box)
    - optional RANSAC threshold shown as solid vertical line
    - legend box order ends with gamma_RANSAC
    """

    uv_measured = np.asarray(uv_measured, dtype=float).reshape(-1, 2)
    uv_projected = np.asarray(uv_projected, dtype=float).reshape(-1, 2)
    if uv_measured.shape != uv_projected.shape:
        raise ValueError(
            f"Shape mismatch: uv_measured {uv_measured.shape} vs uv_projected {uv_projected.shape}"
        )

    # --- per-point error on full arrays (so inliers_idx refers to correct indexing) ---
    err_full = np.linalg.norm(uv_measured - uv_projected, axis=1).astype(np.float64)
    finite_full = np.isfinite(err_full)

    total_points = int(np.sum(finite_full))

    # --- select inliers if provided (stats/hist on inliers; n shown as inliers/total) ---
    if inliers_idx is not None:
        idx = np.asarray(inliers_idx, dtype=int).reshape(-1)
        idx = idx[(idx >= 0) & (idx < err_full.size)]
        idx = idx[finite_full[idx]]
        err = err_full[idx]
        err = err[np.isfinite(err)]
        inlier_count = int(err.size)
    else:
        err = err_full[finite_full]
        inlier_count = None  # means "use total only"

    if err.size == 0:
        print("No finite errors to plot.")
        return

    # --- stats (computed on err = inliers if provided, else all) ---
    e_mean = float(np.mean(err))
    e_med = float(np.median(err))
    e_rms = float(np.sqrt(np.mean(err**2)))
    e_p95 = float(np.percentile(err, 95))

    # x-range: show some tail beyond P95 and include threshold if larger
    xmax = max(e_p95 * 1.8, 1e-6)
    tau_px = None
    if ransac_threshold_px is not None:
        try:
            tau_px = float(ransac_threshold_px)
            xmax = max(xmax, tau_px * 1.25)
        except Exception:
            tau_px = None

    # --- histogram ---
    counts, edges = np.histogram(err, bins=bins, range=(0.0, xmax))
    widths = np.diff(edges)
    lefts = edges[:-1]

    # --- LaTeX style (match plane fitting) ---
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
    })

    fig, ax = plt.subplots()

    # draw bars (then recolor by P95 like plane fitting does)
    bars = ax.bar(
        lefts,
        counts,
        width=widths,
        align="edge",
        alpha=0.85,
        edgecolor="none",
    )

    for left, bar in zip(lefts, bars):
        if left >= e_p95:
            bar.set_color("#d62728")  # same red as plane fitting
        else:
            bar.set_color("#1f77b4")  # blue

    # labels
    ax.set_title(title)
    ax.set_xlabel(r"Reprojection error $e\;[\mathrm{px}]$")
    ax.set_ylabel(r"Count")
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.2)

    # --- legend (stats-only), gamma_RANSAC LAST ---
    if inliers_idx is not None:
        n_str = rf"{int(err.size)}/{int(uv_measured.shape[0])}"
    else:
        n_str = rf"{int(err.size)}"

    handles = [ax.plot([], [], " ")[0] for _ in range(6)]
    labels = [
        rf"$n={n_str}$",
        rf"$\bar{{e}}={e_mean:.3f}\,\mathrm{{px}}$",
        rf"$\tilde{{e}}={e_med:.3f}\,\mathrm{{px}}$",
        rf"$\mathrm{{RMS}}={e_rms:.3f}\,\mathrm{{px}}$",
        rf"$P_{{95}}={e_p95:.3f}\,\mathrm{{px}}$",
        (rf"$\tau_{{\mathrm{{RANSAC}}}}={tau_px:.1f}\,\mathrm{{px}}$"
         if tau_px is not None
         else rf"$\gamma_{{\mathrm{{RANSAC}}}}=\mathrm{{n/a}}$"),
    ]
    ax.legend(handles, labels, loc="upper right", frameon=True, borderpad=0.8)

    fig.tight_layout()
    plt.show(block=True)


# ============================================================
# Main
# ============================================================

def main():
    path = _pick_npz_dialog()
    if not path:
        raise RuntimeError("No file selected.")
    path = os.path.abspath(path)

    data = np.load(path, allow_pickle=False)

    # required
    if "uv_measured" not in data or "uv_projected" not in data:
        raise KeyError("NPZ must contain keys: uv_measured and uv_projected")

    uv_meas = data["uv_measured"]
    uv_proj = data["uv_projected"]

    # optional
    inliers_idx = data["inliers_idx"] if "inliers_idx" in data else None
    tau = float(data["ransac_threshold_px"]) if "ransac_threshold_px" in data else None

    print(f"[loaded] {path}")
    print(f"  uv_measured:  {np.asarray(uv_meas).shape}")
    print(f"  uv_projected: {np.asarray(uv_proj).shape}")
    if inliers_idx is not None:
        print(f"  inliers_idx:  {np.asarray(inliers_idx).shape}")
    print(f"  tau:          {tau}")

    show_pnp_error_histogram(
        uv_measured=uv_meas,
        uv_projected=uv_proj,
        inliers_idx=inliers_idx,
        ransac_threshold_px=tau,
        bins=35,
    )


if __name__ == "__main__":
    main()