# -*- coding: utf-8 -*-
"""
plot_debug_depth_compare.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Load
# ============================================================

def load_data(npz_path):
    data = np.load(npz_path)

    angles = np.asarray(data["angles_deg"], dtype=float)
    dx_iter = np.asarray(data["d_x_iter_mm"], dtype=float)
    dx_ippe = np.asarray(data["d_x_ippe_mm"], dtype=float)
    dx_ippe_kf = np.asarray(data["d_x_ippe_kf_mm"], dtype=float)

    print("mean |dx_iter - dx_ippe|    =", np.mean(np.abs(dx_iter - dx_ippe)))
    print("mean |dx_ippe - dx_ippe_kf| =", np.mean(np.abs(dx_ippe - dx_ippe_kf)))

    return angles, dx_iter, dx_ippe, dx_ippe_kf


def load_extended_data(npz_path_ext):
    data = np.load(npz_path_ext)

    angles = np.asarray(data["angles_deg"], dtype=float)
    dx_iter = np.asarray(data["d_x_iter_mm"], dtype=float)
    dx_ippe = np.asarray(data["d_x_ippe_mm"], dtype=float)
    dx_ippe_kf = np.asarray(data["d_x_ippe_kf_mm"], dtype=float)

    tip_iter_xyz = np.asarray(data["tip_iter_xyz_mm"], dtype=float)
    tip_ippe_xyz = np.asarray(data["tip_ippe_xyz_mm"], dtype=float)
    tip_ippe_kf_xyz = np.asarray(data["tip_ippe_kf_xyz_mm"], dtype=float)

    reproj_iter_px = np.asarray(data["reproj_iter_px"], dtype=float)
    reproj_ippe_px = np.asarray(data["reproj_ippe_px"], dtype=float)

    num_markers_iter = np.asarray(data["num_markers_iter"], dtype=int)
    num_markers_ippe = np.asarray(data["num_markers_ippe"], dtype=int)

    return (
        angles,
        dx_iter,
        dx_ippe,
        dx_ippe_kf,
        tip_iter_xyz,
        tip_ippe_xyz,
        tip_ippe_kf_xyz,
        reproj_iter_px,
        reproj_ippe_px,
        num_markers_iter,
        num_markers_ippe,
    )


# ============================================================
# Binning (Median)
# ============================================================

def binned_median(x, y, bin_width=1.0):
    x = np.asarray(x)
    y = np.asarray(y)

    xmin = np.floor(np.min(x))
    xmax = np.ceil(np.max(x))

    bins = np.arange(xmin, xmax + bin_width, bin_width)

    centers = []
    medians = []

    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])

        if np.any(mask):
            centers.append((bins[i] + bins[i + 1]) / 2)
            medians.append(np.median(y[mask]))

    return np.array(centers), np.array(medians)


# ============================================================
# Plot 1
# ============================================================

def make_plot_boxplot(dx_iter, dx_ippe, dx_ippe_kf):
    fig, ax = plt.subplots(figsize=(8, 6))

    data = [
        dx_iter,
        dx_ippe,
        dx_ippe_kf,
    ]

    labels = [
        "ITERATIVE",
        "IPPE",
        "IPPE + KF",
    ]

    box = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=True,
    )

    # Optional: nicer colors
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Median styling
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_title("Depth estimate distribution (d_x)", fontsize=16)
    ax.set_ylabel(r"$d_x$ [mm]", fontsize=14)

    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    return fig


# ============================================================
# Plot 2: z_tip vs time and d_x vs time
# ============================================================

def make_plot_time(
    dx_iter,
    dx_ippe,
    dx_ippe_kf,
    tip_iter_xyz,
    tip_ippe_xyz,
    tip_ippe_kf_xyz,
):
    t = np.arange(len(dx_iter), dtype=float)

    z_tip_iter = tip_iter_xyz[:, 2]
    z_tip_ippe = tip_ippe_xyz[:, 2]
    z_tip_ippe_kf = tip_ippe_kf_xyz[:, 2]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # -----------------------------------------
    # z_tip vs time
    # -----------------------------------------
    axes[0].plot(
        t,
        z_tip_iter,
        linewidth=1.8,
        label="ITERATIVE z_tip",
    )
    axes[0].plot(
        t,
        z_tip_ippe,
        linewidth=1.8,
        label="IPPE z_tip",
    )
    axes[0].plot(
        t,
        z_tip_ippe_kf,
        linewidth=1.8,
        label="IPPE + KF z_tip",
    )

    axes[0].set_title(r"$z_{tip}$ vs. time", fontsize=15)
    axes[0].set_ylabel(r"$z_{tip}$ [mm]", fontsize=13)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=11)

    # -----------------------------------------
    # d_x vs time
    # -----------------------------------------
    axes[1].plot(
        t,
        dx_iter,
        linewidth=1.8,
        label="ITERATIVE d_x",
    )
    axes[1].plot(
        t,
        dx_ippe,
        linewidth=1.8,
        label="IPPE d_x",
    )
    axes[1].plot(
        t,
        dx_ippe_kf,
        linewidth=1.8,
        label="IPPE + KF d_x",
    )

    axes[1].set_title(r"$d_x$ vs. time", fontsize=15)
    axes[1].set_xlabel("Sample index", fontsize=13)
    axes[1].set_ylabel(r"$d_x$ [mm]", fontsize=13)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=11)

    fig.tight_layout()
    return fig


# ============================================================
# Optional Plot 3: reprojection + used markers
# ============================================================

def make_plot_quality(reproj_iter_px, reproj_ippe_px, num_markers_iter, num_markers_ippe):
    t = np.arange(len(reproj_iter_px), dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(t, reproj_iter_px, linewidth=1.8, label="ITERATIVE reproj")
    axes[0].plot(t, reproj_ippe_px, linewidth=1.8, label="IPPE reproj")
    axes[0].set_title("Reprojection error vs. time", fontsize=15)
    axes[0].set_ylabel("mean reproj [px]", fontsize=13)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=11)

    axes[1].plot(t, num_markers_iter, linewidth=1.8, label="ITERATIVE used markers")
    axes[1].plot(t, num_markers_ippe, linewidth=1.8, label="IPPE used markers")
    axes[1].set_title("Used markers vs. time", fontsize=15)
    axes[1].set_xlabel("Sample index", fontsize=13)
    axes[1].set_ylabel("used markers", fontsize=13)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=11)

    fig.tight_layout()
    return fig


# ============================================================
# Main
# ============================================================

def main():
    npz_path = Path("debug_depth_compare_20260326_184447.npz")
    npz_path_ext = Path("debug_depth_compare_20260326_184447_extended.npz")

    angles, dx_iter, dx_ippe, dx_ippe_kf = load_data(npz_path)

    fig1 = make_plot_boxplot(
        dx_iter=dx_iter,
        dx_ippe=dx_ippe,
        dx_ippe_kf=dx_ippe_kf,
    )

    if npz_path_ext.exists():
        (
            angles_ext,
            dx_iter_ext,
            dx_ippe_ext,
            dx_ippe_kf_ext,
            tip_iter_xyz,
            tip_ippe_xyz,
            tip_ippe_kf_xyz,
            reproj_iter_px,
            reproj_ippe_px,
            num_markers_iter,
            num_markers_ippe,
        ) = load_extended_data(npz_path_ext)

        if len(dx_iter_ext) != len(tip_iter_xyz):
            print("Warning: extended arrays do not have matching lengths.")

        fig2 = make_plot_time(
            dx_iter=dx_iter_ext,
            dx_ippe=dx_ippe_ext,
            dx_ippe_kf=dx_ippe_kf_ext,
            tip_iter_xyz=tip_iter_xyz,
            tip_ippe_xyz=tip_ippe_xyz,
            tip_ippe_kf_xyz=tip_ippe_kf_xyz,
        )

        fig3 = make_plot_quality(
            reproj_iter_px=reproj_iter_px,
            reproj_ippe_px=reproj_ippe_px,
            num_markers_iter=num_markers_iter,
            num_markers_ippe=num_markers_ippe,
        )

    else:
        print(f"Extended file not found: {npz_path_ext}")

    plt.show()


if __name__ == "__main__":
    main()