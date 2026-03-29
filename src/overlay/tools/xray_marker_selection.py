# overlay/tools/xray_marker_selection.py
# -*- coding: utf-8 -*-
"""
xray_marker_selection.py

Marker detection + robust ROI extraction from THREE anchor clicks, without any UI
dependencies and without building a circles_grid.

Public API
----------
- run_xray_marker_detection(...)
- compute_roi_from_grid(...)

Design
------
Detection:
    img_gray -> (optional CLAHE) -> (optional detector mask) -> detect_blobs_hough

ROI (robust to any board rotation / shear / perspective):
    Given 3 anchor circles, define a parallelogram in affine coordinates:

        p = p0 + alpha*u + beta*v

    where p0 is the most right-angled of the 3 anchors, and u/v are its two edges.

    Keep points by:
      (A) affine inclusion: alpha,beta within [0..1] with margin
      (B) lattice proximity gate in PIXEL units using pitch from estimate_pitch_nn

Notes
-----
- No circles_sorted, no circles_grid, no row-wise ordering in ROI output.
- estimate_pitch_nn is imported from overlay.tools.blob_detection (as requested).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from overlay.tools.blob_detection import detect_blobs_hough, estimate_pitch_nn


# ============================================================
# Result container (public)
# ============================================================

@dataclass(frozen=True)
class XrayMarkerDetectionResult:
    """
    Result bundle for the marker DETECTION pipeline (no ROI computation).

    Attributes
    ----------
    img_proc : np.ndarray
        Preprocessed image (e.g., CLAHE applied), uint8 (H,W).
    mask : np.ndarray
        Detector mask, uint8 (H,W), values in {0,255}.
    img_masked : np.ndarray
        Masked image fed into blob detection, uint8 (H,W).
    circles : Optional[np.ndarray]
        Circle detections, (N,3) float32 [x,y,r], or None if detection failed.
    """
    img_proc: np.ndarray
    mask: np.ndarray
    img_masked: np.ndarray
    circles: Optional[np.ndarray]


# ============================================================
# Small internal helpers
# ============================================================

def _apply_clahe(
    img_gray_u8: np.ndarray,
    *,
    use_clahe: bool,
    clip_limit: float,
    tile_grid_size: Tuple[int, int],
) -> np.ndarray:
    """Optionally apply CLAHE to a uint8 grayscale image."""
    if not use_clahe:
        return img_gray_u8
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size),
    )
    return clahe.apply(img_gray_u8)


def _detector_mask(
    img_gray: np.ndarray,
    *,
    blur_ks: int = 11,
    thr_mode: str = "otsu",
    adaptive_block: int = 51,
    adaptive_C: int = -5,
    close_ks: int = 41,
    close_iter: int = 1,
) -> np.ndarray:
    """
    Create a broad support mask for the detector region.

    Returns
    -------
    mask : np.ndarray
        uint8 (H,W), 255 inside detector ROI, 0 outside.
    """
    if img_gray.ndim != 2:
        raise ValueError("img_gray must be grayscale (H,W).")

    h, w = img_gray.shape

    k = int(blur_ks)
    k = k if (k % 2 == 1) else (k + 1)
    blur = cv2.GaussianBlur(img_gray, (k, k), 0)

    if thr_mode == "adaptive":
        blk = int(adaptive_block)
        blk = blk if (blk % 2 == 1) else (blk + 1)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blk,
            int(adaptive_C),
        )
    else:
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if float(np.mean(bw)) > 220.0:
        bw = 255 - bw

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.full((h, w), 255, dtype=np.uint8)

    def _touches_border(c: np.ndarray) -> bool:
        xs = c[:, 0, 0]
        ys = c[:, 0, 1]
        return (xs.min() <= 1) or (ys.min() <= 1) or (xs.max() >= w - 2) or (ys.max() >= h - 2)

    nb = [c for c in cnts if not _touches_border(c)]
    candidates = nb if nb else cnts
    largest = max(candidates, key=cv2.contourArea)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

    kc = int(close_ks)
    if kc > 1:
        kc = kc if (kc % 2 == 1) else (kc + 1)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((kc, kc), np.uint8),
            iterations=int(close_iter),
        )

    return mask


def _pick_right_angle_corner(
    pA: np.ndarray,
    pB: np.ndarray,
    pC: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (p0, pu, pv) where p0 is the corner forming the most orthogonal angle
    with the other two points.
    """
    pts = [np.asarray(pA, float), np.asarray(pB, float), np.asarray(pC, float)]

    def _ortho_score(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        v1 = p1 - p0
        v2 = p2 - p0
        n1 = float(np.linalg.norm(v1)) + 1e-12
        n2 = float(np.linalg.norm(v2)) + 1e-12
        return abs(float(np.dot(v1, v2)) / (n1 * n2))  # 0 => orthogonal

    scores = [
        _ortho_score(pts[0], pts[1], pts[2]),
        _ortho_score(pts[1], pts[0], pts[2]),
        _ortho_score(pts[2], pts[0], pts[1]),
    ]
    i0 = int(np.argmin(scores))
    p0 = pts[i0]
    others = [pts[i] for i in range(3) if i != i0]
    return p0, others[0], others[1]


def _solve_alpha_beta(
    P: np.ndarray,
    p0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Solve (alpha,beta) in: P = p0 + alpha*u + beta*v  for each P (Nx2).

    Returns
    -------
    AB : np.ndarray
        (N,2) float64 array, columns [alpha, beta].
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    A = np.stack([u, v], axis=1)  # 2x2 (columns)
    det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    if abs(det) < 1e-9:
        raise ValueError("Degenerate ROI basis (u and v nearly colinear).")

    Ainv = np.linalg.inv(A)
    D = (np.asarray(P, float) - np.asarray(p0, float)[None, :]).T  # 2xN
    return (Ainv @ D).T


def _lattice_gate_keep(
    *,
    alpha: np.ndarray,
    beta: np.ndarray,
    in_box: np.ndarray,
    Lu: float,
    Lv: float,
    pitch: float,
    nu: int,
    nv: int,
    tol_px: float,
) -> np.ndarray:
    """
    Lattice proximity gate in PIXEL units along u and v.

    au = alpha*|u| should be close to ku*(|u|/nu)
    bv = beta*|v|  should be close to kv*(|v|/nv)
    """
    du = Lu / float(nu)
    dv = Lv / float(nv)

    au = alpha * Lu
    bv = beta * Lv

    ku = np.rint(au / du)
    kv = np.rint(bv / dv)

    ru = np.abs(au - ku * du)
    rv = np.abs(bv - kv * dv)

    on = (ru <= tol_px) & (rv <= tol_px)
    return in_box & on


# ============================================================
# Public API
# ============================================================

def run_xray_marker_detection(
    img_gray: np.ndarray,
    *,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_tiles: Tuple[int, int] = (12, 12),
    use_mask: bool = True,
    blur_ks: int = 11,
    thr_mode: str = "otsu",
    adaptive_block: int = 51,
    adaptive_C: int = -5,
    close_ks: int = 41,
    close_iter: int = 1,
    hough_params=None,
) -> XrayMarkerDetectionResult:
    """
    Detection-only pipeline:
        CLAHE -> detector_mask -> mask image -> detect_blobs_hough

    Parameters
    ----------
    img_gray : np.ndarray
        Input grayscale image (H,W).
    use_clahe, clahe_clip, clahe_tiles : preprocessing
        CLAHE parameters.
    use_mask : bool
        If True, suppress non-detector regions using a broad mask.
    blur_ks, thr_mode, adaptive_block, adaptive_C, close_ks, close_iter : mask params
        Parameters for detector mask construction.
    hough_params : any
        Passed through to detect_blobs_hough(..., params=hough_params).

    Returns
    -------
    XrayMarkerDetectionResult
        Contains debug images and circles (N,3) [x,y,r] or None.
    """
    if img_gray.ndim != 2:
        raise ValueError("img_gray must be grayscale (H,W).")

    img_u8 = img_gray if img_gray.dtype == np.uint8 else np.clip(img_gray, 0, 255).astype(np.uint8)
    img_proc = _apply_clahe(
        img_u8,
        use_clahe=use_clahe,
        clip_limit=clahe_clip,
        tile_grid_size=clahe_tiles,
    )

    if use_mask:
        mask = _detector_mask(
            img_proc,
            blur_ks=blur_ks,
            thr_mode=thr_mode,
            adaptive_block=adaptive_block,
            adaptive_C=adaptive_C,
            close_ks=close_ks,
            close_iter=close_iter,
        )
        img_masked = img_proc.copy()
        img_masked[mask == 0] = 0
    else:
        mask = np.full(img_proc.shape, 255, dtype=np.uint8)
        img_masked = img_proc

    circles = detect_blobs_hough(img_masked, params=hough_params)
    if circles is None or len(circles) == 0:
        return XrayMarkerDetectionResult(
            img_proc=img_proc,
            mask=mask,
            img_masked=img_masked,
            circles=None,
        )

    c = np.asarray(circles, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(c[:, 0]) & np.isfinite(c[:, 1]) & np.isfinite(c[:, 2])
    c = c[finite]
    if c.size == 0:
        c = None

    return XrayMarkerDetectionResult(
        img_proc=img_proc,
        mask=mask,
        img_masked=img_masked,
        circles=c,
    )


def compute_roi_from_grid(
    circles: np.ndarray,
    anchor_idx: Sequence[int],
    *,
    margin_px: float,
    gate_tol_pitch: float = 0.40,
    min_steps: int = 2,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute a marker ROI from circle detections using ONLY 3 anchor indices.
    
    The three anchors define a local affine coordinate system
    
        p = p0 + alpha*u + beta*v
    
    where p0 is the most orthogonal corner and u,v span the ROI edges.
    All detected circles are projected into this affine system and filtered by:
    
    1) geometric inclusion within the affine ROI box
    2) lattice proximity using the estimated grid pitch
    
    The remaining points correspond to markers inside the ROI.
    
    Parameters
    ----------
    circles : np.ndarray
        (N,3) array [x,y,r]. Only x,y are used for ROI computation.
    anchor_idx : Sequence[int]
        Exactly three indices into circles defining the ROI anchors.
    margin_px : float
        Geometric margin around the affine [0..1] box in pixels.
    gate_tol_pitch : float
        Lattice gate tolerance in units of pitch (typical 0.35..0.50).
    min_steps : int
        Minimum number of grid steps along each edge (avoids degenerate 1-step ROIs).
    
    Returns
    -------
    roi_uv : np.ndarray
        (M,2) float64 ROI points [u,v] ordered in grid row-major order
        (rows follow the v-direction, columns follow the u-direction).
    roi_idx : np.ndarray
        (M,) int64 indices into the input circles corresponding to roi_uv.
    dbg : dict
        Diagnostics including pitch estimate, ROI dimensions (nu,nv),
        and gating statistics.
    
    Notes
    -----
    The returned ROI points are already ordered according to the inferred
    grid structure. This allows downstream routines (e.g. homography
    correspondence generation) to consume the points without additional
    row/column sorting.
    """
    c = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    if c.shape[0] < 3:
        raise ValueError("circles must contain at least 3 entries.")
    if anchor_idx is None or len(anchor_idx) != 3:
        raise ValueError("anchor_idx must contain exactly 3 indices.")

    idx = np.asarray(anchor_idx, dtype=int)
    if np.any(idx < 0) or np.any(idx >= c.shape[0]):
        raise ValueError("anchor_idx contains out-of-range indices.")

    xy = c[:, :2]
    if not np.isfinite(xy).all():
        finite = np.isfinite(xy).all(axis=1)
        xy = xy[finite]
        map_back = np.flatnonzero(finite)
    else:
        map_back = None

    if xy.shape[0] < 3:
        raise ValueError("Not enough finite circle centers for ROI computation.")

    if map_back is None:
        A = xy[idx, :]
        idx_back = idx
    else:
        inv = {int(old): int(new) for new, old in enumerate(map_back)}
        idx2 = np.array([inv[int(i)] for i in idx], dtype=int)
        A = xy[idx2, :]
        idx_back = map_back[idx2]

    p0, pu, pv = _pick_right_angle_corner(A[0], A[1], A[2])
    u = pu - p0
    v = pv - p0
    
    # ensure u is the more horizontal direction
    if abs(u[0]) < abs(v[0]):
        pu, pv = pv, pu
        u, v = v, u
    
    # remember axis directions (do NOT flip u/v, only fix index orientation later)
    flip_j = (u[0] < 0)   # u points left  -> reverse column index
    flip_i = (v[1] < 0)   # v points up    -> reverse row index

    Lu = float(np.linalg.norm(u))
    Lv = float(np.linalg.norm(v))
    if Lu <= 1e-9 or Lv <= 1e-9:
        raise ValueError("Anchor geometry is degenerate (edges too short).")

    pitch = float(estimate_pitch_nn(xy))
    if (not np.isfinite(pitch)) or pitch <= 1e-6:
        pitch = 10.0

    AB = _solve_alpha_beta(xy, p0, u, v)
    alpha = AB[:, 0]
    beta = AB[:, 1]

    tol_px = float(gate_tol_pitch) * float(pitch)
    mu = float((margin_px + tol_px) / (Lu + 1e-12))
    mv = float((margin_px + tol_px) / (Lv + 1e-12))
    
    in_box = (
        (alpha >= -mu) & (alpha <= 1.0 + mu) &
        (beta  >= -mv) & (beta  <= 1.0 + mv)
    )
    
    nu0 = int(np.clip(np.rint(Lu / pitch), int(min_steps), 10_000))
    nv0 = int(np.clip(np.rint(Lv / pitch), int(min_steps), 10_000))

    best_keep = None
    best = (nu0, nv0, -1)

    for nu in (max(min_steps, nu0 - 1), nu0, nu0 + 1):
        for nv in (max(min_steps, nv0 - 1), nv0, nv0 + 1):
            keep_tmp = _lattice_gate_keep(
                alpha=alpha,
                beta=beta,
                in_box=in_box,
                Lu=Lu,
                Lv=Lv,
                pitch=pitch,
                nu=int(nu),
                nv=int(nv),
                tol_px=tol_px,
            )
            score = int(np.count_nonzero(keep_tmp))
            if score > best[2]:
                best = (int(nu), int(nv), score)
                best_keep = keep_tmp

    keep = best_keep
    if keep is None:
        raise RuntimeError("ROI gating failed unexpectedly.")

    roi_idx_local = np.flatnonzero(keep).astype(np.int64)

    # --- affine coords of ROI points ---
    alpha_roi = alpha[roi_idx_local]
    beta_roi  = beta[roi_idx_local]
    
    # --- discretize to grid indices ---
    nu = int(best[0])
    nv = int(best[1])
    
    j = np.rint(alpha_roi * nu).astype(np.int32)
    i = np.rint(beta_roi  * nv).astype(np.int32)
    
    if flip_j:
        j = nu - j
    if flip_i:
        i = nv - i
    
    # --- row-major ordering ---
    order = np.lexsort((j, i))
    
    roi_idx_local = roi_idx_local[order]
    roi_uv = xy[roi_idx_local].astype(np.float64)
    
    # map indices back to original circles indexing
    if map_back is None:
        roi_idx = roi_idx_local
    else:
        roi_idx = map_back[roi_idx_local].astype(np.int64)

    dbg = dict(
        pitch=float(pitch),
        Lu=float(Lu),
        Lv=float(Lv),
        nu0=int(nu0),
        nv0=int(nv0),
        nu=int(best[0]),
        nv=int(best[1]),
        mu=float(mu),
        mv=float(mv),
        tol_px=float(tol_px),
        in_box=int(np.count_nonzero(in_box)),
        keep=int(np.count_nonzero(keep)),
        gate_tol_pitch=float(gate_tol_pitch),
        anchor_idx=np.asarray(anchor_idx, dtype=int).tolist(),
        anchor_idx_effective=np.asarray(idx_back, dtype=int).tolist(),
        grid_i=i[order].tolist(),
        grid_j=j[order].tolist(),
    )

    return roi_uv, roi_idx, dbg


