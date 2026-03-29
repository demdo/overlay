# overlay/calib/calib_xray_to_pointer.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ============================================================
# Helpers
# ============================================================

def _as_transform(T: np.ndarray, name: str) -> np.ndarray:
    """
    Validate and normalize a rigid 4x4 homogeneous transform.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4,4), got {T.shape}")
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    """
    Invert a rigid 4x4 homogeneous transform.
    """
    T = _as_transform(T, "T")

    R = T[:3, :3]
    t = T[:3, 3:4]

    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3:4] = -R.T @ t
    return Tinv


def _transform_points(T: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    """
    Apply a rigid 4x4 homogeneous transform to 3D points.

    Parameters
    ----------
    T : np.ndarray
        Shape (4,4).
    points_xyz : np.ndarray
        Shape (N,3).

    Returns
    -------
    np.ndarray
        Transformed points, shape (N,3).
    """
    T = _as_transform(T, "T")
    points_xyz = np.asarray(points_xyz, dtype=np.float64)

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(
            f"points_xyz must have shape (N,3), got {points_xyz.shape}"
        )

    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_xyz, ones], axis=1)
    points_t_h = (T @ points_h.T).T
    return points_t_h[:, :3]


def _extract_translation_xyz(T: np.ndarray) -> np.ndarray:
    """
    Return the translation vector of a 4x4 transform as shape (3,).
    """
    T = _as_transform(T, "T")
    return T[:3, 3].astype(np.float64).copy()


def _extract_translation_z(T: np.ndarray) -> float:
    """
    Return the z-component of the translation of a 4x4 transform.
    """
    T = _as_transform(T, "T")
    return float(T[2, 3])


def _pose_tip_to_xray(
    T_xc: np.ndarray,
    T_tc: np.ndarray,
) -> np.ndarray:
    """
    Compose the transform from tip frame to xray frame.

    Convention
    ----------
    T_ab means: transform from frame a to frame b

    Therefore:
    - T_xc : xray  -> camera
    - T_tc : tip   -> camera

    We need:
    - T_cx : camera -> xray = inv(T_xc)
    - T_tx : tip    -> xray

    Composition:
        T_tx = T_cx @ T_tc
    """
    T_xc = _as_transform(T_xc, "T_xc")
    T_tc = _as_transform(T_tc, "T_tc")

    T_cx = _invert_T(T_xc)

    # T_xc / T_cx come from cam<->xray calibration in meters,
    # T_tc comes from pointer calibration in millimeters.
    # Bring T_cx translation to millimeters before composition.
    T_cx = T_cx.copy()
    T_cx[:3, 3] *= 1e3   # m -> mm

    T_tx = T_cx @ T_tc
    return T_tx


def _build_plane_patch_corners_x(
    d_x_mm: float,
    *,
    half_size_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an axis-aligned square patch on the target plane z_x = d_x.

    The patch is centered at the point where the X-ray axis intersects the
    target plane, i.e. (0,0,d_x).

    Parameters
    ----------
    d_x_mm : float
        Plane depth in X-ray coordinates.
    half_size_mm : float
        Half side length of the square patch in mm.

    Returns
    -------
    patch_corners_x_mm : np.ndarray
        Shape (4,3), corners in X-ray coordinates.
    axis_point_x_mm : np.ndarray
        Shape (3,), center point on X-ray axis.
    """
    d_x_mm = float(d_x_mm)
    half_size_mm = float(half_size_mm)

    if half_size_mm <= 0:
        raise ValueError("half_size_mm must be > 0.")

    axis_point_x_mm = np.array([0.0, 0.0, d_x_mm], dtype=np.float64)
    L = half_size_mm

    patch_corners_x_mm = np.array(
        [
            [-L, -L, d_x_mm],
            [ L, -L, d_x_mm],
            [ L,  L, d_x_mm],
            [-L,  L, d_x_mm],
        ],
        dtype=np.float64,
    )

    return patch_corners_x_mm, axis_point_x_mm


# ============================================================
# Public API
# ============================================================

@dataclass(frozen=True)
class ExtractDepthResult:
    T_tx: np.ndarray
    tip_xyz_x_mm: np.ndarray
    d_x_mm: float


@dataclass(frozen=True)
class TargetPlaneResult:
    d_x_mm: float

    tip_xyz_x_mm: np.ndarray        # (3,)
    axis_point_x_mm: np.ndarray     # (3,)

    patch_corners_x_mm: np.ndarray  # (4,3)
    patch_corners_c_mm: np.ndarray  # (4,3)


def extract_depth(
    T_xc: np.ndarray,
    T_tc: np.ndarray,
) -> ExtractDepthResult:
    """
    Extract tip depth d_x in the xray frame.

    Parameters
    ----------
    T_xc : np.ndarray
        4x4 transform from xray frame to camera frame.
    T_tc : np.ndarray
        4x4 transform from tip frame to camera frame.

    Returns
    -------
    ExtractDepthResult
        T_tx       : tip -> xray
        tip_xyz_x_mm : tip origin expressed in xray coordinates
        d_x_mm     : extracted depth value in xray frame

    Notes
    -----
    With the convention T_ab = "from a to b" and column-vector transforms,

        p_b = T_ab @ p_a

    we need the tip pose in xray coordinates:

        T_tx = T_cx @ T_tc
             = inv(T_xc) @ T_tc

    Since T_tx maps tip -> xray, its translation is directly the tip origin
    expressed in the xray frame. Therefore:

        tip_xyz_x_mm = translation(T_tx)
        d_x_mm       = tip_xyz_x_mm[2]
    """
    T_xc = _as_transform(T_xc, "T_xc")
    T_tc = _as_transform(T_tc, "T_tc")

    T_tx = _pose_tip_to_xray(T_xc=T_xc, T_tc=T_tc)

    tip_xyz_x_mm = _extract_translation_xyz(T_tx)
    d_x_mm = float(tip_xyz_x_mm[2])

    return ExtractDepthResult(
        T_tx=T_tx,
        tip_xyz_x_mm=tip_xyz_x_mm,
        d_x_mm=d_x_mm,
    )


def build_target_plane(
    T_xc: np.ndarray,
    T_tc: np.ndarray,
    *,
    half_size_mm: float = 20.0,
) -> TargetPlaneResult:
    """
    Build a finite square patch of the target plane used for plane-induced
    homography.

    The target plane is defined in X-ray coordinates as

        z_x = d_x

    i.e. parallel to the image intensifier and orthogonal to the X-ray axis.

    The returned patch is centered on the X-ray axis at (0,0,d_x) and
    aligned with the xray x/y axes.

    Parameters
    ----------
    T_xc : np.ndarray
        4x4 transform from xray frame to camera frame.
    T_tc : np.ndarray
        4x4 transform from tip frame to camera frame.
    half_size_mm : float
        Half side length of the square patch in mm.

    Returns
    -------
    TargetPlaneResult
        d_x_mm            : plane depth in X-ray coordinates
        tip_xyz_x_mm      : tip origin in X-ray coordinates
        axis_point_x_mm   : point (0,0,d_x) on X-ray axis
        patch_corners_x_mm: square patch corners in X-ray coordinates
        patch_corners_c_mm: same corners transformed to camera coordinates
    """
    T_xc = _as_transform(T_xc, "T_xc")
    T_tc = _as_transform(T_tc, "T_tc")

    if half_size_mm <= 0:
        raise ValueError("half_size_mm must be > 0.")

    depth_res = extract_depth(T_xc=T_xc, T_tc=T_tc)

    patch_corners_x_mm, axis_point_x_mm = _build_plane_patch_corners_x(
        d_x_mm=depth_res.d_x_mm,
        half_size_mm=half_size_mm,
    )

    patch_corners_c_mm = _transform_points(T_xc, patch_corners_x_mm)

    return TargetPlaneResult(
        d_x_mm=float(depth_res.d_x_mm),
        tip_xyz_x_mm=depth_res.tip_xyz_x_mm.copy(),
        axis_point_x_mm=axis_point_x_mm,
        patch_corners_x_mm=patch_corners_x_mm,
        patch_corners_c_mm=patch_corners_c_mm,
    )