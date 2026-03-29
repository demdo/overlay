# overlay/calib/calib_camera_to_xray_handeye.py

from __future__ import annotations

import cv2
import numpy as np

from overlay.tracking.transforms import (
    as_transform,
    invert_transform,
    make_transform,
    rvec_tvec_to_transform,
)


# ============================================================
# Method mapping
# ============================================================

_HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _normalize_handeye_method(method: str | int) -> int:
    if isinstance(method, str):
        key = method.strip().lower()
        if key not in _HAND_EYE_METHODS:
            raise ValueError(
                f"Unknown hand-eye method '{method}'. "
                f"Supported: {list(_HAND_EYE_METHODS.keys())}"
            )
        return _HAND_EYE_METHODS[key]
    return int(method)


def pose_vectors_to_board_to_xray(
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """
    Convert an OpenCV PnP solution into T_bx.

    When solvePnP is called with board-frame object points and X-ray image points,
    the returned pose is board -> xray.
    """
    return rvec_tvec_to_transform(rvec, tvec)


def compose_camera_to_xray(
    T_bc: np.ndarray,
    T_bx: np.ndarray,
) -> np.ndarray:
    """
    Compute T_cx from one pose pair.

    T_bc : board -> camera
    T_bx : board -> xray

    T_cx = T_bx @ inv(T_bc)
    """
    T_bc = as_transform(T_bc, "T_bc")
    T_bx = as_transform(T_bx, "T_bx")
    return T_bx @ invert_transform(T_bc)


def calibrate_camera_to_xray_handeye(
    T_bc_list: list[np.ndarray],
    T_bx_list: list[np.ndarray],
    *,
    method: str | int = "park",
    return_inverse: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Estimate the fixed camera -> xray transform using OpenCV hand-eye calibration.

    Convention
    ----------
    T_ab means transform from frame a to frame b

    Inputs
    ------
    T_bc_list
        Per-view board -> camera transforms.
    T_bx_list
        Per-view board -> xray transforms.

    Mapping to OpenCV
    -----------------
    OpenCV expects:
        gripper2base
        target2cam
        and returns cam2gripper

    We map:
        gripper = xray
        base    = board
        target  = board
        cam     = camera

    Therefore:
        gripper2base = T_xb = inv(T_bx)
        target2cam   = T_bc
        cam2gripper  = T_cx
    """
    if len(T_bc_list) != len(T_bx_list):
        raise ValueError(
            f"Length mismatch: len(T_bc_list)={len(T_bc_list)} "
            f"!= len(T_bx_list)={len(T_bx_list)}"
        )

    if len(T_bc_list) < 3:
        raise ValueError("Hand-eye calibration requires at least 3 pose pairs.")

    method_cv2 = _normalize_handeye_method(method)

    R_gripper2base: list[np.ndarray] = []
    t_gripper2base: list[np.ndarray] = []
    R_target2cam: list[np.ndarray] = []
    t_target2cam: list[np.ndarray] = []

    for i, (T_bc, T_bx) in enumerate(zip(T_bc_list, T_bx_list)):
        T_bc = as_transform(T_bc, f"T_bc[{i}]")
        T_bx = as_transform(T_bx, f"T_bx[{i}]")

        T_xb = invert_transform(T_bx)

        R_xb = T_xb[:3, :3].copy()
        t_xb = T_xb[:3, 3:4].copy()

        R_bc = T_bc[:3, :3].copy()
        t_bc = T_bc[:3, 3:4].copy()

        R_gripper2base.append(R_xb)
        t_gripper2base.append(t_xb)
        R_target2cam.append(R_bc)
        t_target2cam.append(t_bc)

    R_cx, t_cx = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method_cv2,
    )

    T_cx = make_transform(R_cx, t_cx)

    if return_inverse:
        T_xc = invert_transform(T_cx)
        return T_cx, T_xc

    return T_cx