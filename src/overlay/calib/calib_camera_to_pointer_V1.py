# -*- coding: utf-8 -*-
"""
calib_camera_to_pointer.py

Camera-to-pointer-tip calibration implementation.

Estimates the rigid pose of the pointer tool tip relative to the camera from a
single image of the planar ArUco marker array mounted on the pointer tool.

Project notation
----------------
Throughout this project we use:

    T_ab = transformation from frame a to frame b

i.e.

    x_b = T_ab @ x_a_h

OpenCV solvePnP convention
--------------------------
solvePnP returns the pose of the OBJECT frame in the CAMERA frame:

    x_c = T_pc @ x_p

where:
- p = pointer marker frame
- c = camera frame

Thus, for this tool, solvePnP directly returns the raw transform:

    T_pc

Goal of this module
-------------------
The desired public pose is the pointer TIP (t)pose in the camera frame (c):

    T_tc = T_pc @ T_tp

where:
- T_tp : tip -> pointer   (fixed, known from tool geometry)

Important pointer-frame convention
----------------------------------
The ArUco IDs on the real tool are arranged as:

    top row:     0   1   2   3
    middle row:  4   5   6   7
    bottom row:  8   9  10  11

The pointer frame is defined such that:
- origin = bottom-left corner of marker ID 8
- +x = to the right
- +y = upwards
- +z = out of the marker plane

Thus, the geometric bottom row is [8, 9, 10, 11], not [0, 1, 2, 3].

The known fixed tip position in the pointer frame is:

    p_tip_in_p = [tip_offset_x_mm, tip_offset_y_mm, tip_offset_z_mm]^T

This defines the fixed transform:

    T_tp
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


# ============================================================
# Internal helpers
# ============================================================

def _normalize_dist_coeffs(dist_coeffs: np.ndarray | None) -> np.ndarray:
    """
    Normalize distortion coefficients to a float64 column vector.

    If None is given, a zero-distortion model is assumed.
    """
    if dist_coeffs is None:
        return np.zeros((5, 1), dtype=np.float64)

    dist = np.asarray(dist_coeffs, dtype=np.float64)
    return dist.reshape(-1, 1)


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert grayscale/BGR/BGRA image to grayscale.
    """
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _build_T_4x4(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Build 4x4 homogeneous transform from rotation and translation.

    Convention:
        x_b = T_ab @ x_a_h
    """
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float64).reshape(3, 1)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T


def _transform_points(T: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transform to 3D points.

    Parameters
    ----------
    T : np.ndarray
        Homogeneous transform, shape (4,4).
    points_xyz : np.ndarray
        3D points, shape (N,3).

    Returns
    -------
    np.ndarray
        Transformed 3D points, shape (N,3).
    """
    T = np.asarray(T, dtype=np.float64)
    points_xyz = np.asarray(points_xyz, dtype=np.float64)

    if T.shape != (4, 4):
        raise ValueError("T must have shape (4,4).")
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N,3).")

    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_xyz, ones], axis=1)
    points_t_h = (T @ points_h.T).T
    return points_t_h[:, :3]


def _rvec_tvec_from_T(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert rigid 4x4 transform to OpenCV (rvec, tvec).
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must have shape (4,4), got {T.shape}")

    R = T[:3, :3]
    t = T[:3, 3:4]
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3, 1), t.reshape(3, 1)


def invert_T(T: np.ndarray) -> np.ndarray:
    """
    Invert a rigid 4x4 homogeneous transform.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must have shape (4,4), got {T.shape}")

    R = T[:3, :3]
    t = T[:3, 3:4]

    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3:4] = -R.T @ t
    return Tinv


def _detect_aruco_markers(
    image: np.ndarray,
    dictionary_name: str,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """
    Detect ArUco markers in the given image and refine their corners with an
    explicit manual cornerSubPix step.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR image.
    dictionary_name : str
        OpenCV ArUco dictionary name, e.g. "DICT_5X5_100".

    Returns
    -------
    corners : list[np.ndarray]
        Detected marker corners in OpenCV format.
    ids : np.ndarray | None
        Detected marker ids, shape (N,1), or None.
    """
    gray = _ensure_gray(image)

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module is not available.")

    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    dict_id = getattr(cv2.aruco, dictionary_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    if hasattr(cv2.aruco, "DetectorParameters"):
        detector_params = cv2.aruco.DetectorParameters()
    else:
        detector_params = cv2.aruco.DetectorParameters_create()

    # Use contour-based refinement, as in Parinaz's older code.
    if hasattr(cv2.aruco, "CORNER_REFINE_CONTOUR"):
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR

    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 50
    detector_params.cornerRefinementMinAccuracy = 0.01

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=detector_params,
        )

    # Apply an additional explicit manual subpixel refinement step.
    # This keeps the function fully compatible with the existing code:
    # - same input signature
    # - same return types/shapes
    if ids is not None and len(corners) > 0:
        # cornerSubPix expects float32 grayscale input.
        gray_f = gray
        if gray_f.dtype != np.uint8:
            gray_f = np.asarray(gray_f, dtype=np.uint8)

        # Use a slightly larger search window than the ArUco default.
        win_size = (7, 7)
        zero_zone = (-1, -1)

        # Stop either after enough iterations or once the update is very small.
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            50,
            0.001,
        )

        refined_corners: list[np.ndarray] = []

        for marker_corners in corners:
            # Preserve OpenCV ArUco corner layout: (1, 4, 2).
            c = np.asarray(marker_corners, dtype=np.float32).reshape(4, 1, 2)

            cv2.cornerSubPix(
                gray_f,
                c,
                win_size,
                zero_zone,
                criteria,
            )

            refined_corners.append(c.reshape(1, 4, 2))

        corners = refined_corners

    return corners, ids


def _marker_id_grid(model: "PointerToolModel") -> np.ndarray:
    """
    Return the fixed marker-ID arrangement of the real pointer tool.

    Rows are ordered from bottom to top in pointer coordinates.
    Columns are ordered from left to right.

    For the current tool:
        bottom row:  8  9 10 11
        middle row:  4  5  6  7
        top row:     0  1  2  3
    """
    if int(model.markers_x) != 4 or int(model.markers_y) != 3:
        raise ValueError(
            "The current fixed pointer-tool ID layout is defined for a 4x3 marker array."
        )

    return np.array(
        [
            [8, 9, 10, 11],
            [4, 5, 6, 7],
            [0, 1, 2, 3],
        ],
        dtype=np.int64,
    )


def _build_marker_object_points_pointer(
    model: "PointerToolModel",
) -> dict[int, np.ndarray]:
    """
    Build marker corner coordinates in the POINTER frame.

    Pointer frame definition
    ------------------------
    - origin: bottom-left corner of marker ID 8
    - +x: to the right
    - +y: upwards
    - +z: out of the marker plane

    Returns
    -------
    dict[int, np.ndarray]
        Mapping marker_id -> (4,3) object corner coordinates in mm.

    Notes
    -----
    Corner order matches OpenCV ArUco:
    [top-left, top-right, bottom-right, bottom-left]
    """
    ml = float(model.marker_length_mm)
    ms = float(model.marker_separation_mm)

    id_grid = _marker_id_grid(model)
    out: dict[int, np.ndarray] = {}

    nrows, ncols = id_grid.shape

    for row in range(nrows):       # row=0 is bottom row in pointer frame
        for col in range(ncols):   # col=0 is leftmost column
            marker_id = int(id_grid[row, col])

            x0 = col * (ml + ms)
            y0 = row * (ml + ms)

            obj = np.array(
                [
                    [x0,      y0 + ml, 0.0],  # top-left
                    [x0 + ml, y0 + ml, 0.0],  # top-right
                    [x0 + ml, y0,      0.0],  # bottom-right
                    [x0,      y0,      0.0],  # bottom-left
                ],
                dtype=np.float64,
            )

            out[marker_id] = obj

    return out


def _build_correspondences_pointer(
    corners: list[np.ndarray],
    ids: np.ndarray,
    model: "PointerToolModel",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 3D-2D marker-corner correspondences from detected ArUco markers.

    Returned 3D points are expressed in the POINTER frame.
    """
    object_points_by_id = _build_marker_object_points_pointer(model)

    object_points_xyz = []
    image_points_uv = []
    marker_ids_used = []

    ids_flat = np.asarray(ids, dtype=np.int64).reshape(-1)

    for marker_corners, marker_id in zip(corners, ids_flat):
        marker_id = int(marker_id)
        if marker_id not in object_points_by_id:
            continue

        obj = object_points_by_id[marker_id]
        img = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)

        object_points_xyz.append(obj)
        image_points_uv.append(img)
        marker_ids_used.append(marker_id)

    if not object_points_xyz:
        raise RuntimeError("No detected marker IDs match the pointer tool model.")

    object_points_xyz = np.concatenate(object_points_xyz, axis=0).astype(np.float64)
    image_points_uv = np.concatenate(image_points_uv, axis=0).astype(np.float64)
    marker_ids_used = np.asarray(marker_ids_used, dtype=np.int64)

    return object_points_xyz, image_points_uv, marker_ids_used


def _build_T_tp(model: "PointerToolModel") -> np.ndarray:
    """
    Build T_tp (tip -> pointer).

    The tip origin expressed in the pointer frame is:
        [tip_offset_x_mm, tip_offset_y_mm, tip_offset_z_mm]^T
    """
    return np.array(
        [
            [1.0, 0.0, 0.0, float(model.tip_offset_x_mm)],
            [0.0, 1.0, 0.0, float(model.tip_offset_y_mm)],
            [0.0, 0.0, 1.0, float(model.tip_offset_z_mm)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _build_T_pt(model: "PointerToolModel") -> np.ndarray:
    """
    Build T_pt (pointer -> tip).
    """
    return invert_T(_build_T_tp(model))


def _rotation_distance_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """
    Return the angular distance between two rotation matrices in degrees.
    """
    R_a = np.asarray(R_a, dtype=np.float64).reshape(3, 3)
    R_b = np.asarray(R_b, dtype=np.float64).reshape(3, 3)

    R_rel = R_a.T @ R_b
    trace = float(np.trace(R_rel))
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.degrees(theta))


def _normalize_pose_init_for_solvepnp(
    rvec_init: np.ndarray | None,
    tvec_init: np.ndarray | None,
    use_extrinsic_guess: bool,
    pointer_model: "PointerToolModel",
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """
    Normalize optional pose initialization for solvePnP.

    Public API convention
    ---------------------
    Incoming (rvec_init, tvec_init) are interpreted as the FINAL public pose:

        T_tc

    But solvePnP expects an initialization for the RAW pose:

        T_pc

    Using:
        T_tc = T_pc @ T_tp

    therefore:
        T_pc = T_tc @ T_pt
    """
    if not use_extrinsic_guess:
        return None, None, False

    if rvec_init is None or tvec_init is None:
        return None, None, False

    rvec_tc = np.asarray(rvec_init, dtype=np.float64).reshape(3, 1)
    tvec_tc = np.asarray(tvec_init, dtype=np.float64).reshape(3, 1)

    R_tc, _ = cv2.Rodrigues(rvec_tc)
    T_tc = _build_T_4x4(R_tc, tvec_tc)

    T_pt = _build_T_pt(pointer_model)
    T_pc = T_tc @ T_pt

    rvec_pc, tvec_pc = _rvec_tvec_from_T(T_pc)
    return rvec_pc, tvec_pc, True


def _solve_pointer_pose_ippe_candidates(
    object_points_pointer_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Return all raw IPPE pose candidates T_pc.

    Returns
    -------
    rvecs_pc : list[(3,1)]
    tvecs_pc : list[(3,1)]
    errs     : np.ndarray shape (M,)
        Reprojection error per candidate.
    """
    success, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
        objectPoints=object_points_pointer_xyz,
        imagePoints=image_points_uv,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not success or rvecs is None or tvecs is None or len(rvecs) == 0:
        raise RuntimeError("IPPE solvePnPGeneric failed.")

    if len(rvecs) != len(tvecs):
        raise RuntimeError("IPPE returned inconsistent numbers of candidates.")

    if reproj_errs is not None and len(reproj_errs) == len(rvecs):
        errs = np.asarray(reproj_errs, dtype=np.float64).reshape(-1)
    else:
        uv_meas = np.asarray(image_points_uv, dtype=np.float64).reshape(-1, 2)
        errs = []

        for rv, tv in zip(rvecs, tvecs):
            rv = np.asarray(rv, dtype=np.float64).reshape(3, 1)
            tv = np.asarray(tv, dtype=np.float64).reshape(3, 1)

            uv_proj, _ = cv2.projectPoints(
                object_points_pointer_xyz,
                rv,
                tv,
                K,
                dist,
            )
            uv_proj = uv_proj.reshape(-1, 2)

            err = np.mean(np.linalg.norm(uv_meas - uv_proj, axis=1))
            errs.append(err)

        errs = np.asarray(errs, dtype=np.float64)

    rvecs_pc = [np.asarray(rv, dtype=np.float64).reshape(3, 1) for rv in rvecs]
    tvecs_pc = [np.asarray(tv, dtype=np.float64).reshape(3, 1) for tv in tvecs]

    return rvecs_pc, tvecs_pc, errs


def _solve_pointer_pose(
    object_points_pointer_xyz: np.ndarray,
    image_points_uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    pose_method: str,
    rvec_init: np.ndarray | None,
    tvec_init: np.ndarray | None,
    use_extrinsic_guess: bool,
    pointer_model: PointerToolModel | None = None,
    tip_point_camera_prev_mm: np.ndarray | None = None,
    prev_rotation_camera: np.ndarray | None = None,
    refine_ippe_with_iterative: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool, int, int | None]:
    """
    Estimate the raw pointer marker pose T_pc from 2D-3D correspondences.

    Parameters
    ----------
    object_points_pointer_xyz:
        Pointer-frame 3D marker corner coordinates, shape (N, 3).
    image_points_uv:
        Corresponding image points, shape (N, 2).
    K:
        Camera intrinsics, shape (3, 3).
    dist:
        Distortion coefficients in OpenCV format.
    pose_method:
        Pose solver to use. Supported:
            - "iterative"
            - "ippe"
    rvec_init, tvec_init:
        Optional initialization for the raw marker pose T_pc.
    use_extrinsic_guess:
        Whether to pass the initialization into the iterative solver.
    pointer_model:
        Required for IPPE tip-based candidate selection.
    tip_point_camera_prev_mm:
        Optional tip position from the previous frame in camera coordinates.
        If provided in IPPE mode, candidate selection is done by choosing the
        solution whose final tip position is closest to this previous tip.
        If not provided, IPPE falls back to reprojection-error-based selection.
    prev_rotation_camera:
        Optional rotation matrix of the final tip pose from the previous frame.
        Used to stabilize IPPE candidate selection.
    refine_ippe_with_iterative:
        If True, refine the selected IPPE candidate with SOLVEPNP_ITERATIVE.

    Returns
    -------
    rvec_pc, tvec_pc, used_extrinsic_guess_final, raw_pnp_flag, ippe_solution_index

    Notes
    -----
    - For the iterative solver, `ippe_solution_index` is always None.
    - For IPPE, all candidates are first generated via solvePnPGeneric.
    - If `tip_point_camera_prev_mm` is available, IPPE selects the candidate
      using tip stability instead of marker reprojection error.
    """
    object_points_pointer_xyz = np.asarray(object_points_pointer_xyz, dtype=np.float64)
    image_points_uv = np.asarray(image_points_uv, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)

    if object_points_pointer_xyz.ndim != 2 or object_points_pointer_xyz.shape[1] != 3:
        raise ValueError("object_points_pointer_xyz must have shape (N, 3).")

    if image_points_uv.ndim != 2 or image_points_uv.shape[1] != 2:
        raise ValueError("image_points_uv must have shape (N, 2).")

    if object_points_pointer_xyz.shape[0] != image_points_uv.shape[0]:
        raise ValueError("object_points_pointer_xyz and image_points_uv must have the same length.")

    if object_points_pointer_xyz.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required for pose estimation.")

    if K.shape != (3, 3):
        raise ValueError("K must have shape (3, 3).")

    pose_method = str(pose_method).lower().strip()

    if pose_method == "iterative":
        if use_extrinsic_guess and rvec_init is not None and tvec_init is not None:
            rvec_init = np.asarray(rvec_init, dtype=np.float64).reshape(3, 1)
            tvec_init = np.asarray(tvec_init, dtype=np.float64).reshape(3, 1)

            success, rvec_pc, tvec_pc = cv2.solvePnP(
                object_points_pointer_xyz,
                image_points_uv,
                K,
                dist,
                rvec=rvec_init,
                tvec=tvec_init,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            used_guess_final = True
        else:
            success, rvec_pc, tvec_pc = cv2.solvePnP(
                object_points_pointer_xyz,
                image_points_uv,
                K,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            used_guess_final = False

        if not success:
            raise RuntimeError("ITERATIVE solvePnP failed to converge.")

        rvec_pc = np.asarray(rvec_pc, dtype=np.float64).reshape(3, 1)
        tvec_pc = np.asarray(tvec_pc, dtype=np.float64).reshape(3, 1)

        # Additional VVS refinement, similar to the older code path you found.
        rvec_pc, tvec_pc = cv2.solvePnPRefineVVS(
            object_points_pointer_xyz,
            image_points_uv,
            K,
            dist,
            rvec_pc,
            tvec_pc,
        )

        return (
            np.asarray(rvec_pc, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec_pc, dtype=np.float64).reshape(3, 1),
            used_guess_final,
            int(cv2.SOLVEPNP_ITERATIVE),
            None,
        )

    elif pose_method == "ippe":
        if pointer_model is None:
            raise ValueError("pointer_model must not be None for pose_method='ippe'.")

        rvecs_pc, tvecs_pc, errs = _solve_pointer_pose_ippe_candidates(
            object_points_pointer_xyz=object_points_pointer_xyz,
            image_points_uv=image_points_uv,
            K=K,
            dist=dist,
        )

        if len(rvecs_pc) == 0:
            raise RuntimeError("IPPE returned no valid candidates.")

        # Build fixed transform from tip frame to pointer frame once.
        T_tp = _build_T_tp(pointer_model)

        best_idx = None
        best_cost = None

        for i, (rv_pc, tv_pc) in enumerate(zip(rvecs_pc, tvecs_pc)):
            R_pc, _ = cv2.Rodrigues(rv_pc)
            T_pc = _build_T_4x4(
                R_pc,
                np.asarray(tv_pc, dtype=np.float64).reshape(3, 1),
            )

            T_tc = T_pc @ T_tp
            tip_cam = T_tc[:3, 3].copy()
            R_tc = T_tc[:3, :3].copy()

            reproj_cost = float(errs[i])

            tip_cost = 0.0
            if tip_point_camera_prev_mm is not None:
                tip_prev = np.asarray(
                    tip_point_camera_prev_mm,
                    dtype=np.float64,
                ).reshape(3,)
                tip_cost = float(np.linalg.norm(tip_cam - tip_prev))

            rot_cost = 0.0
            if prev_rotation_camera is not None:
                R_prev = np.asarray(prev_rotation_camera, dtype=np.float64).reshape(3, 3)
                rot_cost = _rotation_distance_deg(R_prev, R_tc)

            # Combine image consistency and temporal consistency.
            total_cost = reproj_cost + 0.15 * tip_cost + 0.03 * rot_cost

            if best_cost is None or total_cost < best_cost:
                best_cost = total_cost
                best_idx = i

        if best_idx is None:
            best_idx = int(np.argmin(np.asarray(errs, dtype=np.float64)))

        rvec_pc = np.asarray(rvecs_pc[best_idx], dtype=np.float64).reshape(3, 1)
        tvec_pc = np.asarray(tvecs_pc[best_idx], dtype=np.float64).reshape(3, 1)

        # First refine the selected IPPE candidate with iterative solvePnP.
        if refine_ippe_with_iterative:
            success_refine, rvec_refined, tvec_refined = cv2.solvePnP(
                object_points_pointer_xyz,
                image_points_uv,
                K,
                dist,
                rvec=rvec_pc.copy(),
                tvec=tvec_pc.copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if success_refine:
                rvec_pc = np.asarray(rvec_refined, dtype=np.float64).reshape(3, 1)
                tvec_pc = np.asarray(tvec_refined, dtype=np.float64).reshape(3, 1)

        # Then apply VVS refinement as final polishing step.
        rvec_pc, tvec_pc = cv2.solvePnPRefineVVS(
            object_points_pointer_xyz,
            image_points_uv,
            K,
            dist,
            rvec_pc,
            tvec_pc,
        )

        return (
            np.asarray(rvec_pc, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec_pc, dtype=np.float64).reshape(3, 1),
            False,
            int(cv2.SOLVEPNP_IPPE),
            best_idx,
        )

    else:
        raise ValueError(
            f"Unknown pose_method '{pose_method}'. Expected 'iterative' or 'ippe'."
        )


# ============================================================
# Public API
# ============================================================

@dataclass(frozen=True)
class PointerToolModel:
    dictionary_name: str
    markers_x: int
    markers_y: int
    marker_length_mm: float
    marker_separation_mm: float
    first_marker: int
    tip_offset_x_mm: float
    tip_offset_y_mm: float
    tip_offset_z_mm: float


@dataclass(frozen=True)
class CameraToPointerResult:
    # Final public pose: tip -> camera
    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    T_4x4: np.ndarray

    marker_ids_detected: np.ndarray
    marker_ids_used: np.ndarray

    object_points_xyz: np.ndarray
    image_points_uv: np.ndarray

    uv_proj: np.ndarray
    reproj_errors_px: np.ndarray
    reproj_mean_px: float
    reproj_median_px: float
    reproj_max_px: float

    tip_point_tip_mm: np.ndarray
    tip_point_pointer_mm: np.ndarray
    tip_point_camera_mm: np.ndarray
    tip_uv: np.ndarray

    used_extrinsic_guess: bool

    pose_method: str
    raw_pnp_flag: int
    ippe_solution_index: int | None


def get_default_pointer_tool_model() -> PointerToolModel:
    """
    Return the default geometric specification of the pointer tool.

    Note
    ----
    `first_marker` is retained for compatibility, but the actual pointer-frame
    origin is fixed by the real tool layout and lies at marker ID 8.
    """
    return PointerToolModel(
        dictionary_name="DICT_5X5_100",
        markers_x=4,
        markers_y=3,
        marker_length_mm=19.304,
        marker_separation_mm=4.064,
        first_marker=8,
        tip_offset_x_mm=146.304,
        tip_offset_y_mm=33.02,
        tip_offset_z_mm=-1.60,
    )


def calibrate_camera_to_pointer(
    image_bgr: np.ndarray,
    camera_intrinsics: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    pointer_model: PointerToolModel | None = None,
    *,
    rvec_init: np.ndarray | None = None,
    tvec_init: np.ndarray | None = None,
    use_extrinsic_guess: bool = False,
    pose_method: str = "iterative",
    tip_point_camera_prev_mm: np.ndarray | None = None,
    prev_rotation_camera: np.ndarray | None = None,
    refine_ippe_with_iterative: bool = True,
) -> CameraToPointerResult:
    """
    Estimate the rigid pose of the pointer TOOL TIP relative to the camera.

    Final returned pose
    -------------------
    The returned pose components all refer to the final transform:

        T_tc

    such that:

        x_c = T_tc @ x_tip

    Internal computation
    --------------------
    solvePnP is first run on marker corners in the pointer frame, yielding:

        T_pc

    This is then converted to the desired tip pose using the fixed offset:

        T_tc = T_pc @ T_tp

    Pose initialization
    -------------------
    If a previous pose is provided via (rvec_init, tvec_init), it is assumed to
    represent the final public pose T_tc. Internally it is converted back to
    T_pc before being passed to the selected pose solver.
    
    Additional IPPE control
    -----------------------
    prev_rotation_camera:
        Optional rotation matrix of the final tip pose from the previous frame.
        Used for temporally consistent IPPE candidate selection.
    
    refine_ippe_with_iterative:
        If True, the selected IPPE candidate is refined with SOLVEPNP_ITERATIVE.
    """
    if image_bgr is None:
        raise ValueError("image_bgr must not be None.")

    if pointer_model is None:
        pointer_model = get_default_pointer_tool_model()

    K = np.asarray(camera_intrinsics, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("camera_intrinsics must be a 3x3 matrix.")

    dist = _normalize_dist_coeffs(dist_coeffs)

    rvec_init_pc, tvec_init_pc, use_extrinsic_guess_norm = _normalize_pose_init_for_solvepnp(
        rvec_init=rvec_init,
        tvec_init=tvec_init,
        use_extrinsic_guess=use_extrinsic_guess,
        pointer_model=pointer_model,
    )

    corners, ids = _detect_aruco_markers(
        image=image_bgr,
        dictionary_name=pointer_model.dictionary_name,
    )

    if ids is None or len(ids) == 0:
        raise RuntimeError("No ArUco markers detected.")

    object_points_pointer_xyz, image_points_uv, marker_ids_used = _build_correspondences_pointer(
        corners=corners,
        ids=ids,
        model=pointer_model,
    )

    if object_points_pointer_xyz.shape[0] < 4:
        raise RuntimeError("At least 4 correspondences are required for PnP.")

    rvec_pc, tvec_pc, used_guess_final, raw_pnp_flag, ippe_solution_index = _solve_pointer_pose(
        object_points_pointer_xyz=object_points_pointer_xyz,
        image_points_uv=image_points_uv,
        K=K,
        dist=dist,
        pose_method=pose_method,
        rvec_init=rvec_init_pc,
        tvec_init=tvec_init_pc,
        use_extrinsic_guess=use_extrinsic_guess_norm,
        pointer_model=pointer_model,
        tip_point_camera_prev_mm=tip_point_camera_prev_mm,
        prev_rotation_camera=prev_rotation_camera,
        refine_ippe_with_iterative=refine_ippe_with_iterative,
    )

    rotation_pc, _ = cv2.Rodrigues(rvec_pc)
    translation_pc = tvec_pc.reshape(3, 1)
    T_pc = _build_T_4x4(rotation_pc, translation_pc)

    T_tp = _build_T_tp(pointer_model)
    T_pt = _build_T_pt(pointer_model)

    T_tc = T_pc @ T_tp
    rotation_tc = T_tc[:3, :3].copy()
    translation_tc = T_tc[:3, 3:4].copy()

    rvec_tc, tvec_tc = _rvec_tvec_from_T(T_tc)

    object_points_tip_xyz = _transform_points(
        T_pt,
        object_points_pointer_xyz,
    )

    proj, _ = cv2.projectPoints(
        object_points_tip_xyz,
        rvec_tc,
        tvec_tc,
        K,
        dist,
    )
    uv_proj = proj.reshape(-1, 2)

    uv_meas = image_points_uv.reshape(-1, 2)
    reproj_errors = np.linalg.norm(uv_meas - uv_proj, axis=1)
    reproj_mean = float(np.mean(reproj_errors))
    reproj_median = float(np.median(reproj_errors))
    reproj_max = float(np.max(reproj_errors))

    tip_point_tip_mm = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    tip_point_pointer_mm = np.array(
        [
            float(pointer_model.tip_offset_x_mm),
            float(pointer_model.tip_offset_y_mm),
            float(pointer_model.tip_offset_z_mm),
        ],
        dtype=np.float64,
    )

    tip_point_camera_mm = translation_tc.reshape(3,).copy()

    tip_uv, _ = cv2.projectPoints(
        tip_point_tip_mm.reshape(1, 3),
        rvec_tc,
        tvec_tc,
        K,
        dist,
    )
    tip_uv = tip_uv.reshape(2,)

    return CameraToPointerResult(
        rvec=rvec_tc,
        tvec=tvec_tc,
        rotation=rotation_tc,
        translation=translation_tc,
        T_4x4=T_tc,
        marker_ids_detected=ids.reshape(-1).astype(np.int64),
        marker_ids_used=marker_ids_used,
        object_points_xyz=object_points_tip_xyz,
        image_points_uv=image_points_uv,
        uv_proj=uv_proj,
        reproj_errors_px=reproj_errors,
        reproj_mean_px=reproj_mean,
        reproj_median_px=reproj_median,
        reproj_max_px=reproj_max,
        tip_point_tip_mm=tip_point_tip_mm,
        tip_point_pointer_mm=tip_point_pointer_mm,
        tip_point_camera_mm=tip_point_camera_mm,
        tip_uv=tip_uv,
        used_extrinsic_guess=used_guess_final,
        pose_method=str(pose_method).lower().strip(),
        raw_pnp_flag=raw_pnp_flag,
        ippe_solution_index=ippe_solution_index,
    )
