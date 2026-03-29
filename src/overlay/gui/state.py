# overlay/gui/state.py

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SessionState:
    # --- X-ray intrinsics ---
    K_xray: Optional[np.ndarray] = None
    
    # --- Camera calibration ---
    K_rgb: Optional[np.ndarray] = None
    rotate_rgb: bool = False

    # --- X-ray marker selection (interactive -> confirmed) ---
    xray_image: Optional[np.ndarray] = None
    xray_image_path: Optional[str] = None
    xray_points_uv: Optional[np.ndarray] = None
    xray_points_confirmed: bool = False
    xray_marker_overlay_bgr: Optional[np.ndarray] = None
    marker_radius_px: Optional[float] = None

    # --- Plane fitting result (interactive -> confirmed) ---
    xray_points_xyz_c: Optional[np.ndarray] = None
    plane_confirmed: bool = False
    
    # --- PnP diagnostics / settings ---
    pnp_ransac_threshold_px: Optional[float] = None

    # --- Final transformations (homogeneous 4x4) ---
    # Convention:
    #   X_x = T_cx @ X_c   (Camera -> X-ray)
    #   X_c = T_xc @ X_x   (X-ray -> Camera)
    T_cx: Optional[np.ndarray] = None  # Camera -> X-ray
    T_xc: Optional[np.ndarray] = None  # X-ray -> Camera
    
    # --- Accepted pointer-tip snapshot from camera-to-pointer measurement ---
    tip_uv_c: Optional[np.ndarray] = None   # (2,) tip position in RGB image [px]
    tip_xyz_c: Optional[np.ndarray] = None  # (3,) tip position in camera frame [mm]
    
    # --- Pointer pose diagnostics (for debugging d_x issues) ---
    T_tc: Optional[np.ndarray] = None  # tip -> camera
    
    # --- Distance from X-ray source to target plane used in plane-induced homography ---
    d_x: Optional[float] = None
    
    # --- Plane-induced homography for X-ray -> camera overlay ---
    H_xc: Optional[np.ndarray] = None
    
    # ----------------
    # Convenience flags
    # ----------------
    @property
    def has_rgb_intrinsics(self) -> bool:
        return self.K_rgb is not None

    @property
    def has_xray_intrinsics(self) -> bool:
        return self.K_xray is not None

    @property
    def has_xray_image(self) -> bool:
        return self.xray_image is not None

    @property
    def has_xray_points(self) -> bool:
        return self.xray_points_uv is not None and len(self.xray_points_uv) > 0

    @property
    def has_xray_points_confirmed(self) -> bool:
        return self.xray_points_confirmed and self.has_xray_points

    @property
    def has_xray_points_xyz(self) -> bool:
        return self.xray_points_xyz_c is not None and len(self.xray_points_xyz_c) > 0

    @property
    def has_plane_confirmed(self) -> bool:
        return self.plane_confirmed and self.has_xray_points_xyz

    @property
    def has_cam_to_xray(self) -> bool:
        return self.T_cx is not None

    @property
    def has_xray_to_cam(self) -> bool:
        return self.T_xc is not None
    
    @property
    def has_d_x(self) -> bool:
        return self.d_x is not None
    
    @property
    def has_H_xc(self) -> bool:
        return self.H_xc is not None
    
    
    
