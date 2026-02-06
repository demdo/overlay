# overlay/gui/state.py

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any
import numpy as np


@dataclass
class SessionState:
    K_rgb: Optional[np.ndarray] = None
    K_xray: Optional[np.ndarray] = None
    
    xray_image: Optional[np.ndarray] = None
    xray_image_path: Optional[str] = None
    xray_points_uv: Optional[np.ndarray] = None
    xray_points_confirmed: bool = False
    
    circles_grid: Optional[np.ndarray] = None     # shape (rows, cols, 3) with (x,y,r) per cell
    pick_radius_px: Optional[float] = None
    
    cb_found: bool = False
    cb_corners_uv: Optional[np.ndarray] = None                 # (N,1,2) or (N,2)
    cb_extremes_uv: Optional[np.ndarray] = None                # (3,2) -> [top_left, top_right, bottom_left]
    cb_rect_uv: Optional[Tuple[int, int, int, int]] = None     # (umin, vmin, umax, vmax)

    pts3d_c: Optional[np.ndarray] = None                       # (M,3) in camera frame (meters)
    plane_model_c: Optional[np.ndarray] = None                 # (4,) [a,b,c,d]
    plane_stats: Optional[Dict[str, float]] = None             # keys: n_inliers, mean, median, p95 (units as you choose)
    plane_redo_seed: int = 0 

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
    def has_circles_grid(self) -> bool:
        return self.circles_grid is not None and self.pick_radius_px is not None
    
    @property
    def has_xray_points_confirmed(self) -> bool:
        return self.xray_points_confirmed
    
    @property
    def has_checkerboard_3d(self) -> bool:
        return self.pts3d_c is not None and len(self.pts3d_c) > 0

    @property
    def has_plane_fit(self) -> bool:
        return self.plane_model_c is not None and self.plane_stats is not None
