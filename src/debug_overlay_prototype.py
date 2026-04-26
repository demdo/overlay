from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from overlay.calib.calib_camera_to_xray import calibrate_camera_to_xray
from overlay.tools.homography import estimate_plane_induced_homography
from overlay.tools.warp import blend_xray_overlay


# ============================================================
# Intrinsics loading
# ============================================================

K_KEY_CANDIDATES = (
    "K_xray",
    "K_x",
    "K",
    "K_refined",
    "K_XRAY_REFINED",
)


# ============================================================
# Helpers
# ============================================================

def _ensure_qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def pick_npz_file() -> Path | None:
    _ensure_qt_app()
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select overlay preview NPZ",
        "",
        "NPZ files (*.npz);;All files (*.*)",
    )
    return Path(path) if path else None


def pick_intrinsics_npz_file() -> Path | None:
    _ensure_qt_app()
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select X-ray intrinsics NPZ",
        "",
        "NPZ files (*.npz);;All files (*.*)",
    )
    return Path(path) if path else None


def load_K_xray_from_npz(path: Path) -> tuple[np.ndarray, str]:
    data = np.load(str(path), allow_pickle=True)

    for key in K_KEY_CANDIDATES:
        if key in data.files:
            K = np.asarray(data[key], dtype=np.float64)
            if K.shape != (3, 3):
                raise ValueError(
                    f"Key '{key}' in intrinsics NPZ must have shape (3,3), got {K.shape}"
                )
            return K, key

    raise ValueError(
        "Intrinsics NPZ does not contain a valid K matrix.\n\n"
        f"Expected one of: {list(K_KEY_CANDIDATES)}\n"
        f"Found keys: {data.files}"
    )


def _as_scalar(x, name: str) -> float:
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(f"{name} must be scalar-like, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def _to_uint8_bgr(img: np.ndarray, name: str) -> np.ndarray:
    img = np.asarray(img)

    if img.ndim == 2:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"{name} must be grayscale or BGR image, got shape {img.shape}")

    return img


def _as_xyz(arr: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N,3), got {pts.shape}")
    return pts


def _as_uv(arr: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N,2), got {pts.shape}")
    return pts


def bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(
        img_rgb.data,
        w,
        h,
        bytes_per_line,
        QImage.Format_RGB888,
    )
    return QPixmap.fromImage(qimg.copy())


def rot_x(rx_rad: float) -> np.ndarray:
    c = np.cos(rx_rad)
    s = np.sin(rx_rad)
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0, c, -s],
         [0.0, s, c]],
        dtype=np.float64,
    )


def rot_y(ry_rad: float) -> np.ndarray:
    c = np.cos(ry_rad)
    s = np.sin(ry_rad)
    return np.array(
        [[c, 0.0, s],
         [0.0, 1.0, 0.0],
         [-s, 0.0, c]],
        dtype=np.float64,
    )


def rot_z(rz_rad: float) -> np.ndarray:
    c = np.cos(rz_rad)
    s = np.sin(rz_rad)
    return np.array(
        [[c, -s, 0.0],
         [s,  c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def euler_xyz_deg_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# ============================================================
# Depth recompute
# ============================================================

def recompute_dx(T_xc_m: np.ndarray, T_tc_mm: np.ndarray) -> float:
    T_cx_m = invert_transform(T_xc_m)
    T_cx_mm = T_cx_m.copy()
    T_cx_mm[:3, 3] *= 1e3  # m -> mm
    T_tx = T_cx_mm @ T_tc_mm
    tip_xyz_x_mm = T_tx[:3, 3]
    return float(tip_xyz_x_mm[2])


# ============================================================
# Data container
# ============================================================

class OverlayDebugData:
    def __init__(
        self,
        npz_path: Path,
        *,
        K_xray_ref: np.ndarray,
        K_xray_source: str,
    ):
        self.npz_path = Path(npz_path)

        self.K_xray_ref = np.asarray(K_xray_ref, dtype=np.float64).reshape(3, 3)
        self.K_xray_source = str(K_xray_source)

        data = np.load(str(npz_path), allow_pickle=True)
        keys = set(data.files)

        required = {
            "xray_gray_u8",
            "K_rgb",
            "xray_points_xyz_c",
            "xray_points_uv",
            "checkerboard_corners_uv",
            "T_tc",
        }
        missing = required - keys
        if missing:
            raise ValueError(f"Missing required keys in NPZ: {sorted(missing)}")

        if "snapshot_rgb_with_tip_bgr" in keys:
            self.camera_bgr = _to_uint8_bgr(
                data["snapshot_rgb_with_tip_bgr"],
                "snapshot_rgb_with_tip_bgr",
            )
            self.camera_source_name = "snapshot_rgb_with_tip_bgr"
        elif "snapshot_rgb_bgr" in keys:
            self.camera_bgr = _to_uint8_bgr(
                data["snapshot_rgb_bgr"],
                "snapshot_rgb_bgr",
            )
            self.camera_source_name = "snapshot_rgb_bgr"
        else:
            raise ValueError(
                "NPZ must contain either 'snapshot_rgb_with_tip_bgr' or 'snapshot_rgb_bgr'."
            )

        self.xray_gray_u8 = np.asarray(data["xray_gray_u8"])
        if self.xray_gray_u8.ndim != 2:
            raise ValueError(
                f"xray_gray_u8 must be grayscale, got shape {self.xray_gray_u8.shape}"
            )
        if self.xray_gray_u8.dtype != np.uint8:
            self.xray_gray_u8 = np.clip(self.xray_gray_u8, 0, 255).astype(np.uint8)

        self.K_rgb = np.asarray(data["K_rgb"], dtype=np.float64)
        if self.K_rgb.shape != (3, 3):
            raise ValueError(f"K_rgb must be (3,3), got {self.K_rgb.shape}")

        self.points_xyz_c_m = _as_xyz(data["xray_points_xyz_c"], "xray_points_xyz_c")
        self.points_uv_x = _as_uv(data["xray_points_uv"], "xray_points_uv")
        self.checkerboard_corners_uv = np.asarray(data["checkerboard_corners_uv"], dtype=np.float64)
        self.T_tc_mm = np.asarray(data["T_tc"], dtype=np.float64)

        if self.checkerboard_corners_uv.shape != (3, 2):
            raise ValueError(
                f"checkerboard_corners_uv must have shape (3,2), got {self.checkerboard_corners_uv.shape}"
            )
        if self.T_tc_mm.shape != (4, 4):
            raise ValueError(f"T_tc must be (4,4), got {self.T_tc_mm.shape}")

        self.alpha_nominal = 0.5
        if "alpha" in keys:
            try:
                self.alpha_nominal = float(
                    np.clip(_as_scalar(data["alpha"], "alpha"), 0.0, 1.0)
                )
            except Exception:
                pass


# ============================================================
# Slider widget row
# ============================================================

class SliderRow(QWidget):
    def __init__(
        self,
        title: str,
        minimum: int,
        maximum: int,
        value: int,
        *,
        callback,
        parent=None,
    ):
        super().__init__(parent)

        self.title = QLabel(title)
        self.value_label = QLabel("")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(callback)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.title)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)


# ============================================================
# Main window
# ============================================================

class OverlayDebugWindow(QMainWindow):
    DX_STEP_MM = 0.1
    ALPHA_STEP = 0.01

    T_STEP_MM = 0.1
    T_RANGE_MM = 20.0

    R_STEP_DEG = 0.05
    R_RANGE_DEG = 5.0

    FX_STEP = 1.0
    FY_STEP = 1.0
    C_STEP = 0.5

    F_RANGE = 400.0
    C_RANGE = 150.0

    def __init__(self, debug_data: OverlayDebugData):
        super().__init__()
        self.data = debug_data

        self.setWindowTitle("Debug Overlay Prototype")
        self.resize(1850, 1100)

        self._overlay_cache = None
        self._current_overlay_bgr: np.ndarray | None = None
        self._current_H_xc: np.ndarray | None = None
        self._current_T_xc_mm: np.ndarray | None = None

        self._current_base_result = None
        self._dx_slider_nominal = 0
        self._alpha_slider_nominal = 0

        self.setUpdatesEnabled(False)
        self._build_ui()
        self._init_slider_ranges_static()
        self._recompute_base_from_intrinsics(reset_dx_to_nominal=True)
        self._recompute_overlay()
        self.setUpdatesEnabled(True)

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        left = QVBoxLayout()
        root.addLayout(left, stretch=1)

        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        left.addWidget(self.lbl_info)

        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMinimumSize(1200, 800)
        self.lbl_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_image.setStyleSheet("background: #202020; border: 1px solid #505050;")
        left.addWidget(self.lbl_image, stretch=1)

        right_wrap = QScrollArea()
        right_wrap.setWidgetResizable(True)
        right_wrap.setMinimumWidth(520)
        root.addWidget(right_wrap)

        right = QWidget()
        right_wrap.setWidget(right)

        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(10)

        title = QLabel("Overlay parameter debugging")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        right_layout.addWidget(title)

        intr_title = QLabel("X-ray intrinsics offsets relative to loaded K_xray")
        intr_title.setStyleSheet("font-weight: 600; padding-top: 8px;")
        right_layout.addWidget(intr_title)

        self.row_dfx = SliderRow("Δf_x [px]", 0, 1, 0, callback=self._on_intrinsics_changed)
        self.row_dfy = SliderRow("Δf_y [px]", 0, 1, 0, callback=self._on_intrinsics_changed)
        self.row_dcx = SliderRow("Δc_x [px]", 0, 1, 0, callback=self._on_intrinsics_changed)
        self.row_dcy = SliderRow("Δc_y [px]", 0, 1, 0, callback=self._on_intrinsics_changed)

        right_layout.addWidget(self.row_dfx)
        right_layout.addWidget(self.row_dfy)
        right_layout.addWidget(self.row_dcx)
        right_layout.addWidget(self.row_dcy)

        self.row_dx = SliderRow("d_x [mm]", 0, 1, 0, callback=self._on_geometry_changed)
        right_layout.addWidget(self.row_dx)

        self.row_alpha = SliderRow("alpha", 0, 1, 0, callback=self._on_alpha_changed)
        right_layout.addWidget(self.row_alpha)

        sep1 = QLabel("Pose translation offsets relative to initial T_xc")
        sep1.setStyleSheet("font-weight: 600; padding-top: 8px;")
        right_layout.addWidget(sep1)

        self.row_tx = SliderRow("Δt_x [mm]", 0, 1, 0, callback=self._on_geometry_changed)
        self.row_ty = SliderRow("Δt_y [mm]", 0, 1, 0, callback=self._on_geometry_changed)
        self.row_tz = SliderRow("Δt_z [mm]", 0, 1, 0, callback=self._on_geometry_changed)
        right_layout.addWidget(self.row_tx)
        right_layout.addWidget(self.row_ty)
        right_layout.addWidget(self.row_tz)

        sep2 = QLabel("Pose rotation offsets relative to initial T_xc")
        sep2.setStyleSheet("font-weight: 600; padding-top: 8px;")
        right_layout.addWidget(sep2)

        self.row_rx = SliderRow("Δr_x [deg]", 0, 1, 0, callback=self._on_geometry_changed)
        self.row_ry = SliderRow("Δr_y [deg]", 0, 1, 0, callback=self._on_geometry_changed)
        self.row_rz = SliderRow("Δr_z [deg]", 0, 1, 0, callback=self._on_geometry_changed)
        right_layout.addWidget(self.row_rx)
        right_layout.addWidget(self.row_ry)
        right_layout.addWidget(self.row_rz)

        self.lbl_stats = QLabel("")
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_stats.setStyleSheet(
            "background: #f3f3f3; border: 1px solid #d0d0d0; padding: 10px;"
        )
        right_layout.addWidget(self.lbl_stats)

        btn_grid = QGridLayout()
        right_layout.addLayout(btn_grid)

        self.btn_reset_intrinsics = QPushButton("Reset intrinsics")
        self.btn_reset_intrinsics.clicked.connect(self._reset_intrinsics)
        btn_grid.addWidget(self.btn_reset_intrinsics, 0, 0)

        self.btn_reset_dx = QPushButton("Reset d_x")
        self.btn_reset_dx.clicked.connect(self._reset_dx)
        btn_grid.addWidget(self.btn_reset_dx, 0, 1)

        self.btn_reset_alpha = QPushButton("Reset alpha")
        self.btn_reset_alpha.clicked.connect(self._reset_alpha)
        btn_grid.addWidget(self.btn_reset_alpha, 1, 0)

        self.btn_reset_translation = QPushButton("Reset translation")
        self.btn_reset_translation.clicked.connect(self._reset_translation)
        btn_grid.addWidget(self.btn_reset_translation, 1, 1)

        self.btn_reset_rotation = QPushButton("Reset rotation")
        self.btn_reset_rotation.clicked.connect(self._reset_rotation)
        btn_grid.addWidget(self.btn_reset_rotation, 2, 0)

        self.btn_reset_pose = QPushButton("Reset full pose")
        self.btn_reset_pose.clicked.connect(self._reset_pose)
        btn_grid.addWidget(self.btn_reset_pose, 2, 1)

        self.btn_reset_all = QPushButton("Reset all")
        self.btn_reset_all.clicked.connect(self._reset_all)
        btn_grid.addWidget(self.btn_reset_all, 3, 0)

        self.btn_save_png = QPushButton("Save current overlay as PNG")
        self.btn_save_png.clicked.connect(self._save_png)
        btn_grid.addWidget(self.btn_save_png, 3, 1)

        right_layout.addStretch(1)

    # --------------------------------------------------------
    # Slider ranges
    # --------------------------------------------------------

    def _set_slider_range(
        self,
        row: SliderRow,
        *,
        minimum_value: float,
        maximum_value: float,
        nominal_value: float,
        step: float,
    ) -> tuple[int, int, int]:
        slider_min = int(round(minimum_value / step))
        slider_max = int(round(maximum_value / step))
        slider_nominal = int(round(nominal_value / step))

        row.slider.blockSignals(True)
        row.slider.setRange(slider_min, slider_max)
        row.slider.setSingleStep(1)
        row.slider.setPageStep(10)
        row.slider.setValue(slider_nominal)
        row.slider.blockSignals(False)

        return slider_min, slider_max, slider_nominal

    def _init_slider_ranges_static(self) -> None:
        self._set_slider_range(
            self.row_dfx,
            minimum_value=-self.F_RANGE,
            maximum_value=+self.F_RANGE,
            nominal_value=0.0,
            step=self.FX_STEP,
        )
        self._set_slider_range(
            self.row_dfy,
            minimum_value=-self.F_RANGE,
            maximum_value=+self.F_RANGE,
            nominal_value=0.0,
            step=self.FY_STEP,
        )
        self._set_slider_range(
            self.row_dcx,
            minimum_value=-self.C_RANGE,
            maximum_value=+self.C_RANGE,
            nominal_value=0.0,
            step=self.C_STEP,
        )
        self._set_slider_range(
            self.row_dcy,
            minimum_value=-self.C_RANGE,
            maximum_value=+self.C_RANGE,
            nominal_value=0.0,
            step=self.C_STEP,
        )

        self._set_slider_range(
            self.row_alpha,
            minimum_value=0.0,
            maximum_value=1.0,
            nominal_value=self.data.alpha_nominal,
            step=self.ALPHA_STEP,
        )
        self._alpha_slider_nominal = int(round(self.data.alpha_nominal / self.ALPHA_STEP))

        tmin = -self.T_RANGE_MM
        tmax = +self.T_RANGE_MM
        self._set_slider_range(self.row_tx, minimum_value=tmin, maximum_value=tmax, nominal_value=0.0, step=self.T_STEP_MM)
        self._set_slider_range(self.row_ty, minimum_value=tmin, maximum_value=tmax, nominal_value=0.0, step=self.T_STEP_MM)
        self._set_slider_range(self.row_tz, minimum_value=tmin, maximum_value=tmax, nominal_value=0.0, step=self.T_STEP_MM)

        rmin = -self.R_RANGE_DEG
        rmax = +self.R_RANGE_DEG
        self._set_slider_range(self.row_rx, minimum_value=rmin, maximum_value=rmax, nominal_value=0.0, step=self.R_STEP_DEG)
        self._set_slider_range(self.row_ry, minimum_value=rmin, maximum_value=rmax, nominal_value=0.0, step=self.R_STEP_DEG)
        self._set_slider_range(self.row_rz, minimum_value=rmin, maximum_value=rmax, nominal_value=0.0, step=self.R_STEP_DEG)

    def _update_dx_slider_range(
        self,
        new_nominal_mm: float,
        *,
        preserve_delta_mm: float | None = None,
        reset_to_nominal: bool = False,
    ) -> None:
        if preserve_delta_mm is None:
            preserve_delta_mm = 0.0

        self._set_slider_range(
            self.row_dx,
            minimum_value=0.95 * new_nominal_mm,
            maximum_value=1.05 * new_nominal_mm,
            nominal_value=new_nominal_mm,
            step=self.DX_STEP_MM,
        )
        self._dx_slider_nominal = int(round(new_nominal_mm / self.DX_STEP_MM))

        if reset_to_nominal:
            target_mm = new_nominal_mm
        else:
            target_mm = new_nominal_mm + preserve_delta_mm

        target_slider = int(round(target_mm / self.DX_STEP_MM))
        target_slider = max(self.row_dx.slider.minimum(), min(self.row_dx.slider.maximum(), target_slider))

        self.row_dx.slider.blockSignals(True)
        self.row_dx.slider.setValue(target_slider)
        self.row_dx.slider.blockSignals(False)

    # --------------------------------------------------------
    # Current values
    # --------------------------------------------------------

    @property
    def current_dfx(self) -> float:
        return float(self.row_dfx.slider.value()) * self.FX_STEP

    @property
    def current_dfy(self) -> float:
        return float(self.row_dfy.slider.value()) * self.FY_STEP

    @property
    def current_dcx(self) -> float:
        return float(self.row_dcx.slider.value()) * self.C_STEP

    @property
    def current_dcy(self) -> float:
        return float(self.row_dcy.slider.value()) * self.C_STEP

    @property
    def current_dx_mm(self) -> float:
        return float(self.row_dx.slider.value()) * self.DX_STEP_MM

    @property
    def current_alpha(self) -> float:
        return float(self.row_alpha.slider.value()) * self.ALPHA_STEP

    @property
    def current_tx_mm(self) -> float:
        return float(self.row_tx.slider.value()) * self.T_STEP_MM

    @property
    def current_ty_mm(self) -> float:
        return float(self.row_ty.slider.value()) * self.T_STEP_MM

    @property
    def current_tz_mm(self) -> float:
        return float(self.row_tz.slider.value()) * self.T_STEP_MM

    @property
    def current_rx_deg(self) -> float:
        return float(self.row_rx.slider.value()) * self.R_STEP_DEG

    @property
    def current_ry_deg(self) -> float:
        return float(self.row_ry.slider.value()) * self.R_STEP_DEG

    @property
    def current_rz_deg(self) -> float:
        return float(self.row_rz.slider.value()) * self.R_STEP_DEG

    # --------------------------------------------------------
    # Intrinsics / base pose recomputation
    # --------------------------------------------------------

    def _build_current_K_xray(self) -> np.ndarray:
        K = self.data.K_xray_ref.copy()
        K[0, 0] += self.current_dfx
        K[1, 1] += self.current_dfy
        K[0, 2] += self.current_dcx
        K[1, 2] += self.current_dcy
        return K

    def _recompute_base_from_intrinsics(self, *, reset_dx_to_nominal: bool) -> None:
        old_nominal = getattr(self.data, "d_x_mm_nominal", None)
        old_current_dx = None if old_nominal is None else self.current_dx_mm

        Kx = self._build_current_K_xray()

        result = calibrate_camera_to_xray(
            K_xray=Kx,
            points_xyz_camera=self.data.points_xyz_c_m,
            points_uv_xray=self.data.points_uv_x,
            pose_method="ippe_handeye",
            refine_with_iterative=False,
            ransac_reprojection_error_px=3.0,
            checkerboard_corners_uv=self.data.checkerboard_corners_uv,
            K_rgb=self.data.K_rgb,
            pitch_mm=2.54,
            steps_per_edge=10,
        )

        self.data.K_xray = Kx
        self.data.T_cx_0 = np.asarray(result.T_cx, dtype=np.float64)
        self.data.T_xc_0 = np.asarray(result.T_xc, dtype=np.float64)

        self.data.T_xc_0_mm = self.data.T_xc_0.copy()
        self.data.T_xc_0_mm[:3, 3] *= 1e3

        self.data.R_xc_0 = self.data.T_xc_0_mm[:3, :3].copy()
        self.data.t_xc_0_mm = self.data.T_xc_0_mm[:3, 3].copy()

        self.data.d_x_mm_nominal = recompute_dx(self.data.T_xc_0, self.data.T_tc_mm)

        self.data.reproj_mean_px = float(result.reproj_mean_px)
        self.data.reproj_median_px = float(result.reproj_median_px)
        self.data.reproj_max_px = float(result.reproj_max_px)

        self.data.candidate_index = result.pose_result.candidate_index
        self.data.candidate_index_rgb = result.pose_result.candidate_index_rgb
        self.data.candidate_index_xray = result.pose_result.candidate_index_xray

        if old_nominal is None or old_current_dx is None:
            preserve_delta = 0.0
        else:
            preserve_delta = old_current_dx - old_nominal

        self._update_dx_slider_range(
            self.data.d_x_mm_nominal,
            preserve_delta_mm=preserve_delta,
            reset_to_nominal=reset_dx_to_nominal,
        )

    # --------------------------------------------------------
    # Pose / homography
    # --------------------------------------------------------

    def _build_current_T_xc_mm(self) -> np.ndarray:
        dR = euler_xyz_deg_to_R(
            self.current_rx_deg,
            self.current_ry_deg,
            self.current_rz_deg,
        )

        R_new = dR @ self.data.R_xc_0

        dt_mm = np.array(
            [self.current_tx_mm, self.current_ty_mm, self.current_tz_mm],
            dtype=np.float64,
        )
        t_new_mm = self.data.t_xc_0_mm + dt_mm

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_new
        T[:3, 3] = t_new_mm
        return T

    def _compute_H_xc(self, T_xc_mm: np.ndarray, d_x_mm: float) -> np.ndarray:
        T_xc_m = T_xc_mm.copy()
        T_xc_m[:3, 3] *= 1e-3

        R_xc = T_xc_m[:3, :3]
        t_xc = T_xc_m[:3, 3]

        return estimate_plane_induced_homography(
            K_c=self.data.K_rgb,
            R_xc=R_xc,
            t_xc=t_xc,
            K_x=self.data.K_xray,
            d_x=float(d_x_mm),
        )

    # --------------------------------------------------------
    # Recompute
    # --------------------------------------------------------

    def _recompute_overlay(self) -> None:
        T_xc_mm = self._build_current_T_xc_mm()
        H_xc = self._compute_H_xc(T_xc_mm, self.current_dx_mm)

        out_bgr, cache = blend_xray_overlay(
            camera_bgr=self.data.camera_bgr,
            xray_gray_u8=self.data.xray_gray_u8,
            H_xc=H_xc,
            alpha=self.current_alpha,
        )

        self._current_T_xc_mm = T_xc_mm
        self._current_H_xc = H_xc
        self._current_overlay_bgr = out_bgr
        self._overlay_cache = cache

        self._update_labels()
        self._update_image()

    def _update_alpha_only(self) -> None:
        if self._overlay_cache is None:
            self._recompute_overlay()
            return

        self._current_overlay_bgr = self._overlay_cache.blend(
            self.data.camera_bgr,
            alpha=self.current_alpha,
        )

        self._update_labels()
        self._update_image()

    # --------------------------------------------------------
    # UI updates
    # --------------------------------------------------------

    def _update_labels(self) -> None:
        d0 = self.data.d_x_mm_nominal
        dx = self.current_dx_mm
        dd = dx - d0
        rel = 100.0 * dd / d0 if abs(d0) > 1e-12 else 0.0

        Kx = self.data.K_xray
        Kref = self.data.K_xray_ref

        self.lbl_info.setText(
            f"File: {self.data.npz_path.name}\n"
            f"X-ray intrinsics: {self.data.K_xray_source}\n"
            f"Camera source: {self.data.camera_source_name}   |   "
            f"Camera: {self.data.camera_bgr.shape[1]} x {self.data.camera_bgr.shape[0]}   |   "
            f"X-ray: {self.data.xray_gray_u8.shape[1]} x {self.data.xray_gray_u8.shape[0]}"
        )

        self.row_dfx.value_label.setText(
            f"current = {Kx[0,0]:.2f}   |   reference = {Kref[0,0]:.2f}   |   Δ = {self.current_dfx:+.1f}"
        )
        self.row_dfy.value_label.setText(
            f"current = {Kx[1,1]:.2f}   |   reference = {Kref[1,1]:.2f}   |   Δ = {self.current_dfy:+.1f}"
        )
        self.row_dcx.value_label.setText(
            f"current = {Kx[0,2]:.2f}   |   reference = {Kref[0,2]:.2f}   |   Δ = {self.current_dcx:+.1f}"
        )
        self.row_dcy.value_label.setText(
            f"current = {Kx[1,2]:.2f}   |   reference = {Kref[1,2]:.2f}   |   Δ = {self.current_dcy:+.1f}"
        )

        self.row_dx.value_label.setText(
            f"current = {dx:.1f} mm   |   initial = {d0:.3f} mm   |   Δ = {dd:+.1f} mm ({rel:+.2f}%)"
        )
        self.row_alpha.value_label.setText(f"current = {self.current_alpha:.2f}")

        self.row_tx.value_label.setText(f"current = {self.current_tx_mm:+.1f} mm")
        self.row_ty.value_label.setText(f"current = {self.current_ty_mm:+.1f} mm")
        self.row_tz.value_label.setText(f"current = {self.current_tz_mm:+.1f} mm")

        self.row_rx.value_label.setText(f"current = {self.current_rx_deg:+.2f} deg")
        self.row_ry.value_label.setText(f"current = {self.current_ry_deg:+.2f} deg")
        self.row_rz.value_label.setText(f"current = {self.current_rz_deg:+.2f} deg")

        t0_mm = self.data.t_xc_0_mm
        t_mm = self._current_T_xc_mm[:3, 3]

        stats = [
            "Current debug parameters",
            f"  d_x [mm]      = {self.current_dx_mm:.1f}",
            f"  alpha         = {self.current_alpha:.2f}",
            "",
            "Current K_xray",
            f"  fx            = {Kx[0,0]:.6f}",
            f"  fy            = {Kx[1,1]:.6f}",
            f"  cx            = {Kx[0,2]:.6f}",
            f"  cy            = {Kx[1,2]:.6f}",
            "",
            "Reference K_xray",
            f"  source        = {self.data.K_xray_source}",
            f"  fx            = {Kref[0,0]:.6f}",
            f"  fy            = {Kref[1,1]:.6f}",
            f"  cx            = {Kref[0,2]:.6f}",
            f"  cy            = {Kref[1,2]:.6f}",
            "",
            "Pose offsets",
            f"  Δt_x [mm]     = {self.current_tx_mm:+.1f}",
            f"  Δt_y [mm]     = {self.current_ty_mm:+.1f}",
            f"  Δt_z [mm]     = {self.current_tz_mm:+.1f}",
            f"  Δr_x [deg]    = {self.current_rx_deg:+.2f}",
            f"  Δr_y [deg]    = {self.current_ry_deg:+.2f}",
            f"  Δr_z [deg]    = {self.current_rz_deg:+.2f}",
            "",
            "Initial T_xc [mm]",
            f"  [{t0_mm[0]: .2f}, {t0_mm[1]: .2f}, {t0_mm[2]: .2f}]",
            "Current T_xc [mm]",
            f"  [{t_mm[0]: .2f}, {t_mm[1]: .2f}, {t_mm[2]: .2f}]",
            "",
            "Initial pose recomputation",
            f"  pose method            = ippe_handeye",
            f"  candidate_index        = {self.data.candidate_index}",
            f"  candidate_index_rgb    = {self.data.candidate_index_rgb}",
            f"  candidate_index_xray   = {self.data.candidate_index_xray}",
            f"  reproj mean [px]       = {self.data.reproj_mean_px:.6f}",
            f"  reproj median [px]     = {self.data.reproj_median_px:.6f}",
            f"  reproj max [px]        = {self.data.reproj_max_px:.6f}",
        ]

        if self._current_H_xc is not None:
            H = self._current_H_xc
            stats += [
                "",
                "H_xc",
                f"  [{H[0,0]: .6f}  {H[0,1]: .6f}  {H[0,2]: .3f}]",
                f"  [{H[1,0]: .6f}  {H[1,1]: .6f}  {H[1,2]: .3f}]",
                f"  [{H[2,0]: .8f}  {H[2,1]: .8f}  {H[2,2]: .6f}]",
            ]

        self.lbl_stats.setText("\n".join(stats))

    def _update_image(self) -> None:
        if self._current_overlay_bgr is None:
            return

        pix = bgr_to_qpixmap(self._current_overlay_bgr)
        pix = pix.scaled(
            self.lbl_image.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.lbl_image.setPixmap(pix)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_image()

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    def _on_intrinsics_changed(self, _value: int) -> None:
        self._recompute_base_from_intrinsics(reset_dx_to_nominal=False)
        self._recompute_overlay()

    def _on_geometry_changed(self, _value: int) -> None:
        self._recompute_overlay()

    def _on_alpha_changed(self, _value: int) -> None:
        self._update_alpha_only()

    def _reset_slider_to_zero(self, row: SliderRow) -> None:
        row.slider.blockSignals(True)
        row.slider.setValue(0)
        row.slider.blockSignals(False)

    def _reset_intrinsics(self) -> None:
        for row in [self.row_dfx, self.row_dfy, self.row_dcx, self.row_dcy]:
            self._reset_slider_to_zero(row)
        self._recompute_base_from_intrinsics(reset_dx_to_nominal=True)
        self._recompute_overlay()

    def _reset_dx(self) -> None:
        self.row_dx.slider.blockSignals(True)
        self.row_dx.slider.setValue(self._dx_slider_nominal)
        self.row_dx.slider.blockSignals(False)
        self._recompute_overlay()

    def _reset_alpha(self) -> None:
        self.row_alpha.slider.blockSignals(True)
        self.row_alpha.slider.setValue(self._alpha_slider_nominal)
        self.row_alpha.slider.blockSignals(False)
        self._update_alpha_only()

    def _reset_translation(self) -> None:
        self._reset_slider_to_zero(self.row_tx)
        self._reset_slider_to_zero(self.row_ty)
        self._reset_slider_to_zero(self.row_tz)
        self._recompute_overlay()

    def _reset_rotation(self) -> None:
        self._reset_slider_to_zero(self.row_rx)
        self._reset_slider_to_zero(self.row_ry)
        self._reset_slider_to_zero(self.row_rz)
        self._recompute_overlay()

    def _reset_pose(self) -> None:
        self._reset_translation()
        self._reset_rotation()
        self._recompute_overlay()

    def _reset_all(self) -> None:
        for row in [self.row_dfx, self.row_dfy, self.row_dcx, self.row_dcy]:
            row.slider.blockSignals(True)
            row.slider.setValue(0)
            row.slider.blockSignals(False)

        self.row_alpha.slider.blockSignals(True)
        self.row_tx.slider.blockSignals(True)
        self.row_ty.slider.blockSignals(True)
        self.row_tz.slider.blockSignals(True)
        self.row_rx.slider.blockSignals(True)
        self.row_ry.slider.blockSignals(True)
        self.row_rz.slider.blockSignals(True)

        self.row_alpha.slider.setValue(self._alpha_slider_nominal)
        self.row_tx.slider.setValue(0)
        self.row_ty.slider.setValue(0)
        self.row_tz.slider.setValue(0)
        self.row_rx.slider.setValue(0)
        self.row_ry.slider.setValue(0)
        self.row_rz.slider.setValue(0)

        self.row_alpha.slider.blockSignals(False)
        self.row_tx.slider.blockSignals(False)
        self.row_ty.slider.blockSignals(False)
        self.row_tz.slider.blockSignals(False)
        self.row_rx.slider.blockSignals(False)
        self.row_ry.slider.blockSignals(False)
        self.row_rz.slider.blockSignals(False)

        self._recompute_base_from_intrinsics(reset_dx_to_nominal=True)
        self._recompute_overlay()

    def _save_png(self) -> None:
        if self._current_overlay_bgr is None:
            QMessageBox.information(self, "Save PNG", "No overlay available.")
            return

        out_name = (
            f"{self.data.npz_path.stem}"
            f"__dfx_{self.current_dfx:+.1f}"
            f"__dfy_{self.current_dfy:+.1f}"
            f"__dcx_{self.current_dcx:+.1f}"
            f"__dcy_{self.current_dcy:+.1f}"
            f"__dx_{self.current_dx_mm:.1f}mm"
            f"__tx_{self.current_tx_mm:+.1f}"
            f"__ty_{self.current_ty_mm:+.1f}"
            f"__tz_{self.current_tz_mm:+.1f}"
            f"__rx_{self.current_rx_deg:+.2f}"
            f"__ry_{self.current_ry_deg:+.2f}"
            f"__rz_{self.current_rz_deg:+.2f}"
            f"__a_{self.current_alpha:.2f}.png"
        )

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save current overlay",
            str(self.data.npz_path.with_name(out_name)),
            "PNG files (*.png);;All files (*.*)",
        )
        if not out_path:
            return

        ok = cv2.imwrite(out_path, self._current_overlay_bgr)
        if not ok:
            QMessageBox.critical(self, "Save PNG", "Failed to save PNG.")
            return

        QMessageBox.information(self, "Save PNG", f"Saved:\n{out_path}")


# ============================================================
# Main
# ============================================================

def main() -> int:
    app = _ensure_qt_app()

    if len(sys.argv) > 1:
        npz_path = Path(sys.argv[1])
    else:
        npz_path = pick_npz_file()

    if npz_path is None:
        return 0

    if len(sys.argv) > 2:
        intr_path = Path(sys.argv[2])
    else:
        intr_path = pick_intrinsics_npz_file()

    if intr_path is None:
        return 0

    try:
        K_xray_ref, k_key = load_K_xray_from_npz(intr_path)
        source = f"{intr_path.name}::{k_key}"

        debug_data = OverlayDebugData(
            npz_path=npz_path,
            K_xray_ref=K_xray_ref,
            K_xray_source=source,
        )

        win = OverlayDebugWindow(debug_data)
        win.show()
        return app.exec()

    except Exception as e:
        QMessageBox.critical(None, "Debug Overlay Prototype", str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())