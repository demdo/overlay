from __future__ import annotations

import sys

import cv2
import numpy as np
import pydicom
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tracking.pose_solvers import solve_pose


# ============================================================
# Config
# ============================================================

N = 11  # first 11 points for detailed printout


# ============================================================
# Qt file dialogs
# ============================================================

def _get_app() -> tuple[QApplication, bool]:
    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True
    return app, owns_app


def _select_file(title: str, name_filter: str) -> str:
    app, owns_app = _get_app()
    dlg = QFileDialog(None, title)
    dlg.setFileMode(QFileDialog.ExistingFile)
    dlg.setNameFilter(name_filter)
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)
    if dlg.exec() != QFileDialog.Accepted:
        if owns_app:
            app.quit()
        raise RuntimeError(f"No file selected for: {title}")
    files = dlg.selectedFiles()
    path = files[0] if files else ""
    if not path:
        if owns_app:
            app.quit()
        raise RuntimeError(f"No file selected for: {title}")
    if owns_app:
        app.quit()
    return path


def _select_npz_file(title: str) -> str:
    return _select_file(title, "NPZ files (*.npz);;All Files (*)")


def _select_xray_image_file() -> str:
    return _select_file(
        "Select X-ray image",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.ima);;All Files (*)",
    )


# ============================================================
# Loaders
# ============================================================

def _find_key(npz, candidates: list[str]) -> str:
    for key in candidates:
        if key in npz.files:
            return key
    raise KeyError(f"None of {candidates} found. Available: {list(npz.files)}")


def _load_pose_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        xyz = np.asarray(npz[_find_key(npz, ["points_xyz", "points_xyz_camera", "xyz"])], dtype=np.float64).reshape(-1, 3)
        uv  = np.asarray(npz[_find_key(npz, ["points_uv",  "points_uv_xray",    "uv" ])], dtype=np.float64).reshape(-1, 2)
    if xyz.shape[0] != uv.shape[0]:
        raise ValueError(f"Point count mismatch: xyz={xyz.shape[0]}, uv={uv.shape[0]}")
    return xyz, uv


def _load_kx_npz(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        K_x = np.asarray(npz[_find_key(npz, ["K_xray", "Kx", "K"])], dtype=np.float64).reshape(3, 3)
    return K_x


def _load_xray_image(path: str) -> np.ndarray:
    if path.lower().endswith((".dcm", ".ima")):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read X-ray image: {path}")
    return img


# ============================================================
# Helpers
# ============================================================

def _rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def _project_points_xray(
    points_xyz_camera: np.ndarray,
    T_cx: np.ndarray,
    K_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_xyz_camera, dtype=np.float64).reshape(-1, 3)
    h   = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    xyz_x = (T_cx @ h.T).T[:, :3]

    X, Y, Z = xyz_x[:, 0], xyz_x[:, 1], xyz_x[:, 2]
    uv = np.full((pts.shape[0], 2), np.nan)
    valid = np.isfinite(Z) & (np.abs(Z) > 1e-12)
    uv[valid, 0] = K_x[0, 0] * (X[valid] / Z[valid]) + K_x[0, 2]
    uv[valid, 1] = K_x[1, 1] * (Y[valid] / Z[valid]) + K_x[1, 2]
    return uv, xyz_x


def _draw_cross(img, u, v, *, color, size=8, thickness=2):
    uu, vv = int(round(u)), int(round(v))
    cv2.line(img, (uu - size, vv), (uu + size, vv), color, thickness, cv2.LINE_AA)
    cv2.line(img, (uu, vv - size), (uu, vv + size), color, thickness, cv2.LINE_AA)


def _draw_labeled_points(img_bgr, pts_uv, *, label_prefix, color, text_offset, n):
    for i in range(min(n, pts_uv.shape[0])):
        u, v = pts_uv[i]
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        _draw_cross(img_bgr, u, v, color=color, size=8, thickness=2)
        cv2.putText(img_bgr, f"{label_prefix}{i}",
                    (int(round(u)) + text_offset[0], int(round(v)) + text_offset[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _sep(title: str = "") -> None:
    if title:
        print(f"\n{'=' * 72}")
        print(title)
        print('=' * 72)
    else:
        print('=' * 72)


# ============================================================
# Main
# ============================================================

def main() -> None:
    _sep("debug_corr  —  comprehensive diagnostic")
    print("Select files in this order:")
    print("  1) pose debug NPZ   (points_xyz + points_uv)")
    print("  2) K_x NPZ          (X-ray intrinsics)")
    print("  3) X-ray image      (.dcm / .ima / .png / ...)")

    pose_path  = _select_npz_file("1) Select pose debug NPZ")
    kx_path    = _select_npz_file("2) Select K_x NPZ")
    xray_path  = _select_xray_image_file()

    xyz, uv_stored = _load_pose_npz(pose_path)
    K_x            = _load_kx_npz(kx_path)
    img            = _load_xray_image(xray_path)
    h_img, w_img   = img.shape[:2]

    # ----------------------------------------------------------------
    # 1. Raw data summary
    # ----------------------------------------------------------------
    _sep("1. RAW DATA SUMMARY")
    print(f"pose NPZ path : {pose_path}")
    print(f"K_x NPZ path  : {kx_path}")
    print(f"X-ray image   : {xray_path}  shape={img.shape}")
    print()
    print(f"N points      : {xyz.shape[0]}")
    print()
    print("K_x:")
    print(K_x)
    print()
    print(f"K_x  fx={K_x[0,0]:.2f}  fy={K_x[1,1]:.2f}  cx={K_x[0,2]:.2f}  cy={K_x[1,2]:.2f}")
    print(f"Image centre  : ({w_img/2:.1f}, {h_img/2:.1f})")
    print(f"cx offset from centre: {K_x[0,2] - w_img/2:+.2f} px")
    print(f"cy offset from centre: {K_x[1,2] - h_img/2:+.2f} px")

    # ----------------------------------------------------------------
    # 2. XYZ statistics
    # ----------------------------------------------------------------
    _sep("2. XYZ POINT STATISTICS  (camera frame of d435i)")
    print(f"  x : min={xyz[:,0].min():.4f}  max={xyz[:,0].max():.4f}  range={xyz[:,0].ptp():.4f} m")
    print(f"  y : min={xyz[:,1].min():.4f}  max={xyz[:,1].max():.4f}  range={xyz[:,1].ptp():.4f} m")
    print(f"  z : min={xyz[:,2].min():.4f}  max={xyz[:,2].max():.4f}  range={xyz[:,2].ptp():.4f} m")
    print()
    print("  z is the depth axis of the d435i (distance from camera).")
    print(f"  => Checkerboard is ~{xyz[:,2].mean():.3f} m from the camera.")
    print()
    print("  First 11 XYZ points:")
    for i in range(min(N, xyz.shape[0])):
        print(f"    {i:3d}: x={xyz[i,0]:8.4f}  y={xyz[i,1]:8.4f}  z={xyz[i,2]:8.4f}")

    # ----------------------------------------------------------------
    # 3. UV statistics
    # ----------------------------------------------------------------
    _sep("3. UV POINT STATISTICS  (X-ray image, RAW coordinates)")
    print(f"  u : min={uv_stored[:,0].min():.1f}  max={uv_stored[:,0].max():.1f}  (image width  = {w_img})")
    print(f"  v : min={uv_stored[:,1].min():.1f}  max={uv_stored[:,1].max():.1f}  (image height = {h_img})")
    print()
    print(f"  u centre of mass: {uv_stored[:,0].mean():.1f}  (image cx = {w_img/2:.1f})")
    print(f"  v centre of mass: {uv_stored[:,1].mean():.1f}  (image cy = {h_img/2:.1f})")
    print()
    print("  First 11 UV points:")
    for i in range(min(N, uv_stored.shape[0])):
        print(f"    {i:3d}: u={uv_stored[i,0]:8.2f}  v={uv_stored[i,1]:8.2f}")

    # ----------------------------------------------------------------
    # 4. Grid ordering check
    # ----------------------------------------------------------------
    _sep("4. GRID ORDERING CHECK")
    print("  Do consecutive XYZ points move in the same direction as UV points?")
    print()
    print("  XYZ delta (camera frame):")
    for i in range(min(5, xyz.shape[0] - 1)):
        dx = xyz[i+1,0] - xyz[i,0]
        dy = xyz[i+1,1] - xyz[i,1]
        dz = xyz[i+1,2] - xyz[i,2]
        print(f"    {i}->{i+1}: dx={dx:+.4f}  dy={dy:+.4f}  dz={dz:+.4f}")
    print()
    print("  UV delta (X-ray image):")
    for i in range(min(5, uv_stored.shape[0] - 1)):
        du = uv_stored[i+1,0] - uv_stored[i,0]
        dv = uv_stored[i+1,1] - uv_stored[i,1]
        print(f"    {i}->{i+1}: du={du:+.2f}  dv={dv:+.2f}")
    print()
    print("  Interpretation:")
    print("  - XYZ x grows → camera-right.  UV u grows → image-right.")
    print("  - XYZ y grows → camera-down.   UV v grows → image-down.")
    print("  - Both should increase in the same physical direction.")

    # ----------------------------------------------------------------
    # 5. Solve PnP
    # ----------------------------------------------------------------
    _sep("5. PNP  (RANSAC + iterative refinement)")

    pose = solve_pose(
        object_points_xyz=xyz,
        image_points_uv=uv_stored,
        K=K_x,
        dist_coeffs=None,
        pose_method="iterative_ransac",
        refine_with_iterative=True,
        ransac_reprojection_error_px=8.0,
        ransac_confidence=0.99,
        ransac_iterations_count=100,
    )

    T_cx = _rvec_tvec_to_transform(pose.rvec, pose.tvec)
    R_cx = T_cx[:3, :3]
    t_cx = T_cx[:3, 3]

    print(f"  reproj mean   : {pose.reproj_mean_px:.3f} px")
    print(f"  reproj median : {pose.reproj_median_px:.3f} px")
    print(f"  reproj max    : {pose.reproj_max_px:.3f} px")
    print(f"  num inliers   : {len(pose.inlier_idx)} / {xyz.shape[0]}")
    print()
    print("  T_cx  (camera → xray frame):")
    print(T_cx)

    # ----------------------------------------------------------------
    # 6. Translation analysis
    # ----------------------------------------------------------------
    _sep("6. TRANSLATION ANALYSIS")
    print("  t_cx = position of camera origin in X-ray frame")
    print()
    print(f"  t_cx = [{t_cx[0]:+.4f},  {t_cx[1]:+.4f},  {t_cx[2]:+.4f}]  m")
    print()
    print(f"  |t_cx|            = {np.linalg.norm(t_cx):.4f} m")
    print(f"  t_cx_z            = {t_cx[2]:+.4f} m  (along z_x = optical axis of X-ray)")
    print()
    print("  Physical expectation:")
    print("  - z_x points from Source toward Intensifier")
    print("  - Camera is at the Intensifier  =>  t_cx_z should be ~ +SDD (~0.98 m)")
    print(f"  - Your SDD = ~0.98 m,  t_cx_z = {t_cx[2]:.4f} m")
    print()
    if abs(t_cx[2] - 0.98) < 0.15:
        print("  => t_cx_z is consistent with SDD  ✓")
    elif abs(t_cx[2]) < 0.5:
        print("  => t_cx_z is much too small — likely the POSE IS WRONG")
        print("     Could indicate: wrong K_x, wrong point correspondences,")
        print("     or the 3D points are not in the camera frame.")
    else:
        print("  => t_cx_z is in an unexpected range — check setup.")

    # ----------------------------------------------------------------
    # 7. Rotation / axis analysis
    # ----------------------------------------------------------------
    _sep("7. ROTATION / AXIS ANALYSIS")

    x_c = R_cx[:, 0]
    y_c = R_cx[:, 1]
    z_c = R_cx[:, 2]

    z_x = np.array([0., 0., 1.])
    y_x = np.array([0., 1., 0.])
    x_x = np.array([1., 0., 0.])

    dot_zz = float(np.dot(z_c, z_x))
    dot_zy = float(np.dot(z_c, y_x))
    dot_zx = float(np.dot(z_c, x_x))

    print("  Camera axes expressed in X-ray frame:")
    print(f"    x_c = [{x_c[0]:+.4f}, {x_c[1]:+.4f}, {x_c[2]:+.4f}]")
    print(f"    y_c = [{y_c[0]:+.4f}, {y_c[1]:+.4f}, {y_c[2]:+.4f}]")
    print(f"    z_c = [{z_c[0]:+.4f}, {z_c[1]:+.4f}, {z_c[2]:+.4f}]")
    print()
    print(f"  dot(z_c, z_x) = {dot_zz:+.4f}   (1=same dir, -1=opposite)")
    print(f"  dot(z_c, y_x) = {dot_zy:+.4f}")
    print(f"  dot(z_c, x_x) = {dot_zx:+.4f}")
    print()
    print("  Physical expectation:")
    print("  - z_x: Source → Intensifier (upward)")
    print("  - z_c: Camera → Table (downward, slightly left)")
    print("  => dot(z_c, z_x) should be NEGATIVE (~-0.9 to -1.0)")
    print()
    if dot_zz < -0.7:
        print(f"  => dot = {dot_zz:.4f}  ✓  z axes are correctly opposed")
    elif dot_zz > 0.7:
        print(f"  => dot = {dot_zz:.4f}  ✗  z axes point the SAME way — T_cx is wrong!")
        print("     Most likely cause: K_x and UV points are from different")
        print("     image coordinate systems (one RAW, one rotated).")
    else:
        print(f"  => dot = {dot_zz:.4f}  ?  z axes are roughly orthogonal — unexpected")

    # ----------------------------------------------------------------
    # 8. Points in X-ray frame
    # ----------------------------------------------------------------
    _sep("8. CHECKERBOARD POINTS IN X-RAY FRAME")

    uv_proj, xyz_x = _project_points_xray(xyz, T_cx, K_x)

    print("  z_x values of checkerboard points (should be ~ d_x = distance PCB from source):")
    z_vals = xyz_x[:, 2]
    print(f"  z_x mean   = {z_vals.mean():.4f} m")
    print(f"  z_x min    = {z_vals.min():.4f} m")
    print(f"  z_x max    = {z_vals.max():.4f} m")
    print(f"  z_x std    = {z_vals.std():.4f} m  (should be ~0 for flat board)")
    print()
    print("  If z_x mean ≈ physical d_x (PCB-to-source distance) => translation correct ✓")
    print("  If z_x mean ≈ camera-to-PCB distance (~0.44 m)      => T_cx is inverted/wrong ✗")

    # ----------------------------------------------------------------
    # 9. Reprojection detail
    # ----------------------------------------------------------------
    _sep("9. REPROJECTION DETAIL  (first 11 points)")

    reproj_errors = np.linalg.norm(uv_proj - uv_stored, axis=1)

    print(f"  {'i':>3}  {'u_stored':>10}  {'v_stored':>10}  {'u_proj':>10}  {'v_proj':>10}  {'err_px':>8}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for i in range(min(N, xyz.shape[0])):
        us, vs = uv_stored[i]
        up, vp = uv_proj[i]
        e = reproj_errors[i]
        print(f"  {i:3d}  {us:10.2f}  {vs:10.2f}  {up:10.2f}  {vp:10.2f}  {e:8.3f}")

    print()
    print(f"  All points — reproj error [px]:")
    print(f"    mean   = {reproj_errors.mean():.3f}")
    print(f"    median = {float(np.median(reproj_errors)):.3f}")
    print(f"    max    = {reproj_errors.max():.3f}")
    print(f"    p95    = {float(np.percentile(reproj_errors, 95)):.3f}")

    # ----------------------------------------------------------------
    # 10. Visualization
    # ----------------------------------------------------------------
    _sep("10. VISUALIZATION")
    print("  RED    = stored UV  (from NPZ)")
    print("  YELLOW = projected XYZ via T_cx + K_x")
    print("  Numbers show point index.")
    print()
    print("  Check: do RED and YELLOW overlap?")
    print("  If yes but z-axis is wrong → T_cx is a valid solution but in wrong convention.")
    print()
    print("  Press any key in the image window to close.")

    img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    _draw_labeled_points(img_vis, uv_stored[:N],  label_prefix="s", color=(0, 0, 255),   text_offset=(6, -6),  n=N)
    _draw_labeled_points(img_vis, uv_proj[:N],    label_prefix="p", color=(0, 255, 255),  text_offset=(6,  14), n=N)

    # draw line between stored and projected to show residual
    for i in range(min(N, xyz.shape[0])):
        us, vs = uv_stored[i]
        up, vp = uv_proj[i]
        if all(np.isfinite([us, vs, up, vp])):
            cv2.line(img_vis,
                     (int(round(us)), int(round(vs))),
                     (int(round(up)), int(round(vp))),
                     (0, 165, 255), 1, cv2.LINE_AA)

    cv2.namedWindow("debug_corr", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("debug_corr", min(w_img, 1000), min(h_img, 1000))
    cv2.imshow("debug_corr", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()