import numpy as np
import cv2

from overlay.tools.homography import estimate_plane_induced_homography
from overlay.tools.warp import blend_xray_overlay


# ============================================================
# PATH
# ============================================================
npz_path = r"overlay_debug_20260323_200400.npz"


# ============================================================
# HELPERS
# ============================================================
def recompute_dx_corrected_mm(T_xc: np.ndarray, T_tc: np.ndarray) -> float:
    """
    T_xc: xray -> camera, translation in m
    T_tc: tip  -> camera, translation in mm

    We want T_tx in mm.
    Therefore convert T_cx translation from m to mm before composition.
    """
    T_xc = np.asarray(T_xc, dtype=np.float64)
    T_tc = np.asarray(T_tc, dtype=np.float64)

    T_cx = np.linalg.inv(T_xc)
    T_cx = T_cx.copy()
    T_cx[:3, 3] *= 1e3   # m -> mm

    T_tx = T_cx @ T_tc
    return float(T_tx[2, 3])


def draw_text(img: np.ndarray, text: str, y: int) -> None:
    cv2.putText(
        img,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# LOAD
# ============================================================
data = np.load(npz_path, allow_pickle=True)

rgb = data["snapshot_rgb_bgr"].copy()
xray = data["xray_gray_u8"].copy()   # alternativ: data["xray_image"]

K_rgb = np.asarray(data["K_rgb"], dtype=np.float64)
K_xray = np.asarray(data["K_xray"], dtype=np.float64)
T_xc = np.asarray(data["T_xc"], dtype=np.float64)
T_tc = np.asarray(data["T_tc"], dtype=np.float64)

d_x_old_mm = float(data["d_x"])

R_xc = T_xc[:3, :3]
t_xc = T_xc[:3, 3]

print("=" * 80)
print("INPUT")
print("=" * 80)
print("old d_x [mm]:", d_x_old_mm)

d_x_new_mm = recompute_dx_corrected_mm(T_xc, T_tc)
print("new d_x [mm]:", d_x_new_mm)
print("delta [mm]:  ", d_x_new_mm - d_x_old_mm)


# ============================================================
# OLD OVERLAY
# ============================================================
H_old = estimate_plane_induced_homography(
    K_c=K_rgb,
    R_xc=R_xc,
    t_xc=t_xc,
    K_x=K_xray,
    d_x=d_x_old_mm,
)

overlay_old = blend_xray_overlay(
    rgb,
    xray,
    H_old,
    alpha=0.5,
)[0]


# ============================================================
# NEW OVERLAY
# ============================================================
H_new = estimate_plane_induced_homography(
    K_c=K_rgb,
    R_xc=R_xc,
    t_xc=t_xc,
    K_x=K_xray,
    d_x=d_x_new_mm,
)

overlay_new = blend_xray_overlay(
    rgb,
    xray,
    H_new,
    alpha=0.5,
)[0]


# ============================================================
# VISUALIZATION
# ============================================================
vis_old = overlay_old.copy()
vis_new = overlay_new.copy()

draw_text(vis_old, f"OLD d_x = {d_x_old_mm:.2f} mm", 40)
draw_text(vis_new, f"NEW d_x = {d_x_new_mm:.2f} mm", 40)
draw_text(vis_new, f"delta = {d_x_new_mm - d_x_old_mm:.2f} mm", 80)

panel = np.hstack([vis_old, vis_new])

cv2.namedWindow("Overlay comparison: old vs new d_x", cv2.WINDOW_NORMAL)
cv2.imshow("Overlay comparison: old vs new d_x", panel)


# ============================================================
# OPTIONAL: DIFFERENCE IMAGE
# ============================================================
diff = cv2.absdiff(overlay_old, overlay_new)
cv2.namedWindow("Difference old vs new", cv2.WINDOW_NORMAL)
cv2.imshow("Difference old vs new", diff)

cv2.waitKey(0)
cv2.destroyAllWindows()