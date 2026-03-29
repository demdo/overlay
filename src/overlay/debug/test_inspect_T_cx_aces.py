from __future__ import annotations

from pathlib import Path
import numpy as np


# ============================================================
# Config
# ============================================================

T_CX_FILE = Path(
    r"C:\Users\domin\Documents\Studium\Master\Masterarbeit\Projekt\Overlay\src\overlay\debug\T_cx_debug.npz"
)


# ============================================================
# Helpers
# ============================================================

def as_transform(T: np.ndarray, name: str = "T") -> np.ndarray:
    """
    Validate and normalize a rigid 4x4 homogeneous transform.

    Parameters
    ----------
    T : np.ndarray
        Transform to validate.
    name : str, optional
        Variable name used in error messages.

    Returns
    -------
    np.ndarray
        Float64 transform of shape (4, 4).
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4,4), got {T.shape}")
    return T

def _load_first_existing_array(npz_path: Path, keys: list[str]) -> np.ndarray:
    data = np.load(npz_path)
    for k in keys:
        if k in data:
            return np.asarray(data[k], dtype=np.float64)
    raise KeyError(f"None of the keys {keys} found in: {npz_path}")


def load_T_cx(npz_path: Path) -> np.ndarray:
    T_cx = _load_first_existing_array(
        npz_path,
        keys=["T_cx", "T", "transform", "T_4x4"],
    )
    return as_transform(T_cx, "T_cx")


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError("Zero-length vector.")
    return v / n


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_u = unit(a)
    b_u = unit(b)
    c = float(np.clip(np.dot(a_u, b_u), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def fmt_vec(name: str, v: np.ndarray) -> str:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return f"{name} = [{v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f}]"


# ============================================================
# Main
# ============================================================

def main() -> None:
    T_cx = load_T_cx(T_CX_FILE)
    
    # ===== Z-FIX: Kompensiere 180° Bilddrehung (EINZIGE ÄNDERUNG) =====
    print("\n=== ANWENDUNG Z-FIX ===")
    R_fix = np.diag([1.0, -1.0, -1.0])  # 180° um X (Y+Z Flip)
    T_cx_fixed = T_cx.copy()
    T_cx_fixed[:3, :3] = T_cx_fixed[:3, :3] @ R_fix
    
    R_cx_fixed = T_cx_fixed[:3, :3]
    z_c_in_x_fixed = R_cx_fixed[:, 2]
    
    print("\nT_cx nach Z-FIX:")
    print(T_cx_fixed)

    # Für weitere Nutzung: T_cx_fixed verwenden statt T_cx
    T_cx = T_cx_fixed  # Override für Rest-Skript
    print("\n=== Z-FIX ANGEWENDET - Rest identisch ===\n")
    # ===== ENDE Z-FIX =====

    R_cx = T_cx[:3, :3]
    t_cx = T_cx[:3, 3]

    # Columns of R_cx:
    # x_c, y_c, z_c expressed in xray frame
    x_c_in_x = R_cx[:, 0]
    y_c_in_x = R_cx[:, 1]
    z_c_in_x = R_cx[:, 2]

    x_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_x = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    print("\n==================================================")
    print("Loaded T_cx (camera -> xray)")
    print("==================================================")
    print(T_cx)

    print("\nR_cx =")
    print(R_cx)

    print("\nt_cx [m] =")
    print(t_cx.reshape(3, 1))

    print("\n--------------------------------------------------")
    print("Camera axes expressed in xray frame")
    print("--------------------------------------------------")
    print(fmt_vec("x_c in x", x_c_in_x))
    print(fmt_vec("y_c in x", y_c_in_x))
    print(fmt_vec("z_c in x", z_c_in_x))

    print("\n--------------------------------------------------")
    print("Dot products with xray axes")
    print("--------------------------------------------------")
    print(f"x_c · x_x = {np.dot(unit(x_c_in_x), x_x): .6f}")
    print(f"x_c · y_x = {np.dot(unit(x_c_in_x), y_x): .6f}")
    print(f"x_c · z_x = {np.dot(unit(x_c_in_x), z_x): .6f}")
    print()
    print(f"y_c · x_x = {np.dot(unit(y_c_in_x), x_x): .6f}")
    print(f"y_c · y_x = {np.dot(unit(y_c_in_x), y_x): .6f}")
    print(f"y_c · z_x = {np.dot(unit(y_c_in_x), z_x): .6f}")
    print()
    print(f"z_c · x_x = {np.dot(unit(z_c_in_x), x_x): .6f}")
    print(f"z_c · y_x = {np.dot(unit(z_c_in_x), y_x): .6f}")
    print(f"z_c · z_x = {np.dot(unit(z_c_in_x), z_x): .6f}")
    print(f"z_c · (-z_x) = {np.dot(unit(z_c_in_x), -z_x): .6f}")

    print("\n--------------------------------------------------")
    print("Angles")
    print("--------------------------------------------------")
    print(f"angle(z_c, +z_x)  = {angle_deg(z_c_in_x, z_x): .3f} deg")
    print(f"angle(z_c, -z_x)  = {angle_deg(z_c_in_x, -z_x): .3f} deg")
    print(f"angle(z_c, +y_x)  = {angle_deg(z_c_in_x, y_x): .3f} deg")
    print(f"angle(z_c, -y_x)  = {angle_deg(z_c_in_x, -y_x): .3f} deg")
    print(f"angle(z_c, +x_x)  = {angle_deg(z_c_in_x, x_x): .3f} deg")
    print(f"angle(z_c, -x_x)  = {angle_deg(z_c_in_x, -x_x): .3f} deg")

    print("\n--------------------------------------------------")
    print("Sanity checks")
    print("--------------------------------------------------")
    det_R = np.linalg.det(R_cx)
    ortho_err = np.linalg.norm(R_cx.T @ R_cx - np.eye(3))
    print(f"det(R_cx)         = {det_R:.6f}")
    print(f"||R^T R - I||_F   = {ortho_err:.6e}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()