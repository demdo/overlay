# overlay/pose/pose_filters.py

from __future__ import annotations

import numpy as np


def _rotation_angle_deg(R: np.ndarray) -> float:
    """
    Return the angle of a relative rotation matrix in degrees.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.degrees(theta))


def _compute_motion_score(
    *,
    tip_step_mm: float,
    rot_step_deg: float,
    tip_ref_mm: float = 0.8,
    rot_ref_deg: float = 0.4,
    w_tip: float = 0.7,
    w_rot: float = 0.3,
) -> float:
    """
    Compute a normalized motion score in [0, 1].
    """
    motion_tip = np.clip(float(tip_step_mm) / float(tip_ref_mm), 0.0, 1.0)
    motion_rot = np.clip(float(rot_step_deg) / float(rot_ref_deg), 0.0, 1.0)

    motion_score = float(w_tip) * motion_tip + float(w_rot) * motion_rot
    return float(np.clip(motion_score, 0.0, 1.0))


class AdaptiveKalmanFilterCV3D:
    """
    Adaptive constant-velocity Kalman filter for 3D position.

    State
    -----
        x = [px, py, pz, vx, vy, vz]^T

    Measurement
    -----------
        z = [px, py, pz]^T

    Notes
    -----
    The filter adapts Q and R based on frame-to-frame motion estimated from:
    - raw 3D tip displacement
    - raw rotation change
    """

    def __init__(
        self,
        dt: float,
        *,
        q_pos_still: float = 1e-4,
        q_vel_still: float = 1e-2,
        r_still: float = 8e-2,
        q_pos_move: float = 5e-3,
        q_vel_move: float = 3e-1,
        r_move: float = 2e-2,
        tip_ref_mm: float = 0.8,
        rot_ref_deg: float = 0.4,
        w_tip: float = 0.7,
        w_rot: float = 0.3,
    ) -> None:
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt must be > 0.")

        self.q_pos_still = float(q_pos_still)
        self.q_vel_still = float(q_vel_still)
        self.r_still = float(r_still)

        self.q_pos_move = float(q_pos_move)
        self.q_vel_move = float(q_vel_move)
        self.r_move = float(r_move)

        self.tip_ref_mm = float(tip_ref_mm)
        self.rot_ref_deg = float(rot_ref_deg)
        self.w_tip = float(w_tip)
        self.w_rot = float(w_rot)

        self.x = np.zeros((6, 1), dtype=np.float64)

        self.F = np.array(
            [
                [1.0, 0.0, 0.0, self.dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, self.dt, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, self.dt],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        self.Q = np.eye(6, dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1e3
        self.I = np.eye(6, dtype=np.float64)

        self.initialized = False

        self._prev_raw_position_mm: np.ndarray | None = None
        self._prev_raw_rotation: np.ndarray | None = None

        self.last_motion_score: float | None = None
        self.last_tip_step_mm: float | None = None
        self.last_rot_step_deg: float | None = None

    def reset(self) -> None:
        self.x[:] = 0.0
        self.P[:] = np.eye(6, dtype=np.float64) * 1e3
        self.initialized = False

        self._prev_raw_position_mm = None
        self._prev_raw_rotation = None

        self.last_motion_score = None
        self.last_tip_step_mm = None
        self.last_rot_step_deg = None

    def initialize(self, pos_xyz: np.ndarray) -> np.ndarray:
        pos_xyz = np.asarray(pos_xyz, dtype=np.float64).reshape(3)

        self.x[:3, 0] = pos_xyz
        self.x[3:, 0] = 0.0

        self.P = np.eye(6, dtype=np.float64)
        self.P[:3, :3] *= 1.0
        self.P[3:, 3:] *= 10.0

        self.initialized = True
        return self.x[:3, 0].copy()

    def _set_adaptive_noise(self, motion_score: float) -> None:
        m = float(np.clip(motion_score, 0.0, 1.0))

        q_pos = (1.0 - m) * self.q_pos_still + m * self.q_pos_move
        q_vel = (1.0 - m) * self.q_vel_still + m * self.q_vel_move
        r_meas = (1.0 - m) * self.r_still + m * self.r_move

        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float64)
        self.R = np.diag([r_meas, r_meas, r_meas]).astype(np.float64)

    def _compute_internal_motion_score(
        self,
        measurement_mm: np.ndarray,
        rotation_camera: np.ndarray | None,
    ) -> float:
        measurement_mm = np.asarray(measurement_mm, dtype=np.float64).reshape(3)

        if self._prev_raw_position_mm is None:
            tip_step_mm = 0.0
        else:
            tip_step_mm = float(np.linalg.norm(measurement_mm - self._prev_raw_position_mm))

        if self._prev_raw_rotation is None or rotation_camera is None:
            rot_step_deg = 0.0
        else:
            R_curr = np.asarray(rotation_camera, dtype=np.float64).reshape(3, 3)
            R_rel = self._prev_raw_rotation.T @ R_curr
            rot_step_deg = _rotation_angle_deg(R_rel)

        motion_score = _compute_motion_score(
            tip_step_mm=tip_step_mm,
            rot_step_deg=rot_step_deg,
            tip_ref_mm=self.tip_ref_mm,
            rot_ref_deg=self.rot_ref_deg,
            w_tip=self.w_tip,
            w_rot=self.w_rot,
        )

        self.last_tip_step_mm = tip_step_mm
        self.last_rot_step_deg = rot_step_deg
        self.last_motion_score = motion_score

        return motion_score

    def predict(self, motion_score: float) -> np.ndarray | None:
        if not self.initialized:
            return None

        self._set_adaptive_noise(motion_score)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3, 0].copy()

    def update(self, meas_xyz: np.ndarray, motion_score: float) -> np.ndarray:
        z = np.asarray(meas_xyz, dtype=np.float64).reshape(3, 1)

        if not self.initialized:
            pos = self.initialize(z.reshape(3))
            self._set_adaptive_noise(motion_score)
            return pos

        self._set_adaptive_noise(motion_score)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

        return self.x[:3, 0].copy()

    def filter(
        self,
        measurement_mm: np.ndarray,
        rotation_camera: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Full adaptive filtering step:
        - compute frame-to-frame motion from raw input
        - adapt Q and R
        - predict
        - update
        - store current raw input as previous state for the next frame
        """
        measurement_mm = np.asarray(measurement_mm, dtype=np.float64).reshape(3)

        motion_score = self._compute_internal_motion_score(
            measurement_mm=measurement_mm,
            rotation_camera=rotation_camera,
        )

        self.predict(motion_score=motion_score)
        pos_filt = self.update(measurement_mm, motion_score=motion_score)

        self._prev_raw_position_mm = measurement_mm.copy()

        if rotation_camera is None:
            self._prev_raw_rotation = None
        else:
            self._prev_raw_rotation = np.asarray(
                rotation_camera,
                dtype=np.float64,
            ).reshape(3, 3).copy()

        return pos_filt
    
    
class PlaneKalmanFilter:
    """
    Kalman filter for a static plane defined by (a, b, c, d) with ax+by+cz+d=0.

    The plane is assumed static (constant model), so the process noise Q is
    very small. The filter mainly smooths repeated RANSAC estimates of the
    same physical plane.

    The normal (a, b, c) is re-normalised after each update to stay on S².
    Sign consistency is enforced so the normal always points toward the camera
    (negative z-component convention, matching the RealSense coordinate frame).

    Measurements whose normal deviates more than outlier_angle_deg from the
    current state are rejected entirely — the state is returned unchanged.
    This handles rare RANSAC failures that land in a false minimum far from
    the true plane orientation.
    """

    def __init__(
        self,
        *,
        process_noise: float = 1e-7,
        measurement_noise: float = 1e-4,
        outlier_angle_deg: float = 1.5,
    ) -> None:
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.outlier_angle_deg = float(outlier_angle_deg)

        self._state: np.ndarray | None = None   # (4,)
        self._P = np.eye(4, dtype=np.float64)   # covariance

        self._Q = np.eye(4, dtype=np.float64) * self.process_noise
        self._R = np.eye(4, dtype=np.float64) * self.measurement_noise

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._state = None
        self._P = np.eye(4, dtype=np.float64)

    @property
    def is_initialized(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> np.ndarray | None:
        """Return a copy of the current filtered plane (a, b, c, d) or None."""
        return self._state.copy() if self._state is not None else None

    def update(self, plane: np.ndarray) -> np.ndarray:
        """
        Ingest a new RANSAC plane estimate and return the filtered plane.

        Parameters
        ----------
        plane : (4,) array  [a, b, c, d]  (need not be normalised)

        Returns
        -------
        filtered_plane : (4,) array, normalised so that ||(a,b,c)|| = 1
        """
        plane = self._normalise(np.asarray(plane, dtype=np.float64))
        plane = self._enforce_sign(plane)

        if self._state is None:
            self._state = plane.copy()
            self._P = np.eye(4, dtype=np.float64)  # reset covariance on first use
            return self._state.copy()

        # --- Ausreißer-Reject ---
        # Messungen die mehr als outlier_angle_deg vom aktuellen State abweichen
        # werden komplett verworfen — typischerweise RANSAC false minima.
        # Der State bleibt unverändert; P wächst weiter durch Q (predict-only).
        angle = float(np.degrees(np.arccos(
            np.clip(np.dot(plane[:3], self._state[:3]), -1.0, 1.0)
        )))
        if angle > self.outlier_angle_deg:
            print(f"[PlaneKF] Outlier rejected: {angle:.3f}° > {self.outlier_angle_deg}°")
            self._P = self._P + self._Q  # covariance wächst weiter
            return self._state.copy()

        # --- predict (static model: state unchanged, P grows by Q) -------
        P_pred = self._P + self._Q

        # --- update -------------------------------------------------------
        y = plane - self._state                        # innovation
        S = P_pred + self._R                           # innovation covariance
        K = P_pred @ np.linalg.inv(S)                 # Kalman gain

        self._state = self._state + K @ y
        self._P = (np.eye(4, dtype=np.float64) - K) @ P_pred

        # Re-normalise: Kalman update moves the normal off S²
        self._state = self._normalise(self._state)

        return self._state.copy()

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise(plane: np.ndarray) -> np.ndarray:
        """Normalise so ||(a, b, c)|| = 1; d is scaled accordingly."""
        n_norm = float(np.linalg.norm(plane[:3]))
        if n_norm < 1e-9:
            raise ValueError("Plane normal is near-zero — invalid plane.")
        return plane / n_norm

    def _enforce_sign(self, plane: np.ndarray) -> np.ndarray:
        """
        Ensure the normal is sign-consistent with the previous state.

        If no previous state exists, fall back to the RealSense convention:
        the board is in front of the camera, so the normal should point
        *toward* the camera, i.e. n_z < 0 in camera frame.
        """
        if self._state is not None:
            if np.dot(plane[:3], self._state[:3]) < 0.0:
                return -plane
        else:
            if plane[2] > 0.0:   # n_z > 0  →  normal points away from camera
                return -plane
        return plane
    
    
    
