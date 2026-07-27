"""Constant-velocity Kalman filter in ``(x, y, a, h)`` image space.

State is 8-dimensional: centre ``(x, y)``, aspect ratio ``a = w / h``, height ``h`` and
their velocities. Process and measurement noise are scaled by the current height, which
is what makes the filter behave sensibly for faces that walk from background to
foreground.

This is the standard SORT/ByteTrack formulation, implemented here so the project has no
dependency on a tracking framework.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg

# 0.95 quantile of the chi-square distribution with N degrees of freedom, used for
# gating implausible associations.
CHI2INV95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilterXYAH:
    """8-state constant-velocity filter for axis-aligned boxes."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Noise is proportional to the object height; these weights come from SORT.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    # -- lifecycle ----------------------------------------------------------
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create a track state from an unassociated measurement ``(x, y, a, h)``."""
        mean_pos = np.asarray(measurement, dtype=np.float32)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = np.array(
            [
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                1e-2,
                2 * self._std_weight_position * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                1e-5,
                10 * self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance the state one frame."""
        h = mean[3]
        std_pos = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-2,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )
        std_vel = np.array(
            [
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                1e-5,
                self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def multi_predict(
        self, means: np.ndarray, covariances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised :meth:`predict` for every active track at once."""
        if means.shape[0] == 0:
            return means, covariances
        h = means[:, 3]
        std_pos = np.stack(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                np.full_like(h, 1e-2),
                self._std_weight_position * h,
            ],
            axis=1,
        )
        std_vel = np.stack(
            [
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                np.full_like(h, 1e-5),
                self._std_weight_velocity * h,
            ],
            axis=1,
        )
        sqr = np.square(np.concatenate([std_pos, std_vel], axis=1))
        motion_cov = np.stack([np.diag(sqr[i]) for i in range(sqr.shape[0])], axis=0)

        means = means @ self._motion_mat.T
        covariances = (
            self._motion_mat @ covariances @ self._motion_mat.T + motion_cov
        )
        return means, covariances

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project the state distribution into measurement space."""
        h = mean[3]
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman correction step with measurement ``(x, y, a, h)``."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov.astype(np.float64), lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._update_mat.T).T.astype(np.float64),
            check_finite=False,
        ).T.astype(np.float32)

        innovation = np.asarray(measurement, dtype=np.float32) - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """Squared Mahalanobis distance from the state to each measurement."""
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        cholesky_factor = np.linalg.cholesky(covariance.astype(np.float64))
        d = measurements.astype(np.float64) - mean.astype(np.float64)
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        return np.sum(z * z, axis=0)


__all__ = ["KalmanFilterXYAH", "CHI2INV95"]
