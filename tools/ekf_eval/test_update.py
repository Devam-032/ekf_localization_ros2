"""Offline check of the EKF association + batch update (no ROS spin needed)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "ekf_localization"))
from ekf_localization.ekf_localization import EKF_LOCALIZATION, normalize_angle  # noqa: E402

LANDMARKS = [[1.3014, -1.74091], [1.98771, 1.13637], [-1.13443, -0.902541], [-1.80846, 0.902126]]
truth = np.array([1.0, 0.5, 0.2])

f = EKF_LOCALIZATION.__new__(EKF_LOCALIZATION)       # skip the ROS constructor
f.sigma_r, f.sigma_alpha, f.gate_range, f.gate_bearing = 0.1, 0.035, 0.4, 0.25
f.reference_cylin = LANDMARKS
f.x, f.y, f.theta = truth                             # only used for logging fields
f.covariance = np.diag([0.05, 0.05, 0.02])

# Prediction off by (+0.25, -0.2, +0.08 rad)
pred = truth + np.array([0.25, -0.2, 0.08])
f.x_predicted, f.y_predicted, f.theta_predicted = pred
f.mu_bar = pred.reshape(3, 1).copy()

# Exact measurements from the true pose
f.approx_linear_distance = [math.hypot(lx - truth[0], ly - truth[1]) for lx, ly in LANDMARKS]
f.approx_angular_position = [normalize_angle(math.atan2(ly - truth[1], lx - truth[0]) - truth[2])
                             for lx, ly in LANDMARKS]

f.z_matrix_acc_to_measurement()
f.z_matrix_acc_to_pos()
f.cylin_pairing()
pairs = f.paired_measurements.shape[1] if f.paired_measurements.size else 0
f.obs_model()

before = np.linalg.norm(pred[:2] - truth[:2])
after = np.linalg.norm(f.mu[:2, 0] - truth[:2])
print(f'pairs={pairs}  pos error {before:.3f} -> {after:.3f} m   '
      f'heading error {pred[2] - truth[2]:+.3f} -> {f.mu[2, 0] - truth[2]:+.3f} rad')
assert pairs == len(LANDMARKS), 'association failed'
assert after < 0.3 * before, 'update did not move toward the truth'
print('OK')
