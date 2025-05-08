import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF
from scipy.spatial.transform import Rotation as R

class VisualInertialEKF(EKF):
    def __init__(self, initial_position, initial_orientation, initial_scale=1.0):
        super().__init__(dim_x=17, dim_z=7)  # 17 states, 7 measurements
        self.dt = 0.01  # IMU update rate (100 Hz)
        
        # Initialize State
        self.x = np.zeros(17)
        self.x[:3] = initial_position  # Initial Position
        self.x[6:10] = initial_orientation  # Initial Quaternion
        self.x[16] = initial_scale  # Initial Scale
       
        # Covariances
        self.P *= 1.0  # Initial covariance
        self.Q = np.eye(17) * 0.01  # Process noise
        self.R = np.eye(7) * 0.1  # Measurement noise

    def predict(self, imu_acc, imu_gyro):
        """ IMU-based state prediction (Propagate motion) """
        self.x = self._state_transition(self.x, imu_acc, imu_gyro, self.dt)

    def update_with_vision(self, vision_meas):
        """ Update state using monocular vision measurements """
        self.update(vision_meas, HJacobian=self._H_jacobian, Hx=self._measurement_model)

    def _state_transition(self, state, imu_acc, imu_gyro, dt):
        """ Predicts the next state based on IMU acceleration and gyroscope data """
        p, v, q, ba, bg, s = state[:3], state[3:6], state[6:10], state[10:13], state[13:16], state[16]
        
        # Remove biases
        acc_corrected = imu_acc - ba
        gyro_corrected = imu_gyro - bg

        # Convert quaternion to rotation matrix
        rot_mat = R.from_quat(q).as_matrix()

        # Update velocity (integrating acceleration)
        v_new = v + rot_mat @ acc_corrected * dt

        # Update position (integrating velocity)
        p_new = p + v * dt + 0.5 * rot_mat @ acc_corrected * dt ** 2

        # Update quaternion using gyroscope data
        omega = np.hstack(([0], gyro_corrected))  # Convert to quaternion format
        dq = 0.5 * self._quaternion_multiply(q, omega * dt)
        q_new = q + dq  # First-order approximation
        q_new /= np.linalg.norm(q_new)  # Normalize quaternion

        return np.hstack([p_new, v_new, q_new, ba, bg, s])

    def _measurement_model(self, state):
        """ Measurement function: Returns expected monocular position and orientation """
        p, q, s = state[:3], state[6:10], state[16]
        return np.hstack([s * p, q])  # Scale-corrected position + quaternion

    def _H_jacobian(self, state):
        """ Jacobian matrix for the measurement function (vision update) """
        H = np.zeros((7, 17))
        H[:3, :3] = np.eye(3) * state[16]  # Partial derivative w.r.t position (scaled)
        H[:3, 16] = state[:3]  # Partial derivative w.r.t scale
        H[3:7, 6:10] = np.eye(4)  # Orientation directly observed
        return H

    @staticmethod
    def _quaternion_multiply(q, r):
        """ Quaternion multiplication """
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = r
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])
