#!/usr/bin/env python3

import rclpy, math
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

time_step = 0.3

class Ekf_Slam_Vw(Node):
    def __init__(self):
        super().__init__("ekf_slam_3rd_approach")

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/ekf_pose', 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laserCB, 10)
        self.control_sub = self.create_subscription(Odometry, '/odom', self.control_CB, 10)
        self.landmark_marker_pub = self.create_publisher(MarkerArray, '/landmark_markers', 10)
        self.trajectory_marker_pub = self.create_publisher(Marker, '/robot_trajectory_marker', 10)
        
        self.trajectory_pub = self.create_publisher(Path, '/robot_path', 10)
        self.robot_path = Path()
        self.robot_path.header.frame_id = "odom"  # Frame to visualize in RViz
        self.count = 0

        self.sigma_r = 0.1
        self.sigma_phi = 0.01

        self.initial_run = True

        self.mu = np.zeros((3, 1))
        self.cov = np.zeros((3, 3))
        self.landmark_count = 0
        self.time_step = time_step

    def laserCB(self, msg: LaserScan):
        while msg.ranges == None:
            self.get_logger().warn("The lidar data is not being subscribed")
        self.laser_points = [0] * len(msg.ranges)
        for i, value in enumerate(msg.ranges[:-1]):
            if not math.isinf(value) and not math.isnan(value):
                self.laser_points[i] = value
            else:
                self.laser_points[i] = msg.range_max
    
    def control_CB(self, msg: Odometry):
        if self.initial_run:
            self.x_dash = msg.pose.pose.position.x
            self.y_dash = msg.pose.pose.position.y
            orientation_q = msg.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (_, _, self.theta_dash) = euler_from_quaternion(orientation_list)
            self.mu[:3,0:1] = np.array([[self.x_dash, self.y_dash, self.theta_dash]]).T
            self.initial_run = False

        self.v = msg.twist.twist.linear.x
        self.w = msg.twist.twist.angular.z

        self.x_dash = msg.pose.pose.position.x
        self.y_dash = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, self.theta_dash) = euler_from_quaternion(orientation_list)

    def mu_prediction(self):
        self.x_delta = -(self.v/self.w)*math.sin(self.mu[2,0]) + (self.v/self.w)*math.sin(normalize_angle(self.mu[2,0] + self.w*self.time_step))
        self.y_delta = (self.v/self.w)*math.cos(self.mu[2,0]) - (self.v/self.w)*math.cos(normalize_angle(self.mu[2,0] + self.w*self.time_step))
        self.theta_delta = self.w * self.time_step
        self.F_x = np.zeros((3, 3+2*self.landmark_count))
        self.F_x[:3,:3] = np.eye(3)

        self.mu_bar = self.mu + np.dot(self.F_x.T, np.array([[self.x_delta, self.y_delta, self.theta_delta]]).T)
        self.mu_bar[2,0] = normalize_angle(self.mu_bar[2,0])

        self.x_bar = self.mu_bar[0,0]
        self.y_bar = self.mu_bar[1,0]
        self.theta_bar = self.mu_bar[2,0]

    def dg_dstate(self):
        G_T_x = np.array([
            [0, 0, -self.y_delta],
            [0, 0, self.x_delta],
            [0, 0, 0]
        ])
        self.G_t = np.eye(3 + 2*self.landmark_count) + np.dot(self.F_x.T, np.dot(G_T_x, self.F_x))

    def dg_dcontrol(self):
        dg0_dv = self.x_delta / self.v
        dg1_dv = self.y_delta / self.v
        dg2_dv = 0

        dg0_dw = ((self.v/(self.w*self.w)) * math.sin(self.mu[2,0])
                  + (self.v*self.time_step/self.w) * math.cos(normalize_angle(self.mu[2,0]+self.w*self.time_step))
                  - (self.v/(self.w*self.w)) * math.sin(normalize_angle(self.mu[2,0]+self.w*self.time_step)))
        dg1_dw = (- (self.v/(self.w*self.w)) * math.cos(self.mu[2,0])
                  + (self.v*self.time_step/self.w) * math.sin(normalize_angle(self.mu[2,0]+self.w*self.time_step))
                  + (self.v/(self.w*self.w)) * math.cos(normalize_angle(self.mu[2,0]+self.w*self.time_step)))
        dg2_dw = self.time_step

        self.V_t_x = np.array([
            [dg0_dv, dg0_dw],
            [dg1_dv, dg1_dw],
            [dg2_dv, dg2_dw]
        ])

    def pred_meas_noise(self):
        self.alpha_1 = 0.0001
        self.alpha_2 = 0.0001

        self.noise = np.diag([self.alpha_1 * (self.v**2), self.alpha_2 * (self.w**2)])
        self.R_x_t = np.dot(self.V_t_x, np.dot(self.noise, self.V_t_x.T))

    def cov_pred(self):
        R_t = np.dot(self.F_x.T, np.dot(self.R_x_t, self.F_x))
        self.cov_bar = np.dot(self.G_t, np.dot(self.cov, self.G_t.T)) + R_t

    def observed_features(self):
        # Compute jump derivatives for each laser ray
        jumps = [0.0] * len(self.laser_points)
        for i in range(1, len(self.laser_points) - 1):
            prev_point = self.laser_points[i - 1]
            next_point = self.laser_points[i + 1]
            if prev_point > 0.2 and next_point > 0.2:
                derivative = (next_point - prev_point) / 2.0
            else:
                derivative = 0.0
            if abs(derivative) > 0.1:
                jumps[i] = derivative
        
        # Process jumps to group rays into cylinder detections.
        self.approx_linear_distance = []
        self.approx_angular_position = []
        cylin_active = False
        no_of_rays = 0
        sum_ray_indices = 0
        sum_depth = 0.0

        i = 0
        while i < len(jumps):
            if jumps[i] < 0 and not cylin_active:
                cylin_active = True
                no_of_rays = 1
                sum_ray_indices = i
                sum_depth = self.laser_points[i]
            elif cylin_active and abs(jumps[i]) < 1e-6:
                no_of_rays += 1
                sum_ray_indices += i
                sum_depth += self.laser_points[i]
            elif jumps[i] > 0 and cylin_active:
                avg_index = sum_ray_indices / no_of_rays
                avg_depth = sum_depth / no_of_rays
                approx_ang = normalize_angle(avg_index * 0.01745)
                self.approx_angular_position.append(approx_ang)
                self.approx_linear_distance.append(avg_depth + 0.25)  # offset as before
                cylin_active = False
                no_of_rays = 0
                sum_ray_indices = 0
                sum_depth = 0.0
            i += 1

        self.obs_curr = np.vstack((self.approx_linear_distance, self.approx_angular_position))

    def h_mat(self):
        self.delta_x_cylin = np.zeros(self.landmark_count)
        self.delta_y_cylin = np.zeros(self.landmark_count)
        self.h_theta = np.zeros(self.landmark_count)

        for i in range(self.landmark_count):
            self.delta_x_cylin[i] = self.mu_bar[3 + 2 * i, 0] - self.mu_bar[0, 0]
            self.delta_y_cylin[i] = self.mu_bar[4 + 2 * i, 0] - self.mu_bar[1, 0]
            self.h_theta[i] = normalize_angle(math.atan2(self.delta_y_cylin[i], self.delta_x_cylin[i]) - self.mu_bar[2, 0])
        self.q = self.delta_x_cylin**2 + self.delta_y_cylin**2
        self.h = np.vstack((np.sqrt(self.q), self.h_theta))
        
    def z_pairing_check(self):
        self.z_curr = np.empty_like(self.h)
        self.z_curr[:] = np.nan

        num_obs = self.obs_curr.shape[1]
        self.curr_match = [False] * num_obs
        self.prev_match = [False] * self.landmark_count

        range_threshold = 0.5
        bearing_threshold = 0.5

        self.get_logger().info("Running new z_pairing_check()")
        for i in range(self.landmark_count):
            best_obs = None
            best_diff = float('inf')
            for j in range(num_obs):
                if self.curr_match[j]:
                    continue
                dr = abs(self.h[0, i] - self.obs_curr[0, j])
                da = abs(normalize_angle(self.h[1, i] - self.obs_curr[1, j]))
                if dr < range_threshold and da < bearing_threshold:
                    diff = math.sqrt(dr**2 + da**2)
                    if diff < best_diff:
                        best_diff = diff
                        best_obs = j
            if best_obs is not None:
                self.z_curr[:, i] = self.obs_curr[:, best_obs]
                self.curr_match[best_obs] = True
                self.prev_match[i] = True
            else:
                self.z_curr[:, i] = np.array([0.0, 0.0])
                
        self.get_logger().info(f"curr_match: {self.curr_match}")
        self.get_logger().info(f"prev_match: {self.prev_match}")

    def correction_step(self):
        Q_t = np.array([
            [self.sigma_r**2, 0],
            [0, self.sigma_phi**2]
        ])

        F_dash_x = np.zeros((5, 3+2*self.landmark_count))
        F_dash_x[:3, :3] = np.eye(3)

        self.mu_bar_copy = self.mu_bar.copy()
        self.cov_bar_copy = self.cov_bar.copy()

        for i in range(self.landmark_count):
            if self.prev_match[i]:
                F_dash_x[3:5, 3+2*i:5+2*i] = np.eye(2)

                low_H_00 = -math.sqrt(self.q[i]) * self.delta_x_cylin[i]
                low_H_10 = self.delta_y_cylin[i]
                low_H_01 = -math.sqrt(self.q[i]) * self.delta_y_cylin[i]  
                low_H_11 = -self.delta_x_cylin[i]
                low_H_02 = 0
                low_H_12 = -self.q[i]
                low_H_03 = -low_H_00
                low_H_13 = -low_H_10
                low_H_04 = -low_H_01
                low_H_14 = -low_H_11

                low_H_t = (1/self.q[i]) * np.array([
                    [low_H_00, low_H_01, low_H_02, low_H_03, low_H_04],
                    [low_H_10, low_H_11, low_H_12, low_H_13, low_H_14]
                ])

                H_t = np.dot(low_H_t, F_dash_x)
                S_t = np.dot(H_t, np.dot(self.cov_bar, H_t.T)) + Q_t
                K_gain = np.dot(self.cov_bar, np.dot(H_t.T, np.linalg.inv(S_t)))

                Innovation_mat = np.array([
                    [self.z_curr[0, i] - self.h[0, i]],
                    [normalize_angle(self.z_curr[1, i] - self.h[1, i])]
                ])

                self.mu_bar = self.mu_bar + np.dot(K_gain, Innovation_mat)
                Identity = np.eye(3+2*self.landmark_count)
                self.cov_bar = np.dot((Identity - np.dot(K_gain, H_t)), self.cov_bar)

        for i in range(self.landmark_count):
            if not self.prev_match[i]:
                self.mu_bar[3+2*i:5+2*i, 0] = self.mu_bar_copy[3+2*i:5+2*i, 0]
                self.cov_bar[3+2*i:5+2*i, 3+2*i:5+2*i] = self.cov_bar_copy[3+2*i:5+2*i, 3+2*i:5+2*i]

    def add_new_feature(self):
        prev_detection = self.landmark_count
        for i in range(len(self.curr_match)):
            if self.curr_match[i] == False:
                self.landmark_count += 1

        first_new_cylin = True
        self.cov_dash = np.eye(3+2*self.landmark_count)
        cov_bar_shape, _ = np.shape(self.cov_bar)
        self.cov_dash[:cov_bar_shape, :cov_bar_shape] = self.cov_bar
        self.cov_bar = self.cov_dash

        extra_landmarks = None
        for i in range(len(self.curr_match)):
            if self.curr_match[i] == False:
                r_new_cylin = self.obs_curr[0, i]
                theta_new_cylin = self.obs_curr[1, i]
                x_new_cylin = self.x_bar + r_new_cylin * math.cos(normalize_angle(theta_new_cylin + self.theta_bar))
                y_new_cylin = self.y_bar + r_new_cylin * math.sin(normalize_angle(theta_new_cylin + self.theta_bar))
                if first_new_cylin:
                    extra_landmarks = np.array([[x_new_cylin], [y_new_cylin]])
                    first_new_cylin = False
                else:
                    extra_landmarks = np.vstack((extra_landmarks, np.array([[x_new_cylin], [y_new_cylin]])))
        if (prev_detection - self.landmark_count) != 0:              
            self.mu_bar = np.vstack((self.mu_bar, extra_landmarks))

    def publish_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        x = self.mu[0, 0]
        y = self.mu[1, 0]
        theta = self.mu[2, 0]
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0

        quat = quaternion_from_euler(0, 0, theta)
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        cov_small = self.cov[0:3, 0:3].flatten()
        covariance_full = [0.0] * 36
        for i in range(9):
            covariance_full[i] = cov_small[i]
        msg.pose.covariance = covariance_full

        self.pose_pub.publish(msg)
        print(f"covariance_shape : {np.shape(self.cov)}")

    def publish_landmark_markers(self):
        marker_array = MarkerArray()
        for i in range(self.landmark_count):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "odom"
            marker.ns = "landmarks"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            if self.mu.shape[0] >= 3 + 2*i + 2:
                x = self.mu[3 + 2*i, 0]
                y = self.mu[4 + 2*i, 0]
            else:
                x, y = 0.0, 0.0
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)
        self.landmark_marker_pub.publish(marker_array)

    def publish_trajectory(self):
        """Append the current pose to the trajectory and publish the updated path."""
        clock = Clock()
        traj_pose = PoseStamped()
        traj_pose.header.stamp = clock.now().to_msg()
        traj_pose.header.frame_id = "odom"
        traj_pose.pose.position.x = self.mu[0, 0]
        traj_pose.pose.position.y = self.mu[1, 0]
        traj_pose.pose.position.z = 0.0
        quat = quaternion_from_euler(0, 0, self.mu[2, 0])
        traj_pose.pose.orientation.x = quat[0]
        traj_pose.pose.orientation.y = quat[1]
        traj_pose.pose.orientation.z = quat[2]
        traj_pose.pose.orientation.w = quat[3]
        
        self.robot_path.poses.append(traj_pose)
        self.robot_path.header.stamp = clock.now().to_msg()
        self.trajectory_pub.publish(self.robot_path)

    def run(self):
        self.mu_prediction()
        self.dg_dstate()
        self.dg_dcontrol()
        self.pred_meas_noise()
        self.cov_pred()
        self.observed_features()
        self.h_mat()
        self.z_pairing_check()
        self.correction_step()
        self.add_new_feature()
        self.mu = self.mu_bar
        self.cov = self.cov_bar
        error = self.mu[:3,:] - np.array([[self.x_dash, self.y_dash, self.theta_dash]]).T
        self.publish_pose()
        self.publish_landmark_markers()
        self.publish_trajectory()
        print(f"error : {error}")

def main(args=None):
    rclpy.init(args=args)
    node = Ekf_Slam_Vw()
    timer_period = time_step
    node.create_timer(timer_period, node.run)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
