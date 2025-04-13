#!/usr/bin/env python3

import rclpy, math
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseWithCovarianceStamped,PoseStamped
from nav_msgs.msg import Odometry,Path
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from scipy.optimize import linear_sum_assignment

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

import logging

# Configure logging: logs will be written to ekf_slam_debug.log
logging.basicConfig(
    filename='ekf_slam_debug.log',
    filemode='w',  # 'w' to overwrite every time, or 'a' to append
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EKF_SLAM(Node):

    def __init__(self):
        super().__init__("ekf_slam")
        self.position_pub = self.create_publisher(PoseWithCovarianceStamped,'/ekf_slam_pose',10)
        self.lidar_sub = self.create_subscription(LaserScan,'/scan',self.laserCB,10)
        self.odom_sub = self.create_subscription(Odometry,'/odom',self.odomCB,10)

        self.x_prev, self.y_prev, self.theta_prev = 0.0, 0.0, 0.0
        self.mu_bar = np.zeros((3,1))
        self.total_landmarks = 0
        self.final_covariance = np.zeros((3,3))
        self.mu = np.zeros((3,1))
        self.sigma_r = 0.01  # standard deviation in the disance measurement
        self.sigma_alpha = 0.01  # standard deviation in the angle measurement
        self.iteration = 0
        self.trajectory_pub = self.create_publisher(Path, '/robot_path', 10)
        self.robot_path = Path()
        self.robot_path.header.frame_id = "odom"  # Frame to visualize in RViz
        self.real_traj_pub = self.create_publisher(Path, '/real_robot_path', 10)
        self.real_robot_path = Path()
        self.real_robot_path.header.frame_id = "odom"
        

    def laserCB(self, msg: LaserScan):
        """Entry point to the slam implementation, subscribes to the lidar data."""
        while msg.ranges is None:
            self.get_logger().warn("The lidar data is not being subscribed")
        self.laser_points = [0] * len(msg.ranges)
        for i, value in enumerate(msg.ranges[:-1]):
            if not math.isinf(value) and not math.isnan(value):
                self.laser_points[i] = value
            else:
                self.laser_points[i] = msg.range_max

    def odomCB(self, msg: Odometry):
        """Subscribes to the odom data to generate the control parameters."""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, self.theta) = euler_from_quaternion(orientation_list)
        # print(self.x,self.x_prev)

    def pose_predictor(self):
        """Calculate the predicted pose by generating control parameters."""
        self.F_x = np.zeros((3,3+2*self.total_landmarks))
        self.main_Identity = np.eye(3+2*self.total_landmarks)
        self.F_x[:3,:3] = np.eye(3,3)
        self.delta_trans = math.sqrt((self.x - self.x_prev)**2 + (self.y - self.y_prev)**2)
        self.delta_rot1 = normalize_angle(math.atan2(self.y - self.y_prev, self.x - self.x_prev) - self.theta_prev)
        self.delta_rot2 = normalize_angle(self.theta - self.theta_prev - self.delta_rot1)

        self.x_delta = self.delta_trans * math.cos(normalize_angle(self.theta + self.delta_rot1))
        self.y_delta = self.delta_trans * math.sin(normalize_angle(self.theta + self.delta_rot1))
        self.theta_delta = normalize_angle(self.delta_rot1 + self.delta_rot2)

        delta_mat = np.array([
            [self.x_delta],
            [self.y_delta],
            [self.theta_delta]
        ])

        # print("F_x shape: ", np.shape(self.F_x))
        # print("self.mu shape: ", np.shape(self.mu))
        # print("delta_mat shape: ", np.shape(delta_mat))

        self.mu_bar = self.mu + np.dot(self.F_x.T,delta_mat)

    def state_cov_calc(self):
        """Calculate the covariance matrix (Jacobian w.r.t. state)."""
        self.G_x_t = np.eye(3)
        self.G_x_t[:3, :3] = np.array([
            [1, 0, -self.delta_trans * math.sin(normalize_angle(self.theta_prev + self.delta_rot1))],
            [0, 1,  self.delta_trans * math.cos(normalize_angle(self.theta_prev + self.delta_rot1))],
            [0, 0, 1]
        ])

        self.G_t = self.main_Identity + np.dot(self.F_x.T,np.dot(self.G_x_t,self.F_x))

    def control_cov_calc(self):
        """Calculate the Jacobian w.r.t. control parameters."""
        
        self.V_t = np.array([
            [-self.delta_trans * math.sin(normalize_angle(self.theta_prev + self.delta_rot1)),
              math.cos(normalize_angle(self.theta_prev + self.delta_rot1)), 0],
            [ self.delta_trans * math.cos(normalize_angle(self.theta_prev + self.delta_rot1)),
              math.sin(normalize_angle(self.theta_prev + self.delta_rot1)), 0],
            [1, 0, 1]
        ])

    def predicted_covariance(self):
        """Calculate the predicted covariance."""
        alpha1 = 0.05
        alpha2 = 0.01
        alpha3 = 0.05
        alpha4 = 0.01
        self.rot1_variance = alpha1 * (self.delta_rot1 ** 2) + alpha2 * (self.delta_trans ** 2)
        self.trans_variance = alpha3 * (self.delta_trans ** 2) + alpha4 * ((self.delta_rot1 ** 2) + (self.delta_rot2 ** 2))
        self.rot2_variance = alpha1 * (self.delta_rot2 ** 2) + alpha2 * (self.delta_trans ** 2)
        noise = np.diag([self.rot1_variance, self.trans_variance, self.rot2_variance])
        R_t = np.dot(self.V_t, np.dot(noise, self.V_t.T))
        self.noise_cov = np.dot(self.F_x.T, np.dot(R_t, self.F_x))
        # print(self.G_t)
        self.covariance_bar = np.dot(self.G_t, np.dot(self.final_covariance, self.G_t.T)) + self.noise_cov
        _, self.size = np.shape(self.covariance_bar)

    def observed_landmarks(self):
        """
        Process the lidar scan (self.laser_points) to detect jumps.
        Each jump is assumed to correspond to a cylinder edge.
        The average ray index and depth for each detected cylinder region
        are stored in self.approx_linear_distance and self.approx_angular_position.
        """

        jumps = [0.0] * len(self.laser_points)
        for i in range(1, len(self.laser_points) - 1):
            prev_point = self.laser_points[i - 1]
            next_point = self.laser_points[i + 1]
            if prev_point > 0.2 and next_point > 0.2:
                derivative = (next_point - prev_point) / 2.0
            else:
                derivative = 0.0
            if abs(derivative) > 0.15:
                jumps[i] = derivative
            # Else, jumps[i] remains 0.0
        
        # self.get_logger().info(f"Detected jumps: {jumps}")

        # Process the jumps to group rays into cylinder detections.
        self.approx_linear_distance = []
        self.approx_angular_position = []
        cylin_active = False
        no_of_rays = 0
        sum_ray_indices = 0
        sum_depth = 0.0

        i = 0
        while i < len(jumps):
            # Start of a cylinder detection: falling edge (derivative < 0)
            if jumps[i] < 0 and not cylin_active:
                cylin_active = True
                no_of_rays = 1
                sum_ray_indices = i
                sum_depth = self.laser_points[i]
            # If we are in a cylinder region and the derivative is near zero
            elif cylin_active and abs(jumps[i]) < 1e-6:
                no_of_rays += 1
                sum_ray_indices += i
                sum_depth += self.laser_points[i]
            # End of the cylinder region: rising edge (derivative > 0)
            elif jumps[i] > 0 and cylin_active:
                # Compute average index and depth
                avg_index = sum_ray_indices / no_of_rays
                avg_depth = sum_depth / no_of_rays
                # Convert ray index to angle (assuming a fixed angular resolution, e.g., 1° = 0.01745 rad)
                approx_ang = normalize_angle(avg_index * 0.01745)
                self.approx_angular_position.append(approx_ang)
                # Optionally add an offset equal to the radious of the cylinder (here +0.25 as in your code)
                self.approx_linear_distance.append(avg_depth + 0.25)
                # Reset for the next cylinder detection
                cylin_active = False
                no_of_rays = 0
                sum_ray_indices = 0
                sum_depth = 0.0
            i += 1
        
        self.num_meas = len(self.approx_linear_distance)
        self.z_curr = np.vstack((self.approx_linear_distance,self.approx_angular_position)) #measurement matrix for the current observation
#        self.get_logger().info(f"Number of cylinder measurements: {self.num_meas}")


    def z_mat_from_previous_state(self):
        """Calculates the z_matrix based upon the previous state matrix and current predicted pose"""
        i = 0
        meas_dist_list = []
        meas_ang_list = []
        diff_X_list = []
        diff_Y_list = []
        self.index = 3+2*self.total_landmarks
        self.flag1 = 0
        #print("Index limit:", index)
        while(i<(self.index-3)/2):
            x_land_curr = self.mu_bar[3+2*i][0]
            y_land_curr = self.mu_bar[4+2*i][0]
            print(f"x_land_curr = {x_land_curr},y_land_curr = {y_land_curr},x_robot = {self.mu_bar[0,0]},y_robot = {self.mu_bar[1,0]},theta = {self.mu_bar[2,0]}")
            x_diff = self.mu_bar[0,0] - x_land_curr
            y_diff = self.mu_bar[1,0] - y_land_curr
            dist = math.sqrt(math.pow((x_diff),2) + math.pow((y_diff),2))
            theta = normalize_angle(normalize_angle(math.atan2(-y_diff,-x_diff))- normalize_angle(self.mu_bar[2,0]))
            # print(f"theta = {theta}")
            meas_dist_list.append(dist)
            meas_ang_list.append(theta)
            diff_X_list.append(x_diff)
            diff_Y_list.append(y_diff)
            self.flag1 = 1 
            i+=1 

        self.z_prev = np.vstack((meas_dist_list,meas_ang_list))
        self.diff_estim = np.vstack((diff_X_list,diff_Y_list))

    def meas_association(self):
        """The current measurement is associated with the state matrix to get the correct indexing of the features"""

        self.z_curr_flags = [False]*self.num_meas

        if self.flag1 != 1:
            return #all the cylinders observed for this iteration are new.

        
    def compute_pairwise_costs(self, lam=0.1):
        """
        Build a cost matrix for all pairs (i, j) where i indexes columns of self.z_prev
        and j indexes columns of self.z_curr.
        Both self.z_prev and self.z_curr are 2 x m (or 2 x n) matrices, with each column = [dist, angle].
        
        The cost for matching a column i in z_prev with a column j in z_curr is defined as:
          cost(i,j) = lam * |d_prev - d_curr| + (1 - lam) * |theta_prev - theta_curr|
        """
        m = self.z_prev.shape[1]
        n = self.z_curr.shape[1]
        cost_mat = np.zeros((m, n), dtype=float)
        for i in range(m):
            for j in range(n):
                dist_diff = abs(self.z_prev[0, i] - self.z_curr[0, j])
                angle_diff = abs(normalize_angle(self.z_prev[1, i]) - normalize_angle(self.z_curr[1, j]))
                cost_mat[i, j] = lam * dist_diff + (1 - lam) * angle_diff
        return cost_mat

    def match_matrices(self, lam=0.1, large_cost=999999.0, match_threshold=1):
        """
        Matches columns of self.z_prev (2 x m) with columns of self.z_curr (2 x n)
        using the Hungarian algorithm. A match is only accepted if its cost is below match_threshold.
        
        The results are stored in instance variables:
          - self.matched_curr: 2 x m array, where for each column i in z_prev, if a match is accepted,
                               matched_curr[:, i] contains the corresponding column from z_curr; otherwise zeros.
          - self.unmatched_prev: 2 x X array for columns in z_prev that remain unmatched.
          - self.unmatched_curr: 2 x Y array for columns in z_curr that remain unmatched.
          - self.match_vector: an array of length m where match_vector[i] = j if z_prev col i is matched with z_curr col j,
                               or -1 if no acceptable match is found.
          - self.match_vector_curr: an array of length n where for each column j in z_curr, if matched then the corresponding
                               index from z_prev is stored; otherwise -10.
        
        If m != n, dummy rows/columns (with a high cost) are added so that the Hungarian algorithm produces a complete assignment.
        """
        m = self.z_prev.shape[1]
        n = self.z_curr.shape[1]
        
        # Compute the raw cost matrix (m x n)
        cost_matrix_raw = self.compute_pairwise_costs(lam=lam)
        
        # Create a square cost matrix of size = max(m, n) filled with large_cost.
        size = max(m, n)
        cost_matrix = np.full((size, size), large_cost, dtype=float)
        cost_matrix[:m, :n] = cost_matrix_raw
        
        # Solve assignment using the Hungarian algorithm.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Initialize match_vector for z_prev columns (length m) and matched_curr.
        match_vector = np.full(m, -1, dtype=int)
        matched_curr = np.zeros_like(self.z_prev)
        
        # Initialize a match_vector for z_curr columns (length n), default to -10.
        match_vector_curr = np.full(n, -10, dtype=int)
        
        # Sets to track which indices have been used.
        used_rows_prev = set()
        used_cols_curr = set()
        
        # Lists to collect unmatched indices.
        unmatched_prev_indices = []
        unmatched_curr_indices = []
        
        # Process each assignment pair.
        for (i, j) in zip(row_ind, col_ind):
            if i < m and j < n:
                # Accept match only if its cost is below the threshold.
                if cost_matrix[i, j] <= match_threshold:
                    match_vector[i] = j
                    matched_curr[:, i] = self.z_curr[:, j]
                    used_rows_prev.add(i)
                    used_cols_curr.add(j)
                    # Record the reverse match: for z_curr column j, assign i.
                    match_vector_curr[j] = i
                else:
                    # Cost too high: mark both as unmatched.
                    unmatched_prev_indices.append(i)
                    unmatched_curr_indices.append(j)
            elif i < m and j >= n:
                unmatched_prev_indices.append(i)
            elif i >= m and j < n:
                unmatched_curr_indices.append(j)
            # else: dummy to dummy, no effect.
        
        # Any leftover indices not assigned: mark them unmatched.
        for i in range(m):
            if i not in used_rows_prev and i not in unmatched_prev_indices:
                unmatched_prev_indices.append(i)
        for j in range(n):
            if j not in used_cols_curr and j not in unmatched_curr_indices:
                unmatched_curr_indices.append(j)
        
        # Build unmatched matrices.
        unmatched_prev = self.z_prev[:, unmatched_prev_indices] if unmatched_prev_indices else np.zeros((2, 0))
        unmatched_curr = self.z_curr[:, unmatched_curr_indices] if unmatched_curr_indices else np.zeros((2, 0))
        
        # Store results as instance variables.
        self.matched_curr = matched_curr
        self.unmatched_prev = unmatched_prev
        self.unmatched_curr = unmatched_curr
        self.match_vector = match_vector
        self.match_vector_curr = match_vector_curr

    def correction_step(self):
        if self.flag1 == 0:
            self.mu = self.mu_bar
            self.final_covariance = self.covariance_bar
            return

        self.mu_sub_mat = 0
        self.final_indices  = []
        self.H_t = np.zeros((2, np.shape(self.covariance_bar)[0]))
        

        for i in range(np.shape(self.matched_curr)[1]):
            if(self.match_vector[i]==-1):
                continue
            else:
                self.F_h = np.zeros((5, np.shape(self.covariance_bar)[0]))
                self.F_h[:3, :3] = np.eye(3)
                self.F_h[3:5,2*i+3:2*i+5] = np.eye(2)
                meas_dist = self.matched_curr[0,i]
                meas_angle = self.matched_curr[1,i]

                z_curr = np.array([
                    [meas_dist],
                    [meas_angle]
                ])

                prev_dist = self.z_prev[0,i]
                prev_angle = self.z_prev[1,i]
                
                h_mat = np.array([
                    [prev_dist],
                    [prev_angle]
                ])

                x_diff = self.diff_estim[0,i]
                y_diff = self.diff_estim[1,i]

                H_small = np.array([
                    [x_diff/prev_dist,y_diff/prev_dist,0,-x_diff/prev_dist,-y_diff/prev_dist],
                    [-y_diff/prev_dist,x_diff/prev_dist,-1,y_diff/prev_dist,-x_diff/prev_dist]
                ])

                self.H_t = np.dot(H_small,self.F_h)

                q_matrix = np.array([
                    [self.sigma_r**2, 0],
                    [0, self.sigma_alpha**2]
                ])

                S = np.dot(self.H_t, np.dot(self.covariance_bar, self.H_t.T)) + q_matrix
                k_gain = np.dot(self.covariance_bar, np.dot(self.H_t.T, np.linalg.inv(S)))

                Innovation_matrix = np.array([
                    [z_curr[0, 0] - h_mat[0, 0]],
                    [normalize_angle(z_curr[1, 0] - h_mat[1, 0])]
                ])
                print(f"z_t = {z_curr}, h_t = {h_mat}")

                self.mu_bar = self.mu_bar + np.dot(k_gain, Innovation_matrix)
                _,Identity_dimension = np.shape(np.dot(k_gain, self.H_t))
                Identity = np.eye(Identity_dimension)
                self.covariance_bar = np.dot((Identity - np.dot(k_gain, self.H_t)), self.covariance_bar)
                self.mu_bar[2,0]=normalize_angle(self.mu_bar[2,0])
                self.covariance_bar[2,2]=normalize_angle(self.covariance_bar[2,2])
            
        self.mu = self.mu_bar
        self.final_covariance = self.covariance_bar
        obs = np.array([[self.x],
                        [self.y],
                        [self.theta]])
        err = self.mu[:3] - obs
        print(f"err: {err}")

    def add_new_cylin(self):
        # Count new landmarks from unmatched measurements.
        total_landmarks_new = sum(1 for i in range(len(self.match_vector_curr)) if self.match_vector_curr[i] == -10)
        new_total = self.total_landmarks + total_landmarks_new

        # Allocate new state and covariance for existing + new landmarks.
        mu_new = np.zeros((3 + 2 * new_total, 1))
        cov_new = np.eye(3 + 2 * new_total)

        # Copy existing state and covariance.
        mu_new[:np.shape(self.mu)[0], :] = self.mu
        cov_new[:np.shape(self.final_covariance)[0], :np.shape(self.final_covariance)[0]] = self.final_covariance

        # For each new cylinder, add its state.
        for i in range(len(self.match_vector_curr)):
            if self.match_vector_curr[i] == -10:
                dist = self.z_curr[0, i]
                ang = self.z_curr[1, i]
                x_new = self.mu_bar[0, 0] + dist * math.cos(normalize_angle(self.mu_bar[2, 0] + ang))
                y_new = self.mu_bar[1, 0] + dist * math.sin(normalize_angle(self.mu_bar[2, 0] + ang))
                index = self.total_landmarks  # insert at next available index
                mu_new[3 + 2 * index : 3 + 2 * index + 2, 0] = np.array([x_new, y_new])
                self.total_landmarks += 1
                print(self.z_curr)

        
        self.mu = mu_new
        self.final_covariance = cov_new

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

    def publish_real_trajectory(self):
        """Publish the trajectory based directly on odometry data."""
        clock = Clock()
        real_pose = PoseStamped()
        real_pose.header.stamp = clock.now().to_msg()
        real_pose.header.frame_id = "odom"
        real_pose.pose.position.x = self.x
        real_pose.pose.position.y = self.y
        real_pose.pose.position.z = 0.0
        quat = quaternion_from_euler(0, 0, self.theta)
        real_pose.pose.orientation.x = quat[0]
        real_pose.pose.orientation.y = quat[1]
        real_pose.pose.orientation.z = quat[2]
        real_pose.pose.orientation.w = quat[3]
        self.real_robot_path.poses.append(real_pose)
        self.real_robot_path.header.stamp = clock.now().to_msg()
        self.real_traj_pub.publish(self.real_robot_path)
        

    def run(self):
        self.pose_predictor()
        self.state_cov_calc()
        self.control_cov_calc()
        self.predicted_covariance()
        self.observed_landmarks()
        self.z_mat_from_previous_state()
        self.meas_association()
        self.compute_pairwise_costs()
        self.match_matrices()
        self.correction_step()
        self.add_new_cylin()
        self.publish_trajectory()
        self.publish_real_trajectory()
        # print(f"mu = {self.mu}")
        # print(f"iteration = {self.iteration} ")
        self.iteration+=1
        # logger.info("mu = %s", self.mu)
        # logger.info("iteration = %s", self.iteration)
        
        self.x_prev = self.x
        self.y_prev = self.y
        self.theta_prev = self.theta

        # self.get_logger().info(f"self.final_covariance = {self.match_vector_curr} and total_landmarks = {self.total_landmarks}")
        # self.get_logger().info(f"z_curr = {self.match_vector_curr}")

def main(args=None):
    rclpy.init(args=args)
    node = EKF_SLAM()
    # Create a timer to call the run() method periodically (e.g., every 0.1 seconds)
    timer_period = 0.5  # seconds
    node.create_timer(timer_period, node.run)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
