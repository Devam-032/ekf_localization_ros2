#!/usr/bin/env python3

import rclpy,math
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion,quaternion_from_euler

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

class EKF_SLAM(Node):

    def __init__(self):

        super().__init__("ekf_slam")

        self.position_pub = self.create_publisher(PoseWithCovarianceStamped,'/ekf_slam_pose',10)

        self.lidar_sub = self.create_subscription(LaserScan,'/scan',self.laserCB,10)
        self.odom_sub = self.create_subscription(Odometry,'/odom',self.odomCB,10)

        self.x_prev,self.y_prev,self.theta_prev = 0.0,0.0,0.0
        self.mu_bar = np.zeros((3,1))
        self.total_landmarks = 0
        self.final_covariance = np.zeros((3,3))
        self.mu = np.zeros((3,1)) 
        self.sigma_r = 0.1 # This is to define standard deviation in the distance measurement
        self.sigma_alpha = 0.01  # This is to define the standard deviation in the angle measurement

    def laserCB(self,msg:LaserScan):
        """This functioon serves as the entry point to the slam implementation, as well as subscribes to the lidar data."""
        while(msg.ranges==None):
            self.get_logger().warn("The lidar data is not being subscribed")
        self.laser_points = [0]*len(msg.ranges)
        for i, value in enumerate(msg.ranges[:-1]):
            if not math.isinf(value) and not math.isnan(value):
                self.laser_points[i] = value
            else:
                self.laser_points[i] = msg.range_max

    def odomCB(self,msg:Odometry):
        """This function serves the purpose of subscribing to the odom data which helps to generate the control parameters."""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, self.theta) = euler_from_quaternion (orientation_list)

    def pose_predictor(self):
        """This function is used to calculate the predicted pose by generating different contol parameters."""
        self.delta_trans = math.sqrt((self.x-self.x_prev)**2+(self.y-self.y_prev)**2)
        self.delta_rot1 = normalize_angle(math.atan2(self.y - self.y_prev, self.x - self.x_prev) - self.theta_prev)
        self.delta_rot2 = normalize_angle(self.theta - self.theta_prev - self.delta_rot1)

        self.x_predicted = self.x_prev + self.delta_trans*math.cos(self.theta + self.delta_rot1)
        self.y_predicted = self.y_prev + self.delta_trans*math.sin(self.theta + self.delta_rot1)
        self.theta_predicted = normalize_angle(self.theta_prev + self.delta_rot1 + self.delta_rot2)

        self.mu_bar[:3,:] = np.array([
            [self.x_predicted],
            [self.y_predicted],
            [self.theta_predicted]
        ])

    def state_cov_calc(self):
        """This function performs the calculation of the covariance matrix formed by diff. w.r.t. states for robot"""
        self.G_t = np.eye(3+2*self.total_landmarks)

        self.G_t[:3,:3] = np.array([
            [1 , 0  , -self.delta_trans*math.sin(self.theta_prev+self.delta_rot1)],
            [0 , 1 , self.delta_trans*math.cos(self.theta_prev+self.delta_rot1)],
            [0 , 0, 1]
        ])
        
    def control_cov_calc(self):
        """This function performs the calculation of the covariance matrix formed by diff. w.r.t. controls for robot"""   
        self.V_t = np.zeros((3+2*self.total_landmarks,3))

        self.V_t[:3, :] = np.array([
            [-self.delta_trans * math.sin(self.theta_prev + self.delta_rot1), math.cos(self.theta_prev + self.delta_rot1), 0],
            [ self.delta_trans * math.cos(self.theta_prev + self.delta_rot1), math.sin(self.theta_prev + self.delta_rot1), 0],
            [1, 0, 1]
        ])


    def predicted_covariance(self):
        """This function calculates the predicted covariance"""
        alpha1 = 0.05
        alpha2 = 0.01
        alpha3 = 0.05
        alpha4 = 0.01
        self.rot1_variance = alpha1 * pow((self.delta_rot1),2) + alpha2 * pow((self.delta_trans),2)
        self.trans_variance = alpha3 * pow((self.delta_trans),2) + alpha4 * (pow((self.delta_rot1),2) + pow((self.delta_rot2),2))
        self.rot2_variance = alpha1 * pow((self.delta_rot2),2) + alpha2 * pow((self.delta_trans),2)
        Identity_mat = np.eye(3+2*self.total_landmarks)
        noise = np.diag([self.rot1_variance, self.trans_variance, self.rot2_variance]) #M_t matrix
        R_t = np.dot(self.V_t, np.dot(noise, self.V_t.T))
        self.noise_cov = np.dot(Identity_mat.T,np.dot(R_t,Identity_mat))

        #print("G_t shape:", self.G_t.shape)
        #print("final_covariance shape:", self.final_covariance.shape)
        #print("G_t.T shape:", self.G_t.T.shape)
        #print("V_t shape:", self.V_t.shape)
        #print("noise shape:", noise.shape)
        #print("V_t.T shape:", self.V_t.T.shape)

        self.covariance_bar = np.dot(self.G_t, np.dot(self.final_covariance, self.G_t.T)) + self.noise_cov
        
        _,self.size = np.shape(self.covariance_bar)

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
            if abs(derivative) > 0.1:
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
        #print("Index limit:", index)
        while(i<(self.index-3)/2):
            x_land_curr = self.mu_bar[3+2*i][0]
            y_land_curr = self.mu_bar[4+2*i][0]
            x_diff = self.x_predicted - x_land_curr
            y_diff = self.y_predicted - y_land_curr
            dist = math.sqrt(math.pow((x_diff),2) + math.pow((y_diff),2))
            theta = math.atan2(y_diff,x_diff)
            meas_dist_list.append(dist)
            meas_ang_list.append(theta)
            diff_X_list.append(x_diff)
            diff_Y_list.append(y_diff)
            i+=1 #ek bhul

        self.z_prev = np.vstack((meas_dist_list,meas_ang_list))
        self.diff_estim = np.vstack((diff_X_list,diff_Y_list))

    def cylin_pairing(self):
        """Pairing of the cylinders based on the distance tolerance"""
        
        tolerance = 0.2
        self.total_meas = [False]*self.num_meas
        self.left_unmatched_from_prev = [True]*self.total_landmarks
        self.z_curr_updated = []
        self.indices = []

        paired_meas_dist = []
        paired_meas_angle = []
        paired_estim_dist = []
        paired_estim_angle = []
        paired_estim_diff_x = []
        paired_estim_diff_y = []

        if self.z_prev.shape[1] == 0:
            return  # or handle the case appropriately


        for i in range(self.z_prev.shape[1]):
            for j in range(self.z_curr.shape[1]):
                if (abs(self.z_prev[0, i] - self.z_curr[0, j]) < 10):
                    self.total_meas[j] = True
                    self.left_unmatched_from_prev[i] = False
                    self.z_curr_updated.append(self.z_curr[:,j])
                    paired_meas_dist.append(self.z_curr[0, j])
                    paired_meas_angle.append(self.z_curr[1, j])
                    paired_estim_dist.append(self.z_prev[0, i])
                    paired_estim_angle.append(self.z_prev[1, i])
                    paired_estim_diff_x.append(self.diff_estim[0, i])
                    paired_estim_diff_y.append(self.diff_estim[1, i])
                    # print("Paired measurements:", paired_meas_dist)
                    # print("Paired estimations:", paired_estim_dist)
                    break
            if not self.left_unmatched_from_prev[i]:
                self.indices.append(i)
                
        
        if paired_meas_dist and paired_meas_angle:
            self.paired_measurements = np.vstack((paired_meas_dist, paired_meas_angle))
        else:
            self.paired_measurements = np.array([])
        
        if paired_estim_dist and paired_estim_angle:
            self.paired_estimations = np.vstack((paired_estim_dist, paired_estim_angle))
        else:
            self.paired_estimations = np.array([])
        
        if paired_estim_diff_x and paired_estim_diff_y:
            self.paired_estim_diff = np.vstack((paired_estim_diff_x, paired_estim_diff_y))
        else:
            self.paired_estim_diff = np.array([])
        #self.get_logger().info(f"Paired measurements: {self.paired_measurements}, Paired estimations: {self.paired_estimations}")

    def z_to_states_for_new_cylin(self):

        state_mat = np.empty((2,1))

        self.check_flag = 1
        for i in range(len(self.total_meas)):
            if not self.total_meas[i]:
                print(self.total_meas[i])
                x_new = self.x_predicted + (self.z_curr[0,i]*math.cos(self.z_curr[1,i]))
                y_new = self.y_predicted + (self.z_curr[0,i]*math.sin(self.z_curr[1,i]))
                self.check_flag = 0
                state_mat = np.vstack((state_mat, x_new,y_new))
                #self.get_logger().info(f"New cylinder detected at x: {x_new}, y: {y_new}")

        self.mu_new_cylin = state_mat

    def correction_step(self):
        """This function performs the correction step"""
        if self.z_prev.shape[1] == 0:
            return  # or handle the case appropriately
        dime = 3+2*self.total_landmarks
        H_t_mat = np.zeros((2, 3))
        self.cov_sub_mat = np.eye((self.index))
        self.cov_sub_mat[:3,:3] = (self.covariance_bar[:3,:3])
        self.mu = self.mu_bar
        self.mu_sub_mat = np.zeros((self.index, 1))
        self.mu_sub_mat[:3,0] = self.mu[:3,0]
        
        self.final_covariance = self.covariance_bar
        self.mu = self.mu_bar
        
        
        

        for i in range(len(self.indices)):
            cylinder_original_placeholder = self.indices[i]
            self.cov_sub_mat[3+2*i:5+2*i,3+2*i:5+2*i] = self.final_covariance[3+2*cylinder_original_placeholder:5+2*cylinder_original_placeholder,3+2*cylinder_original_placeholder:5+2*cylinder_original_placeholder]
            self.mu_sub_mat[i:i+2,0] = self.mu_bar[cylinder_original_placeholder:cylinder_original_placeholder+2,0]

        for i in range(len(self.indices)):
            #Have to make changes to the cylin_pairing function to store the paired estim diff as well similar to the localization code.
            meas_dist = self.paired_measurements[0, i]
            meas_ang = self.paired_measurements[1, i]

            z_matrix = np.array([[meas_dist], [meas_ang]])
            h_matrix = np.array([[self.paired_estimations[0, i]], [self.paired_estimations[1, i]]])

            H_t_mat[:2, :3] = np.array([
                    [(-self.paired_estim_diff[0, i] / self.paired_estimations[0, i]),
                     (-self.paired_estim_diff[1, i] / self.paired_estimations[0, i]), 0],
                    [(self.paired_estim_diff[1, i] / (self.paired_estimations[0, i] ** 2)),
                     (-self.paired_estim_diff[0, i] / (self.paired_estimations[0, i] ** 2)), -1]
                ])
            
            H_obs = np.array(-H_t_mat[:2, :2])

            H_t_mat = np.hstack((H_t_mat, H_obs))

            q_matrix = np.array([
                    [self.sigma_r**2, 0],
                    [0, self.sigma_alpha**2]
                ])
            
            S = np.dot(H_t_mat, np.dot(self.cov_sub_mat[:5+2*i,:5+2*i], H_t_mat.T)) + q_matrix
            k_gain = np.dot(self.cov_sub_mat[:5+2*i,:5+2*i], np.dot(H_t_mat.T, np.linalg.inv(S)))

            Innovation_matrix = np.array([
                    [z_matrix[0, 0] - h_matrix[0, 0]],
                    [normalize_angle(z_matrix[1, 0] - h_matrix[1, 0])]
                ])
            update = np.dot(k_gain, Innovation_matrix).squeeze()  # or .flatten()
            self.mu_sub_mat[:5+2*i, 0] = self.mu_sub_mat[:5+2*i, 0] + update
            Identity = np.eye(5+2*i)
            self.cov_sub_mat[:5+2*i,:5+2*i] = np.dot((Identity - np.dot(k_gain, H_t_mat)), self.cov_sub_mat[:5+2*i,:5+2*i])
            
        self.mu[:3,0] = self.mu_bar[:3,0]
        self.final_covariance[:3,:3] = self.cov_sub_mat[:3,:3]

        for i in range(len(self.indices)):
            self.final_covariance[
                3 + 2 * self.indices[i] : 5 + 2 * self.indices[i],
                3 + 2 * self.indices[i] : 5 + 2 * self.indices[i]
            ]         = self.cov_sub_mat[3+2*i:5+2*i,3+2*i:5+2*i]

            self.mu[3+2*self.indices[i]:5+2*self.indices[i],0] = self.mu_sub_mat[3+2*i:5+2*i,0]

        self.mu_bar = self.mu
        self.x_prev = self.x
        self.y_prev = self.y
        self.theta_prev = self.theta

        


    def add_new_cylin(self):

        if self.check_flag == 1:
            return #no new cylins observed

        for i in range(len(self.total_meas)):
            if not self.total_meas[i]:
                self.total_landmarks+=1

        self.mu = np.vstack((self.mu,self.mu_new_cylin))
        final_cov = (np.eye(3+2*self.total_landmarks))*100
        final_cov[:(np.shape(self.final_covariance)[0]),:(np.shape(self.final_covariance)[0])] = self.final_covariance

        self.final_covariance = final_cov
        self.mu_bar = self.mu
        # self.get_logger().info(f"Covariance = {self.final_covariance}")
        # self.get_logger().info(f"mu = {self.mu}")
        self.get_logger().info(f"The shape of the {np.shape(self.final_covariance)}")


    def run(self):
        self.pose_predictor()
        self.state_cov_calc()
        self.control_cov_calc()
        self.predicted_covariance()
        self.observed_landmarks()
        self.z_mat_from_previous_state()
        self.cylin_pairing()
        self.z_to_states_for_new_cylin()
        self.correction_step()
        self.add_new_cylin()

def main(args = None):
    rclpy.init(args=args)
    node = EKF_SLAM()
    timer_period = .4
    node.create_timer(timer_period,node.run)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()