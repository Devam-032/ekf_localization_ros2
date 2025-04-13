#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math as m

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from gazebo_msgs.msg import ModelStates
from tf_transformations import euler_from_quaternion, quaternion_from_euler

def scale(angle):
    return (angle + m.pi) % (2 * m.pi) - m.pi

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_localization_node')
        self.create_subscription(ModelStates, '/gazebo/model_states', self.obj_original_pose, 10)
        self.create_subscription(Odometry, '/odom', self.pos_measurement_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.pos_lidar_callback, 10)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/filtered_data', 10)

        # Initialize state variables
        self.m_prev_x = 0.0
        self.m_prev_y = 0.0
        self.m_prev_0 = 0.0  # previous orientation
        self.Dt = 0.0
        self.Dr1 = 0.0
        self.Dr2 = 0.0
        self.m_x = 0.0
        self.m_y = 0.0
        self.m_0 = 0.0

        self.covariance_prev = np.eye(3) * 0.001  # initial covariance matrix

        # Motion model noise parameters
        self.alpha1 = 0.05  
        self.alpha2 = 0.01  
        self.alpha3 = 0.05
        self.alpha4 = 0.01  

        self.lidar_distance = []  # average lidar ranges
        self.lidar_angle = []     # corresponding angles

        self.observed = []        # observed coordinates of landmarks
        self.original_coord = []  # reference coordinates of landmarks

        # Measurement noise parameters
        self.sigma_r = 0.1      # range measurement noise
        self.sigma_phi = 0.01  # bearing measurement noise
        self.zm = []

        self.meas = np.array([])    # measured (actual) landmarks [range; angle]
        self.o = np.array([])       # expected (reference) landmarks [range; angle]
        self.difff = np.array([])   # differences in x and y for Jacobian use

        # Create a timer to update at 4 Hz (0.25 sec period)
        self.timer = self.create_timer(0.5, self.timer_callback)

    def obj_original_pose(self, msg:ModelStates):
        self.original_coord = []
        for i, name in enumerate(msg.name):
            if name not in ["ground_plane", "waffle", "mappp"]:
                pos = msg.pose[i].position
                self.original_coord.append([pos.x, pos.y])

    def pos_measurement_callback(self, msg:Odometry):
        self.m_x = msg.pose.pose.position.x
        self.m_y = msg.pose.pose.position.y
        thetaa = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([thetaa.x, thetaa.y, thetaa.z, thetaa.w])
        self.m_0 = scale(yaw)
        self.Dt = m.sqrt((self.m_x - self.m_prev_x)**2 + (self.m_y - self.m_prev_y)**2)
        self.Dr1 = scale(m.atan2(self.m_y - self.m_prev_y, self.m_x - self.m_prev_x) - self.m_prev_0)
        self.Dr2 = scale(self.m_0 - self.m_prev_0 - self.Dr1)

    def state_covariance(self):
        self.Gt = np.array([
            [1, 0, -self.Dt * m.sin(self.m_prev_0 + self.Dr1)],
            [0, 1,  self.Dt * m.cos(self.m_prev_0 + self.Dr1)],
            [0, 0, 1]
        ])

    def control_covariance(self):
        self.Vt = np.array([
            [-self.Dt * m.sin(self.m_prev_0 + self.Dr1), m.cos(self.m_prev_0 + self.Dr1), 0],
            [ self.Dt * m.cos(self.m_prev_0 + self.Dr1), m.sin(self.m_prev_0 + self.Dr1), 0],
            [1, 0, 1]
        ])

    def process_noise(self):
        self.Mt = np.array([
            [self.alpha1 * (self.Dr1**2) + self.alpha2 * (self.Dt**2), 0, 0],
            [0, self.alpha3 * (self.Dt**2) + (self.alpha4 * ((self.Dr2**2) + (self.Dr1**2))), 0],  # process noise
            [0, 0,(self.alpha1 * (self.Dr1**2)) + (self.alpha2 * (self.Dt**2))]
        ])

    def mu_mean(self):
        self.x_bel = self.m_prev_x + self.Dt * m.cos(scale(self.m_prev_0 + self.Dr1))
        self.y_bel = self.m_prev_y + self.Dt * m.sin(scale(self.m_prev_0 + self.Dr1))
        self.theta_bel = scale(self.m_prev_0 + self.Dr1 + self.Dr2)
        self.muBar = np.array([[self.x_bel],
                               [self.y_bel],
                               [self.theta_bel]])
        #print(self.muBar)

    def sigma_covariance(self):
        self.state_covariance()    # Ensure Gt is updated
        self.control_covariance()  # Ensure Vt is updated
        self.sigmaBar = self.Gt @ self.covariance_prev @ self.Gt.T + self.Vt @ self.Mt @ self.Vt.T
        #print(self.sigmaBar)
                     

    def pos_lidar_callback(self, scan):
        n = len(scan.ranges)
        self.x = [0.0] * n
        self.lidar_distance = []
        self.lidar_angle = []
        self.diff = [0.0] * n

        # Process each range value and compute the difference with the previous value
        for i in range(n):
            if m.isnan(scan.ranges[i]) or m.isinf(scan.ranges[i]):
                self.x[i] = 3.0
            else:
                self.x[i] = scan.ranges[i]
            if i > 0 and self.x[i] > 0.2 and self.x[i-1] > 0.2:
                diff_val = self.x[i] - self.x[i - 1]
            else:
                diff_val = 0.0
            self.diff[i] = diff_val if abs(diff_val) > 0.15 else 0.0

        obj_detected = 0
        add_ranges = 0.0
        indices = 0
        add_index = 0
        j = -1
        objects = 0

        while j < (len(self.diff) - 1):                             # Process the diff array to estimate average angles and ranges for objects
            if self.diff[j] < 0 and obj_detected == 0:              #a sudden decrease in range is detected
                obj_detected = 1
                indices += 1                                        # add indices  
                add_index += j                                      # and lidar angles 
                add_ranges += self.x[j]                             # and ranges
            elif self.diff[j] < 0 and obj_detected == 1:
                obj_detected = 0                                    #reset all if another sudden decrease in ranges is detected
                indices = 0
                add_index = 0
                add_ranges = 0.0
                j -= 1
            elif self.diff[j] == 0 and obj_detected == 1:
                indices += 1                                        # if no change in ranges but object is detected
                add_ranges += self.x[j]                             # add all the paramters
                add_index += j
            elif self.diff[j] > 0 and obj_detected == 1:
                obj_detected = 0
                avg_angle = scale((add_index * (m.pi/180)) / indices)
                self.lidar_angle.append(avg_angle)                              #take average to get approx near values
                self.lidar_distance.append(add_ranges / indices)
                objects += 1
                indices = 0
                add_index = 0
                add_ranges = 0.0
            else:
                continue
            j+=1
        self.zm = np.vstack([self.lidar_distance, self.lidar_angle])
        # self.get_logger().info(f"Number of landmarks in original coordinates: {len(self.original_coord)}")


    def obs_meas_difference(self):
        """Matching the measured and reference coordinates of landmarks"""
        self.orig_dist = []
        self.orig_angle = []
        self.diff_x = []
        self.diff_y = []
        self.meas_dist = []
        self.meas_angle = []
        self.o_dist =  []
        self.o_angle = []
        self.difff_x = []
        self.difff_y = []
        object = -1
        for i in range(len(self.original_coord)):
            diff_x = self.original_coord[i][0] - self.x_bel
            diff_y = self.original_coord[i][1] - self.y_bel
            dist = m.sqrt(diff_x**2 + diff_y**2)
            angle = scale(m.atan2(diff_y, diff_x) - self.theta_bel)
            self.orig_dist.append(dist)
            self.orig_angle.append(angle)
            self.diff_x.append(diff_x)
            self.diff_y.append(diff_y)
            #print(self.original_coord[i][0])
        self.orig = np.vstack((self.orig_dist, self.orig_angle))
        self.diff_1 = np.vstack((self.diff_x, self.diff_y))
        
        #print(f"Original coordinates: {self.orig}")

        if (len(self.original_coord)) == 0 or (len(self.lidar_distance)) == 0:
            self.get_logger().warn("No landmarks detected!")
            return
        for i in range(len(self.orig[1])):
            for j in range(len(self.zm[1])):
                if abs(self.orig[0][i] - self.zm[0][j]) < 0.25 and abs(self.orig[1][i] - self.zm[1][0]) < 0.25:
                    self.meas_dist.append(self.zm[0][j])
                    self.meas_angle.append(self.zm[1][j])
                    self.o_dist.append(self.orig[0][i])
                    self.o_angle.append(self.orig[1][i])
                    self.difff_x.append(self.diff_1[0][i])
                    self.difff_y.append(self.diff_1[1][i])
                    object += j
        
        if self.meas_angle and self.meas_dist:      
            self.meas = np.vstack((self.meas_dist, self.meas_angle))
            self.get_logger().info(f"No. of landmarks, {object}")
        else:
            self.meas = np.array([])
        if self.o_dist and self.o_angle:
            self.o = np.vstack((self.o_dist, self.o_angle))
        else:
            self.o = np.array([])
        if self.difff_x and self.difff_y:
            self.difff = np.vstack((self.difff_x, self.difff_y))
        else:
            self.difff = np.array([])
        
        #print(f"o:{self.o}, m: {self.meas}")

    def correction_matrices(self):
        """Correction step using landmark observations"""
        if self.meas.size == 0:                  # If no landmarks observed, just use prediction
            #self.get_logger().warn("No observed landmarks found!")
            self.Mu = self.muBar
            self.covariance_prev = self.sigmaBar
        
        else:
            for i in range(self.meas.shape[1]):            # Process each landmark sequentially
                measDist = self.meas[0][i]                     # Expected measurement
                measAngle = scale(self.meas[1][i])
                
                Zt = np.array([[measDist],               # Create measurement vectors of actual coordinates
                            [measAngle]])
                
                hx = np.array([[self.o[0][i]],               # Create measurement vectors of expected coordinates
                            [scale(self.o[1][i])]])
                
                dx = self.difff[0][i]
                dy = self.difff[1][i]
                r = self.o[0][i]                            # expected range

                Ht = np.array([
                    [-dx / r, -dy / r, 0],
                    [dy / (r**2), -dx / (r**2), -1]
                ])
                
                
                Qt = np.array([[self.sigma_r**2, 0],               # Measurement noise covariance
                            [0, self.sigma_phi**2]])
                
                #print(self.sigmaBar)
                S_t = Ht @ self.sigmaBar @ Ht.T + Qt     # Innovation covariance   
                
                innovation = np.array([[Zt[0][0] - hx[0][0]],
                                    [scale(Zt[1][0] - hx[1][0])]])    #innovation matrix for error between observed and calculated
                
                K_t = self.sigmaBar @ Ht.T @ (np.linalg.inv(S_t))             # Calculate Kalman gain
        
                    
                correction = K_t @ innovation                # Update state estimate        
                self.Mu = self.muBar + correction  
                
                I_KH = np.eye(3) - (K_t @ Ht)             
                self.covariance_prev = I_KH @ self.sigmaBar   #updated covariance
                
                print(f"Z_t: {S_t}, {i}")    
        #print(self.covariance_prev)
        
        # Update previous state values
        self.m_prev_x = self.m_x
        self.m_prev_y = self.m_y
        self.m_prev_0 = self.m_0

    def publish_mu_sigma(self):
        if self.pub.get_subscription_count() == 0:
            #self.get_logger().warn("No subscribers to /filtered_data yet!")
            return

        pose = PoseWithCovarianceStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "odom"

        pose.pose.pose.position.x = self.Mu[0, 0]
        pose.pose.pose.position.y = self.Mu[1, 0]
        quat = quaternion_from_euler(0, 0, self.Mu[2, 0])
        pose.pose.pose.orientation.z = quat[2]
        pose.pose.pose.orientation.w = quat[3]

        full_covariance = np.zeros((6, 6))
        full_covariance[:3, :3] = self.covariance_prev
        pose.pose.covariance = full_covariance.ravel().tolist()

        self.pub.publish(pose)
        #self.get_logger().info(f"Published estimate: x={self.Mu[0,0]:.3f}, y={self.Mu[1,0]:.3f}, theta={self.Mu[2,0]:.3f}")

    def timer_callback(self):
        self.control_covariance()
        self.state_covariance()
        self.process_noise()
        self.mu_mean()
        self.sigma_covariance()
        self.obs_meas_difference()
        self.correction_matrices()
        self.publish_mu_sigma()

def main(args=None):
    rclpy.init(args=args)
    ekf_localization_node = EKFNode()
    rclpy.spin(ekf_localization_node)
    ekf_localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()