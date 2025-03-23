#!/usr/bin/env python3

import rclpy,math
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion,quaternion_from_euler
from ekf_localization.ekf_localization import normalize_angle

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
        control_covariance = np.diag([self.rot1_variance, self.trans_variance, self.rot2_variance]) #M_t matrix

        self.covariance_bar = np.dot(self.G_t, np.dot(self.final_covariance, self.G_t.T)) + np.dot(self.V_t, np.dot(control_covariance, self.V_t.T))
        
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
                # Optionally add an offset (here +0.25 as in your code)
                self.approx_linear_distance.append(avg_depth + 0.25)
                # Reset for the next cylinder detection
                cylin_active = False
                no_of_rays = 0
                sum_ray_indices = 0
                sum_depth = 0.0
            i += 1
        
        num_meas = len(self.approx_linear_distance)
        self.get_logger().info(f"Number of cylinder measurements: {num_meas}")
        self.total_landmarks = num_meas
        self.final_covariance = np.eye(3+2*self.total_landmarks)
        self.final_covariance[:self.size,:self.size] = self.covariance_bar
        self.final_covariance[self.size:,self.size:] *= 10

    def robot_pose_publisher(self):
        clock = Clock()
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = clock.now().to_msg()

        pose_msg.header.frame_id = "odom"  # Use the appropriate frame of reference

        # Setting the pose based on the mean (mu)
        pose_msg.pose.pose.position.x = self.mu_bar[0, 0]
        pose_msg.pose.pose.position.y = self.mu_bar[1, 0]
        pose_msg.pose.pose.position.z = 0.0  # Assume planar navigation

        # Convert orientation from Euler to quaternion
        quat = quaternion_from_euler(0, 0, self.mu_bar[2, 0])
        pose_msg.pose.pose.orientation.x = quat[0]
        pose_msg.pose.pose.orientation.y = quat[1]
        pose_msg.pose.pose.orientation.z = quat[2]
        pose_msg.pose.pose.orientation.w = quat[3]

        # Fill in the covariance (flattened row-major order)
        covariance_flat = self.final_covariance.flatten()
        pose_msg.pose.covariance = [float(covariance_flat[i]) if i < len(covariance_flat) else 0.0 for i in range(36)]



        # Publish the message
        self.position_pub.publish(pose_msg)

    def run(self):
        self.pose_predictor()
        self.state_cov_calc()
        self.control_cov_calc()
        self.predicted_covariance()
        self.observed_landmarks()
        self.robot_pose_publisher()  

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
        