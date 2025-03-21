#!/usr/bin/env python3
import rclpy,math
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import Twist,PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ekf_localization.ekf_localization import EKF_LOCALIZATION
from tf_transformations import euler_from_quaternion,quaternion_from_euler


def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi 

class EKF_SLAM(Node):

    def __init__(self):
        super().__init__('ekf_slam')

        #Publishers
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped,"/pose_with_cov_stamped",10)

        #Subscribers
        self.odom_sub = self.create_subscription(Odometry,'/odom',self.odomCb,10)
        self.laser_sub = self.create_subscription(LaserScan,'/scan',self.laserCb,10)

        #Variable_Initializations
        self.x_prev,self.y_prev,self.theta_prev =0,0,0
        self.mu_bar = np.array([[0],[0],[0]])
        self.G_t = np.eye(3)
        self.covariance_bar = np.zeros((3, 3))
        self.final_covariance = np.zeros((3, 3))
        self.R_t = np.zeros((3, 3))


    def laserCb(self,msg:LaserScan):
        """This Function filters the lidar scan data and then stores it in a class variable."""
        while(msg.ranges==None):
            self.get_logger().warn("The lidar data is not being subscribed")
        self.laser_points = [0]*len(msg.ranges)
        for i, value in enumerate(msg.ranges[:-1]):
            if not math.isinf(value) and not math.isnan(value):
                self.laser_points[i] = value
            else:
                self.laser_points[i] = msg.range_max

    def odomCb(self,msg:Odometry):
        self.x=msg.pose.pose.position.x
        self.y=msg.pose.pose.position.y

        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, self.yaw) = euler_from_quaternion (orientation_list)
        self.theta = normalize_angle(self.yaw)
        self.delta_trans = math.sqrt((self.x-self.x_prev)**2+(self.y-self.y_prev)**2)
        self.delta_rot1 = normalize_angle(math.atan2(self.y - self.y_prev, self.x - self.x_prev) - self.theta_prev)
        self.delta_rot2 = normalize_angle(self.theta - self.theta_prev - self.delta_rot1)

    def pose_predictor(self):
        """This step is to predict the pose of the robot using the odometry motion model."""
        self.x_predicted = self.x_prev + self.delta_trans*math.cos(self.theta + self.delta_rot1)
        self.y_predicted = self.y_prev + self.delta_trans*math.sin(self.theta + self.delta_rot1)
        self.theta_predicted = normalize_angle(self.theta_prev + self.delta_rot1 + self.delta_rot2)


        self.mu_bar[:3] = np.array([
            [self.x_predicted],
            [self.y_predicted],
            [self.theta_predicted]
        ])

    def state_cov_pred(self):
        """This function serves as the calculation of state_covariance."""
        self.G_t[:3,:3] = np.array([
            [1 , 0  , -self.delta_trans*math.sin(self.theta_prev+self.delta_rot1)],
            [0 , 1 , self.delta_trans*math.cos(self.theta_prev+self.delta_rot1)],
            [0 , 0, 1]
        ]) #W.R.T. STATES(POSITION,ORIENTATION)

    def cont_cov_pred(self):
        """This function is used to obtain the covariance in the control signals given to the robot."""
        self.V = np.array([
            [-self.delta_trans*(math.sin(self.theta_prev+self.delta_rot1)) , math.cos(self.theta_prev + self.delta_rot1) , 0],
            [self.delta_trans*math.cos(self.theta_prev + self.delta_rot1) , math.sin(self.theta_prev + self.delta_rot1) , 0],
            [1 , 0 , 1]
        ]) #W.R.T. CONTROLS U=[DEL_R1,DEL_T,DEL_R2]

    def prediction_covariance_calc(self):
        """This function is used to get the exact prediction covariance."""
        alpha1 = 0.05
        alpha2 = 0.01
        alpha3 = 0.05
        alpha4 = 0.01
        self.rot1_variance = alpha1 * abs(self.delta_rot1) + alpha2 * abs(self.delta_trans)
        self.trans_variance = alpha3 * abs(self.delta_trans) + alpha4 * (abs(self.delta_rot1) + abs(self.delta_rot2))
        self.rot2_variance = alpha1 * abs(self.delta_rot2) + alpha2 * abs(self.delta_trans)
        control_covariance = np.diag([self.rot1_variance, self.trans_variance, self.rot2_variance]) #M_t matrix
        self.R_t[:3,:3] = np.dot(self.V, np.dot(control_covariance, self.V.T))

        self.covariance_bar[:3,:3] = np.dot(self.G_t, np.dot(self.final_covariance, self.G_t.T)) + self.R_t

    def feature_detection(self):
        """
        Process the lidar scan (self.laser_points) to detect jumps.
        Each jump is assumed to correspond to a cylinder edge.
        The average ray index and depth for each detected cylinder region
        are stored in self.approx_linear_distance and self.approx_angular_position.
        """
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

    def publish_pose_with_covariance(self):
        self.mu = self.mu_bar
        self.final_covariance = self.covariance_bar
        clock = Clock()
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = clock.now().to_msg()

        pose_msg.header.frame_id = "odom"  # Use the appropriate frame of reference

        # Setting the pose based on the mean (mu)
        pose_msg.pose.pose.position.x = float(self.mu[0, 0])
        pose_msg.pose.pose.position.y = float(self.mu[1, 0])

        pose_msg.pose.pose.position.z = 0.0  # Assume planar navigation

        # Convert orientation from Euler to quaternion
        quat = quaternion_from_euler(0, 0, self.mu[2, 0])
        pose_msg.pose.pose.orientation.x = quat[0]
        pose_msg.pose.pose.orientation.y = quat[1]
        pose_msg.pose.pose.orientation.z = quat[2]
        pose_msg.pose.pose.orientation.w = quat[3]

        # Fill in the covariance (flattened row-major order)
        covariance_flat = self.final_covariance.flatten()
        pose_msg.pose.covariance = [float(covariance_flat[i]) if i < len(covariance_flat) else 0.0 for i in range(36)]


        # Publish the message
        self.pose_pub.publish(pose_msg)

        self.theta_prev = self.theta
        self.x_prev = self.x
        self.y_prev =self.y
        

    def run(self):
        self.pose_predictor()
        self.state_cov_pred()
        self.cont_cov_pred()
        self.prediction_covariance_calc()
        self.publish_pose_with_covariance()

def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = EKF_SLAM()
    timer_period = .5  # seconds
    node.create_timer(timer_period, node.run)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
