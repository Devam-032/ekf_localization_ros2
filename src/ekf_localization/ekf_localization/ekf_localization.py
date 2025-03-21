#!/usr/bin/env python3
import rclpy,math
import numpy as np
import rclpy.logging
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped,PoseStamped,Point
from nav_msgs.msg import Odometry,Path
from tf_transformations import euler_from_quaternion,quaternion_from_euler
from rclpy.clock import Clock
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64MultiArray,Bool

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi 

class EKF_LOCALIZATION(Node):

    def __init__(self):
        super().__init__("Localization_script")
        self.lidar_sub_  = self.create_subscription(LaserScan,'/scan',self.laserCb,10)
        # self.ref_cylin_sub_ = self.create_subscription(ModelStates,'/gazebo/model_states',self.reference_coordinates,10)
        self.odom_sub_ = self.create_subscription(Odometry,"/odom",self.odomCb,10)
        self.delta_trans = 0.0
        self.delta_rot1 = 0.0
        self.delta_rot2 = 0.0
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.theta = 0.0
        self.sigma_r = 0.1 # This is to define standard deviation in the distance measurement
        self.sigma_alpha = 0.01  # This is to define the standard deviation in the angle measurement
        self.theta_prev = 0.0
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped,"/pose_with_cov_stamped",10)
        self.ref_marker_pub = self.create_publisher(MarkerArray, '/reference_cylinder_markers', 10)
        self.final_covariance = np.eye(3)
        self.ref_flag_pub = self.create_publisher(Bool, '/ref_coords_ready', 10)
        # self.reference_cylin = [
        #     [1.3014, -1.74091],
        #     [1.98771, 1.13637],
        #     [-1.13443, -0.902541],
        #     [-1.37644, 2.26725],
        #     [3.5403, -1.8486],
        #     [1.05571, 3.07389],
        #     [3.58534, 2.92355],
        #     [4.6648, 1.68058],
        #     [4.81043, -2.61818],
        #     [3.30091, -4.2055],
        #     [0.414463, -4.33209],
        #     [-1.32098, -3.92076],
        #     [-2.85487, -2.88071],
        #     [-3.23012, 0.1117],
        #     [-3.15908, 3.68903],
        #     [-0.413215, 3.60968],
        #     [-1.80846, 0.902126]
        # ]

        self.reference_cylin=[]
        self.ref_coords_received = False

        self.trajectory_pub = self.create_publisher(Path, '/robot_path', 10)
        self.robot_path = Path()
        self.robot_path.header.frame_id = "odom"  # Frame to visualize in RViz
        
        self.real_traj_pub = self.create_publisher(Path, '/real_robot_path', 10)
        self.real_robot_path = Path()
        self.real_robot_path.header.frame_id = "odom"

        self.marker_pub = self.create_publisher(MarkerArray, '/estimated_cylinder_markers', 10)
        self.error_ellipse_pub = self.create_publisher(Marker, '/error_ellipse', 10)

        self.ref_coords_sub = self.create_subscription(Float64MultiArray,'/reference_coords',self.reference_coords_callback,10)

    def reference_coords_callback(self, msg: Float64MultiArray):
    # Check if we've already processed a message
        if self.ref_coords_received:
            return  # Do nothing if already received once
        self.ref_coords_received = True

        # Assume msg.data is a flat list: [x1, y1, x2, y2, ...]
        data = msg.data
        if len(data) % 2 != 0:
            self.get_logger().error("Reference coordinates data length is not even!")
            return
        new_coords = []
        for i in range(0, len(data), 2):
            new_coords.append([data[i], data[i+1]])
        self.reference_cylin = new_coords
        self.get_logger().info(f"Updated reference coordinates: {self.reference_cylin}")

        flag_msg = Bool()
        flag_msg.data = True
        self.ref_flag_pub.publish(flag_msg)
        self.get_logger().info("Published reference flag on /ref_coords_ready")

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

    # def reference_coordinates(self, msg):
    #     a2 = len(msg.pose) - 3
    #     self.reference_cylin = []  # Initialize as an empty list

    #     for i in range(a2):
    #         # Append each coordinate as a list
    #         self.reference_cylin.append([msg.pose[2+i].position.x, msg.pose[2+i].position.y])

    def odomCb(self,msg:Odometry):
        """This function is used to initialize the position and orientation of the robot and also initialize the parameters req. for Odom model."""
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


        self.mu_bar = np.array([
            [self.x_predicted],
            [self.y_predicted],
            [self.theta_predicted]
        ])

    def state_covariance_calc(self):
        """This function serves as the calculation of state_covariance."""
        self.G_t = np.array([
            [1 , 0  , -self.delta_trans*math.sin(self.theta_prev+self.delta_rot1)],
            [0 , 1 , self.delta_trans*math.cos(self.theta_prev+self.delta_rot1)],
            [0 , 0, 1]
        ]) #W.R.T. STATES(POSITION,ORIENTATION)
    
    def control_covariance_calc(self):
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

        self.covariance = np.dot(self.G_t, np.dot(self.final_covariance, self.G_t.T)) + np.dot(self.V, np.dot(control_covariance, self.V.T))

    def observed_cylinders(self):
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

        # For debugging, print number of detected measurements:
        # num_meas = len(self.approx_linear_distance)
        # self.get_logger().info(f"Number of cylinder measurements: {num_meas}")

    def z_matrix_acc_to_measurement(self):
        """
        Build the measurement matrix z_meas with shape (2, N), where
        first row contains measured distances and the second row measured angles.
        """
        self.z_meas = np.vstack((self.approx_linear_distance, self.approx_angular_position))
        # self.get_logger().info(f"z_meas shape: {self.z_meas.shape}")

    def z_matrix_acc_to_pos(self):
        """
        For each reference cylinder (predefined in self.reference_cylin),
        compute the estimated distance and relative angle from the predicted pose.
        Also store the differences in x and y.
        """
        dist_estim = []
        angle_estim = []
        self.x_estim_diff = []
        self.y_estim_diff = []
        for ref in self.reference_cylin:
            x_ref, y_ref = ref
            diff_x = self.x_predicted - x_ref
            diff_y = self.y_predicted - y_ref
            self.x_estim_diff.append(diff_x)
            self.y_estim_diff.append(diff_y)
            dist_estim.append(math.sqrt(diff_x**2 + diff_y**2))
            angle = math.atan2(y_ref - self.y_predicted, x_ref - self.x_predicted)
            angle_estim.append(normalize_angle(angle - self.theta_predicted))
        self.z_estim = np.vstack((dist_estim, angle_estim))
        self.diff_estim = np.vstack((self.x_estim_diff, self.y_estim_diff))

    def cylin_pairing(self):
        """
        Pair the estimated measurements (z_estim) with the observed measurements (z_meas).
        Note: z_meas has shape (2, N_meas) and z_estim has shape (2, N_estim).
        The number of detected cylinders is given by z_meas.shape[1].
        """
        if not hasattr(self, 'z_estim') or not hasattr(self, 'z_meas'):
            self.get_logger().error("z_estim and/or z_meas not computed.")
            return
        
        tolerance = 0.15  # Use a reasonable tolerance for matching
        paired_meas_dist = []
        paired_meas_angle = []
        paired_estim_dist = []
        paired_estim_angle = []
        paired_estim_diff_x = []
        paired_estim_diff_y = []

        num_estim = self.z_estim.shape[1]
        num_meas = self.z_meas.shape[1]
        
        for i in range(num_estim):
            for j in range(num_meas):
                if np.allclose(self.z_estim[:, i], self.z_meas[:, j], atol=tolerance):
                    paired_meas_dist.append(self.z_meas[0, j])
                    paired_meas_angle.append(self.z_meas[1, j])
                    paired_estim_dist.append(self.z_estim[0, i])
                    paired_estim_angle.append(self.z_estim[1, i])
                    paired_estim_diff_x.append(self.diff_estim[0, i])
                    paired_estim_diff_y.append(self.diff_estim[1, i])
                    break
        
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

    def obs_model(self):
        """
        Update the state estimate using the EKF measurement update.
        If no measurement pairs were found, the prediction is kept.
        (Note: The Jacobian H_t_mat is set as a dummy 2x3 matrix here;
         adjust as needed for your observation model.)
        """
        if self.paired_measurements.size == 0:
            self.mu = self.mu_bar
            self.final_covariance = self.covariance
        else:
            num_pairs = self.paired_measurements.shape[1]
            for i in range(num_pairs):
                meas_dist = self.paired_measurements[0, i]
                meas_ang = self.paired_measurements[1, i]
                z_matrix = np.array([[meas_dist], [meas_ang]])
    
                h_matrix = np.array([[self.paired_estimations[0, i]], [self.paired_estimations[1, i]]])
    
                # Here we use a dummy observation Jacobian for illustration.
                H_t_mat = np.array([
                    [(-self.paired_estim_diff[0, i] / self.paired_estimations[0, i]), (-self.paired_estim_diff[1, i] / self.paired_estimations[0, i]), 0],
                    [(self.paired_estim_diff[1, i] / ((self.paired_estimations[0, i]) ** 2)), (-self.paired_estim_diff[0, i] / ((self.paired_estimations[0, i]) ** 2)), -1]
                ])

    
                q_matrix = np.array([
                    [self.sigma_r**2, 0],
                    [0, self.sigma_alpha**2]
                ])
    
                S = np.dot(H_t_mat, np.dot(self.covariance, H_t_mat.T)) + q_matrix
                k_gain = np.dot(self.covariance, np.dot(H_t_mat.T, np.linalg.inv(S)))
    
                Innovation_matrix = np.array([
                    [z_matrix[0, 0] - h_matrix[0, 0]],
                    [normalize_angle(z_matrix[1, 0] - h_matrix[1, 0])]
                ])
    
                self.mu = self.mu_bar + np.dot(k_gain, Innovation_matrix)
                Identity = np.eye(3)
                self.final_covariance = np.dot((Identity - np.dot(k_gain, H_t_mat)), self.covariance)
    
        self.obs_bot_position = np.array([
            [self.x],
            [self.y],
            [self.theta]
        ])
        self.error_in_estim_positions = self.mu - self.obs_bot_position
        self.x_prev = self.x
        self.y_prev = self.y
        self.theta_prev = self.theta

    def publish_pose_with_covariance(self):
        clock = Clock()
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = clock.now().to_msg()

        pose_msg.header.frame_id = "odom"  # Use the appropriate frame of reference

        # Setting the pose based on the mean (mu)
        pose_msg.pose.pose.position.x = self.mu[0, 0]
        pose_msg.pose.pose.position.y = self.mu[1, 0]
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

    def publish_cylinder_markers(self):
        marker_array = MarkerArray()
        clock = Clock()
        # Check if any paired estimations exist
        if hasattr(self, 'paired_estimations') and self.paired_estimations.size != 0:
            num_markers = self.paired_estimations.shape[1]
            for i in range(num_markers):
                marker = Marker()
                marker.header.stamp = clock.now().to_msg()
                marker.header.frame_id = "odom"
                marker.ns = "estimated_cylinders"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                # Get measurement: distance and angle relative to robot's predicted pose
                dist = self.paired_estimations[0, i]
                ang = self.paired_estimations[1, i]
                # Compute the estimated global coordinates:
                x_est = self.x_predicted + dist * math.cos(self.theta_predicted + ang)
                y_est = self.y_predicted + dist * math.sin(self.theta_predicted + ang)
                marker.pose.position.x = x_est
                marker.pose.position.y = y_est
                marker.pose.position.z = 0.1  # slightly above the ground
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
        else:
            # If there are no paired estimations, you might choose to publish an empty MarkerArray to clear previous markers.
            marker_array.markers = []
            # print('noooooo')
        
        self.marker_pub.publish(marker_array)

    def publish_reference_cylinder_markers(self):
        marker_array = MarkerArray()
        clock = Clock()
        for i, ref in enumerate(self.reference_cylin):
            marker = Marker()
            marker.header.stamp = clock.now().to_msg()
            marker.header.frame_id = "odom"
            marker.ns = "reference_cylinders"
            marker.id = i
            marker.type = Marker.CUBE  # You can choose SPHERE or CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = ref[0]
            marker.pose.position.y = ref[1]
            marker.pose.position.z = 0.1  # Slightly above the ground
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            # Use a distinct color (e.g., blue) for reference cylinders
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)
        self.ref_marker_pub.publish(marker_array)

    def compute_error_ellipse(self, cov):
        """
        Extract the 2x2 covariance submatrix for x and y (from the 3x3 covariance matrix)
        and compute the ellipse parameters.
        Returns semi-major axis (a), semi-minor axis (b) and orientation angle (in radians).
        """
        cov_2d = np.array([
            [cov[0, 0], cov[0, 1]],
            [cov[1, 0], cov[1, 1]]
        ])
        eigenvals, eigenvecs = np.linalg.eig(cov_2d)
        sort_indices = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[sort_indices]
        eigenvecs = eigenvecs[:, sort_indices]
        a = math.sqrt(eigenvals[0])
        b = math.sqrt(eigenvals[1])
        angle = math.atan2(eigenvecs[1, 0], eigenvecs[0, 0])
        return a, b, angle

    def generate_ellipse_points(self, center, a, b, angle, num_points=36):
        """
        Generate a list of geometry_msgs/Point forming an ellipse.
        'center' is a tuple (x, y) for the ellipse center.
        'a' and 'b' are the semi-axes and 'angle' is the orientation.
        """
        points = []
        for t in np.linspace(0, 2 * math.pi, num_points):
            x = a * math.cos(t)
            y = b * math.sin(t)
            # Rotate the point by the ellipse angle
            x_rot = x * math.cos(angle) - y * math.sin(angle)
            y_rot = x * math.sin(angle) + y * math.cos(angle)
            pt = Point()
            pt.x = center[0] + x_rot
            pt.y = center[1] + y_rot
            pt.z = 0.2  # Adjust if needed
            points.append(pt)
        # Close the ellipse
        points.append(points[0])
        return points

    def publish_error_ellipse(self):
        """
        Publishes the error ellipse as a visualization_msgs/Marker.
        Uses the x-y submatrix of self.final_covariance and self.mu for the estimated pose.
        """
        # Compute ellipse parameters from the final covariance
        b, a, angle = self.compute_error_ellipse(self.final_covariance)
        center = (self.mu[0, 0], self.mu[1, 0])
        ellipse_points = self.generate_ellipse_points(center, a, b, angle)
        
        marker = Marker()
        marker.header.stamp = Clock().now().to_msg()
        marker.header.frame_id = "odom"
        marker.ns = "error_ellipse"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005  # Line thickness
        # Set color to green
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.points = ellipse_points
        marker.lifetime.sec = 0  # Remains until updated
        self.error_ellipse_pub.publish(marker)


    def run(self):
        # Execute the prediction step (using odometry data)
        self.pose_predictor()
        self.state_covariance_calc()
        self.control_covariance_calc()
        self.prediction_covariance_calc()
        
        # Process LIDAR data and generate measurements
        self.observed_cylinders()
        self.z_matrix_acc_to_measurement()
        self.z_matrix_acc_to_pos()
        self.cylin_pairing()
        
        # Perform the EKF update step with the observed data
        self.obs_model()
        
        # Finally, publish the updated pose with covariance
        self.publish_pose_with_covariance()
        self.publish_trajectory()
        self.publish_real_trajectory()
        self.publish_cylinder_markers()
        self.publish_reference_cylinder_markers()
        self.publish_error_ellipse()



def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = EKF_LOCALIZATION()
    # Create a timer to call the run() method periodically (e.g., every 0.1 seconds)
    timer_period = .5  # seconds
    node.create_timer(timer_period, node.run)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
