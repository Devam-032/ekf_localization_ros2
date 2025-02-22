#!/usr/bin/env python3
import rclpy,math
import numpy as np
import rclpy.logging
from  rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion,quaternion_from_euler
from rclpy.clock import Clock

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi 

class EKF_LOCALIZATION(Node):

    def __init__(self):
        super().__init__("Localization_script")
        self.lidar_sub_  = self.create_subscription(LaserScan,'/scan',self.laserCb,10)
        self.ref_cylin_sub_ = self.create_subscription(ModelStates,'/gazebo/model_states',self.reference_coordinates,10)
        self.odom_sub_ = self.create_subscription(Odometry,"/bumperbot_controller/odom",self.odomCb,10)
        self.delta_trans = 0.0
        self.delta_rot1 = 0.0
        self.delta_rot2 = 0.0
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.sigma_r = 0.1 # This is to define standard deviation in the distance measurement
        self.sigma_alpha = 0.01  # This is to define the standard deviation in the angle measurement
        self.theta_prev = 0.0
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped,"/pose_with_cov_stamped",10)

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

    def reference_coordinates(self, msg):
        a2 = len(msg.pose) - 3
        self.reference_cylin = []  # Initialize as an empty list

        for i in range(a2):
            # Append each coordinate as a list
            self.reference_cylin.append([msg.pose[2+i].position.x, msg.pose[2+i].position.y])

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
        """To get the observed cylinder distance and angle w.r.t. the lidar sensor."""
        jumps=[0]*len(self.laser_points)
        jumps_index=[]
        for i in range(1,len(self.laser_points)-1):
            next_point = self.laser_points[i+1]
            prev_point = self.laser_points[i-1]
            if(prev_point>.2 and next_point>.2):
                derivative = (next_point-prev_point)/2
            if(abs(derivative)>.3):
                jumps[i] = derivative
                jumps_index.append(i)
    
        cylin_detected,no_of_rays,sum_ray_indices,sum_depth,i=0,0,0,0,0,-1
        self.approx_linear_distance,self.approx_angular_position =[],[]
        while(i<len(jumps)-1):
            i+=1
            if(jumps[i]<0 and cylin_detected==0): #a falling edge detected in the lidar's scan and currently not on a cylinder
                cylin_detected = 1 #the first cylinder has been detected
                no_of_rays += 1 #increment the number of rays that are falling on the cylinder
                sum_ray_indices += i #sum up the indices of the rays
                sum_depth += self.laser_points[i]  #sum of the distance travelled by the rays falling on cylinder

            elif(jumps[i]<0 and cylin_detected==1): # a second falling edge has been detected so ignore the previous data if already on the cylinder
                cylin_detected = 0  # now reset the value of cylinder_detected
                no_of_rays = 0 # reset the no. of rays falling on the cylinders
                sum_ray_indices = 0 # reset
                sum_depth = 0 # reset
                i-=1 # decrementing the index so that this falling edge can be checked again and can be passed to the 1st if statement

            elif jumps[i] > 0 and cylin_detected == 1:
                cylin_detected = 0  # Reset the cylinder detection flag
                # Calculate the approximate angular distance in radians
                approx_ang = sum_ray_indices * 0.01745 / no_of_rays
                # Adjust the angle if it exceeds π
                normalize_angle(approx_ang)  
                self.approx_angular_position.append(approx_ang)
                self.approx_linear_distance.append(sum_depth / no_of_rays)
                no_of_rays = 0  #reset
                sum_ray_indices = 0  #reset
                sum_depth = 0  #reset

            elif(jumps==0 and cylin_detected==1):
                no_of_rays+=1
                sum_depth+=self.laser_points[i]
                sum_ray_indices+=i

            else:
                pass #do_nothing
    
    def z_matrix_acc_to_measurement(self):
        self.z_meas = np.vstack((self.approx_linear_distance, self.approx_angular_position))

    def z_matrix_acc_to_pos(self):
        dist_estim = []
        angle_estim = []
        self.x_estim_diff = []
        self.y_estim_diff = []
        for i in range(len(self.reference_cylin)):
            self.x_estim_diff.append(self.x_predicted - self.reference_cylin[i][0])
            self.y_estim_diff.append(self.y_predicted - self.reference_cylin[i][1])
            dist_estim.append(math.sqrt((self.x_predicted - self.reference_cylin[i][0])**2 +
                                        (self.y_predicted - self.reference_cylin[i][1])**2))
            angle = math.atan2(self.reference_cylin[i][1] - self.y_predicted,
                            self.reference_cylin[i][0] - self.x_predicted)
            angle_estim.append(normalize_angle(angle - normalize_angle(self.theta_predicted)))
        self.z_estim = np.vstack((dist_estim, angle_estim))
        self.diff_estim = np.vstack((self.x_estim_diff, self.y_estim_diff))

    def cylin_pairing(self):
        # Check that both matrices have been computed.
        if not hasattr(self, 'z_estim') or not hasattr(self, 'z_meas'):
            print("Error: z_estim and/or z_meas not computed.")
            return
        
        tolerance = 1e-6  # Tolerance for comparing floating point values.
        
        # Lists to store paired measurement data.
        paired_meas_dist = []
        paired_meas_angle = []
        
        # Lists to store corresponding estimated data.
        paired_estim_dist = []
        paired_estim_angle = []

        # Lists to store corresponding estimated differences.
        paired_estim_diff_x = []
        paired_estim_diff_y = [] 

        num_estim = self.z_estim.shape[1]
        num_meas = self.z_meas.shape[1]
        
        # Compare every column in the estimated matrix with every column in the measurement matrix.
        for i in range(num_estim):
            for j in range(num_meas):
                # Compare both the distance and angular components.
                if np.allclose(self.z_estim[:, i], self.z_meas[:, j], atol=tolerance):
                    # When a match is found, store the measurement values...
                    paired_meas_dist.append(self.z_meas[0, j])
                    paired_meas_angle.append(self.z_meas[1, j])
                    # ...and store the corresponding estimation values.
                    paired_estim_dist.append(self.z_estim[0, i])
                    paired_estim_angle.append(self.z_estim[1, i])
                    # ...and store the corresponding differences.
                    paired_estim_diff_x.append(self.diff_estim[0, i])
                    paired_estim_diff_y.append(self.diff_estim[1, i])
                    break  # Found a match for z_estim column i; move to next.
        
        # Create matrices from the paired data (if any pairs were found).
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
        
        for i in range(len(self.paired_measurements)):

            meas_dist = self.paired_measurements[i][0]
            meas_ang = self.paired_measurements[i][1]
            z_matrix = np.array([
                [meas_dist],
                [meas_ang]
            ])

            h_matrix = np.array([
                [self.paired_estimations[i][0]],
                [self.paired_estimations[i][1]]
            ])

            H_t_mat = np.array([
                [(-self.paired_estim_diff[i][0]/(self.paired_estimations[0])),(-self.paired_estim_diff[i][1]/(self.paired_estimations[i][0])),0],
                [(self.paired_estim_diff[i][1]/((self.paired_estimations[0])**2)),(-self.paired_estim_diff[i][0]/((self.paired_estimations[i][0])**2)),-1]
            ])
               
            q_matrix = np.array([
                [self.sigma_r**2 , 0 ],
                [0 , self.sigma_alpha**2]
            ])

            self.covariance = np.array(self.covariance)

            k_gain = np.dot(self.covariance,np.dot(H_t_mat.T,np.linalg.inv(np.dot(H_t_mat,np.dot(self.covariance,H_t_mat.T)))+q_matrix)) # to claculate the kalman gain for each observed cylinder

            Innovation_matrix = np.array([
                [z_matrix[0, 0] - h_matrix[0, 0]],  # distance difference
                [normalize_angle(z_matrix[1, 0] - h_matrix[1, 0])]  # angle difference
            ])

            self.mu =  self.mu_bar + np.dot(k_gain , Innovation_matrix)

            Identity = np.eye(3)

            self.final_covariance = np.dot((Identity - np.dot(k_gain,H_t_mat)),self.covariance) # final covariance calculation

        if (len(self.match_pairs_left)==0):
            self.mu = self.mu_bar
            self.final_covariance = self.covariance
            #(len(self.match_pairs_left))
        #print(self.final_covariance)
        self.obs_bot_position = np.array([
            [self.x],
            [self.y],
            [self.theta]
        ])

        self.error_in_estim_positions = self.mu-self.obs_bot_position
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
        pose_msg.pose.pose.position.z = 0  # Assume planar navigation

        # Convert orientation from Euler to quaternion
        quat = quaternion_from_euler(0, 0, self.mu[2, 0])
        pose_msg.pose.pose.orientation.x = quat[0]
        pose_msg.pose.pose.orientation.y = quat[1]
        pose_msg.pose.pose.orientation.z = quat[2]
        pose_msg.pose.pose.orientation.w = quat[3]

        # Fill in the covariance (flattened row-major order)
        covariance_flat = self.final_covariance.flatten()
        pose_msg.pose.covariance = [covariance_flat[i] if i < len(covariance_flat) else 0 for i in range(36)]

        # Publish the message
        self.pose_pub.publish(pose_msg)