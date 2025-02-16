#!/usr/bin/env python3
import rclpy,math
import numpy as np
import rclpy.logging
from  rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

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
        self.theta_prev = 0.0

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
        control_covariance = np.diag([self.rot1_variance, self.trans_variance, self.rot2_variance])

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
    
        cylin_detected,no_of_rays,sum_ray_indices,sum_depth,count,i=0,0,0,0,0,-1
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
        self.z_meas = np.vstack(self.approx_linear_distance,self.approx_angular_position)
    
    def z_matrix_acc_to_pos(self):
        dist_estim = []
        angle_estim = []
        for i in range(len(self.reference_cylin)):
            dist_estim.append(math.sqrt((self.x_predicted - self.reference_cylin[i][0])**2 + (self.y_predicted - self.reference_cylin[i][1])**2))
            angle_estim.append(normalize_angle(math.atan2(self.reference_cylin[i][1]-self.y_predicted,self.reference_cylin[i][0]-self.x_predicted) - normalize_angle(self.theta_predicted)))
        self.z_estim = np.vstack(dist_estim,angle_estim)
 
    def cylin_pairing(self):
        # Ensure both z_estim and z_meas have been computed
        if not hasattr(self, 'z_estim') or not hasattr(self, 'z_meas'):
            print("Error: z_estim and/or z_meas not computed.")
            return
        
        tolerance = 1e-6
        # Lists to store paired distance and angle values.
        paired_dist = []
        paired_alpha = []
        
        num_estim = self.z_estim.shape[1]
        num_meas = self.z_meas.shape[1]
        
        for i in range(num_estim):
            for j in range(num_meas):
                # Compare both the distance and angular components.
                if np.allclose(self.z_estim[:, i], self.z_meas[:, j], atol=tolerance):
                    # Append the measurement values (distance and alpha) to the paired lists.
                    paired_dist.append(self.z_meas[0, j])
                    paired_alpha.append(self.z_meas[1, j])
                    break  # Stop after finding the first matching column in z_meas.
        
        # Combine paired values into a new 2 x N matrix (if any pairings were found).
        if paired_dist and paired_alpha:
            self.paired_measurements = np.vstack((paired_dist, paired_alpha))
        else:
            self.paired_measurements = np.array([])
        
        print("Paired measurements (distance and angle):")
        print(self.paired_measurements)

    