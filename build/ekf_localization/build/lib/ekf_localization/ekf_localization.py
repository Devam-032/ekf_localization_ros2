#!/usr/bin/env python3
import rclpy,math
from  rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

def normalize_angle(angle):
    return (angle + math.pi)
