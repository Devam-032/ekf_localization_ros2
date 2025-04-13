#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf_transformations import euler_from_quaternion
from nav_msgs.msg import Odometry

class Odom_sub(Node):
    
    def __init__(self):
        super().__init__("odom_sub")
        self.odom_sub_ = self.create_subscription(Odometry,"/bumperbot_controller/odom",self.OdomCb,10)

    def OdomCb(self,msg:Odometry):
        orientation_q = msg.pose.pose.orientation
        yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])[2]
        print(yaw)

def main():
    rclpy.init()
    node = Odom_sub()
    rclpy.spin(node)
    rclpy.shutdown()