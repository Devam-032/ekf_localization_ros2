#!/usr/bin/env python3
 
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class PubSub(Node):

    def __init__(self):
        super().__init__("pubsub_script")
        self.twist_pub_ = self.create_publisher(Twist,"/cmd_vel",10)
        self.string_sub = self.create_subscription(String,"/chatter",self.pubsubCb,10)

        self.msg = Twist()
    
    def pubsubCb(self,msg1):

        x = msg1.data
        self.msg.linear.x = 1.0
        self.msg.angular.z = 1.0
        self.twist_pub_.publish(self.msg)

def main():
    rclpy.init()
    node = PubSub()
    rclpy.spin(node)
    rclpy.shutdown()