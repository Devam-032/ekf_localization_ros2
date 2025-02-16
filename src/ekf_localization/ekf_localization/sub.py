#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Mysub(Node):

    def __init__(self):
        super().__init__("listener1")
        self.string_sub = self.create_subscription(String,"/chatter",self.msgCb,10)
    
    def msgCb(self,msg):
        print(msg.data)

def main():
    rclpy.init()
    node = Mysub()
    rclpy.spin(node)
    rclpy.shutdown()