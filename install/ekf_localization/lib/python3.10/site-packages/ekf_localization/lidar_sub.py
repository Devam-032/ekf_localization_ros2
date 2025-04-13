#!/usr/bin/env python3 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class LaserSub(Node):

    def __init__(self):
        super().__init__("laser_sub")
        self.lidar_sub = self.create_subscription(LaserScan,"/scan",self.laserCb,10)

    def laserCb(self,msg:LaserScan):
        scan_data = [0]*(len(msg.ranges))
        for i in range(len(msg.ranges)-1):
            scan_data[i] = msg.ranges[i]
        print(scan_data)

def main():
    rclpy.init()
    node = LaserSub()
    rclpy.spin(node)
    rclpy.shutdown()