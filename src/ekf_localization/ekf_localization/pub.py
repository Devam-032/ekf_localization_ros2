#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MyPub(Node):
    
    def __init__(self):
        super().__init__("talker1")
        self.msg_pub = self.create_publisher(String,"/chatter",10)
        self.create_timer(.05,self.timer_callback)
        self.get_logger().info("Hello from ROS2")
        self.count = 0

    def timer_callback(self):
        # self.get_logger().info(f"Hi Devam {self.count}")
        msg = String()
        msg.data = f"Hi Devam {self.count}"
        self.count+=1
        self.msg_pub.publish(msg)
    
def main(args=None):
    rclpy.init(args=args)
    node = MyPub()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=="__main__":
    main()