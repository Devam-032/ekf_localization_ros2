#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
import math

class MapFlipTransformer(Node):
    def __init__(self):
        super().__init__('map_flip_transformer')
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.publish_static_transform()

    def publish_static_transform(self):
        # Create a TransformStamped for the 180° rotation around the X-axis
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()

        # The "parent" is the flipped frame; the "child" is the real map
        # (You can swap if you prefer the other direction.)
        t.header.frame_id = "map_flipped"  # This is the frame you'll use in RViz
        t.child_frame_id = "map"          # Your original map frame

        # No translation: keep the same origin
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # 180° rotation around the X-axis => quaternion (x=1, y=0, z=0, w=0)
        # Because sin(pi/2)=1, cos(pi/2)=0
        t.transform.rotation.x = 1.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 0.0

        # Publish the static transform
        self.static_broadcaster.sendTransform(t)
        self.get_logger().info("Published 180° rotation around X-axis from map_flipped -> map.")

def main(args=None):
    rclpy.init(args=args)
    node = MapFlipTransformer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
