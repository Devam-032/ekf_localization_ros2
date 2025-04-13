#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion, TransformStamped
import yaml
import cv2
import numpy as np
import os
import math
import tf2_ros

class StaticMapServer(Node):
    def __init__(self, yaml_file):
        super().__init__('static_map_server')
        # Publisher for the occupancy grid map
        self.publisher_ = self.create_publisher(OccupancyGrid, 'map', 10)
        
        # Parameter to control publishing frequency (Hz)
        self.declare_parameter('publish_rate', 0.2)  # 0.2 Hz => publish every 5 seconds
        publish_rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)
        
        self.yaml_file = yaml_file
        # self.get_logger().info("Static Map Server starting. Loading map from: " + yaml_file)
        self.load_map()

        # Setup static transform broadcaster to publish the identity transform
        # between "map" (parent) and "odom" (child) so that they overlap initially.
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.publish_static_transform()

    def load_map(self):
        # Load YAML file
        with open(self.yaml_file, 'r') as f:
            self.map_yaml = yaml.safe_load(f)
        
        # Get the image file name from the YAML and build its full path
        image_file = self.map_yaml['image']
        yaml_dir = os.path.dirname(self.yaml_file)
        self.image_path = os.path.join(yaml_dir, image_file)
        
        # Load map parameters
        self.resolution = float(self.map_yaml['resolution'])
        self.origin = self.map_yaml['origin']  # Expected format: [x, y, theta]
        self.negate = int(self.map_yaml.get('negate', 0))
        self.occupied_thresh = float(self.map_yaml.get('occupied_thresh', 0.65))
        self.free_thresh = float(self.map_yaml.get('free_thresh', 0.196))
        
        # Load the PGM image in grayscale
        map_img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if map_img is None:
            self.get_logger().error("Failed to load map image from: " + self.image_path)
            return
        self.map_img = map_img
        self.height, self.width = self.map_img.shape
        self.get_logger().info(f"Loaded map image: {self.width} x {self.height} pixels.")

    def convert_to_occupancy_grid(self):
        # Normalize image pixel values to the range [0, 1]
        normalized = self.map_img.astype(np.float32) / 255.0
        # If the 'negate' flag is set in the YAML, invert the image
        if self.negate:
            normalized = 1.0 - normalized
        
        # Initialize occupancy grid array
        occupancy = np.zeros_like(normalized, dtype=np.int8)
        # Assign cell values based on thresholds
        occupancy[normalized > self.occupied_thresh] = 100  # Occupied
        occupancy[normalized < self.free_thresh] = 0          # Free
        unknown_mask = (normalized >= self.free_thresh) & (normalized <= self.occupied_thresh)
        occupancy[unknown_mask] = -1                          # Unknown

        return occupancy.flatten().tolist()

    def timer_callback(self):
        msg = OccupancyGrid()
        current_time = self.get_clock().now().to_msg()
        msg.header.stamp = current_time
        
        # The occupancy grid is defined in the "map" frame.
        msg.header.frame_id = "map"
        
        msg.info.map_load_time = current_time
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        
        # Set map origin (position and orientation)
        msg.info.origin.position.x = self.origin[0]
        msg.info.origin.position.y = self.origin[1]
        msg.info.origin.position.z = 0.0

        theta = self.origin[2]
        # Convert yaw (theta) to quaternion
        qz = math.sin(theta / 2.0)
        qw = math.cos(theta / 2.0)
        msg.info.origin.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        
        msg.data = self.convert_to_occupancy_grid()
        self.publisher_.publish(msg)
        #self.get_logger().info("Published occupancy grid map in 'map' frame.")

    def publish_static_transform(self):
        # Create a TransformStamped message for the static transform
        t = TransformStamped()
        current_time = self.get_clock().now().to_msg()
        t.header.stamp = current_time
        t.header.frame_id = "map"  # Parent frame
        t.child_frame_id = "odom"  # Child frame
        
        # Set zero translation so that odom overlaps map
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # Set identity rotation (no rotation difference)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        
        # Publish the static transform
        self.static_broadcaster.sendTransform(t)
        # self.get_logger().info("Published static identity transform from 'map' (parent) to 'odom' (child).")

def main(args=None):
    rclpy.init(args=args)
    # Update the path to your map.yaml file as needed.
    yaml_file = '/home/devam/FYP_ROS2/src/ekf_localization/maps/map.yaml'
    node = StaticMapServer(yaml_file)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
