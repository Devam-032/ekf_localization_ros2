#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64MultiArray, Bool
import numpy as np
import cv2
import math

class ObstacleExtractorSequential(Node):
    def __init__(self):
        super().__init__('obstacle_extractor_sequential')
        # Subscribe to the /map topic
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        # Publisher to publish the flattened world coordinates
        self.ref_coords_pub = self.create_publisher(Float64MultiArray, '/reference_coords', 10)
        # Subscriber for the reference flag from the EKF node
        self.ref_flag_sub = self.create_subscription(Bool, '/ref_coords_ready', self.ref_flag_callback, 10)
        self.stop_publishing = False

        # We'll store the computed world coordinates here (list of (x,y) tuples)
        self.world_coords_list = None

        # Create a timer to publish the reference coordinates periodically (e.g., every 1 second)
        self.pub_timer = self.create_timer(1.0, self.publish_ref_coords_periodically)

    def map_callback(self, msg: OccupancyGrid):
        # Process the map every time (or you can limit it to run only once if /map is static)
        # For continuous publishing, we remove the one-shot guard.
        # Extract map metadata
        height = msg.info.height
        width = msg.info.width
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        self.get_logger().info(f"Map received: {width}x{height}, resolution: {resolution} m/cell.")

        # Convert flattened occupancy grid data to a 2D numpy array
        grid = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # Create a grayscale image:
        # Free space (0) -> white (255)
        # Occupied (100) -> black (0)
        # Unknown (-1) -> gray (127)
        image = np.zeros_like(grid, dtype=np.uint8)
        image[grid == 0] = 255
        image[grid == 100] = 0
        image[grid == -1] = 127

        # --- Step 1: Show the Grayscale Map (disabled)
        # cv2.imshow("Step 1: Grayscale Map", image)
        # cv2.waitKey(0)

        # --- Step 2: Apply Gaussian Blur ---
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        # cv2.imshow("Step 2: Gaussian Blurred", blurred)
        # cv2.waitKey(0)

        # --- Step 3: Thresholding to Create a Binary Image ---
        ret, binary = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY_INV)
        self.get_logger().info(f"Binary threshold used: 25, ret: {ret}")
        # cv2.imshow("Step 3: Binary Image", binary)
        # cv2.waitKey(0)

        # --- Step 4: Apply Dilation to Separate Obstacles ---
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        # cv2.imshow("Step 4: Dilated Image", dilated)
        # cv2.waitKey(0)

        # --- Step 5: Connected Components Analysis ---
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
        self.get_logger().info(f"Connected components detected: {num_labels - 1} (excluding background)")

        # Create a color image for final visualization (disabled for continuous publishing)
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # List to hold world coordinates of obstacles
        world_coords_list = []

        # Process each detected obstacle (skip background label 0)
        for i in range(1, num_labels):
            left, top, w, h, area = stats[i]
            cx, cy = centroids[i]

            # Filter out small noise or too large areas (e.g., walls)
            if area < 18 or area > 100:
                continue

            # Filter out obstacles touching the map borders (assuming these are walls)
            if left == 0 or top == 0 or (left + w) >= width or (top + h) >= height:
                self.get_logger().info(f"Skipping obstacle {i} (touches map border).")
                continue

            # Convert grid (pixel) coordinates to Gazebo world coordinates.
            # Assuming map origin is bottom-left:
            world_x = origin_x + cx * resolution
            world_y = origin_y + (height - cy) * resolution

            world_coords_list.append((world_x, world_y))

            self.get_logger().info(
                f"Obstacle {i}: Area = {area} cells, Bounding Box = ({left}, {top}, {w}, {h}), "
                f"Centroid (grid) = ({cx:.2f}, {cy:.2f}), World = ({world_x:.2f}, {world_y:.2f})"
            )
            # Draw the bounding box and centroid (disabled)
            cv2.rectangle(vis_image, (left, top), (left + w, top + h), (255, 0, 0), 2)
            cv2.circle(vis_image, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        # --- Step 6: Display the Final Visualization (disabled)
        # cv2.imshow("Step 6: Obstacle Detection", vis_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Print the world coordinates list
        self.get_logger().info("World Coordinates of detected obstacles:")
        for coord in world_coords_list:
            self.get_logger().info(f"{coord}")

        print("World Coordinates List:")
        print(world_coords_list)
        
        # Store the computed coordinates in an instance variable for later publishing
        self.world_coords_list = world_coords_list

    def publish_ref_coords_periodically(self):
        """
        This timer callback publishes the world coordinates (if available) as a flattened list.
        It stops publishing if the stop flag is set.
        """
        if self.world_coords_list is None:
            # Haven't computed yet, so do nothing.
            return

        # Create and publish the message
        flat_list = [coord for pair in self.world_coords_list for coord in pair]
        ref_msg = Float64MultiArray()
        ref_msg.data = flat_list
        self.ref_coords_pub.publish(ref_msg)
        self.get_logger().info("Published reference coordinates.")

        # If the stop flag is set, shutdown the timer and node.
        if self.stop_publishing:
            self.get_logger().info("Stop flag received. Shutting down publisher node.")
            self.pub_timer.cancel()
            self.destroy_node()

    def ref_flag_callback(self, msg: Bool):
        """
        Callback for the /ref_coords_ready topic.
        When a True flag is received, set stop_publishing to True.
        """
        if msg.data:
            self.stop_publishing = True
            self.get_logger().info("Received reference flag. Will stop publishing after current cycle.")

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleExtractorSequential()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
