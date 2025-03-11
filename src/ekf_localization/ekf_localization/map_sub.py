#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
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
        self.received_map = False

    def map_callback(self, msg: OccupancyGrid):
        if self.received_map:
            return  # Process only once for a static map
        self.received_map = True

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

        # --- Step 1: Show the Grayscale Map ---
        cv2.imshow("Step 1: Grayscale Map", image)
        cv2.waitKey(0)  # Wait until a key is pressed

        # --- Step 2: Apply Gaussian Blur ---
        # Using a kernel size of (9, 9) and sigma of 2, as in your code
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        cv2.imshow("Step 2: Gaussian Blurred", blurred)
        cv2.waitKey(0)

        # --- Step 3: Thresholding to Create a Binary Image ---
        # Here, we use a threshold value of 25 to generate the binary image.
        # Since obstacles are dark in the grayscale map, we use THRESH_BINARY_INV.
        ret, binary = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY_INV)
        self.get_logger().info(f"Binary threshold used: 25, ret: {ret}")
        cv2.imshow("Step 3: Binary Image", binary)
        cv2.waitKey(0)

        # --- Step 4: Apply Dilation to Separate Obstacles ---
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        cv2.imshow("Step 4: Dilated Image", dilated)
        cv2.waitKey(0)

        # --- Step 5: Connected Components Analysis ---
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
        self.get_logger().info(f"Connected components detected: {num_labels - 1} (excluding background)")

        # Create a color image for final visualization.
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Process each detected obstacle (skip background label 0)
        for i in range(1, num_labels):
            left, top, w, h, area = stats[i]
            cx, cy = centroids[i]
            # Filter out small noise by checking if area is above a threshold.
            if area < 18:
                continue
            if area>100:
                continue
            # Filter out obstacles touching the map borders (assuming these are walls)
            if left == 0 or top == 0 or (left + w) >= width or (top + h) >= height:
                self.get_logger().info(f"Skipping obstacle {i} (touches map border).")
                continue

            # Convert grid (pixel) coordinates to world coordinates.
            world_x = origin_x + cx * resolution
            world_y = origin_y + cy * resolution

            self.get_logger().info(
                f"Obstacle {i}: Area = {area} cells, Bounding Box = ({left}, {top}, {w}, {h}), "
                f"Centroid (grid) = ({cx:.2f}, {cy:.2f}), World = ({world_x:.2f}, {world_y:.2f})"
            )
            # Draw the bounding box (blue) and centroid (red) on the visualization image.
            cv2.rectangle(vis_image, (left, top), (left + w, top + h), (255, 0, 0), 2)
            cv2.circle(vis_image, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        # --- Step 6: Display the Final Visualization ---
        cv2.imshow("Step 6: Obstacle Detection", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleExtractorSequential()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
