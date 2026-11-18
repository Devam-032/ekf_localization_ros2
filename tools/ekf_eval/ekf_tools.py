"""
Helpers for evaluating ekf_localization_ros2 in Gazebo.

    python3 ekf_tools.py noisy_odom          # /odom (Gazebo ground truth) -> /odom_noisy
    python3 ekf_tools.py drive               # follow a fixed loop through the landmark field
    python3 ekf_tools.py log <out.csv> <s>   # ground truth / noisy odom / EKF -> CSV

Gazebo's TurtleBot3 /odom is the true pose, so it serves as ground truth. To give the EKF
something to correct, noisy_odom integrates the true motion increments through the
odometry motion model (rot1, trans, rot2) with Gaussian noise plus a small systematic
heading bias, like real wheel odometry. The EKF is launched with odom_topic:=/odom_noisy.
"""
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class NoisyOdom(Node):
    # Odometry motion model noise (Thrun et al.): std of rot = A1*|rot| + A2*trans, etc.
    A1, A2, A3, A4 = 0.05, 0.02, 0.03, 0.01
    HEADING_BIAS = 0.012           # rad per metre travelled (systematic, e.g. wheel radius mismatch)

    def __init__(self):
        super().__init__('noisy_odom')
        self.rng = np.random.default_rng(7)
        self.prev = None
        self.est = None
        self.pub = self.create_publisher(Odometry, '/odom_noisy', 10)
        self.create_subscription(Odometry, '/odom', self.cb, 50)

    def cb(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        th = yaw_of(msg.pose.pose.orientation)
        if self.prev is None:
            self.prev = self.est = (x, y, th)
        px, py, pth = self.prev
        trans = math.hypot(x - px, y - py)
        rot1 = wrap(math.atan2(y - py, x - px) - pth) if trans > 1e-4 else 0.0
        rot2 = wrap(th - pth - rot1)
        n = self.rng.normal
        rot1_n = rot1 + n(0, self.A1 * abs(rot1) + self.A2 * trans)
        trans_n = trans + n(0, self.A3 * trans + self.A4 * (abs(rot1) + abs(rot2)))
        rot2_n = rot2 + n(0, self.A1 * abs(rot2) + self.A2 * trans) + self.HEADING_BIAS * trans
        ex, ey, eth = self.est
        ex += trans_n * math.cos(eth + rot1_n)
        ey += trans_n * math.sin(eth + rot1_n)
        eth = wrap(eth + rot1_n + rot2_n)
        self.est, self.prev = (ex, ey, eth), (x, y, th)

        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id
        out.pose.pose.position.x, out.pose.pose.position.y = ex, ey
        out.pose.pose.orientation.z, out.pose.pose.orientation.w = math.sin(eth / 2), math.cos(eth / 2)
        out.twist = msg.twist
        self.pub.publish(out)


class Driver(Node):
    # Loop through free space; every leg keeps >= 0.7 m from the cylinder landmarks
    ROUTE = [(1.0, 0.0), (2.6, -0.4), (2.6, -2.9), (0.6, -3.1), (-0.2, -1.9), (-2.0, -1.8),
             (-2.4, -0.6), (-0.6, 1.0), (0.2, 2.3), (2.6, 2.0), (3.2, 0.6), (1.4, 0.1), (0.0, 0.0)]
    V, K_W, W_MAX, TOL = 0.18, 1.8, 1.2, 0.15

    def __init__(self):
        super().__init__('loop_driver')
        self.i = 0
        self.pose = None
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.cb, 20)
        self.create_timer(0.05, self.step)
        self.done = False

    def cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))

    def step(self):
        if self.pose is None or self.done:
            return
        x, y, th = self.pose
        gx, gy = self.ROUTE[self.i]
        if math.hypot(gx - x, gy - y) < self.TOL:
            self.i += 1
            if self.i == len(self.ROUTE):
                self.pub.publish(Twist())
                self.done = True
                return
            gx, gy = self.ROUTE[self.i]
        err = wrap(math.atan2(gy - y, gx - x) - th)
        cmd = Twist()
        cmd.angular.z = max(-self.W_MAX, min(self.W_MAX, self.K_W * err))
        cmd.linear.x = self.V * max(0.0, math.cos(err)) ** 2   # turn in place on sharp corners
        self.pub.publish(cmd)


class Logger(Node):
    def __init__(self, path):
        super().__init__('ekf_eval_logger')
        self.gt = self.od = None
        self.t0 = time.time()
        self.f = open(path, 'w')
        self.f.write('t,gt_x,gt_y,gt_yaw,odom_x,odom_y,odom_yaw,ekf_x,ekf_y,ekf_yaw,cov_xx,cov_yy,cov_xy\n')
        self.rows = 0
        self.create_subscription(Odometry, '/odom', lambda m: setattr(self, 'gt', m.pose.pose), 50)
        self.create_subscription(Odometry, '/odom_noisy', lambda m: setattr(self, 'od', m.pose.pose), 50)
        self.create_subscription(PoseWithCovarianceStamped, '/pose_with_cov_stamped', self.ekf_cb, 10)

    def ekf_cb(self, msg):
        if self.gt is None or self.od is None:
            return
        e, c = msg.pose.pose, msg.pose.covariance
        g, o = self.gt, self.od
        self.f.write(f'{time.time() - self.t0:.2f},'
                     f'{g.position.x:.4f},{g.position.y:.4f},{yaw_of(g.orientation):.4f},'
                     f'{o.position.x:.4f},{o.position.y:.4f},{yaw_of(o.orientation):.4f},'
                     f'{e.position.x:.4f},{e.position.y:.4f},{yaw_of(e.orientation):.4f},'
                     # the node packs its 3x3 (x, y, yaw) covariance row-major into c[0:9]
                     f'{c[0]:.6f},{c[4]:.6f},{c[1]:.6f}\n')
        self.rows += 1


def main():
    rclpy.init()
    mode = sys.argv[1]
    if mode == 'noisy_odom':
        rclpy.spin(NoisyOdom())
    elif mode == 'drive':
        node = Driver()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        print('DRIVE_DONE', flush=True)
    elif mode == 'log':
        node = Logger(sys.argv[2])
        end = time.time() + float(sys.argv[3])
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.f.close()
        print(f'logged {node.rows} EKF updates', flush=True)


if __name__ == '__main__':
    main()
