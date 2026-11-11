#!/usr/bin/env python3
"""
camera_node.py
==============
Module 1 — Image Acquisition
Captures frames from a USB camera and publishes them as ROS2 Image messages.

Publishes:
  /camera/image_raw  (sensor_msgs/Image)

Parameters:
  device  (string)  — camera device path  (default: '/dev/video0')
  width   (int)     — capture width        (default: 640)
  height  (int)     — capture height       (default: 480)
  fps     (int)     — frames per second    (default: 30)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('device', '/dev/video2')
        self.declare_parameter('width',  640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps',    30)

        device = self.get_parameter('device').value
        width  = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps    = self.get_parameter('fps').value

        # ── OpenCV camera ────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.cap    = cv2.VideoCapture(device)

        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open camera: {device}")
            raise RuntimeError(f"Camera device {device} not found.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,          fps)

        actual_w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f"Camera opened: {device} | {actual_w}x{actual_h} @ {actual_fps:.1f} fps"
        )

        # ── Publisher ────────────────────────────────────────────────────────
        self.pub = self.create_publisher(Image, '/rgb', 10)

        # ── Timer — fires at target fps ──────────────────────────────────────
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)

    # ─────────────────────────────────────────────────────────────────────────

    def timer_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn("Failed to grab frame from camera.")
            return

        # Convert OpenCV BGR image → ROS2 Image message
        msg                 = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'

        self.pub.publish(msg)

    # ─────────────────────────────────────────────────────────────────────────

    def destroy_node(self):
        """Release camera resource on shutdown."""
        if self.cap.isOpened():
            self.cap.release()
            self.get_logger().info("Camera released.")
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
