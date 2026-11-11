#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

class Viewer(Node):
    def __init__(self):
        super().__init__('viewer')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.callback,
            10  # QoS par défaut (RELIABLE)
        )

    def callback(self, msg):
        # Décompresser l'image
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow("Camera view", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = Viewer()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()