#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.bridge = CvBridge()
        
        # Charger le modèle YOLO (modifie le chemin si besoin)
        self.model = YOLO('/home/sidibe/yolo_project/yolo11n.pt')
        
        # Souscrire au topic RGB de la caméra (Isaac Sim)
        self.sub = self.create_subscription(Image, '/rgb', self.image_callback, 10)
        
        # Publisher pour l'image annotée
        self.pub_img = self.create_publisher(Image, '/yolo/rgb', 10)
        
        # Publisher pour les markers 3D (optionnel)
        self.pub_markers = self.create_publisher(MarkerArray, '/yolo/markers', 10)
        
        self.get_logger().info('YOLO node started, listening to /rgb')
    
    def image_callback(self, msg):
        try:
            # Convertir ROS Image -> OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Conversion error: {e}')
            return
        
        # Détection YOLO
        results = self.model(frame, verbose=False)
        annotated = results[0].plot()  # image avec bounding boxes
        
        # Publier l'image annotée
        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = msg.header
        self.pub_img.publish(out_msg)
        
        # Optionnel : envoyer des markers pour RViz (position 2D approximative)
        markers = MarkerArray()
        for i, box in enumerate(results[0].boxes):
            # Coordonnées du centre du bounding box (pixels)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.model.names[cls]
            
            # Créer un marker (position en 2D, mais RViz le verra dans le plan image)
            marker = Marker()
            marker.header = msg.header
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = cx / msg.width - 0.5   # normalisé entre -0.5 et 0.5
            marker.pose.position.y = 0.0 - (cy / msg.height - 0.5)
            marker.pose.position.z = 1.0
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.text = f"{label} {conf:.2f}"
            markers.markers.append(marker)
        
        self.pub_markers.publish(markers)
    
def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()