#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from message_filters import Subscriber, ApproximateTimeSynchronizer

class YoloPersonDistance(Node):
    def __init__(self):
        super().__init__('yolo_person_distance')
        self.bridge = CvBridge()
        # Charger le modèle YOLO (modifier le chemin si besoin)
        self.model = YOLO('/home/sidibe/yolo_project/yolo11n.pt')
        # ID de la classe 'person' dans COCO
        self.person_class_id = 0

        # Souscriptions synchronisées pour RGB et Depth
        self.rgb_sub = Subscriber(self, Image, '/rgb')
        self.depth_sub = Subscriber(self, Image, '/depth')
        # Tolérance de synchronisation : 0.1 seconde
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.image_depth_callback)

        # Publishers
        self.pub_img = self.create_publisher(Image, '/yolo/rgb', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/yolo/markers', 10)

        self.get_logger().info('YOLO personne + distance démarré (topics /rgb et /depth)')

    def image_depth_callback(self, rgb_msg, depth_msg):
        # Conversion RGB → OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Erreur conversion RGB: {e}')
            return

        # Conversion Depth → OpenCV (matrice float32)
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        except Exception as e:
            self.get_logger().error(f'Erreur conversion Depth: {e}')
            return

        # Vérifier que les dimensions correspondent
        if frame.shape[:2] != depth.shape[:2]:
            self.get_logger().warn(f'Tailles différentes: RGB {frame.shape[:2]} vs Depth {depth.shape[:2]}')
            return

        # Inférence YOLO (seulement sur RGB)
        results = self.model(frame, verbose=False, classes=[self.person_class_id])
        annotated = results[0].plot()  # image avec boxes (uniquement personnes)

        # Préparer les markers pour RViz
        markers = MarkerArray()
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            # Centre de la boîte en pixels
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Lecture de la distance au centre de la boîte
            # Attention: depth est en float32, peut contenir des NaN ou 0 pour les pixels sans valeur
            distance = depth[cy, cx]
            if np.isnan(distance) or distance <= 0:
                # Alternative : moyenne sur la boîte entière (ignorer les zéros)
                roi = depth[y1:y2, x1:x2]
                valid = roi[roi > 0]
                if len(valid) > 0:
                    distance = np.mean(valid)
                else:
                    distance = -1.0  # valeur invalide

            conf = float(box.conf[0])
            label = f"personne {conf:.2f} | {distance:.2f} m" if distance > 0 else f"personne {conf:.2f} | distance inconnue"

            # Dessiner la distance sur l'image (en plus du plot YOLO)
            cv2.putText(annotated, f"{distance:.2f}m", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Créer un marker texte pour RViz
            marker = Marker()
            marker.header = rgb_msg.header
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            # Position en coordonnées normalisées (entre -0.5 et 0.5) pour rester dans le champ de vue
            marker.pose.position.x = (cx / rgb_msg.width) - 0.5
            marker.pose.position.y = 0.0 - (cy / rgb_msg.height - 0.5)
            marker.pose.position.z = 1.0
            marker.scale.z = 0.08
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.text = label
            markers.markers.append(marker)

        # Publier l'image annotée
        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = rgb_msg.header
        self.pub_img.publish(out_msg)

        # Publier les markers
        self.pub_markers.publish(markers)

def main(args=None):
    rclpy.init(args=args)
    node = YoloPersonDistance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()