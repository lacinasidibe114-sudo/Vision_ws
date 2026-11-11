#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from message_filters import Subscriber, ApproximateTimeSynchronizer

class YoloPersonController(Node):
    def __init__(self):
        super().__init__('yolo_person_controller')
        self.bridge = CvBridge()
        self.model = YOLO('/home/sidibe/yolo_project/yolo11n.pt')
        self.person_class_id = 0

        # Souscriptions synchronisées RGB + Depth
        self.rgb_sub = Subscriber(self, Image, '/rgb')
        self.depth_sub = Subscriber(self, Image, '/depth')
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.image_depth_callback)

        # Publisher commandes vitesse
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Publishers pour visualisation
        self.pub_img = self.create_publisher(Image, '/yolo/rgb', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/yolo/markers', 10)

        # Paramètres de contrôle
        self.angular_gain = 1.5    # gain proportionnel pour l'angle
        self.linear_gain = 0.5     # gain proportionnel pour la distance
        self.stop_distance = 0.5    # 50 cm
        self.angle_threshold = 0.1  # rad (~5.7°)

        self.get_logger().info('Nœud de contrôle YOLO + navigation démarré')

    def image_depth_callback(self, rgb_msg, depth_msg):
        # Conversion
        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        except Exception as e:
            self.get_logger().error(f'Erreur conversion: {e}')
            return

        if frame.shape[:2] != depth.shape[:2]:
            self.get_logger().warn('Tailles RGB/Depth différentes')
            return

        # Détection YOLO (seulement personnes)
        results = self.model(frame, verbose=False, classes=[self.person_class_id])
        annotated = results[0].plot()

        persons = []  # liste de (distance, angle, x1, y1, x2, y2, conf)
        h, w = frame.shape[:2]
        center_x = w / 2.0

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Distance au centre de la boîte
            dist = depth[cy, cx]
            if np.isnan(dist) or dist <= 0:
                # Moyenne sur la boîte
                roi = depth[y1:y2, x1:x2]
                valid = roi[roi > 0]
                if len(valid) > 0:
                    dist = np.mean(valid)
                else:
                    continue  # on ignore cette personne si pas de distance valide

            # Angle horizontal par rapport au centre (radians)
            # On suppose que l'image fait 120° de champ horizontal (à adapter selon votre caméra)
            # Approximation : angle = (pixel_center_offset / largeur) * champ_horizontal_rad
            # Ici champ_horizontal_rad = 1.047 (60°) par défaut.
            # Vous devez ajuster selon les specs de votre caméra.
            fov_horizontal_rad = 1.047  # 60° en rad
            offset_x = (cx - center_x) / w  # entre -0.5 et 0.5
            angle = offset_x * fov_horizontal_rad

            conf = float(box.conf[0])
            persons.append((dist, angle, x1, y1, x2, y2, conf))

        if not persons:
            # Aucune personne détectée → arrêt du robot
            self.stop_robot()
            # Publier quand même l'image annotée (sans personne)
            out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
            out_msg.header = rgb_msg.header
            self.pub_img.publish(out_msg)
            self.pub_markers.publish(MarkerArray())
            return

        # Choisir la personne la plus proche
        persons.sort(key=lambda x: x[0])  # tri par distance croissante
        dist, angle, x1, y1, x2, y2, conf = persons[0]

        # Afficher la cible sur l'image
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)  # cadre rouge pour la cible
        label = f"Cible: {dist:.2f}m | angle: {angle:.2f}rad"
        cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Commande du robot
        twist = Twist()
        if dist > self.stop_distance:
            # Asservissement : d'abord orienter vers la cible
            if abs(angle) > self.angle_threshold:
                twist.angular.z = -self.angular_gain * angle   # signe selon orientation
                twist.linear.x = 0.0
            else:
                # Angle correct, avancer proportionnellement à la distance (saturé)
                linear = self.linear_gain * (dist - self.stop_distance)
                linear = min(linear, 0.3)  # vitesse max 0.3 m/s
                twist.linear.x = linear
                twist.angular.z = 0.0
        else:
            # Arrêt atteint
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info(f"Arrêt à {dist:.2f} m de la personne")

        self.cmd_pub.publish(twist)

        # Publier l'image annotée
        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = rgb_msg.header
        self.pub_img.publish(out_msg)

        # Markers pour RViz (optionnel)
        markers = MarkerArray()
        marker = Marker()
        marker.header = rgb_msg.header
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = (cx / w) - 0.5
        marker.pose.position.y = 0.0 - (cy / h - 0.5)
        marker.pose.position.z = 1.0
        marker.scale.z = 0.08
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.text = f"Cible: {dist:.2f}m"
        markers.markers.append(marker)
        self.pub_markers.publish(markers)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = YoloPersonController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()