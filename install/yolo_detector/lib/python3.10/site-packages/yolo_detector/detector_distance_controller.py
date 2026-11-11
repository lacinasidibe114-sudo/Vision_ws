#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Détection des personnes en mouvement avec YOLO + MOG2 (BackgroundSubtractor) pour ROS2.
Souscrit à /rgb (image), applique MOG2 et YOLO, publie l'image annotée sur /yolo/rgb.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class PersonMotionDetector(Node):
    def __init__(self):
        super().__init__('person_motion_detector')
        self.bridge = CvBridge()

        # Charger YOLO (chemin absolu ou relatif)
        # Utilisez votre modèle existant, par exemple yolo11n.pt
        self.model = YOLO('/home/sidibe/yolo_project/yolo11n.pt')

        # Paramètres
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('history', 500)
        self.declare_parameter('var_threshold', 16)
        self.declare_parameter('movement_ratio', 0.15)
        self.declare_parameter('conf_threshold', 0.5)

        image_topic = self.get_parameter('image_topic').value
        history = self.get_parameter('history').value
        var_threshold = self.get_parameter('var_threshold').value
        self.movement_ratio = self.get_parameter('movement_ratio').value
        self.conf_threshold = self.get_parameter('conf_threshold').value

        # Créer le soustracteur de fond MOG2
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True
        )

        # Subscription à l'image
        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)

        # Publisher pour l'image annotée
        self.pub_img = self.create_publisher(Image, '/yolo/rgb', 10)

        self.get_logger().info(
            f'PersonMotionDetector démarré sur {image_topic} (seuil mouvement: {self.movement_ratio})'
        )

    def image_callback(self, msg):
        # Conversion ROS -> OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Erreur conversion: {e}')
            return

        # ---------- Étape 1 : Masque de mouvement avec MOG2 ----------
        fgmask = self.fgbg.apply(frame)
        # Supprimer les ombres (127) : ne garder que les vrais mouvements (255)
        fgmask = np.where(fgmask == 255, 255, 0).astype(np.uint8)
        # Post-traitement : médian puis dilatation
        fgmask = cv2.medianBlur(fgmask, 5)
        fgmask = cv2.dilate(fgmask, None, iterations=2)

        # ---------- Étape 2 : Détection des personnes avec YOLO ----------
        results = self.model(frame, verbose=False, conf=self.conf_threshold, classes=[0])  # classe 0 = personne

        # Copie pour affichage
        output = frame.copy()

        # ---------- Étape 3 : Pour chaque personne, décider mouvement ----------
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                # Extraire la région du masque correspondant à la boîte
                roi_motion = fgmask[y1:y2, x1:x2]
                if roi_motion.size == 0:
                    continue

                motion_pixels = np.count_nonzero(roi_motion)
                total_pixels = roi_motion.size
                ratio = motion_pixels / total_pixels

                # Choisir la couleur
                if ratio > self.movement_ratio:
                    color = (0, 0, 255)        # rouge : en mouvement
                    label = "Moving"
                else:
                    color = (0, 255, 0)        # vert : immobile
                    label = "Still"

                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Optionnel : afficher le masque de mouvement dans une petite fenêtre (décommenter pour debug)
        # cv2.imshow("Motion mask", fgmask)
        # cv2.waitKey(1)

        # Publication de l'image annotée
        out_msg = self.bridge.cv2_to_imgmsg(output, 'bgr8')
        out_msg.header = msg.header
        self.pub_img.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PersonMotionDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()