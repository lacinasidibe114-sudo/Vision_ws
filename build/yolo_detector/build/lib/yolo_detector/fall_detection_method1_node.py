#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthode 1 - Détection de chute par YOLO + MediaPipe (Aspect Ratio)
Article: Kothari & Chakurkar, MethodsX 2025

Ce nœud ROS2 s'abonne à un topic image (/rgb par défaut), détecte les personnes,
calcule le rapport d'aspect de la boîte englobante (height/width) et utilise
MediaPipe pour classifier l'activité (debout, assis, marche, allongé).
Une chute est confirmée si l'aspect ratio < 0.9 pendant 3 frames consécutives.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO


# ------------------------------------------------------------
# Utilitaires géométriques (reprises du code original)
# ------------------------------------------------------------
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Indices des landmarks MediaPipe (simplifiés)
LANDMARK_NAMES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
}

class MediaPipePoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame_rgb):
        results = self.pose.process(frame_rgb)
        return results.pose_landmarks if results.pose_landmarks else None

    def close(self):
        self.pose.close()

def draw_landmarks(image, landmarks, color=(0, 255, 0), connection_color=(255, 128, 0)):
    mp.solutions.drawing_utils.draw_landmarks(
        image, landmarks, mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
            color=color, thickness=2, circle_radius=2),
        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
            color=connection_color, thickness=2)
    )


# ------------------------------------------------------------
# Détecteur de chute Méthode 1
# ------------------------------------------------------------
class Method1FallDetector:
    def __init__(self, yolo_model_path, conf_threshold=0.80):
        self.yolo = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold
        self.pose = MediaPipePoseDetector()

        # Paramètres de l'article
        self.aspect_ratio_threshold = 0.9   # < 0.9 → chute potentielle
        self.fall_confirm_frames = 3
        self.fall_counter = 0

    def detect_activity(self, landmarks, h, w):
        """
        Classification d'activité : Debout, Assis, Marche, Allongé.
        Basée sur les proportions corporelles (article, Step 2).
        """
        lm = landmarks.landmark
        L = LANDMARK_NAMES

        def get_xy(part):
            idx = L[part]
            return (lm[idx].x * w, lm[idx].y * h)

        l_sh = get_xy('LEFT_SHOULDER')
        r_sh = get_xy('RIGHT_SHOULDER')
        l_hip = get_xy('LEFT_HIP')
        r_hip = get_xy('RIGHT_HIP')
        l_knee = get_xy('LEFT_KNEE')
        r_knee = get_xy('RIGHT_KNEE')

        h_upper = (calculate_distance(l_sh, l_hip) +
                   calculate_distance(r_sh, r_hip)) / 2.0
        h_leg = (calculate_distance(l_hip, l_knee) +
                 calculate_distance(r_hip, r_knee)) / 2.0

        if h_upper < 1e-5:
            return "Inconnu"

        ratio = h_leg / h_upper

        if ratio < 0.5:
            return "Assis"
        elif ratio > 1.2:
            return "Marche"
        else:
            shoulder_y = (lm[L['LEFT_SHOULDER']].y + lm[L['RIGHT_SHOULDER']].y) / 2.0
            hip_y = (lm[L['LEFT_HIP']].y + lm[L['RIGHT_HIP']].y) / 2.0
            if abs(shoulder_y - hip_y) < 0.05:
                return "Allonge"
            return "Debout"

    def check_fall_aspect_ratio(self, bbox):
        """Calcule le rapport d'aspect (height/width). Retourne (is_fall, aspect_ratio)."""
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1
        if w < 1:
            return False, 0.0
        aspect_ratio = h / w
        return (aspect_ratio < self.aspect_ratio_threshold), aspect_ratio

    def process_frame(self, frame_bgr):
        """
        Traite une image BGR.
        Retourne (frame_annotated, fall_detected, info_text).
        """
        h, w = frame_bgr.shape[:2]
        fall_detected = False
        activity = "Aucune personne"
        aspect_ratio_val = 0.0

        # Détection YOLO (personnes, classe 0)
        results = self.yolo(frame_bgr, verbose=False, classes=[0], conf=self.conf_threshold)

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 1) Rapport d'aspect
                is_fall_ar, aspect_ratio_val = self.check_fall_aspect_ratio((x1, y1, x2, y2))

                # 2) Pose estimation sur la ROI
                roi = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if roi.size > 0:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    landmarks = self.pose.process(roi_rgb)
                    if landmarks:
                        draw_landmarks(roi, landmarks)
                        roi_h, roi_w = roi.shape[:2]
                        activity = self.detect_activity(landmarks, roi_h, roi_w)

                # 3) Compteur de chute (confirmation sur plusieurs frames)
                if is_fall_ar:
                    self.fall_counter += 1
                    color = (0, 0, 255)  # rouge
                else:
                    self.fall_counter = max(0, self.fall_counter - 1)
                    color = (0, 255, 0)  # vert

                if self.fall_counter >= self.fall_confirm_frames:
                    fall_detected = True

                # Annotation
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                label = f"Personne {conf:.2f} AR:{aspect_ratio_val:.2f}"
                cv2.putText(frame_bgr, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Afficher l'activité détectée
                cv2.putText(frame_bgr, f"Act: {activity}", (x1, y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)

        info_text = f"Act: {activity} | AR: {aspect_ratio_val:.2f}"
        return frame_bgr, fall_detected, info_text

    def release(self):
        self.pose.close()


# ------------------------------------------------------------
# Nœud ROS2
# ------------------------------------------------------------
class Method1FallNode(Node):
    def __init__(self):
        super().__init__('method1_fall_node')
        self.bridge = CvBridge()

        # Paramètres ROS2
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('yolo_model', '/home/sidibe/yolo_project/yolov8n.pt')
        self.declare_parameter('conf_threshold', 0.80)
        self.declare_parameter('aspect_ratio_threshold', 0.9)
        self.declare_parameter('fall_confirm_frames', 3)

        image_topic = self.get_parameter('image_topic').value
        yolo_path = self.get_parameter('yolo_model').value
        conf_thr = self.get_parameter('conf_threshold').value

        self.detector = Method1FallDetector(yolo_path, conf_thr)
        # Appliquer les seuils personnalisés
        self.detector.aspect_ratio_threshold = self.get_parameter('aspect_ratio_threshold').value
        self.detector.fall_confirm_frames = self.get_parameter('fall_confirm_frames').value

        # Subscriber / Publishers
        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.alert_pub = self.create_publisher(Bool, '/method1/alert', 10)
        self.img_pub = self.create_publisher(Image, '/method1/output', 10)
        self.status_pub = self.create_publisher(String, '/method1/status', 10)

        self.get_logger().info(f"Method1 (YOLO+MediaPipe AR) node démarré sur {image_topic}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Erreur conversion: {e}")
            return

        # Redimension optionnel pour performance
        frame = cv2.resize(frame, (980, 740))
        annotated, fall, info = self.detector.process_frame(frame)

        # Publier alerte
        alert_msg = Bool()
        alert_msg.data = fall
        self.alert_pub.publish(alert_msg)
        if fall:
            self.get_logger().warn("⚠️ CHUTE DÉTECTÉE (Méthode 1)")

        # Publier statut
        self.status_pub.publish(String(data=info))

        # Ajouter des annotations visuelles
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if fall:
            cv2.putText(annotated, "FALL ALERT", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Publier image annotée
        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = msg.header
        self.img_pub.publish(out_msg)

    def destroy_node(self):
        self.detector.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Method1FallNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()