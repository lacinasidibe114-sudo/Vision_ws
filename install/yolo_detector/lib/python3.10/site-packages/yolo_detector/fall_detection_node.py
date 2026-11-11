#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Détection de chute selon la Méthode 1 (YOLO + MediaPipe + aspect ratio)
Article: Kothari & Chakurkar, MethodsX 2025
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# ----------------------------------------------------------------------
# Utilitaires (distance, landmarks, etc.)
# ----------------------------------------------------------------------
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Mapping des landmarks MediaPipe (indices simplifiés)
LANDMARK_NAMES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28
}

class PoseDetector:
    """Wrapper simple pour MediaPipe Pose."""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, image_rgb):
        """Retourne les landmarks (objet MediaPipe) ou None."""
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            return results.pose_landmarks
        return None

    def close(self):
        self.pose.close()

def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=2):
    """Dessine les landmarks sur l'image (optionnel)."""
    h, w = image.shape[:2]
    for lm in landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), radius, color, -1)

# ----------------------------------------------------------------------
# Détecteur de chute (Méthode 1)
# ----------------------------------------------------------------------
class FallDetectorMethod1:
    def __init__(self, yolo_model_path, conf_threshold=0.80):
        self.yolo = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold
        self.pose = PoseDetector()

        # Paramètres article
        self.aspect_ratio_threshold = 0.9
        self.fall_confirm_frames = 3
        self.fall_counter = 0

    def detect_activity(self, landmarks, h, w):
        """Classification activité (debout, assis, marche, allongé)."""
        lm = landmarks.landmark
        # Récupération des coordonnées (x,y)
        def get_xy(part):
            return (lm[LANDMARK_NAMES[part]].x * w,
                    lm[LANDMARK_NAMES[part]].y * h)

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
            shoulder_y = (lm[LANDMARK_NAMES['LEFT_SHOULDER']].y +
                          lm[LANDMARK_NAMES['RIGHT_SHOULDER']].y) / 2.0
            hip_y = (lm[LANDMARK_NAMES['LEFT_HIP']].y +
                     lm[LANDMARK_NAMES['RIGHT_HIP']].y) / 2.0
            if abs(shoulder_y - hip_y) < 0.05:
                return "Allonge"
            return "Debout"

    def check_fall_aspect_ratio(self, bbox):
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1
        if w < 1:
            return False, 0.0
        aspect_ratio = h / w
        return (aspect_ratio < self.aspect_ratio_threshold), aspect_ratio

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        fall_detected = False
        activity = "Aucune"
        aspect_ratio_val = 0.0

        results = self.yolo(frame, verbose=False, classes=[0])  # personne

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Aspect ratio
            is_fall, aspect_ratio_val = self.check_fall_aspect_ratio((x1, y1, x2, y2))

            # Pose estimation sur la ROI
            roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if roi.size > 0:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pose_result = self.pose.process(roi_rgb)
                if pose_result:
                    draw_landmarks(roi, pose_result)
                    roi_h, roi_w = roi.shape[:2]
                    activity = self.detect_activity(pose_result, roi_h, roi_w)

            # Compteur de chute
            if is_fall:
                self.fall_counter += 1
                color = (0, 0, 255)  # rouge
            else:
                self.fall_counter = max(0, self.fall_counter - 1)
                color = (0, 255, 0)  # vert

            if self.fall_counter >= self.fall_confirm_frames:
                fall_detected = True

            # Annotation
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Personne {conf:.2f} AR:{aspect_ratio_val:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        info_text = f"Act: {activity} | AR: {aspect_ratio_val:.2f}"
        return frame, fall_detected, info_text

    def release(self):
        self.pose.close()

# ----------------------------------------------------------------------
# Nœud ROS2
# ----------------------------------------------------------------------
class FallDetectionNode(Node):
    def __init__(self):
        super().__init__('fall_detection_node')
        self.bridge = CvBridge()

        # Paramètres
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('yolo_model', '/home/sidibe/yolo_project/yolo11n.pt')
        self.declare_parameter('conf_threshold', 0.80)

        image_topic = self.get_parameter('image_topic').value
        yolo_path = self.get_parameter('yolo_model').value
        conf_thr = self.get_parameter('conf_threshold').value

        self.detector = FallDetectorMethod1(yolo_path, conf_thr)

        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.pub_img = self.create_publisher(Image, '/yolo/fall_detection', 10)

        self.get_logger().info(f"Fall detection node started (Méthode 1) sur {image_topic}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Conversion error: {e}")
            return

        # Redimension pour performance (optionnel)
        frame = cv2.resize(frame, (980, 740))
        annotated, fall_detected, info = self.detector.process_frame(frame)

        if fall_detected:
            self.get_logger().warn("⚠️ CHUTE DÉTECTÉE !")

        # Ajout d'un panneau d'information
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if fall_detected:
            cv2.putText(annotated, "FALL ALERT", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = msg.header
        self.pub_img.publish(out_msg)

    def destroy_node(self):
        self.detector.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FallDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()