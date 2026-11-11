#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthode 4 - Détection de chute par vote basé sur 6 features de pose
Article: Mali & Jaiswal, IIT Kharagpur 2025

Ce nœud ROS2 extrait les landmarks MediaPipe, calcule 6 features :
  1. Height Ratio (tronc)
  2. Torso-Leg Angle
  3. Knee-Ankle Position
  4. Head-Floor Distance
  5. Upper Body Alignment
  6. Movement Analysis
Un buffer de 20 frames accumule les votes (feature seuillée) et déclare
une chute si plus de 60% des frames sont positives.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque

import mediapipe as mp

# ------------------------------------------------------------
# Utilitaires géométriques (reprises du code original)
# ------------------------------------------------------------
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle(p1, p2, p3):
    """Angle en degrés entre les vecteurs p1p2 et p2p3."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = np.hypot(v1[0], v1[1])
    mag2 = np.hypot(v2[0], v2[1])
    if mag1 * mag2 < 1e-6:
        return 0.0
    cos_angle = dot / (mag1 * mag2)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

# Indices simplifiés des landmarks (MediaPipe Pose)
LANDMARK_NAMES = {
    'NOSE': 0,
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

def draw_landmarks(image, landmarks, color=(0,255,0), connection_color=(255,128,0)):
    mp.solutions.drawing_utils.draw_landmarks(
        image, landmarks, mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
            color=color, thickness=2, circle_radius=2),
        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
            color=connection_color, thickness=2)
    )

# ------------------------------------------------------------
# Détecteur de chute Méthode 4
# ------------------------------------------------------------
class Method4FallDetector:
    def __init__(self):
        self.pose = MediaPipePoseDetector()
        self.buffer_size = 20               # fenêtre glissante
        self.vote_buffer = deque(maxlen=self.buffer_size)

        # Seuils des features (article, Section IV.A)
        self.height_ratio_threshold = 0.5
        self.torso_leg_angle_low = 60       # degrés
        self.torso_leg_angle_high = 120
        self.knee_ankle_threshold = 0.1     # normalisé par hauteur
        self.head_floor_threshold = 0.3     # normalisé
        self.movement_threshold = 0.05      # pixels/frame normalisé
        self.vote_threshold = 3             # nombre minimal de features positives pour voter

        self.initial_height = None          # hauteur de référence du tronc
        self.prev_landmarks = None

    def compute_features(self, landmarks, h_img, w_img):
        """
        Calcule les 6 features et retourne (features_dict, votes).
        """
        lm = landmarks.landmark
        L = LANDMARK_NAMES
        votes = 0

        # Récupération des coordonnées (x, y) en pixels
        def pt(idx):
            return (lm[idx].x * w_img, lm[idx].y * h_img)

        l_sh = pt(L['LEFT_SHOULDER'])
        r_sh = pt(L['RIGHT_SHOULDER'])
        l_hip = pt(L['LEFT_HIP'])
        r_hip = pt(L['RIGHT_HIP'])
        l_knee = pt(L['LEFT_KNEE'])
        r_knee = pt(L['RIGHT_KNEE'])
        l_ankle = pt(L['LEFT_ANKLE'])
        r_ankle = pt(L['RIGHT_ANKLE'])
        nose = pt(L['NOSE'])

        # Points médians
        sh_mid = ((l_sh[0] + r_sh[0])/2, (l_sh[1] + r_sh[1])/2)
        hip_mid = ((l_hip[0] + r_hip[0])/2, (l_hip[1] + r_hip[1])/2)
        knee_mid = ((l_knee[0] + r_knee[0])/2, (l_knee[1] + r_knee[1])/2)
        ankle_mid = ((l_ankle[0] + r_ankle[0])/2, (l_ankle[1] + r_ankle[1])/2)

        # Feature 1 : Height Ratio (hauteur du tronc / hauteur initiale)
        current_height = calculate_distance(sh_mid, hip_mid)
        if self.initial_height is None or self.initial_height < 1e-6:
            self.initial_height = current_height
        height_ratio = current_height / max(self.initial_height, 1e-6)
        if height_ratio < self.height_ratio_threshold:
            votes += 1

        # Feature 2 : Torso-Leg Angle (angle tronc-jambe)
        torso_leg_angle = calculate_angle(sh_mid, hip_mid, knee_mid)
        if self.torso_leg_angle_low <= torso_leg_angle <= self.torso_leg_angle_high:
            votes += 1

        # Feature 3 : Knee-Ankle Position (distance verticale genou-cheville / hauteur image)
        knee_ankle_dist = abs(knee_mid[1] - ankle_mid[1]) / h_img
        if knee_ankle_dist < self.knee_ankle_threshold:
            votes += 1

        # Feature 4 : Head-Floor Distance (distance nez-sol normalisée)
        head_floor_dist = (h_img - nose[1]) / h_img
        if head_floor_dist < self.head_floor_threshold:
            votes += 1

        # Feature 5 : Upper Body Alignment (tête désalignée par rapport au tronc)
        upper_misaligned = sh_mid[1] < nose[1]   # épaules plus hautes que nez ?
        if upper_misaligned:
            votes += 1

        # Feature 6 : Movement Analysis (déplacement moyen des keypoints entre frames)
        movement_speed = 0.0
        if self.prev_landmarks is not None:
            prev_lm = self.prev_landmarks.landmark
            total = 0.0
            for idx in [0, 11, 12, 23, 24]:   # nez, épaules, hanches
                dx = (lm[idx].x - prev_lm[idx].x) * w_img
                dy = (lm[idx].y - prev_lm[idx].y) * h_img
                total += np.hypot(dx, dy)
            movement_speed = total / 5.0 / max(h_img, 1)
            if movement_speed > self.movement_threshold:
                votes += 1

        # Stockage pour la frame suivante
        self.prev_landmarks = landmarks

        features = {
            'height_ratio': height_ratio,
            'torso_leg_angle': torso_leg_angle,
            'knee_ankle_dist': knee_ankle_dist,
            'head_floor_dist': head_floor_dist,
            'upper_misaligned': upper_misaligned,
            'movement_speed': movement_speed,
            'votes': votes
        }
        return features, votes

    def process_frame(self, frame_bgr):
        """
        Traite une image BGR.
        Retourne (frame_annotated, fall_detected, info_text).
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks = self.pose.process(frame_rgb)

        fall_detected = False
        features_info = {}
        votes = 0

        if landmarks:
            # Dessiner les landmarks sur l'image (toujours)
            draw_landmarks(frame_bgr, landmarks)

            features_info, votes = self.compute_features(landmarks, h, w)
            is_fall_frame = votes >= self.vote_threshold
            self.vote_buffer.append(is_fall_frame)

            if len(self.vote_buffer) == self.buffer_size:
                fall_count = sum(self.vote_buffer)
                if fall_count > self.buffer_size * 0.6:   # 60% de votes positifs
                    fall_detected = True

            # Affichage des features sur l'image
            y_pos = 100
            for feat_name, feat_val in features_info.items():
                if feat_name == 'votes':
                    color = (0, 0, 255) if feat_val >= self.vote_threshold else (0, 255, 0)
                    cv2.putText(frame_bgr, f"Votes: {feat_val}/6 (seuil={self.vote_threshold})",
                                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    if isinstance(feat_val, float):
                        val_str = f"{feat_val:.3f}"
                    else:
                        val_str = str(feat_val)
                    cv2.putText(frame_bgr, f"{feat_name}: {val_str}",
                                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (200, 200, 200), 1)
                y_pos += 20

            buf_falls = sum(self.vote_buffer)
            cv2.putText(frame_bgr, f"Buffer: {buf_falls}/{len(self.vote_buffer)} ({100*buf_falls/self.buffer_size:.0f}%)",
                        (10, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

        else:
            # Aucune personne détectée : on réinitialise le buffer ?
            # On peut choisir de garder l'état ou de le vider partiellement
            pass

        info = f"Votes: {votes}/6 | Buffer: {sum(self.vote_buffer)}/{len(self.vote_buffer)}"
        return frame_bgr, fall_detected, info

    def release(self):
        self.pose.close()


# ------------------------------------------------------------
# Nœud ROS2
# ------------------------------------------------------------
class Method4FallNode(Node):
    def __init__(self):
        super().__init__('method4_fall_node')
        self.bridge = CvBridge()

        # Paramètres ROS2
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('buffer_size', 20)
        self.declare_parameter('vote_threshold', 3)
        self.declare_parameter('fall_ratio', 0.6)

        image_topic = self.get_parameter('image_topic').value
        buffer_size = self.get_parameter('buffer_size').value
        vote_thr = self.get_parameter('vote_threshold').value
        self.fall_ratio = self.get_parameter('fall_ratio').value

        self.detector = Method4FallDetector()
        self.detector.buffer_size = buffer_size
        self.detector.vote_threshold = vote_thr
        self.detector.vote_buffer = deque(maxlen=buffer_size)

        # Subscriptions / Publications
        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.alert_pub = self.create_publisher(Bool, '/method4/alert', 10)
        self.img_pub = self.create_publisher(Image, '/method4/output', 10)
        self.status_pub = self.create_publisher(String, '/method4/status', 10)

        self.get_logger().info(f"Method4 (Pose Voting) node démarré sur {image_topic}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Erreur conversion: {e}")
            return

        # Redimension pour performance
        frame = cv2.resize(frame, (980, 740))
        annotated, fall, info = self.detector.process_frame(frame)

        # Publier alerte
        alert_msg = Bool()
        alert_msg.data = fall
        self.alert_pub.publish(alert_msg)
        if fall:
            self.get_logger().warn("⚠️ CHUTE DÉTECTÉE (Méthode 4)")

        # Publier status
        self.status_pub.publish(String(data=info))

        # Ajouter des annotations visuelles
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if fall:
            cv2.putText(annotated, "FALL ALERT", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Publier image
        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = msg.header
        self.img_pub.publish(out_msg)

    def destroy_node(self):
        self.detector.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Method4FallNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()