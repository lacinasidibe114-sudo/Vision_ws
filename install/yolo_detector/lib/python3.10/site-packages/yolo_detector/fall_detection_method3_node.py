#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthode 3 - ElderFallGuard: Détection de chute par MediaPipe + Random Forest
Article: Riahi et al., arXiv 2025

Ce nœud ROS2 implémente la logique temporelle :
   - Pose 6 (prone / chute) détectée pendant > 3 secondes
   - ET mouvement très faible (< seuil) pendant > 2 secondes
   -> ALERTE (avec cooldown de 30 secondes)

Si un modèle Random Forest est fourni, il est utilisé ; sinon, un classifieur
heuristique (basé sur des règles géométriques) est employé.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os
import joblib

import mediapipe as mp

# ------------------------------------------------------------
# Constantes et utilitaires (inspirés du code original)
# ------------------------------------------------------------
POSE_NAMES = {
    0: "Debout", 1: "Marche", 2: "Assis", 3: "Penche",
    4: "Allonge (intentionnel)", 5: "Perte d'equilibre",
    6: "Prone/Chute (POSE6)", 7: "Supine",
    8: "En train de se relever", 9: "A genoux",
    10: "Accroupi", 11: "Bras leves"
}

def extract_keypoints_66(landmarks):
    """
    Extrait 66 features : (x, y, visibility) pour 33 keypoints MediaPipe.
    Retourne un vecteur numpy de taille 66.
    """
    features = []
    for lm in landmarks.landmark:
        features.append(lm.x)
        features.append(lm.y)
    return np.array(features)

def heuristic_pose_classifier(landmarks):
    """Classification heuristique des 12 poses (fallback si pas de RF)."""
    lm = landmarks.landmark
    nose_y = lm[0].y
    shoulder_y = (lm[11].y + lm[12].y) / 2
    hip_y = (lm[23].y + lm[24].y) / 2
    knee_y = (lm[25].y + lm[26].y) / 2
    ankle_y = (lm[27].y + lm[28].y) / 2
    wrist_y = (lm[15].y + lm[16].y) / 2

    body_height = abs(ankle_y - nose_y)
    torso_height = abs(hip_y - shoulder_y)
    is_horizontal = body_height < 0.25
    shoulders_near_hips = abs(shoulder_y - hip_y) < 0.08
    knees_bent = abs(knee_y - hip_y) < abs(ankle_y - hip_y) * 0.5

    if is_horizontal and shoulder_y > 0.6:
        return 6, "Prone/Chute (POSE6)"
    if is_horizontal and shoulder_y < 0.5:
        return 7, "Supine"
    if shoulders_near_hips and ankle_y > hip_y:
        return 6, "Prone/Chute (POSE6)"
    if body_height > 0.4 and not knees_bent:
        hip_diff = abs(lm[23].x - lm[24].x)
        if hip_diff > 0.15:
            return 1, "Marche"
        return 0, "Debout"
    if knees_bent and abs(hip_y - knee_y) < 0.1:
        if torso_height > 0.15:
            return 2, "Assis"
        return 10, "Accroupi"
    if shoulder_y > hip_y * 0.95 and body_height > 0.2:
        return 3, "Penche"
    if wrist_y < shoulder_y - 0.15:
        return 11, "Bras leves"
    if knee_y > 0.7 and hip_y > 0.5:
        return 9, "A genoux"
    return 0, "Debout"

class MediaPipePoseDetector:
    """Wrapper minimal pour MediaPipe Pose."""
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
    """Dessine les landmarks et connexions (similaire au code original)."""
    mp.solutions.drawing_utils.draw_landmarks(
        image, landmarks, mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=2, circle_radius=2),
        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=connection_color, thickness=2)
    )

# ------------------------------------------------------------
# Détecteur de chute Méthode 3
# ------------------------------------------------------------
class ElderFallGuardDetector:
    def __init__(self, model_path=None):
        self.pose = MediaPipePoseDetector()
        self.rf_model = None
        if model_path and os.path.exists(model_path):
            self.rf_model = joblib.load(model_path)
            print(f"[ElderFallGuard] Modèle Random Forest chargé: {model_path}")
        else:
            print("[ElderFallGuard] Aucun modèle RF fourni → utilisation du classifieur heuristique.")

        # Variables temporelles (article)
        self.pose6_start_time = None
        self.motion_drop_start_time = None
        self.prev_landmarks = None
        self.pose6_duration_threshold = 3.0      # secondes
        self.motion_drop_threshold = 2.0         # secondes
        self.motion_threshold = 0.02             # mouvement moyen faible
        self.fall_cooldown = 30.0                # secondes entre deux alertes
        self.last_alert_time = 0.0

    def calculate_motion(self, landmarks):
        """Moyenne du déplacement des keypoints entre deux frames."""
        if self.prev_landmarks is None:
            return 1.0
        motion = 0.0
        count = 0
        for curr, prev in zip(landmarks.landmark, self.prev_landmarks.landmark):
            motion += np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2)
            count += 1
        return motion / count if count > 0 else 0.0

    def classify_pose(self, landmarks):
        """Retourne (pose_id, pose_name) en utilisant RF ou heuristique."""
        if self.rf_model is not None:
            features = extract_keypoints_66(landmarks).reshape(1, -1)
            pose_id = self.rf_model.predict(features)[0]
            return pose_id, POSE_NAMES.get(pose_id, f"Pose{pose_id}")
        else:
            return heuristic_pose_classifier(landmarks)

    def process_frame(self, frame_bgr):
        """
        Traite une image BGR.
        Retourne (frame_annotated, fall_detected, info_text)
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks = self.pose.process(frame_rgb)

        fall_detected = False
        pose_id = -1
        pose_name = "Aucune personne"
        motion = 0.0
        pose6_time = 0.0
        motion_drop_time = 0.0
        current_time = time.time()

        if landmarks:
            draw_landmarks(frame_bgr, landmarks, color=(0,255,0), connection_color=(255,128,0))
            pose_id, pose_name = self.classify_pose(landmarks)
            motion = self.calculate_motion(landmarks)
            self.prev_landmarks = landmarks

            # Logique temporelle pose 6 (prone)
            if pose_id == 6:
                if self.pose6_start_time is None:
                    self.pose6_start_time = current_time
                pose6_time = current_time - self.pose6_start_time
            else:
                self.pose6_start_time = None

            # Logique chute de mouvement
            if motion < self.motion_threshold:
                if self.motion_drop_start_time is None:
                    self.motion_drop_start_time = current_time
                motion_drop_time = current_time - self.motion_drop_start_time
            else:
                self.motion_drop_start_time = None

            # Décision de chute
            if (pose6_time > self.pose6_duration_threshold and
                motion_drop_time > self.motion_drop_threshold):
                if current_time - self.last_alert_time > self.fall_cooldown:
                    fall_detected = True
                    self.last_alert_time = current_time

            # Affichage de la pose
            color = (0,0,255) if pose_id == 6 else (0,255,0)
            cv2.putText(frame_bgr, f"Pose: {pose_name}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        info = (f"Pose: {pose_name} | Mvt: {motion:.3f} | "
                f"P6: {pose6_time:.1f}s | Drop: {motion_drop_time:.1f}s")
        return frame_bgr, fall_detected, info

    def release(self):
        self.pose.close()

# ------------------------------------------------------------
# Nœud ROS2
# ------------------------------------------------------------
class ElderFallGuardNode(Node):
    def __init__(self):
        super().__init__('elderfallguard_node')
        self.bridge = CvBridge()

        # Paramètres ROS2
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('model_path', '')
        self.declare_parameter('pose6_threshold', 3.0)
        self.declare_parameter('motion_drop_threshold', 2.0)
        self.declare_parameter('motion_threshold', 0.02)
        self.declare_parameter('fall_cooldown', 30.0)

        image_topic = self.get_parameter('image_topic').value
        model_path = self.get_parameter('model_path').value
        # Mettre à jour les seuils dans le détecteur après création
        self.detector = ElderFallGuardDetector(model_path if model_path else None)

        # Appliquer les seuils depuis les paramètres
        self.detector.pose6_duration_threshold = self.get_parameter('pose6_threshold').value
        self.detector.motion_drop_threshold = self.get_parameter('motion_drop_threshold').value
        self.detector.motion_threshold = self.get_parameter('motion_threshold').value
        self.detector.fall_cooldown = self.get_parameter('fall_cooldown').value

        # Subscriber / Publishers
        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.alert_pub = self.create_publisher(Bool, '/elderfallguard/alert', 10)
        self.img_pub = self.create_publisher(Image, '/elderfallguard/output', 10)
        self.status_pub = self.create_publisher(String, '/elderfallguard/status', 10)

        self.get_logger().info(f"ElderFallGuard node démarré sur {image_topic}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Erreur conversion image: {e}")
            return

        # Redimension optionnel pour performance
        frame = cv2.resize(frame, (980, 740))
        annotated, fall, info = self.detector.process_frame(frame)

        # Publier alerte
        alert_msg = Bool()
        alert_msg.data = fall
        self.alert_pub.publish(alert_msg)
        if fall:
            self.get_logger().warn("⚠️ CHUTE DÉTECTÉE (ElderFallGuard)")

        # Publier statut texte
        self.status_pub.publish(String(data=info))

        # Ajouter quelques annotations
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
    node = ElderFallGuardNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()