#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthode 2 - Détection de chute par MediaPipe + LSTM
Article: Sirikongtham & Nimkoompai, IJACSA 2025

Ce nœud ROS2 extrait les keypoints MediaPipe (33 points × 4 valeurs = 132)
sur une séquence glissante de 30 frames, puis utilise un modèle LSTM
(ou une heuristique de repli) pour classer fall vs non-fall.
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
import sys

import mediapipe as mp

# Paramètres du modèle
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 33
VALUES_PER_LANDMARK = 4          # x, y, z, visibility
INPUT_FEATURES = NUM_LANDMARKS * VALUES_PER_LANDMARK  # = 132


def extract_keypoints_132(landmarks):
    """
    Extrait un vecteur de 132 valeurs à partir des landmarks MediaPipe.
    Si landmarks est None, retourne un vecteur nul.
    Ordre : pour chaque landmark (0 à 32) : x, y, z, visibility.
    """
    if landmarks is None:
        return np.zeros(INPUT_FEATURES, dtype=np.float32)
    features = []
    for lm in landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(features, dtype=np.float32)


def heuristic_fall_check(keypoints):
    """
    Heuristique de secours (utilisée si aucun modèle LSTM n'est chargé).
    Se base sur la position verticale du torse et l'horizontalité.
    Retourne (fall_detected, confidence).
    """
    if np.all(keypoints == 0):
        return False, 0.0

    # Indices : chaque landmark a 4 valeurs : x, y, z, visibility
    # Landmarks utiles : nez(0), épaules(11,12), hanches(23,24), chevilles(27,28)
    nose_y = keypoints[0 * 4 + 1]
    l_shoulder_y = keypoints[11 * 4 + 1]
    r_shoulder_y = keypoints[12 * 4 + 1]
    l_hip_y = keypoints[23 * 4 + 1]
    r_hip_y = keypoints[24 * 4 + 1]
    l_ankle_y = keypoints[27 * 4 + 1]
    r_ankle_y = keypoints[28 * 4 + 1]

    shoulder_y = (l_shoulder_y + r_shoulder_y) / 2.0
    hip_y = (l_hip_y + r_hip_y) / 2.0
    ankle_y = (l_ankle_y + r_ankle_y) / 2.0
    body_height = abs(ankle_y - nose_y)

    if body_height < 0.01:
        return False, 0.0

    torso_hip_diff = abs(shoulder_y - hip_y)
    vertical_ratio = torso_hip_diff / body_height

    # Détection d'horizontalité (personne allongée)
    is_horizontal = (abs(nose_y - hip_y) < 0.15 and
                     abs(shoulder_y - ankle_y) < 0.25)

    confidence = 0.0
    if is_horizontal:
        confidence = min(1.0, (0.25 - abs(shoulder_y - ankle_y)) / 0.25)
    if vertical_ratio < 0.15:
        confidence = max(confidence, 0.8)

    return confidence > 0.5, confidence


def build_lstm_model():
    """
    Construit le modèle LSTM décrit dans l'article.
    Nécessite tensorflow. Sera appelé seulement si l'utilisateur veut
    entraîner, mais pour l'inférence on charge un modèle pré-entraîné.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(SEQUENCE_LENGTH, INPUT_FEATURES)),
        Dropout(0.2),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


class MediaPipePoseDetector:
    """Wrapper simple pour MediaPipe Pose."""
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


def draw_landmarks(image, landmarks, color=(0, 255, 0), thickness=2):
    """Dessine les landmarks et les connexions MediaPipe."""
    mp.solutions.drawing_utils.draw_landmarks(
        image, landmarks, mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
            color=color, thickness=thickness, circle_radius=2),
        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
            color=(255, 128, 0), thickness=thickness)
    )


# ----------------------------------------------------------------------
# Détecteur de chute (Méthode 2)
# ----------------------------------------------------------------------
class Method2FallDetector:
    def __init__(self, model_path=None):
        self.pose = MediaPipePoseDetector()
        self.sequence = []               # liste de vecteurs 132 (jusqu'à SEQUENCE_LENGTH)
        self.confidence_threshold = 0.50

        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(model_path)
                print(f"[Method2] Modèle LSTM chargé: {model_path}")
            except Exception as e:
                print(f"[Method2] Erreur chargement modèle: {e}, utilisation heuristique")
        else:
            print("[Method2] Aucun modèle LSTM fourni → utilisation heuristique")

    def process_frame(self, frame_bgr):
        """Traite une image BGR. Retourne (frame_annoté, fall, info_text)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks = self.pose.process(frame_rgb)

        # Dessiner les landmarks sur l'image (même si non utilisés directement)
        if landmarks:
            draw_landmarks(frame_bgr, landmarks)

        keypoints = extract_keypoints_132(landmarks)
        self.sequence.append(keypoints)

        # Garder seulement les SEQUENCE_LENGTH dernières frames
        if len(self.sequence) > SEQUENCE_LENGTH:
            self.sequence.pop(0)

        fall_detected = False
        confidence = 0.0

        if len(self.sequence) == SEQUENCE_LENGTH:
            if self.model is not None:
                # Inférence LSTM
                input_seq = np.expand_dims(self.sequence, axis=0)   # (1, 30, 132)
                prediction = self.model.predict(input_seq, verbose=0)[0]
                # Le modèle sort [prob_fall, prob_non_fall]
                fall_prob = prediction[0]
                confidence = fall_prob
                fall_detected = fall_prob > self.confidence_threshold
            else:
                # Heuristique de secours (seulement sur la dernière frame)
                fall_detected, confidence = heuristic_fall_check(keypoints)

        status = f"Conf: {confidence:.2f} | Seq: {len(self.sequence)}/{SEQUENCE_LENGTH}"
        if fall_detected:
            status += " | CHUTE!"

        return frame_bgr, fall_detected, status

    def release(self):
        self.pose.close()


# ----------------------------------------------------------------------
# Nœud ROS2
# ----------------------------------------------------------------------
class Method2FallNode(Node):
    def __init__(self):
        super().__init__('method2_fall_node')
        self.bridge = CvBridge()

        # Paramètres ROS2
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.50)
        self.declare_parameter('sequence_length', 30)

        image_topic = self.get_parameter('image_topic').value
        model_path = self.get_parameter('model_path').value
        self.conf_thr = self.get_parameter('confidence_threshold').value
        seq_len = self.get_parameter('sequence_length').value
        # Ajuster la variable globale SEQUENCE_LENGTH si nécessaire
        global SEQUENCE_LENGTH
        SEQUENCE_LENGTH = seq_len

        self.detector = Method2FallDetector(model_path if model_path else None)
        self.detector.confidence_threshold = self.conf_thr

        # Subscriptions / Publications
        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.alert_pub = self.create_publisher(Bool, '/method2/alert', 10)
        self.img_pub = self.create_publisher(Image, '/method2/output', 10)
        self.status_pub = self.create_publisher(String, '/method2/status', 10)

        self.get_logger().info(f"Method2 (LSTM) node démarré sur {image_topic}")

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
            self.get_logger().warn("⚠️ CHUTE DÉTECTÉE (Méthode 2)")

        # Publier status
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
    node = Method2FallNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()