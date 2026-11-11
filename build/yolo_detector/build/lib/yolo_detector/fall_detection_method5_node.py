#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthode 5 - Détection de chute hybride HFDMIA-Pose
Article: Liu et al., Scientific Reports 2025

Ce nœud ROS2 combine YOLOv8 pour la détection de personnes,
MediaPipe pour l'extraction de 14 joints simplifiés, et un
algorithme hybride à deux étages :
  1. Feature instantanée (vitesse tête/centre et vitesse cheville)
  2. Feature d'état (angles du tronc avec temporisation)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from collections import deque

import mediapipe as mp
from ultralytics import YOLO

# ------------------------------------------------------------
# Utilitaires (reprises du code original)
# ------------------------------------------------------------
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Indices simplifiés des landmarks MediaPipe (pour 14 joints)
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
# Détecteur de chute Méthode 5
# ------------------------------------------------------------
class Method5FallDetector:
    def __init__(self, yolo_model_path, conf_threshold=0.5):
        self.yolo = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold
        self.pose = MediaPipePoseDetector()

        # Paramètres de l'article (Section III)
        self.chi = 1.5          # seuil rapport vitesses tête/centre
        self.alpha = 0.0        # vitesse minimale cheville (normalisée)
        self.beta = 0.3         # vitesse maximale cheville
        self.M = 5              # taille de la fenêtre d'historique (M frames)

        self.epsilon1 = 45.0    # seuil angle supérieur (theta_u)
        self.epsilon2 = 45.0    # seuil angle inférieur (theta_d)
        self.T1 = 1.0           # temps (secondes) pour theta_u sous seuil
        self.T2 = 1.5           # temps (secondes) pour theta_d sous seuil

        # Variables d'état
        self.position_history = deque(maxlen=self.M + 1)   # M+1 frames pour calcul vitesses
        self.instantaneous_fall_detected = False
        self.theta_u_fall_start = None
        self.theta_d_fall_start = None

    def get_joint_positions(self, landmarks, h_img, w_img):
        """
        Extrait les 14 joints simplifiés (pixels) selon l'article.
        Retourne un dictionnaire avec les noms des joints et leurs coordonnées (x,y).
        """
        lm = landmarks.landmark
        L = LANDMARK_NAMES
        joints = {}
        # Points de base
        for name, key in [('head', 'NOSE'),
                          ('r_shoulder', 'RIGHT_SHOULDER'),
                          ('l_shoulder', 'LEFT_SHOULDER'),
                          ('r_hip', 'RIGHT_HIP'),
                          ('l_hip', 'LEFT_HIP'),
                          ('r_knee', 'RIGHT_KNEE'),
                          ('l_knee', 'LEFT_KNEE'),
                          ('r_ankle', 'RIGHT_ANKLE'),
                          ('l_ankle', 'LEFT_ANKLE')]:
            idx = L[key]
            joints[name] = np.array([lm[idx].x * w_img, lm[idx].y * h_img])
        # Points composites
        joints['neck'] = (joints['l_shoulder'] + joints['r_shoulder']) / 2.0
        joints['center'] = (joints['neck'] + joints['r_hip'] + joints['l_hip']) / 3.0
        joints['crotch'] = (joints['r_hip'] + joints['l_hip']) / 2.0
        return joints

    def compute_velocities(self):
        """
        Calcule les normes de vitesse (pixels/frame) pour la tête, le centre et la cheville.
        Utilise l'historique des positions (M+1 frames). Retourne (v_head, v_center, v_ankle).
        """
        if len(self.position_history) < 2:
            return None, None, None

        history = list(self.position_history)
        v_head = np.zeros(2)
        v_center = np.zeros(2)
        v_ankle = np.zeros(2)
        count = 0
        for i in range(1, len(history)):
            v_head += history[i]['head'] - history[i-1]['head']
            v_center += history[i]['center'] - history[i-1]['center']
            ankle_curr = (history[i]['r_ankle'] + history[i]['l_ankle']) / 2.0
            ankle_prev = (history[i-1]['r_ankle'] + history[i-1]['l_ankle']) / 2.0
            v_ankle += ankle_curr - ankle_prev
            count += 1

        if count == 0:
            return None, None, None

        return (np.linalg.norm(v_head) / count,
                np.linalg.norm(v_center) / count,
                np.linalg.norm(v_ankle) / count)

    def check_instantaneous_feature(self):
        """
        Vérifie la condition : V_head / V_center > chi ET V_ankle ∈ [alpha, beta]
        Les vitesses sont normalisées par la diagonale de l'image (max_dim).
        """
        v_head, v_center, v_ankle = self.compute_velocities()
        if v_head is None:
            return False

        # Normalisation (taille typique de l'image ~ 740 pixels)
        # On utilise une constante pour éviter la dépendance à la résolution.
        # Dans l'article ils utilisent probablement des pixels normalisés.
        max_dim = 740.0
        v_h_norm = v_head / max_dim
        v_c_norm = v_center / max_dim
        v_a_norm = v_ankle / max_dim

        ratio = v_h_norm / (v_c_norm + 1e-8)
        return (ratio > self.chi) and (self.alpha <= v_a_norm <= self.beta)

    def compute_trunk_angles(self, joints):
        """
        Calcule les angles theta_u (tête-cou horizontal) et theta_d (crotch-genou horizontal).
        Retourne (theta_u, theta_d) en degrés.
        """
        neck = joints['neck']
        head = joints['head']
        crotch = joints['crotch']
        knee_mid = (joints['r_knee'] + joints['l_knee']) / 2.0

        horizontal = np.array([1.0, 0.0])

        # theta_u : angle entre le vecteur tête -> cou et l'horizontale
        upper_vec = head - neck
        if np.linalg.norm(upper_vec) < 1e-6:
            theta_u = 90.0
        else:
            cos_u = np.dot(upper_vec, horizontal) / np.linalg.norm(upper_vec)
            theta_u = np.degrees(np.arccos(np.clip(abs(cos_u), -1.0, 1.0)))

        # theta_d : angle entre le vecteur genou -> crotch et l'horizontale
        lower_vec = knee_mid - crotch
        if np.linalg.norm(lower_vec) < 1e-6:
            theta_d = 90.0
        else:
            cos_d = np.dot(lower_vec, horizontal) / np.linalg.norm(lower_vec)
            theta_d = np.degrees(np.arccos(np.clip(abs(cos_d), -1.0, 1.0)))

        return theta_u, theta_d

    def check_state_feature(self, theta_u, theta_d):
        """
        Vérifie si theta_u < epsilon1 pendant plus de T1 secondes,
        ou theta_d < epsilon2 pendant plus de T2 secondes.
        Retourne True si l'une des deux conditions est satisfaite.
        """
        current_time = time.time()

        # Condition sur theta_u
        if theta_u < self.epsilon1:
            if self.theta_u_fall_start is None:
                self.theta_u_fall_start = current_time
            elif current_time - self.theta_u_fall_start > self.T1:
                return True
        else:
            self.theta_u_fall_start = None

        # Condition sur theta_d
        if theta_d < self.epsilon2:
            if self.theta_d_fall_start is None:
                self.theta_d_fall_start = current_time
            elif current_time - self.theta_d_fall_start > self.T2:
                return True
        else:
            self.theta_d_fall_start = None

        return False

    def process_frame(self, frame_bgr):
        """
        Traite une image BGR.
        Retourne (frame_annotated, fall_detected, info_text).
        """
        h, w = frame_bgr.shape[:2]
        fall_detected = False
        info_text = "Aucune personne"
        theta_u_val = 0.0
        theta_d_val = 0.0
        inst_fall = False
        state_fall = False

        # Détection YOLO des personnes (classe 0)
        results = self.yolo(frame_bgr, verbose=False, classes=[0], conf=self.conf_threshold)

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Extraire la ROI de la personne
                roi = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if roi.size == 0:
                    continue

                # Pose estimation sur la ROI
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                landmarks = self.pose.process(roi_rgb)

                if landmarks is None:
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (128,128,128), 2)
                    continue

                # Dessiner les landmarks (coordonnées relatives à la ROI)
                draw_landmarks(roi, landmarks)

                # Obtenir les joints en coordonnées globales (pixels)
                roi_h, roi_w = roi.shape[:2]
                joints_roi = self.get_joint_positions(landmarks, roi_h, roi_w)
                # Convertir en coordonnées globales
                global_joints = {}
                for name, pos in joints_roi.items():
                    global_joints[name] = pos + np.array([x1, y1])

                # Stocker dans l'historique
                self.position_history.append(global_joints)

                # 1. Feature instantanée
                inst_fall = self.check_instantaneous_feature()
                if inst_fall:
                    self.instantaneous_fall_detected = True

                # 2. Feature d'état (angles)
                theta_u_val, theta_d_val = self.compute_trunk_angles(global_joints)
                state_fall = self.check_state_feature(theta_u_val, theta_d_val)

                # 3. Décision hybride
                # Selon l'article, une chute est confirmée si les deux features sont vraies
                # (ou si la feature d'état est vraie après détection instantanée)
                if (self.instantaneous_fall_detected and state_fall) or state_fall:
                    fall_detected = True

                if not state_fall:
                    self.instantaneous_fall_detected = False

                # Annotation
                color = (0, 0, 255) if fall_detected else (0, 255, 0)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                # Affichage des angles
                cv2.putText(frame_bgr, f"theta_u: {theta_u_val:.1f}",
                            (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)
                cv2.putText(frame_bgr, f"theta_d: {theta_d_val:.1f}",
                            (x1, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)

                info_text = f"Inst: {'OUI' if inst_fall else 'NON'} | Etat: {'OUI' if state_fall else 'NON'}"
                break  # ne traiter que la première personne (celle détectée)

        # Construction du message de statut
        status = f"θu:{theta_u_val:.0f} θd:{theta_d_val:.0f} | {info_text}"
        return frame_bgr, fall_detected, status

    def release(self):
        self.pose.close()


# ------------------------------------------------------------
# Nœud ROS2
# ------------------------------------------------------------
class Method5FallNode(Node):
    def __init__(self):
        super().__init__('method5_fall_node')
        self.bridge = CvBridge()

        # Paramètres ROS2
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('yolo_model', '/home/sidibe/yolo_project/yolov8n.pt')
        self.declare_parameter('conf_threshold', 0.5)
        # Paramètres de l'algorithme (optionnels, pour ajustement)
        self.declare_parameter('chi', 1.5)
        self.declare_parameter('alpha', 0.0)
        self.declare_parameter('beta', 0.3)
        self.declare_parameter('M', 5)
        self.declare_parameter('epsilon1', 45.0)
        self.declare_parameter('epsilon2', 45.0)
        self.declare_parameter('T1', 1.0)
        self.declare_parameter('T2', 1.5)

        image_topic = self.get_parameter('image_topic').value
        yolo_path = self.get_parameter('yolo_model').value
        conf_thr = self.get_parameter('conf_threshold').value

        self.detector = Method5FallDetector(yolo_path, conf_thr)

        # Appliquer les paramètres personnalisés
        self.detector.chi = self.get_parameter('chi').value
        self.detector.alpha = self.get_parameter('alpha').value
        self.detector.beta = self.get_parameter('beta').value
        self.detector.M = self.get_parameter('M').value
        self.detector.position_history = deque(maxlen=self.detector.M + 1)
        self.detector.epsilon1 = self.get_parameter('epsilon1').value
        self.detector.epsilon2 = self.get_parameter('epsilon2').value
        self.detector.T1 = self.get_parameter('T1').value
        self.detector.T2 = self.get_parameter('T2').value

        # Publishers / Subscriber
        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.alert_pub = self.create_publisher(Bool, '/method5/alert', 10)
        self.img_pub = self.create_publisher(Image, '/method5/output', 10)
        self.status_pub = self.create_publisher(String, '/method5/status', 10)

        self.get_logger().info(f"Method5 (HFDMIA-Pose) node démarré sur {image_topic}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Erreur conversion: {e}")
            return

        # Redimension pour performance (optionnel)
        frame = cv2.resize(frame, (980, 740))
        annotated, fall, info = self.detector.process_frame(frame)

        # Publication alerte
        alert_msg = Bool()
        alert_msg.data = fall
        self.alert_pub.publish(alert_msg)
        if fall:
            self.get_logger().warn("⚠️ CHUTE DÉTECTÉE (Méthode 5)")

        # Publication statut
        self.status_pub.publish(String(data=info))

        # Ajout d'une ligne d'info sur l'image
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if fall:
            cv2.putText(annotated, "FALL ALERT", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Publication image
        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = msg.header
        self.img_pub.publish(out_msg)

    def destroy_node(self):
        self.detector.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Method5FallNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()