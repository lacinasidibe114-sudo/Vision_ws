#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suivi de personne (RGB seule) avec détection de chute – version sans PID interne.
Le robot reste immobile jusqu'à détection d'une personne. Les commandes de vitesse
linéaire/angulaire sont calculées par des gains proportionnels simples, puis
converties en vitesses de roues et envoyées à l'Arduino via liaison série.
"""

import math
import serial
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp


# =====================================================================
#  Utilitaires
# =====================================================================
def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def clamp(value, vmin, vmax):
    return max(vmin, min(vmax, value))

# Indices des landmarks MediaPipe
LANDMARK_NAMES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
}

# =====================================================================
#  MediaPipe Pose
# =====================================================================
class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, image_rgb):
        results = self.pose.process(image_rgb)
        return results.pose_landmarks if results.pose_landmarks else None

    def close(self):
        self.pose.close()

def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=2):
    h, w = image.shape[:2]
    for lm in landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), radius, color, -1)

# =====================================================================
#  Détecteur de chute (Method 1)
# =====================================================================
class FallDetectorMethod1:
    def __init__(self, aspect_ratio_threshold=0.9, fall_confirm_frames=3):
        self.pose = PoseDetector()
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.fall_confirm_frames = fall_confirm_frames
        self.fall_counter = 0

    def reset_counter(self):
        self.fall_counter = 0

    def detect_activity(self, landmarks, h, w):
        lm = landmarks.landmark
        def get_xy(part):
            return (lm[LANDMARK_NAMES[part]].x * w,
                    lm[LANDMARK_NAMES[part]].y * h)

        l_sh, r_sh = get_xy('LEFT_SHOULDER'), get_xy('RIGHT_SHOULDER')
        l_hip, r_hip = get_xy('LEFT_HIP'), get_xy('RIGHT_HIP')
        l_knee, r_knee = get_xy('LEFT_KNEE'), get_xy('RIGHT_KNEE')

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

        shoulder_y = (lm[LANDMARK_NAMES['LEFT_SHOULDER']].y +
                      lm[LANDMARK_NAMES['RIGHT_SHOULDER']].y) / 2.0
        hip_y = (lm[LANDMARK_NAMES['LEFT_HIP']].y +
                 lm[LANDMARK_NAMES['RIGHT_HIP']].y) / 2.0
        if abs(shoulder_y - hip_y) < 0.05:
            return "Allonge"
        return "Debout"

    def check_aspect_ratio(self, bbox):
        x1, y1, x2, y2 = bbox
        h, w = (y2 - y1), (x2 - x1)
        if w < 1:
            return False, 0.0
        ar = h / float(w)
        return (ar < self.aspect_ratio_threshold), ar

    def analyze(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        is_fall_ar, aspect_ratio = self.check_aspect_ratio(bbox)

        activity = "Aucune"
        roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if roi.size > 0:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            landmarks = self.pose.process(roi_rgb)
            if landmarks:
                draw_landmarks(roi, landmarks)
                rh, rw = roi.shape[:2]
                activity = self.detect_activity(landmarks, rh, rw)

        if is_fall_ar or activity == "Allonge":
            self.fall_counter += 1
        else:
            self.fall_counter = max(0, self.fall_counter - 1)

        fall_confirmed = self.fall_counter >= self.fall_confirm_frames
        return {
            'fall': fall_confirmed,
            'aspect_ratio': aspect_ratio,
            'activity': activity,
        }

    def release(self):
        self.pose.close()


# =====================================================================
#  Nœud ROS 2 principal (sans PID)
# =====================================================================
class PersonFollowerFallNode(Node):

    def __init__(self):
        super().__init__('person_follower_fall_node')
        self.bridge = CvBridge()

        # ------------------------- Paramètres -------------------------
        p = self.declare_parameter
        p('image_topic', '/rgb')
        p('yolo_model', '/home/sidibe/yolo_project/yolo11n.pt')
        p('conf_threshold', 0.80)
        p('tracker_cfg', 'bytetrack.yaml')

        # Distance (mode bbox_height recommandé)
        p('distance_mode', 'bbox_height')
        p('target_bbox_height_ratio', 0.80)   # calibration à 1 m
        p('bbox_height_deadband', 0.04)       # tolérance (fraction de hauteur)
        p('calib_log', True)

        # Gains proportionnels (remplacent les PID)
        p('linear_gain', 0.8)      # m/s par erreur de distance (ou ratio)
        p('angular_gain', 1.2)     # rad/s par erreur angulaire normalisée
        p('max_linear_speed', 0.35)
        p('max_angular_speed', 1.0)

        # Paramètres roues (conversion (v, ω) -> consignes roues)
        p('wheel_base', 0.30)      # empattement (m)
        p('wheel_radius', 0.05)    # rayon roue (m)

        # Série vers Arduino
        p('serial_port', '/dev/ttyUSB0')
        p('serial_baud', 115200)

        # Suivi / sécurité
        p('lost_frames_threshold', 15)
        p('watchdog_timeout', 0.5)

        g = lambda n: self.get_parameter(n).value
        self.distance_mode = g('distance_mode')
        self.target_bbox_ratio = g('target_bbox_height_ratio')
        self.calib_log = g('calib_log')
        self.linear_gain = g('linear_gain')
        self.angular_gain = g('angular_gain')
        self.max_lin = g('max_linear_speed')
        self.max_ang = g('max_angular_speed')
        self.wheel_base = g('wheel_base')
        self.wheel_radius = g('wheel_radius')
        self.lost_threshold = g('lost_frames_threshold')
        self.watchdog_timeout = g('watchdog_timeout')
        self.conf_threshold = g('conf_threshold')
        self.tracker_cfg = g('tracker_cfg')

        # ------------------------- Modules ----------------------------
        self.yolo = YOLO(g('yolo_model'))
        self.fall_detector = FallDetectorMethod1()

        # Communication série avec l'Arduino
        try:
            self.ser = serial.Serial(g('serial_port'), g('serial_baud'), timeout=0.05)
            time.sleep(2)  # laisser le temps à l'Arduino de reset
            self.get_logger().info(f"Série ouverte sur {g('serial_port')}")
        except Exception as e:
            self.get_logger().error(f"Impossible d'ouvrir le port série : {e}")
            self.ser = None

        # ------------------------- État -------------------------------
        self.target_id = None
        self.lost_counter = 0
        self.last_target_cx = None
        self.calib_frame_count = 0
        self.last_frame_time = self.get_clock().now()
        self.last_control_time = self.get_clock().now()

        # ------------------------- ROS I/O ----------------------------
        self.sub_img = self.create_subscription(
            Image, g('image_topic'), self.image_callback, qos_profile_sensor_data)
        self.pub_img = self.create_publisher(Image, '~/image', 10)
        self.pub_alert = self.create_publisher(Bool, '~/fall_alert', 10)
        self.pub_status = self.create_publisher(String, '~/status', 10)

        # Watchdog
        self.watchdog = self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info(
            f"Nœud démarré (sans PID) – mode distance '{self.distance_mode}'")

    # ----------------------- Callback image ---------------------------
    def image_callback(self, msg):
        self.last_frame_time = self.get_clock().now()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Conversion image échouée : {e}")
            return

        h, w = frame.shape[:2]

        # Détection + suivi (personnes, classe 0)
        results = self.yolo.track(
            frame, persist=True, classes=[0],
            conf=self.conf_threshold, tracker=self.tracker_cfg, verbose=False)
        persons = self._extract_persons(results)

        # Sélection / maintien de la cible
        target = self._update_target(persons, img_cx=w / 2.0)

        if target is not None:
            self.lost_counter = 0
            tid, bbox, conf = target
            self.last_target_cx = (bbox[0] + bbox[2]) / 2.0

            fall_info = self.fall_detector.analyze(frame, bbox)

            # Calcul de l'erreur d'approche (linéaire)
            approach_err, signal_ok, dist_txt = self._compute_approach(bbox, frame.shape)

            # Calcul des commandes vitesse (proportionnel)
            linear, angular = self._compute_commands(
                frame_width=w, bbox=bbox,
                approach_error=approach_err,
                signal_ok=signal_ok,
                fall=fall_info['fall']
            )

            # Envoi des consignes à l'Arduino
            self._send_to_arduino(linear, angular)

            self._annotate_target(frame, bbox, conf, dist_txt, fall_info)
            self._publish_status(
                f"SUIVI id={tid} | act={fall_info['activity']} | {dist_txt}")
            self._publish_alert(fall_info['fall'])

            if fall_info['fall']:
                self.get_logger().warn("CHUTE DÉTECTÉE – arrêt du robot")
        else:
            self.lost_counter += 1
            self._handle_target_lost(frame)

        # Publication image annotée
        self._draw_others(frame, persons)
        out = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        out.header = msg.header
        self.pub_img.publish(out)

    # ----------------------- Calcul des commandes (sans PID) ----------
    def _compute_commands(self, frame_width, bbox, approach_error, signal_ok, fall):
        if fall:
            return 0.0, 0.0

        # Erreur angulaire normalisée [-1,1]
        x1, _, x2, _ = bbox
        cx = (x1 + x2) / 2.0
        err_angle = (cx - frame_width/2.0) / (frame_width/2.0)

        # Vitesse linéaire seulement si mesure fiable et pas trop désaxé
        if signal_ok and abs(err_angle) <= 0.5:
            linear = self.linear_gain * approach_error
        else:
            linear = 0.0

        # Vitesse angulaire proportionnelle à l'erreur
        angular = self.angular_gain * err_angle

        # Saturation
        linear = clamp(linear, -self.max_lin, self.max_lin)
        angular = clamp(angular, -self.max_ang, self.max_ang)

        return linear, angular

    def _compute_approach(self, bbox, frame_shape):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        bbox_h = y2 - y1

        if self.distance_mode == 'metric':
            # Non recommandé sans calibration précise
            self.get_logger().warn("Mode métrique non conseillé – utilisez 'bbox_height'")
            if bbox_h <= 1:
                return 0.0, False, "d=?"
            # Ces paramètres devraient être chargés depuis les paramètres
            focal = 600.0      # À remplacer par un paramètre
            person_h = 1.7
            distance = (person_h * focal) / bbox_h
            err = distance - 1.0
            return err, True, f"d={distance:.2f}m"

        # Mode 'bbox_height' (recommandé)
        ratio = bbox_h / float(h)

        if self.calib_log:
            self.calib_frame_count += 1
            if self.calib_frame_count % 30 == 0:
                self.get_logger().info(f"[CALIB] h_ratio = {ratio:.3f}")

        # Si la personne est coupée (haut ou bas), la mesure n'est pas fiable
        clipped_top = y1 <= 2
        clipped_bottom = y2 >= h - 2
        if clipped_top != clipped_bottom:
            return 0.0, False, f"h={ratio:.2f} (cadrage partiel)"

        err = self.target_bbox_ratio - ratio   # >0 => trop loin, <0 => trop près
        return err, True, f"h={ratio:.2f}/{self.target_bbox_ratio:.2f}"

    # ----------------------- Envoi série vers Arduino -----------------
    def _send_to_arduino(self, linear, angular):
        """Convertit (v, ω) en vitesses de roues et envoie la commande série."""
        if self.ser is None:
            return

        # Cinématique différentielle
        # v_gauche = (v - (L/2)*ω) / r
        # v_droite = (v + (L/2)*ω) / r
        L = self.wheel_base
        R = self.wheel_radius
        v_left  = (linear - (L/2.0) * angular) / R
        v_right = (linear + (L/2.0) * angular) / R

        # Limites arbitraires (à ajuster selon votre Arduino)
        max_wheel_speed = 2.0   # rad/s (exemple)
        v_left  = clamp(v_left,  -max_wheel_speed, max_wheel_speed)
        v_right = clamp(v_right, -max_wheel_speed, max_wheel_speed)

        # Format de commande attendu par l'Arduino :
        #   r<vitesse_droite>,l<vitesse_gauche>
        # Les vitesses doivent être en rad/s (signe indique direction)
        # Exemple : "r1.23,l-1.20"
        cmd = f"r{v_right:.2f},l{v_left:.2f}\n"
        try:
            self.ser.write(cmd.encode())
        except Exception as e:
            self.get_logger().error(f"Erreur envoi série : {e}")

    # ----------------------- Logique de cible -------------------------
    def _extract_persons(self, results):
        persons = []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return persons
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        for tid, bb, c in zip(ids, xyxy, confs):
            persons.append((int(tid), tuple(bb), float(c)))
        return persons

    def _update_target(self, persons, img_cx):
        if not persons:
            return None

        if self.target_id is not None:
            for pers in persons:
                if pers[0] == self.target_id:
                    return pers
            return None  # perdu cette frame

        def center_offset(pers):
            x1, _, x2, _ = pers[1]
            return abs((x1 + x2) / 2.0 - img_cx)

        best = min(persons, key=center_offset)
        self.target_id = best[0]
        self.fall_detector.reset_counter()
        self.get_logger().info(f"Cible verrouillée : ID {self.target_id}")
        return best

    def _handle_target_lost(self, frame):
        # Robot immobile en attente
        self._send_to_arduino(0.0, 0.0)

        if self.lost_counter < self.lost_threshold:
            self._publish_status(f"Cible perdue temporairement ({self.lost_counter})")
            return

        if self.target_id is not None:
            self.get_logger().warn(f"Cible ID {self.target_id} perdue – arrêt, en attente")
            self.target_id = None

        self._publish_status("En attente d'une personne... (robot immobile)")
        self._publish_alert(False)

    # ----------------------- Watchdog ---------------------------------
    def watchdog_callback(self):
        dt = (self.get_clock().now() - self.last_frame_time).nanoseconds * 1e-9
        if dt > self.watchdog_timeout:
            self._send_to_arduino(0.0, 0.0)

    # ----------------------- Publications -----------------------------
    def _publish_status(self, text):
        self.pub_status.publish(String(data=text))

    def _publish_alert(self, value):
        self.pub_alert.publish(Bool(data=bool(value)))

    # ----------------------- Annotations ------------------------------
    def _annotate_target(self, frame, bbox, conf, dist_txt, fall_info):
        x1, y1, x2, y2 = bbox
        color = (0, 0, 255) if fall_info['fall'] else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = (f"CIBLE id{self.target_id} {conf:.2f} | "
                 f"{fall_info['activity']} | AR:{fall_info['aspect_ratio']:.2f} | "
                 f"{dist_txt}")
        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if fall_info['fall']:
            cv2.putText(frame, "FALL ALERT", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    def _draw_others(self, frame, persons):
        for tid, bbox, _ in persons:
            if tid == self.target_id:
                continue
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
            cv2.putText(frame, f"id{tid}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # ----------------------- Arrêt propre -----------------------------
    def destroy_node(self):
        if self.ser:
            self.ser.close()
        self.fall_detector.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PersonFollowerFallNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()