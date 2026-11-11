#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suivi de personne + maintien de distance + détection de chute (ROS 2)
=====================================================================
CONFIGURATION : caméra RGB SEULE, commande sur /cmd_vel.

Ce nœud combine trois comportements dans une seule boucle de perception :

  1. SUIVI (tracking) avec verrouillage d'identité
     YOLO + ByteTrack (intégré à Ultralytics) attribue un ID stable à chaque
     personne. On verrouille UNE cible et on la garde image après image,
     même si d'autres personnes entrent dans le champ -> on ne perd pas la
     personne suivie.

  2. MAINTIEN DE DISTANCE (consigne ~ 1 m) — sans capteur de profondeur
     Deux modes possibles (paramètre 'distance_mode') :

       'bbox_height' (RECOMMANDÉ en RGB seule, mode par défaut)
           On régule la TAILLE APPARENTE de la personne : on maintient la
           hauteur de sa boîte englobante à une valeur cible (fraction de la
           hauteur de l'image). Calibration une seule fois : se placer à 1 m,
           lire la valeur affichée, la mettre dans 'target_bbox_height_ratio'.
           Avantage : aucune calibration de focale, robuste, stable.

       'metric'
           Distance estimée en mètres :  d = hauteur_réelle * focale / h_bbox.
           Nécessite une focale calibrée et que la personne tienne en entier
           dans le cadre — peu fiable à courte distance. À éviter si possible.

     Un PID génère la vitesse linéaire (avancer/reculer) et la vitesse
     angulaire (garder la personne centrée).

  3. DÉTECTION DE CHUTE (Méthode 1)
     YOLO + MediaPipe + aspect ratio, d'après Kothari & Chakurkar,
     MethodsX 2025. Appliquée UNIQUEMENT sur la cible suivie.

Sécurités intégrées :
  - Chute détectée            -> arrêt immédiat du robot + alerte.
  - Cadrage non fiable         -> on gèle l'avance (mais on continue à viser).
  - Cible perdue              -> arrêt puis rotation lente de recherche.
  - Plus d'image (watchdog)   -> arrêt du robot.

Topics
------
  Entrées :
    image_topic        (sensor_msgs/Image)       image RGB
  Sorties :
    cmd_vel_topic      (geometry_msgs/Twist)     commande de vitesse robot
    ~/image            (sensor_msgs/Image)       image annotée (debug)
    ~/fall_alert       (std_msgs/Bool)           True si chute confirmée
    ~/status           (std_msgs/String)         état courant (texte)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
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
    """Distance euclidienne 2D entre deux points (x, y)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def clamp(value, vmin, vmax):
    return max(vmin, min(vmax, value))


# Indices des landmarks MediaPipe utilisés pour la classification d'activité
LANDMARK_NAMES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28,
}


# =====================================================================
#  Régulateur PID (asservissement)
# =====================================================================
class PIDController:
    """PID simple avec zone morte, anti-windup et saturation de sortie."""

    def __init__(self, kp, ki, kd, out_min, out_max, deadband=0.0,
                 integral_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.deadband = deadband
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, error, dt):
        # Zone morte : pas de correction tant que l'erreur est petite
        if abs(error) < self.deadband:
            self._integral = 0.0
            self._prev_error = 0.0
            return 0.0

        if dt <= 0.0:
            dt = 1e-3

        self._integral += error * dt
        self._integral = clamp(self._integral,
                               -self.integral_limit, self.integral_limit)
        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return clamp(output, self.out_min, self.out_max)


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
#  Détecteur de chute (Méthode 1) — logique pure, sans inférence YOLO
# =====================================================================
class FallDetectorMethod1:
    """
    Reçoit une bbox + la ROI image et décide s'il y a chute.
    L'inférence YOLO est faite en amont (par le tracker), pour n'avoir
    qu'UN seul passage YOLO par image.
    """

    def __init__(self, aspect_ratio_threshold=0.9, fall_confirm_frames=3):
        self.pose = PoseDetector()
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.fall_confirm_frames = fall_confirm_frames
        self.fall_counter = 0

    def reset_counter(self):
        self.fall_counter = 0

    def detect_activity(self, landmarks, h, w):
        """Classification d'activité (Assis / Marche / Allongé / Debout)."""
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
        """
        Renvoie un dict : fall (bool confirmé), aspect_ratio, activity.
        Met aussi à jour le compteur de confirmation temporelle.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        is_fall_ar, aspect_ratio = self.check_aspect_ratio(bbox)

        # Pose sur la ROI de la cible
        activity = "Aucune"
        roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if roi.size > 0:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            landmarks = self.pose.process(roi_rgb)
            if landmarks:
                draw_landmarks(roi, landmarks)
                rh, rw = roi.shape[:2]
                activity = self.detect_activity(landmarks, rh, rw)

        # Confirmation temporelle (anti faux-positif)
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
#  Nœud ROS 2
# =====================================================================
class PersonFollowerFallNode(Node):

    def __init__(self):
        super().__init__('person_follower_fall_node')
        self.bridge = CvBridge()

        # ------------------------- Paramètres -------------------------
        p = self.declare_parameter
        # Topics
        p('image_topic', '/rgb')
        p('cmd_vel_topic', '/diff_drive_controller/cmd_vel_unstamped')
        # Modèle / détection
        p('yolo_model', '/home/sidibe/yolo_project/yolo11n.pt')
        p('conf_threshold', 0.80)
        p('tracker_cfg', 'bytetrack.yaml')       # ou 'botsort.yaml'
        # Distance (RGB seule)
        p('distance_mode', 'bbox_height')         # 'bbox_height' (reco) | 'metric'
        p('target_bbox_height_ratio', 0.80)       # mode bbox_height : À CALIBRER à 1 m
        p('bbox_height_deadband', 0.04)           # tolérance (fraction de hauteur)
        p('target_distance', 1.0)                 # mode metric : consigne (m)
        p('distance_deadband', 0.10)              # mode metric : tolérance (m)
        p('focal_length_px', 600.0)               # mode metric : À CALIBRER
        p('person_height_m', 1.7)                 # mode metric
        p('calib_log', True)                      # logge la valeur pour calibrer
        # Asservissement
        p('max_linear_speed', 0.35)               # m/s
        p('max_angular_speed', 1.0)               # rad/s
        p('angle_deadband', 0.05)                 # erreur normalisée [-1,1]
        p('kp_lin', 0.8); p('ki_lin', 0.0); p('kd_lin', 0.05)
        p('kp_ang', 1.2); p('ki_ang', 0.0); p('kd_ang', 0.10)
        # Suivi / sécurité
        p('lost_frames_threshold', 15)            # frames avant "cible perdue"
        p('search_angular_speed', 0.4)            # rad/s rotation de recherche
        p('watchdog_timeout', 0.5)                # s sans image -> arrêt

        g = lambda n: self.get_parameter(n).value
        self.distance_mode = g('distance_mode')
        self.target_bbox_ratio = g('target_bbox_height_ratio')
        self.target_distance = g('target_distance')
        self.focal_length_px = g('focal_length_px')
        self.person_height_m = g('person_height_m')
        self.calib_log = g('calib_log')
        self.max_lin = g('max_linear_speed')
        self.max_ang = g('max_angular_speed')
        self.lost_threshold = g('lost_frames_threshold')
        self.search_ang = g('search_angular_speed')
        self.watchdog_timeout = g('watchdog_timeout')
        self.conf_threshold = g('conf_threshold')
        self.tracker_cfg = g('tracker_cfg')

        # ------------------------- Modules ----------------------------
        self.yolo = YOLO(g('yolo_model'))
        self.fall_detector = FallDetectorMethod1()

        # Zone morte du PID linéaire selon le mode (unités différentes)
        lin_deadband = (g('bbox_height_deadband')
                        if self.distance_mode == 'bbox_height'
                        else g('distance_deadband'))
        self.pid_lin = PIDController(
            g('kp_lin'), g('ki_lin'), g('kd_lin'),
            -self.max_lin, self.max_lin, deadband=lin_deadband)
        self.pid_ang = PIDController(
            g('kp_ang'), g('ki_ang'), g('kd_ang'),
            -self.max_ang, self.max_ang, deadband=g('angle_deadband'))

        # ------------------------- État -------------------------------
        self.target_id = None
        self.lost_counter = 0
        self.last_target_cx = None      # pour orienter la recherche
        self.calib_frame_count = 0
        self.last_frame_time = self.get_clock().now()
        self.last_control_time = self.get_clock().now()

        # ------------------------- ROS I/O ----------------------------
        self.sub_img = self.create_subscription(
            Image, g('image_topic'), self.image_callback, qos_profile_sensor_data)
        self.pub_cmd = self.create_publisher(Twist, g('cmd_vel_topic'), 10)
        self.pub_img = self.create_publisher(Image, '~/image', 10)
        self.pub_alert = self.create_publisher(Bool, '~/fall_alert', 10)
        self.pub_status = self.create_publisher(String, '~/status', 10)

        # Watchdog : arrête le robot si plus d'image
        self.watchdog = self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info(
            f"Person-Follower + Fall-Detection démarré "
            f"(RGB seule, mode='{self.distance_mode}')")
        if self.distance_mode == 'bbox_height':
            self.get_logger().info(
                f"CALIBRATION : place-toi à 1 m, lis 'h_ratio' affiché, "
                f"puis règle target_bbox_height_ratio "
                f"(actuel = {self.target_bbox_ratio:.2f}).")

    # ----------------------- Callback image ---------------------------
    def image_callback(self, msg):
        self.last_frame_time = self.get_clock().now()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        h, w = frame.shape[:2]

        # 1) Détection + suivi (personnes uniquement, classe 0)
        results = self.yolo.track(
            frame, persist=True, classes=[0],
            conf=self.conf_threshold, tracker=self.tracker_cfg, verbose=False)
        persons = self._extract_persons(results)

        # 2) Sélection / maintien de la cible verrouillée
        target = self._update_target(persons, img_cx=w / 2.0)

        # 3) Traitement de la cible (chute + distance + asservissement)
        if target is not None:
            self.lost_counter = 0
            tid, bbox, conf = target
            x1, y1, x2, y2 = bbox
            self.last_target_cx = (x1 + x2) / 2.0

            fall_info = self.fall_detector.analyze(frame, bbox)

            # Erreur d'approche : > 0 => avancer, < 0 => reculer
            approach_err, signal_ok, dist_txt = self._compute_approach(
                bbox, frame.shape)

            self._control(frame_width=w, bbox=bbox,
                          approach_error=approach_err,
                          signal_ok=signal_ok,
                          fall=fall_info['fall'])

            self._annotate_target(frame, bbox, conf, dist_txt, fall_info)
            self._publish_status(
                f"SUIVI id={tid} | act={fall_info['activity']} | {dist_txt}")
            self._publish_alert(fall_info['fall'])

            if fall_info['fall']:
                self.get_logger().warn("CHUTE DÉTECTÉE — robot arrêté")
        else:
            self.lost_counter += 1
            self._handle_target_lost(frame)

        # 4) Annotations globales + publication image debug
        self._draw_others(frame, persons)
        out = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        out.header = msg.header
        self.pub_img.publish(out)

    # ----------------------- Estimation d'approche --------------------
    def _compute_approach(self, bbox, frame_shape):
        """
        Renvoie (approach_error, signal_ok, texte_affichage).
        approach_error > 0  => trop loin  => avancer
        approach_error < 0  => trop près  => reculer
        signal_ok = False   => mesure non fiable => on gèle l'avance.
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        bbox_h = y2 - y1

        if self.distance_mode == 'metric':
            if bbox_h <= 1:
                return 0.0, False, "d=?"
            distance = (self.person_height_m * self.focal_length_px) / float(bbox_h)
            err = distance - self.target_distance
            return err, True, f"d={distance:.2f}m"

        # ---- mode 'bbox_height' (recommandé RGB) ----
        ratio = bbox_h / float(h)

        # Log périodique pour faciliter la calibration à 1 m
        if self.calib_log:
            self.calib_frame_count += 1
            if self.calib_frame_count % 30 == 0:
                self.get_logger().info(f"[CALIB] h_ratio = {ratio:.3f}")

        clipped_top = y1 <= 2
        clipped_bottom = y2 >= h - 2

        # Un seul bord coupé => hauteur peu fiable => on ne fait pas avancer
        if clipped_top != clipped_bottom:
            return 0.0, False, f"h={ratio:.2f} (cadrage partiel)"

        # ratio cible - ratio courant :
        #   personne plus petite (loin) => err > 0 => avancer
        #   personne plus grande (proche) => err < 0 => reculer
        err = self.target_bbox_ratio - ratio
        return err, True, f"h={ratio:.2f}/{self.target_bbox_ratio:.2f}"

    # ----------------------- Logique de cible -------------------------
    def _extract_persons(self, results):
        """Liste de (track_id, bbox(x1,y1,x2,y2), conf) avec ID valide."""
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
        """Garde l'ID verrouillé ; sinon acquiert la personne la plus centrée."""
        if not persons:
            return None

        if self.target_id is not None:
            for pers in persons:
                if pers[0] == self.target_id:
                    return pers
            return None  # ID absent cette frame -> comptera comme "perdu"

        def center_offset(pers):
            x1, _, x2, _ = pers[1]
            return abs((x1 + x2) / 2.0 - img_cx)

        best = min(persons, key=center_offset)
        self.target_id = best[0]
        self.fall_detector.reset_counter()
        self.pid_lin.reset()
        self.pid_ang.reset()
        self.get_logger().info(f"Cible verrouillée : ID {self.target_id}")
        return best

    def _handle_target_lost(self, frame):
        """Cible non vue : on attend un peu, puis on tourne pour la rechercher."""
        if self.lost_counter < self.lost_threshold:
            self._publish_twist(0.0, 0.0)
            self._publish_status(f"Cible momentanément perdue ({self.lost_counter})")
            return

        if self.target_id is not None:
            self.get_logger().warn(f"Cible ID {self.target_id} perdue — recherche")
            self.target_id = None

        direction = 1.0
        if self.last_target_cx is not None:
            w = frame.shape[1]
            direction = -1.0 if self.last_target_cx > w / 2.0 else 1.0
        self._publish_twist(0.0, direction * self.search_ang)
        self._publish_status("Recherche de la cible…")
        self._publish_alert(False)

    # ----------------------- Asservissement ---------------------------
    def _control(self, frame_width, bbox, approach_error, signal_ok, fall):
        now = self.get_clock().now()
        dt = (now - self.last_control_time).nanoseconds * 1e-9
        self.last_control_time = now
        dt = clamp(dt, 1e-3, 0.5)

        # SÉCURITÉ : chute -> arrêt complet
        if fall:
            self.pid_lin.reset()
            self.pid_ang.reset()
            self._publish_twist(0.0, 0.0)
            return

        # Angulaire : garder la personne centrée (erreur normalisée [-1, 1])
        x1, _, x2, _ = bbox
        cx = (x1 + x2) / 2.0
        err_x = (cx - frame_width / 2.0) / (frame_width / 2.0)
        # personne à droite (err_x>0) -> tourner à droite (angular.z<0)
        angular = -self.pid_ang.update(err_x, dt)

        # Linéaire : seulement si la mesure est fiable ET grossièrement aligné
        if signal_ok and abs(err_x) <= 0.5:
            linear = self.pid_lin.update(approach_error, dt)
        else:
            linear = 0.0
            self.pid_lin.reset()

        self._publish_twist(linear, angular)

    def _publish_twist(self, linear, angular):
        cmd = Twist()
        cmd.linear.x = float(clamp(linear, -self.max_lin, self.max_lin))
        cmd.angular.z = float(clamp(angular, -self.max_ang, self.max_ang))
        self.pub_cmd.publish(cmd)

    # ----------------------- Watchdog ---------------------------------
    def watchdog_callback(self):
        dt = (self.get_clock().now() - self.last_frame_time).nanoseconds * 1e-9
        if dt > self.watchdog_timeout:
            self._publish_twist(0.0, 0.0)

    # ----------------------- Publication ------------------------------
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
        """Trace en gris les autres personnes (non suivies)."""
        for tid, bbox, _ in persons:
            if tid == self.target_id:
                continue
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
            cv2.putText(frame, f"id{tid}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # ----------------------- Arrêt propre -----------------------------
    def destroy_node(self):
        self._publish_twist(0.0, 0.0)   # robot à l'arrêt en quittant
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