#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from message_filters import Subscriber, ApproximateTimeSynchronizer
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
#  CONTRÔLEUR FLOU (inchangé - logique de décision)
# ============================================================================
class FuzzyPersonFollower:
    def __init__(self, safety_distance=0.5):
        self.safety_distance = safety_distance
        self._build_universes()
        self._build_membership_functions()
        self._build_rules()
        self.controller = ctrl.ControlSystem(self.rules)
        self.sim = ctrl.ControlSystemSimulation(self.controller)

    def _build_universes(self):
        self.d_rel     = ctrl.Antecedent(np.linspace(0.5, 5.0, 500), 'd_rel')
        self.theta_rel = ctrl.Antecedent(np.linspace(-np.pi/3, np.pi/3, 500), 'theta_rel')
        self.delta_d   = ctrl.Antecedent(np.linspace(-0.4, 0.4, 500), 'delta_d')
        self.v_cmd     = ctrl.Consequent(np.linspace(0.0, 1.0, 500), 'v_cmd')
        self.omega_cmd = ctrl.Consequent(np.linspace(-1.2, 1.2, 500), 'omega_cmd')

    def _build_membership_functions(self):
        d, t, dd = self.d_rel, self.theta_rel, self.delta_d
        v, w = self.v_cmd, self.omega_cmd

        d['Tres_Proche'] = fuzz.trapmf(d.universe, [0.5, 0.5, 0.8, 1.2])
        d['Proche']      = fuzz.trimf(d.universe,  [0.8, 1.2, 2.0])
        d['Moyenne']     = fuzz.trimf(d.universe,  [1.5, 2.5, 3.5])
        d['Eloignee']    = fuzz.trapmf(d.universe, [3.0, 3.5, 5.0, 5.0])

        t['Gauche'] = fuzz.trimf(t.universe, [-np.pi/3, -np.pi/3, 0])
        t['Centre'] = fuzz.trimf(t.universe, [-np.pi/6, 0, np.pi/6])
        t['Droite'] = fuzz.trimf(t.universe, [0, np.pi/3, np.pi/3])

        dd['S_approche'] = fuzz.trimf(dd.universe, [-0.4, -0.4, 0])
        dd['Stable']     = fuzz.trimf(dd.universe, [-0.15, 0, 0.15])
        dd['S_eloigne']  = fuzz.trimf(dd.universe, [0, 0.4, 0.4])

        v['Stop']   = fuzz.trapmf(v.universe, [0, 0, 0.05, 0.1])
        v['Lent']   = fuzz.trimf(v.universe,  [0.05, 0.15, 0.35])
        v['Moyen']  = fuzz.trimf(v.universe,  [0.25, 0.45, 0.65])
        v['Rapide'] = fuzz.trapmf(v.universe, [0.55, 0.8, 1.0, 1.0])

        w['T_G']    = fuzz.trapmf(w.universe, [-1.2, -1.2, -0.6, -0.3])
        w['L_G']    = fuzz.trimf(w.universe,  [-0.5, -0.25, 0])
        w['Neutre'] = fuzz.trimf(w.universe,  [-0.2, 0, 0.2])
        w['L_D']    = fuzz.trimf(w.universe,  [0, 0.25, 0.5])
        w['T_D']    = fuzz.trapmf(w.universe, [0.3, 0.6, 1.2, 1.2])

    def _build_rules(self):
        d, t, dd = self.d_rel, self.theta_rel, self.delta_d
        v, w = self.v_cmd, self.omega_cmd
        R = []
        R.append(ctrl.Rule(d['Tres_Proche'] & t['Gauche'], (v['Stop'], w['T_D'])))
        R.append(ctrl.Rule(d['Tres_Proche'] & t['Centre'], (v['Stop'], w['Neutre'])))
        R.append(ctrl.Rule(d['Tres_Proche'] & t['Droite'], (v['Stop'], w['T_G'])))
        R.append(ctrl.Rule(d['Proche'] & t['Centre'] & dd['S_approche'], (v['Lent'], w['Neutre'])))
        R.append(ctrl.Rule(d['Proche'] & t['Centre'] & (dd['Stable'] | dd['S_eloigne']), (v['Lent'], w['Neutre'])))
        R.append(ctrl.Rule(d['Proche'] & t['Gauche'], (v['Lent'], w['L_D'])))
        R.append(ctrl.Rule(d['Proche'] & t['Droite'], (v['Lent'], w['L_G'])))
        R.append(ctrl.Rule(d['Moyenne'] & t['Centre'] & dd['S_approche'], (v['Moyen'], w['Neutre'])))
        R.append(ctrl.Rule(d['Moyenne'] & t['Centre'] & dd['Stable'], (v['Moyen'], w['Neutre'])))
        R.append(ctrl.Rule(d['Moyenne'] & t['Centre'] & dd['S_eloigne'], (v['Moyen'], w['Neutre'])))
        R.append(ctrl.Rule(d['Moyenne'] & t['Gauche'], (v['Moyen'], w['L_G'])))
        R.append(ctrl.Rule(d['Moyenne'] & t['Droite'], (v['Moyen'], w['L_D'])))
        R.append(ctrl.Rule(d['Eloignee'] & dd['S_eloigne'] & t['Gauche'], (v['Rapide'], w['L_G'])))
        R.append(ctrl.Rule(d['Eloignee'] & dd['S_eloigne'] & t['Centre'], (v['Rapide'], w['Neutre'])))
        R.append(ctrl.Rule(d['Eloignee'] & dd['S_eloigne'] & t['Droite'], (v['Rapide'], w['L_D'])))
        R.append(ctrl.Rule(d['Eloignee'] & dd['Stable'] & t['Gauche'], (v['Moyen'], w['L_G'])))
        R.append(ctrl.Rule(d['Eloignee'] & dd['Stable'] & t['Centre'], (v['Moyen'], w['Neutre'])))
        R.append(ctrl.Rule(d['Eloignee'] & dd['Stable'] & t['Droite'], (v['Moyen'], w['L_D'])))
        R.append(ctrl.Rule(d['Eloignee'] & dd['S_approche'], (v['Lent'], w['Neutre'])))
        R.append(ctrl.Rule(t['Centre'], (w['Neutre'],)))
        R.append(ctrl.Rule(~d['Tres_Proche'] & ~t['Centre'], (v['Lent'], w['Neutre'])))
        self.rules = R

    def compute(self, d_val, theta_val, dd_val, d_brut=None):
        d_val     = np.clip(d_val,     0.5, 5.0)
        theta_val = np.clip(theta_val, -np.pi/3, np.pi/3)
        dd_val    = np.clip(dd_val,    -0.4, 0.4)

        self.sim.input['d_rel']     = d_val
        self.sim.input['theta_rel'] = theta_val
        self.sim.input['delta_d']   = dd_val
        try:
            self.sim.compute()
            v = float(np.clip(self.sim.output['v_cmd'],     0.0, 1.0))
            w = float(np.clip(self.sim.output['omega_cmd'], -1.2, 1.2))
        except Exception:
            v, w = 0.0, 0.0
        if d_brut is not None and d_brut < self.safety_distance:
            v, w = 0.0, 0.0
        return v, w

    def membership_degrees(self, d_val, theta_val, dd_val):
        deg = {}
        deg['d_rel'] = {
            name: float(fuzz.interp_membership(self.d_rel.universe,
                                               self.d_rel[name].mf, d_val))
            for name in self.d_rel.terms
        }
        deg['theta_rel'] = {
            name: float(fuzz.interp_membership(self.theta_rel.universe,
                                               self.theta_rel[name].mf, theta_val))
            for name in self.theta_rel.terms
        }
        deg['delta_d'] = {
            name: float(fuzz.interp_membership(self.delta_d.universe,
                                               self.delta_d[name].mf, dd_val))
            for name in self.delta_d.terms
        }
        return deg


# ============================================================================
#  NŒUD ROS2 HYBRIDE : RGB + DEPTH (Isaac Sim)
# ============================================================================
class HybridFuzzyFollowerNode(Node):
    def __init__(self):
        super().__init__('hybrid_fuzzy_follower_node')

        # ===== PARAMÈTRES ROS2 =====
        self.declare_parameter('rgb_topic', '/rgb')
        self.declare_parameter('depth_topic', '/depth')
        self.declare_parameter('focal_length_px', 600.0)      # Pour estimation monoculaire
        self.declare_parameter('real_person_height', 1.77)     # Hauteur moyenne personne
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('ema_alpha', 0.25)
        self.declare_parameter('tracker_config', 'bytetrack.yaml')
        self.declare_parameter('yolo_model', '/home/sidibe/yolo_project/yolo11n.pt')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('img_width', 640)
        self.declare_parameter('img_height', 480)
        self.declare_parameter('use_depth_primary', True)      # Switch entre méthodes
        self.declare_parameter('depth_fallback_mono', True)    # Fallback automatique

        # ===== RÉCUPÉRATION PARAMÈTRES =====
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.focal_length = self.get_parameter('focal_length_px').value
        self.real_height = self.get_parameter('real_person_height').value
        self.safety_dist = self.get_parameter('safety_distance').value
        self.ema_alpha = self.get_parameter('ema_alpha').value
        self.tracker_cfg = self.get_parameter('tracker_config').value
        self.yolo_path = self.get_parameter('yolo_model').value
        self.conf_thr = self.get_parameter('conf_threshold').value
        self.img_w = self.get_parameter('img_width').value
        self.img_h = self.get_parameter('img_height').value
        self.use_depth_primary = self.get_parameter('use_depth_primary').value
        self.depth_fallback_mono = self.get_parameter('depth_fallback_mono').value

        # ===== UTILITAIRES =====
        self.bridge = CvBridge()

        # ===== CHARGEMENT YOLO =====
        self.get_logger().info(f"🤖 Chargement YOLO depuis {self.yolo_path}")
        self.model = YOLO(self.yolo_path)

        # ===== CONTRÔLEUR FLOU =====
        self.fuzzy = FuzzyPersonFollower(safety_distance=self.safety_dist)

        # ===== ÉTAT INTERNE =====
        self.d_filt = 2.0
        self.theta_filt = 0.0
        self.prev_d_filt = 2.0
        self.prev_time = time.time()
        self.tracked_id = None

        # ===== STATISTIQUES COMPARAISON =====
        self.depth_valid_count = 0
        self.mono_valid_count = 0
        self.error_sum = 0.0
        self.compare_count = 0

        # ===== SOUSCRIPTIONS SYNCHRONISÉES (Isaac Sim) =====
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.image_depth_callback)

        # ===== PUBLISHERS =====
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.img_pub = self.create_publisher(Image, '/yolo/hybrid_output', 10)

        self.get_logger().info(f"✅ Nœud hybride démarré | RGB:{self.rgb_topic} + Depth:{self.depth_topic}")
        self.get_logger().info(f"📊 Mode: {'Depth primaire' if self.use_depth_primary else 'Monoculaire primaire'} | Fallback: {'ON' if self.depth_fallback_mono else 'OFF'}")

    # ------------------------------------------------------------------------
    #  ESTIMATION MONOCULAIRE (méthode originale Code 2)
    # ------------------------------------------------------------------------
    def bbox_to_distance_mono(self, bbox_height_px):
        """Estimation de distance par géométrie projective"""
        if bbox_height_px <= 0:
            return 5.0
        return (self.focal_length * self.real_height) / bbox_height_px

    def bbox_to_angle(self, center_x):
        """Conversion pixel → angle horizontal"""
        dx = center_x - self.img_w / 2.0
        return np.arctan2(dx, self.focal_length)

    # ------------------------------------------------------------------------
    #  MESURE DE PROFONDEUR (méthode Code 1 - Isaac Sim)
    # ------------------------------------------------------------------------
    def get_depth_measurement(self, depth_img, x1, y1, x2, y2, cx, cy):
        """
        Extrait la distance depuis l'image de profondeur Isaac Sim.
        Retourne: (distance, is_valid, method_used)
        """
        # 1. Lecture directe au centre du bbox
        if 0 <= cy < depth_img.shape[0] and 0 <= cx < depth_img.shape[1]:
            d_center = depth_img[cy, cx]
            if not np.isnan(d_center) and d_center > 0.3 and d_center < 10.0:
                return d_center, True, "center"

        # 2. Fallback : moyenne sur la ROI du bbox
        roi = depth_img[y1:y2, x1:x2]
        valid_pixels = roi[(roi > 0.3) & (roi < 10.0)]
        if len(valid_pixels) > 10:  # Au moins 10 pixels valides
            return np.median(valid_pixels), True, "roi_median"

        # 3. Échec : pas de mesure depth fiable
        return None, False, "invalid"

    # ------------------------------------------------------------------------
    #  CALLBACK PRINCIPALE SYNCHRONISÉE
    # ------------------------------------------------------------------------
    def image_depth_callback(self, rgb_msg, depth_msg):
        # ===== CONVERSIONS CV_BRIDGE =====
        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')  # Isaac Sim: float32 en mètres
        except Exception as e:
            self.get_logger().error(f"❌ Erreur conversion: {e}")
            return

        # Vérification cohérence dimensions
        if frame.shape[:2] != depth.shape[:2]:
            self.get_logger().warn(f"⚠️ Tailles RGB{frame.shape[:2]} != Depth{depth.shape[:2]}")
            depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

        frame = cv2.resize(frame, (self.img_w, self.img_h))
        depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

        # ===== DÉTECTION + TRACKING YOLO =====
        try:
            results = self.model.track(frame, persist=True, tracker=self.tracker_cfg,
                                       classes=[0], conf=self.conf_thr, verbose=False)
        except Exception:
            results = self.model.track(frame, persist=True, classes=[0],
                                       conf=self.conf_thr, verbose=False)

        # Initialisation variables de sortie
        v_cmd, w_cmd = 0.0, 0.0
        d_raw, theta_raw = 5.0, 0.0
        delta_d = 0.0
        annotated = frame.copy()
        d_depth, d_mono = None, None
        depth_valid, mono_valid = False, False

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # Gestion des IDs de tracking
            if hasattr(boxes, 'id') and boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = None

            # Sélection de la cible : ID suivi ou plus grande aire
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            idx = None
            if self.tracked_id is not None and track_ids is not None:
                mask = (track_ids == self.tracked_id)
                if np.any(mask):
                    idx = np.where(mask)[0][0]
                else:
                    idx = int(torch.argmax(areas))
                    self.tracked_id = track_ids[idx]
            else:
                idx = int(torch.argmax(areas))
                if track_ids is not None:
                    self.tracked_id = track_ids[idx]

            if idx is not None:
                # Extraction bbox
                x1, y1, x2, y2 = map(int, boxes.xyxy[idx].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                bbox_h = y2 - y1

                # Affichage bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if self.tracked_id is not None:
                    cv2.putText(annotated, f"ID:{self.tracked_id}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # ===== COMPARAISON DES DEUX MÉTHODES =====
                # 1. Mesure Depth (Isaac Sim)
                d_depth, depth_valid, depth_method = self.get_depth_measurement(depth, x1, y1, x2, y2, cx, cy)
                if depth_valid:
                    self.depth_valid_count += 1

                # 2. Estimation Monoculaire
                d_mono = self.bbox_to_distance_mono(bbox_h)
                mono_valid = (0.3 < d_mono < 10.0)
                if mono_valid:
                    self.mono_valid_count += 1

                # 3. Choix de la distance finale selon configuration
                if self.use_depth_primary:
                    if depth_valid:
                        d_raw = d_depth
                        method_label = "DEPTH"
                        color_dist = (0, 255, 0)  # Vert
                    elif self.depth_fallback_mono and mono_valid:
                        d_raw = d_mono
                        method_label = "MONO(fallback)"
                        color_dist = (0, 165, 255)  # Orange
                    else:
                        d_raw = 5.0
                        method_label = "NONE"
                        color_dist = (0, 0, 255)  # Rouge
                else:
                    if mono_valid:
                        d_raw = d_mono
                        method_label = "MONO"
                        color_dist = (255, 255, 0)  # Jaune
                    elif self.depth_fallback_mono and depth_valid:
                        d_raw = d_depth
                        method_label = "DEPTH(fallback)"
                        color_dist = (0, 165, 255)
                    else:
                        d_raw = 5.0
                        method_label = "NONE"
                        color_dist = (0, 0, 255)

                # Statistiques de comparaison (quand les deux sont valides)
                if depth_valid and mono_valid:
                    error = abs(d_depth - d_mono)
                    self.error_sum += error
                    self.compare_count += 1
                    if self.compare_count % 50 == 0:
                        avg_err = self.error_sum / self.compare_count
                        self.get_logger().info(f"📊 Erreur moyenne Depth vs Mono: {avg_err:.3f}m")

                # ===== CALCUL DE L'ANGLE =====
                theta_raw = self.bbox_to_angle(cx)

                # ===== FILTRAGE TEMPOREL (EMA) =====
                self.d_filt = self.ema_alpha * d_raw + (1 - self.ema_alpha) * self.d_filt
                self.theta_filt = self.ema_alpha * theta_raw + (1 - self.ema_alpha) * self.theta_filt

                now = time.time()
                dt = max(now - self.prev_time, 1e-6)
                delta_d = (self.d_filt - self.prev_d_filt) / dt
                delta_d = np.clip(delta_d, -0.4, 0.4)
                self.prev_time, self.prev_d_filt = now, self.d_filt

                # ===== CONTRÔLE FLOU =====
                v_cmd, w_cmd = self.fuzzy.compute(self.d_filt, self.theta_filt, delta_d, d_brut=d_raw)

                # ===== AFFICHAGE COMPARATIF SUR L'IMAGE =====
                y_offset = 30
                # Distance Depth
                depth_str = f"{d_depth:.2f}m [{depth_method}]" if depth_valid else "N/A"
                cv2.putText(annotated, f"Depth: {depth_str}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if depth_valid else (100, 100, 100), 2)
                # Distance Monoculaire
                mono_str = f"{d_mono:.2f}m" if mono_valid else "N/A"
                cv2.putText(annotated, f"Mono : {mono_str}", (10, y_offset + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0) if mono_valid else (100, 100, 100), 2)
                # Distance utilisée + méthode
                cv2.putText(annotated, f"UTIL: {d_raw:.2f}m [{method_label}]", (10, y_offset + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_dist, 2)
                # Écart entre méthodes (si les deux valides)
                if depth_valid and mono_valid:
                    diff = d_depth - d_mono
                    color_diff = (0, 255, 0) if abs(diff) < 0.2 else (0, 165, 255) if abs(diff) < 0.5 else (0, 0, 255)
                    cv2.putText(annotated, f"Δ: {diff:+.2f}m", (10, y_offset + 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_diff, 2)

                # Angle et métriques de contrôle
                cv2.putText(annotated, f"Angle: {np.degrees(theta_raw):.1f}°", (10, y_offset + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(annotated, f"Δd: {delta_d:+.2f} m/s", (10, y_offset + 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Commandes finales
                cmd_text = f"v:{v_cmd:.2f} | w:{w_cmd:.2f}"
                cv2.putText(annotated, cmd_text, (10, y_offset + 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Degrés d'activation floue (optionnel - debug)
                if self.get_logger().get_effective_level() <= 10:  # DEBUG
                    deg = self.fuzzy.membership_degrees(self.d_filt, self.theta_filt, delta_d)
                    debug_y = y_offset + 180
                    cv2.putText(annotated, f"[DEBUG] d_rel: P={deg['d_rel']['Proche']:.2f} M={deg['d_rel']['Moyenne']:.2f}",
                                (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
                    cv2.putText(annotated, f"[DEBUG] theta: C={deg['theta_rel']['Centre']:.2f}",
                                (10, debug_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)

        else:
            # ===== AUCUNE PERSONNE DÉTECTÉE =====
            self.d_filt = min(5.0, self.d_filt + 0.1)  # Retour progressif à distance max
            self.theta_filt *= 0.9
            self.tracked_id = None
            cv2.putText(annotated, "Aucune personne", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated, f"Mode: {method_label if 'method_label' in locals() else 'WAIT'}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)

        # ===== PUBLICATION COMMANDE TWIST =====
        twist = Twist()
        twist.linear.x = v_cmd
        twist.angular.z = w_cmd
        self.cmd_pub.publish(twist)

        # ===== PUBLICATION IMAGE ANNOTÉE =====
        # Légende méthode en haut à droite
        cv2.rectangle(annotated, (self.img_w - 220, 5), (self.img_w - 5, 95), (20, 20, 20), -1)
        cv2.putText(annotated, "METHOD COMPARE", (self.img_w - 215, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Depth OK: {self.depth_valid_count}", (self.img_w - 215, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(annotated, f"Mono  OK: {self.mono_valid_count}", (self.img_w - 215, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        if self.compare_count > 0:
            avg_err = self.error_sum / self.compare_count
            cv2.putText(annotated, f"Avg Err: {avg_err:.2f}m", (self.img_w - 215, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = rgb_msg.header
        self.img_pub.publish(out_msg)


# ============================================================================
#  ENTRY POINT
# ============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = HybridFuzzyFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()