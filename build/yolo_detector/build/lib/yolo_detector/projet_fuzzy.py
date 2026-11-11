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
import warnings
warnings.filterwarnings("ignore")



# ============================================================================
#  1. CONTRÔLEUR FLOU (identique au code original)
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
#  2. NŒUD ROS2
# ============================================================================
class FuzzyFollowerNode(Node):
    def __init__(self):
        super().__init__('fuzzy_follower_node')

        # Paramètres ROS2
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('focal_length_px', 600.0)
        self.declare_parameter('real_person_height', 1.77)
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('ema_alpha', 0.25)
        self.declare_parameter('tracker_config', 'bytetrack.yaml')
        self.declare_parameter('yolo_model', '/home/sidibe/yolo_project/yolo11n.pt')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('img_width', 640)
        self.declare_parameter('img_height', 480)

        # Récupération
        self.image_topic = self.get_parameter('image_topic').value
        self.focal_length = self.get_parameter('focal_length_px').value
        self.real_height = self.get_parameter('real_person_height').value
        self.safety_dist = self.get_parameter('safety_distance').value
        self.ema_alpha = self.get_parameter('ema_alpha').value
        self.tracker_cfg = self.get_parameter('tracker_config').value
        self.yolo_path = self.get_parameter('yolo_model').value
        self.conf_thr = self.get_parameter('conf_threshold').value
        self.img_w = self.get_parameter('img_width').value
        self.img_h = self.get_parameter('img_height').value

        # Bridge
        self.bridge = CvBridge()

        # Chargement YOLO
        self.get_logger().info(f"Chargement YOLO depuis {self.yolo_path}")
        self.model = YOLO(self.yolo_path)

        # Contrôleur flou
        self.fuzzy = FuzzyPersonFollower(safety_distance=self.safety_dist)

        # État interne
        self.d_filt = 2.0
        self.theta_filt = 0.0
        self.prev_d_filt = 2.0
        self.prev_time = time.time()
        self.tracked_id = None

        # Publishers / Subscriber
        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/diff_drive_controller/cmd_vel_unstamped', 10)
        self.img_pub = self.create_publisher(Image, '/yolo/fuzzy_output', 10)

        self.get_logger().info(f"Nœud flou démarré, abonné à {self.image_topic}")

    def bbox_to_distance(self, bbox_height_px):
        if bbox_height_px <= 0:
            return 5.0
        return (self.focal_length * self.real_height) / bbox_height_px

    def bbox_to_angle(self, center_x):
        dx = center_x - self.img_w / 2.0
        return np.arctan2(dx, self.focal_length)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Erreur conversion: {e}")
            return

        frame = cv2.resize(frame, (self.img_w, self.img_h))

        # Tracking avec ByteTrack
        # Utiliser le tracker par défaut si le fichier n'existe pas
        try:
            results = self.model.track(frame, persist=True, tracker=self.tracker_cfg,
                                       classes=[0], conf=self.conf_thr, verbose=False)
        except Exception:
            # Si fichier non trouvé, laisser Ultralytics utiliser son tracker par défaut
            results = self.model.track(frame, persist=True, classes=[0],
                                       conf=self.conf_thr, verbose=False)

        v_cmd, w_cmd = 0.0, 0.0
        d_raw, theta_raw = 5.0, 0.0
        delta_d = 0.0
        annotated = frame.copy()

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # Récupérer les IDs de suivi
            if hasattr(boxes, 'id') and boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = None

            # Calcul des aires
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            idx = None
            # Choix de la cible : même ID ou plus grande aire
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
                x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                if self.tracked_id is not None:
                    cv2.putText(annotated, f"ID:{self.tracked_id}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

                bbox_h = y2 - y1
                center_x = (x1 + x2) / 2.0
                d_raw = self.bbox_to_distance(bbox_h)
                theta_raw = self.bbox_to_angle(center_x)

                # Lissage
                self.d_filt = self.ema_alpha * d_raw + (1 - self.ema_alpha) * self.d_filt
                self.theta_filt = self.ema_alpha * theta_raw + (1 - self.ema_alpha) * self.theta_filt

                now = time.time()
                dt = max(now - self.prev_time, 1e-6)
                delta_d = (self.d_filt - self.prev_d_filt) / dt
                delta_d = np.clip(delta_d, -0.4, 0.4)
                self.prev_time, self.prev_d_filt = now, self.d_filt

                v_cmd, w_cmd = self.fuzzy.compute(self.d_filt, self.theta_filt, delta_d, d_brut=d_raw)

                # Affichage des degrés d'activation (optionnel)
                deg = self.fuzzy.membership_degrees(self.d_filt, self.theta_filt, delta_d)
                text_d = f"d: Tp={deg['d_rel']['Tres_Proche']:.2f} P={deg['d_rel']['Proche']:.2f} M={deg['d_rel']['Moyenne']:.2f} E={deg['d_rel']['Eloignee']:.2f}"
                text_theta = f"θ: G={deg['theta_rel']['Gauche']:.2f} C={deg['theta_rel']['Centre']:.2f} D={deg['theta_rel']['Droite']:.2f}"
                text_dd = f"Δd: A-={deg['delta_d']['S_approche']:.2f} S={deg['delta_d']['Stable']:.2f} E+={deg['delta_d']['S_eloigne']:.2f}"
                cv2.putText(annotated, text_d, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                cv2.putText(annotated, text_theta, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                cv2.putText(annotated, text_dd, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        else:
            # Aucune personne -> réinitialisation progressive
            self.d_filt = min(5.0, self.d_filt + 0.1)
            self.theta_filt *= 0.9
            self.tracked_id = None
            cv2.putText(annotated, "Aucune personne", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Publier la commande Twist (même si v_cmd=0)
        twist = Twist()
        twist.linear.x = v_cmd
        twist.angular.z = w_cmd
        self.cmd_pub.publish(twist)

        # Affichage des métriques principales
        color = (0, 255, 255) if d_raw >= self.safety_dist else (0, 0, 255)
        metrics = [
            f"Dist: {d_raw:.2f}m (filt:{self.d_filt:.2f})",
            f"Angle: {np.degrees(theta_raw):.1f}°",
            f"Δd: {delta_d:+.2f} m/s",
            f"v_cmd: {v_cmd:.2f} | w_cmd: {w_cmd:.2f}"
        ]
        for i, txt in enumerate(metrics):
            cv2.putText(annotated, txt, (10, 30 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Publier l'image annotée
        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = msg.header
        self.img_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FuzzyFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()