#!/usr/bin/env python3
"""
fuzzy_follower_lidar.py  (v2 — distance robuste)
================================================
Suivi de personne par fusion RPLidar A1 + Caméra RGB
Contrôleur flou skfuzzy + YOLO + ByteTrack

Améliorations v2 (par rapport à v1) — toutes ciblées sur la stabilité
de la distance, sans rien changer aux entrées du contrôleur flou :

  1. CLUSTERING sur les points LiDAR dans la fenêtre
     → on isole la personne (objet le plus proche) au lieu de prendre
       la médiane brute qui mélange personne + arrière-plan.

  2. ÉLARGISSEMENT PROGRESSIF de la fenêtre angulaire
     → si trop peu de points trouvés, on agrandit (jusqu'à ×2.5)
       au lieu d'abandonner immédiatement.

  3. REJET D'OUTLIER TEMPOREL
     → un saut de distance > max_jump_speed m/s est rejeté
       (garde la valeur précédente le temps d'une frame).

  4. CONVENTION D'ANGLE configurable (lidar_angle_sign)
     → utile si le LiDAR est monté à l'envers ou si la convention
       ROS standard (CCW+) ne correspond pas à la caméra (CW+).

  5. DIAGNOSTIC PÉRIODIQUE
     → log toutes les ~1s : source effective, nb de points utilisés,
       distances brute/cluster/finale. Aide à comprendre où ça coince.
"""

import threading
import time
import warnings

import cv2
import numpy as np
import torch
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

warnings.filterwarnings("ignore")


# ============================================================================
#  1. CONTRÔLEUR FLOU  (inchangé — fonctionne, donc on n'y touche pas)
# ============================================================================

class FuzzyPersonFollower:
    def __init__(self, safety_distance: float = 0.5):
        self.safety_distance = safety_distance
        self._build_universes()
        self._build_membership_functions()
        self._build_rules()
        self.controller = ctrl.ControlSystem(self.rules)
        self.sim = ctrl.ControlSystemSimulation(self.controller)

    def _build_universes(self):
        self.d_rel     = ctrl.Antecedent(np.linspace(0.5,        5.0,  500), 'd_rel')
        self.theta_rel = ctrl.Antecedent(np.linspace(-np.pi/3, np.pi/3, 500), 'theta_rel')
        self.delta_d   = ctrl.Antecedent(np.linspace(-0.4,       0.4,  500), 'delta_d')
        self.v_cmd     = ctrl.Consequent(np.linspace(0.0,        1.0,  500), 'v_cmd')
        self.omega_cmd = ctrl.Consequent(np.linspace(-1.2,       1.2,  500), 'omega_cmd')

    def _build_membership_functions(self):
        d, t, dd = self.d_rel, self.theta_rel, self.delta_d
        v, w     = self.v_cmd, self.omega_cmd

        d['Tres_Proche'] = fuzz.trapmf(d.universe, [0.5, 0.5, 0.8, 1.2])
        d['Proche']      = fuzz.trimf(d.universe,  [0.8, 1.2, 2.0])
        d['Moyenne']     = fuzz.trimf(d.universe,  [1.5, 2.5, 3.5])
        d['Eloignee']    = fuzz.trapmf(d.universe, [3.0, 3.5, 5.0, 5.0])

        t['Gauche'] = fuzz.trimf(t.universe, [-np.pi/3, -np.pi/3, 0])
        t['Centre'] = fuzz.trimf(t.universe, [-np.pi/6,  0,        np.pi/6])
        t['Droite'] = fuzz.trimf(t.universe, [ 0,         np.pi/3,  np.pi/3])

        dd['S_approche'] = fuzz.trimf(dd.universe, [-0.4, -0.4,  0.0])
        dd['Stable']     = fuzz.trimf(dd.universe, [-0.15, 0.0,  0.15])
        dd['S_eloigne']  = fuzz.trimf(dd.universe, [ 0.0,  0.4,  0.4])

        v['Stop']   = fuzz.trapmf(v.universe, [0.0,  0.0,  0.05, 0.10])
        v['Lent']   = fuzz.trimf(v.universe,  [0.05, 0.15, 0.35])
        v['Moyen']  = fuzz.trimf(v.universe,  [0.25, 0.45, 0.65])
        v['Rapide'] = fuzz.trapmf(v.universe, [0.55, 0.80, 1.00, 1.00])

        w['T_G']    = fuzz.trapmf(w.universe, [-1.2, -1.2, -0.6, -0.3])
        w['L_G']    = fuzz.trimf(w.universe,  [-0.5, -0.25, 0.0])
        w['Neutre'] = fuzz.trimf(w.universe,  [-0.2,  0.0,  0.2])
        w['L_D']    = fuzz.trimf(w.universe,  [ 0.0,  0.25, 0.5])
        w['T_D']    = fuzz.trapmf(w.universe, [ 0.3,  0.60, 1.2, 1.2])

    def _build_rules(self):
        d, t, dd = self.d_rel, self.theta_rel, self.delta_d
        v, w     = self.v_cmd, self.omega_cmd
        R = [
            ctrl.Rule(d['Tres_Proche'] & t['Gauche'],  (v['Stop'],   w['T_D'])),
            ctrl.Rule(d['Tres_Proche'] & t['Centre'],  (v['Stop'],   w['Neutre'])),
            ctrl.Rule(d['Tres_Proche'] & t['Droite'],  (v['Stop'],   w['T_G'])),
            ctrl.Rule(d['Proche'] & t['Centre'] & dd['S_approche'],           (v['Lent'],  w['Neutre'])),
            ctrl.Rule(d['Proche'] & t['Centre'] & (dd['Stable'] | dd['S_eloigne']), (v['Lent'],  w['Neutre'])),
            ctrl.Rule(d['Proche'] & t['Gauche'], (v['Lent'],  w['L_D'])),
            ctrl.Rule(d['Proche'] & t['Droite'], (v['Lent'],  w['L_G'])),
            ctrl.Rule(d['Moyenne'] & t['Centre'] & dd['S_approche'], (v['Moyen'], w['Neutre'])),
            ctrl.Rule(d['Moyenne'] & t['Centre'] & dd['Stable'],     (v['Moyen'], w['Neutre'])),
            ctrl.Rule(d['Moyenne'] & t['Centre'] & dd['S_eloigne'],  (v['Moyen'], w['Neutre'])),
            ctrl.Rule(d['Moyenne'] & t['Gauche'], (v['Moyen'], w['L_G'])),
            ctrl.Rule(d['Moyenne'] & t['Droite'], (v['Moyen'], w['L_D'])),
            ctrl.Rule(d['Eloignee'] & dd['S_eloigne'] & t['Gauche'],  (v['Rapide'], w['L_G'])),
            ctrl.Rule(d['Eloignee'] & dd['S_eloigne'] & t['Centre'],  (v['Rapide'], w['Neutre'])),
            ctrl.Rule(d['Eloignee'] & dd['S_eloigne'] & t['Droite'],  (v['Rapide'], w['L_D'])),
            ctrl.Rule(d['Eloignee'] & dd['Stable']    & t['Gauche'],  (v['Moyen'],  w['L_G'])),
            ctrl.Rule(d['Eloignee'] & dd['Stable']    & t['Centre'],  (v['Moyen'],  w['Neutre'])),
            ctrl.Rule(d['Eloignee'] & dd['Stable']    & t['Droite'],  (v['Moyen'],  w['L_D'])),
            ctrl.Rule(d['Eloignee'] & dd['S_approche'],               (v['Lent'],   w['Neutre'])),
            ctrl.Rule(t['Centre'],                                       (w['Neutre'],)),
            ctrl.Rule(~d['Tres_Proche'] & ~t['Centre'],                  (v['Lent'],   w['Neutre'])),
        ]
        self.rules = R

    def compute(self, d_val, theta_val, dd_val, d_brut=None):
        d_val     = float(np.clip(d_val,     0.5,        5.0))
        theta_val = float(np.clip(theta_val, -np.pi/3, np.pi/3))
        dd_val    = float(np.clip(dd_val,    -0.4,       0.4))
        self.sim.input['d_rel']     = d_val
        self.sim.input['theta_rel'] = theta_val
        self.sim.input['delta_d']   = dd_val
        try:
            self.sim.compute()
            v = float(np.clip(self.sim.output['v_cmd'],     0.0,  1.0))
            w = float(np.clip(self.sim.output['omega_cmd'], -1.2, 1.2))
        except Exception:
            v, w = 0.0, 0.0
        if d_brut is not None and d_brut < self.safety_distance:
            v, w = 0.0, 0.0
        return v, w

    def membership_degrees(self, d_val, theta_val, dd_val):
        def _deg(antecedent, val):
            return {
                name: float(fuzz.interp_membership(
                    antecedent.universe, antecedent[name].mf, val
                ))
                for name in antecedent.terms
            }
        return {
            'd_rel':     _deg(self.d_rel,     d_val),
            'theta_rel': _deg(self.theta_rel, theta_val),
            'delta_d':   _deg(self.delta_d,   dd_val),
        }


# ============================================================================
#  2. NŒUD ROS 2
# ============================================================================

class FuzzyFollowerNode(Node):

    # ------------------------------------------------------------------ init
    def __init__(self):
        super().__init__('fuzzy_follower_node')

        # ── Paramètres ROS 2 ──────────────────────────────────────────
        self.declare_parameter('image_topic',        '/rgb')
        self.declare_parameter('scan_topic',         '/scan')
        self.declare_parameter('focal_length_px',    600.0)
        self.declare_parameter('real_person_height', 1.77)
        self.declare_parameter('safety_distance',    0.5)
        self.declare_parameter('ema_alpha',          0.35)   # ↑ un peu plus lissé (était 0.25)
        self.declare_parameter('tracker_config',     'bytetrack.yaml')
        self.declare_parameter('yolo_model',         '/home/sidibe/yolo_project/yolo11n.pt')
        self.declare_parameter('conf_threshold',     0.5)
        self.declare_parameter('img_width',          640)
        self.declare_parameter('img_height',         480)

        # Calibration LiDAR ↔ Caméra
        self.declare_parameter('lidar_angle_offset_deg', 0.0)
        self.declare_parameter('lidar_angle_sign',       1.0)    # ±1
        self.declare_parameter('lidar_max_age_s',        0.5)

        # ── Paramètres de clustering (les leviers de stabilité) ──────
        self.declare_parameter('window_base_deg',     8.0)        # demi-largeur de base de la fenêtre
        self.declare_parameter('cluster_width',       0.40)       # largeur du cluster personne (m)
        self.declare_parameter('min_cluster_points',  3)          # points minimum pour valider
        self.declare_parameter('max_jump_speed',      3.5)        # m/s — au-delà → outlier

        # Plage LiDAR utilisable (RPLidar A1 : 0.15–12 m, mais on borne à la zone utile)
        self.declare_parameter('lidar_min_range',     0.15)
        self.declare_parameter('lidar_max_range',     8.0)

        # Diagnostic
        self.declare_parameter('debug_log_period',    30)         # 1 log toutes les N frames

        # ── Lecture paramètres ─────────────────────────────────────────
        gp = self.get_parameter
        self.image_topic      = gp('image_topic').value
        self.scan_topic       = gp('scan_topic').value
        self.focal_length     = gp('focal_length_px').value
        self.real_height      = gp('real_person_height').value
        self.safety_dist      = gp('safety_distance').value
        self.ema_alpha        = gp('ema_alpha').value
        self.tracker_cfg      = gp('tracker_config').value
        self.yolo_path        = gp('yolo_model').value
        self.conf_thr         = gp('conf_threshold').value
        self.img_w            = gp('img_width').value
        self.img_h            = gp('img_height').value
        self.lidar_offset_rad = np.radians(gp('lidar_angle_offset_deg').value)
        self.lidar_sign       = float(gp('lidar_angle_sign').value)
        self.lidar_max_age_s  = gp('lidar_max_age_s').value
        self.window_base_deg  = gp('window_base_deg').value
        self.cluster_width    = gp('cluster_width').value
        self.min_cluster_pts  = gp('min_cluster_points').value
        self.max_jump_speed   = gp('max_jump_speed').value
        self.lidar_min_range  = gp('lidar_min_range').value
        self.lidar_max_range  = gp('lidar_max_range').value
        self.debug_log_period = gp('debug_log_period').value

        # ── ROS bridge ────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── YOLO ──────────────────────────────────────────────────────
        self.get_logger().info(f"Chargement YOLO : {self.yolo_path}")
        self.model = YOLO(self.yolo_path)

        # ── Contrôleur flou ───────────────────────────────────────────
        self.fuzzy = FuzzyPersonFollower(safety_distance=self.safety_dist)

        # ── État ──────────────────────────────────────────────────────
        self.d_filt      = 2.0
        self.theta_filt  = 0.0
        self.prev_d_filt = 2.0
        self.prev_d_raw  = 2.0        # pour rejet outlier
        self.prev_time   = time.time()
        self.tracked_id  = None
        self._frame_cnt  = 0

        # Buffer LiDAR thread-safe
        self._scan_lock       = threading.Lock()
        self._last_scan       = None
        self._last_scan_stamp = 0.0

        # ── Subscribers / Publishers ──────────────────────────────────
        self.img_sub  = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._scan_callback, 10)
        self.cmd_pub = self.create_publisher(
            Twist, '/diff_drive_controller/cmd_vel_unstamped', 10)
        self.img_pub = self.create_publisher(
            Image, '/yolo/fuzzy_output', 10)

        self.get_logger().info(
            f"Nœud v2 démarré — image:{self.image_topic}  scan:{self.scan_topic}  "
            f"ema_α={self.ema_alpha}  cluster_width={self.cluster_width}m"
        )

    # ------------------------------------------------------------------ LiDAR callback
    def _scan_callback(self, msg: LaserScan) -> None:
        with self._scan_lock:
            self._last_scan       = msg
            self._last_scan_stamp = time.time()

    # ------------------------------------------------------------------ Collecte des distances dans une fenêtre
    def _collect_in_window(self, scan: LaserScan, center_deg: float, half_deg: float):
        """Retourne la liste des distances valides dans [center-half, center+half]."""
        distances = []
        for i, r in enumerate(scan.ranges):
            if not np.isfinite(r) or r == 0.0:
                continue
            if r < self.lidar_min_range or r > self.lidar_max_range:
                continue
            ang_deg = np.degrees(scan.angle_min + i * scan.angle_increment)
            # Différence angulaire normalisée [-180, 180]
            diff = (ang_deg - center_deg + 180.0) % 360.0 - 180.0
            if abs(diff) <= half_deg:
                distances.append(r)
        return distances

    # ------------------------------------------------------------------ Extraction de la personne (CLUSTERING)
    def _extract_person_distance(self, distances):
        """
        Isole la personne dans la liste de mesures LiDAR.
        Stratégie : la personne est l'objet le plus proche dans la fenêtre.
        On garde tous les points dans [d_min, d_min + cluster_width] et on
        prend la médiane de ce sous-ensemble. Cela élimine les points de
        l'arrière-plan (mur, sol, autres personnes plus loin).

        Returns
        -------
        (d_person, n_cluster_points) ou (None, 0)
        """
        if len(distances) < self.min_cluster_pts:
            return None, 0

        sorted_d = sorted(distances)
        d_min    = sorted_d[0]
        cluster  = [d for d in sorted_d if d <= d_min + self.cluster_width]

        if len(cluster) < self.min_cluster_pts:
            return None, 0

        return float(np.median(cluster)), len(cluster)

    # ------------------------------------------------------------------ Distance LiDAR (avec élargissement progressif)
    def _get_lidar_distance(self, angle_camera_rad: float):
        """
        Distance personne via LiDAR.
        Tente d'abord avec la fenêtre de base, élargit si peu de points.

        Returns
        -------
        dict avec : d, source ('lidar' | None), n_points, window_used, scan_age
        """
        info = {'d': None, 'source': None, 'n_points': 0,
                'window_used': 0.0, 'scan_age': float('inf')}

        with self._scan_lock:
            scan       = self._last_scan
            stamp_wall = self._last_scan_stamp

        if scan is None:
            return info

        info['scan_age'] = time.time() - stamp_wall
        if info['scan_age'] > self.lidar_max_age_s:
            return info

        # Conversion angle caméra → angle LiDAR
        angle_lidar_rad = self.lidar_sign * angle_camera_rad + self.lidar_offset_rad
        angle_lidar_deg = np.degrees(angle_lidar_rad)

        # Essai avec fenêtres croissantes
        windows_try = [self.window_base_deg,
                       self.window_base_deg * 1.5,
                       self.window_base_deg * 2.5]

        for half_deg in windows_try:
            raw = self._collect_in_window(scan, angle_lidar_deg, half_deg)
            d_person, n = self._extract_person_distance(raw)
            if d_person is not None:
                info['d']           = d_person
                info['source']      = 'lidar'
                info['n_points']    = n
                info['window_used'] = 2.0 * half_deg
                return info

        return info

    # ------------------------------------------------------------------ Helpers visuels
    def _bbox_to_distance(self, bbox_height_px: float) -> float:
        if bbox_height_px <= 0:
            return 5.0
        return (self.focal_length * self.real_height) / bbox_height_px

    def _bottom_center_to_angle(self, cx: float) -> float:
        return float(np.arctan2(cx - self.img_w / 2.0, self.focal_length))

    # ------------------------------------------------------------------ Image callback
    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Erreur conversion image : {e}")
            return

        frame = cv2.resize(frame, (self.img_w, self.img_h))
        self._frame_cnt += 1

        # Tracking
        try:
            results = self.model.track(
                frame, persist=True, tracker=self.tracker_cfg,
                classes=[0], conf=self.conf_thr, verbose=False)
        except Exception:
            results = self.model.track(
                frame, persist=True,
                classes=[0], conf=self.conf_thr, verbose=False)

        v_cmd, w_cmd = 0.0, 0.0
        d_raw        = 5.0
        theta_raw    = 0.0
        delta_d      = 0.0
        source       = 'none'
        lidar_info   = {'d': None, 'source': None, 'n_points': 0,
                        'window_used': 0.0, 'scan_age': float('inf')}
        d_visual     = None
        annotated    = frame.copy()

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # Choix de la cible
            track_ids = (boxes.id.cpu().numpy().astype(int)
                         if (hasattr(boxes, 'id') and boxes.id is not None) else None)
            areas = ((boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) *
                     (boxes.xyxy[:, 3] - boxes.xyxy[:, 1]))

            if self.tracked_id is not None and track_ids is not None:
                mask = (track_ids == self.tracked_id)
                if np.any(mask):
                    idx = int(np.where(mask)[0][0])
                else:
                    idx = int(torch.argmax(areas))
                    self.tracked_id = track_ids[idx]
            else:
                idx = int(torch.argmax(areas))
                if track_ids is not None:
                    self.tracked_id = track_ids[idx]

            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[idx].tolist()]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if self.tracked_id is not None:
                cv2.putText(annotated, f"ID:{self.tracked_id}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            bbox_h    = float(y2 - y1)
            bottom_cx = (x1 + x2) / 2.0
            theta_raw = self._bottom_center_to_angle(bottom_cx)
            cv2.circle(annotated, (int(bottom_cx), y2), 5, (0, 128, 255), -1)

            # ── DISTANCE : LiDAR (clustering) puis fallback visuel ─────
            lidar_info = self._get_lidar_distance(theta_raw)
            d_visual   = self._bbox_to_distance(bbox_h)

            if lidar_info['d'] is not None:
                d_candidate = lidar_info['d']
                source      = 'lidar'
            else:
                d_candidate = d_visual
                source      = 'visual'

            # ── REJET OUTLIER TEMPOREL ─────────────────────────────────
            # Si la nouvelle mesure saute de plus de max_jump_speed m/s
            # par rapport à la précédente, on la considère aberrante.
            now = time.time()
            dt  = max(now - self.prev_time, 1e-6)
            jump = abs(d_candidate - self.prev_d_raw) / dt
            if jump > self.max_jump_speed and self._frame_cnt > 5:
                d_raw  = self.prev_d_raw          # ignore le saut
                source = source + '_rej'
            else:
                d_raw = d_candidate

            # ── EMA sur distance et angle ──────────────────────────────
            self.d_filt     = self.ema_alpha * d_raw     + (1 - self.ema_alpha) * self.d_filt
            self.theta_filt = self.ema_alpha * theta_raw + (1 - self.ema_alpha) * self.theta_filt

            # ── Dérivée ────────────────────────────────────────────────
            delta_d = float(np.clip(
                (self.d_filt - self.prev_d_filt) / dt, -0.4, 0.4))
            self.prev_time   = now
            self.prev_d_filt = self.d_filt
            self.prev_d_raw  = d_raw

            # ── Contrôleur flou ────────────────────────────────────────
            v_cmd, w_cmd = self.fuzzy.compute(
                self.d_filt, self.theta_filt, delta_d, d_brut=d_raw)

            # ── Affichage degrés d'activation ─────────────────────────
            deg = self.fuzzy.membership_degrees(self.d_filt, self.theta_filt, delta_d)
            dr, dt_, dd_ = deg['d_rel'], deg['theta_rel'], deg['delta_d']
            cv2.putText(annotated,
                        f"d: Tp={dr['Tres_Proche']:.2f} P={dr['Proche']:.2f} "
                        f"M={dr['Moyenne']:.2f} E={dr['Eloignee']:.2f}",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(annotated,
                        f"θ: G={dt_['Gauche']:.2f} C={dt_['Centre']:.2f} D={dt_['Droite']:.2f}",
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(annotated,
                        f"Δd: A={dd_['S_approche']:.2f} S={dd_['Stable']:.2f} E={dd_['S_eloigne']:.2f}",
                        (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            # ── DIAGNOSTIC PÉRIODIQUE ──────────────────────────────────
            if self._frame_cnt % self.debug_log_period == 0:
                d_vis_str  = f"{d_visual:.2f}m"  if d_visual is not None else "n/a"
                d_lid_str  = (f"{lidar_info['d']:.2f}m"
                              if lidar_info['d'] is not None else "n/a")
                self.get_logger().info(
                    f"[DIAG] θ={np.degrees(theta_raw):+5.1f}°  "
                    f"d_lidar={d_lid_str} (pts={lidar_info['n_points']}, "
                    f"win=±{lidar_info['window_used']/2:.0f}°, "
                    f"age={lidar_info['scan_age']*1000:.0f}ms)  "
                    f"d_visual={d_vis_str}  "
                    f"d_raw={d_raw:.2f}m  d_filt={self.d_filt:.2f}m  "
                    f"src={source}"
                )

        else:
            # Personne perdue → réinit progressive
            self.d_filt     = min(5.0, self.d_filt + 0.1)
            self.theta_filt *= 0.9
            self.tracked_id  = None
            source           = 'none'
            cv2.putText(annotated, "Aucune personne detectee", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ── Publication Twist ──────────────────────────────────────────
        twist           = Twist()
        twist.linear.x  = v_cmd
        twist.angular.z = w_cmd
        self.cmd_pub.publish(twist)

        # ── Annotations principales ────────────────────────────────────
        safe  = d_raw >= self.safety_dist
        color = (0, 255, 255) if safe else (0, 0, 255)
        if not safe:
            cv2.putText(annotated, "STOP - TROP PROCHE!",
                        (self.img_w // 2 - 100, self.img_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Information détaillée sur la source LiDAR (utile pour le debug)
        if lidar_info['source'] == 'lidar':
            extra = f"  pts={lidar_info['n_points']}  win=±{lidar_info['window_used']/2:.0f}°"
        else:
            extra = "  (LiDAR n/a)"

        metrics = [
            f"Dist: {d_raw:.2f}m (filt:{self.d_filt:.2f}) [{source}]{extra}",
            f"Angle: {np.degrees(theta_raw):.1f} deg",
            f"Delta_d: {delta_d:+.2f} m/s",
            f"v={v_cmd:.2f}  w={w_cmd:.2f}",
        ]
        for i, txt in enumerate(metrics):
            cv2.putText(annotated, txt, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # ── Publication image annotée ──────────────────────────────────
        out_msg        = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = msg.header
        self.img_pub.publish(out_msg)


# ============================================================================
#  3. MAIN
# ============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = FuzzyFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()