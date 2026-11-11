#!/usr/bin/env python3
"""
fuzzy_follower_lidar.py
=======================
Nœud ROS 2 — Suivi de personne par fusion LiDAR + Caméra RGB
Contrôleur flou (skfuzzy) + YOLO + ByteTrack + RPLidar A1

Architecture :
  - Thread 1 (30 Hz) : image_callback  → YOLO → angle θ depuis Bottom Center
  - Thread 2 (10 Hz) : _scan_callback  → buffer LiDAR thread-safe (mutex)
  - Fusion pondérée  : d_lidar (si frais) sinon d_visuelle, avec poids par âge
  - Fenêtre angulaire adaptative : ±(2 + 1.5 × d_filt)°, borné [5°, 15°]
  - Sécurité         : arrêt immédiat sur d_brut < safety_distance (non filtré)

Dépendances :
  pip install ultralytics scikit-fuzzy numpy opencv-python
  ros-humble-rplidar-ros (ou équivalent)
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
#  1. CONTRÔLEUR FLOU
# ============================================================================

class FuzzyPersonFollower:
    """
    Contrôleur flou Mamdani à 3 entrées / 2 sorties.

    Entrées
    -------
    d_rel     : distance à la personne [0.5, 5.0] m
    theta_rel : angle horizontal       [-π/3, π/3] rad
    delta_d   : dérivée de la distance [-0.4, 0.4] m/s

    Sorties
    -------
    v_cmd     : vitesse linéaire  [0.0, 1.0] m/s
    omega_cmd : vitesse angulaire [-1.2, 1.2] rad/s
    """

    def __init__(self, safety_distance: float = 0.5):
        self.safety_distance = safety_distance
        self._build_universes()
        self._build_membership_functions()
        self._build_rules()
        self.controller = ctrl.ControlSystem(self.rules)
        self.sim = ctrl.ControlSystemSimulation(self.controller)

    # ------------------------------------------------------------------
    def _build_universes(self):
        self.d_rel     = ctrl.Antecedent(np.linspace(0.5,        5.0,  500), 'd_rel')
        self.theta_rel = ctrl.Antecedent(np.linspace(-np.pi / 3, np.pi / 3, 500), 'theta_rel')
        self.delta_d   = ctrl.Antecedent(np.linspace(-0.4,       0.4,  500), 'delta_d')
        self.v_cmd     = ctrl.Consequent(np.linspace(0.0,        1.0,  500), 'v_cmd')
        self.omega_cmd = ctrl.Consequent(np.linspace(-1.2,       1.2,  500), 'omega_cmd')

    # ------------------------------------------------------------------
    def _build_membership_functions(self):
        d, t, dd = self.d_rel, self.theta_rel, self.delta_d
        v, w     = self.v_cmd, self.omega_cmd

        # --- distance ---
        d['Tres_Proche'] = fuzz.trapmf(d.universe, [0.5, 0.5, 0.8, 1.2])
        d['Proche']      = fuzz.trimf(d.universe,  [0.8, 1.2, 2.0])
        d['Moyenne']     = fuzz.trimf(d.universe,  [1.5, 2.5, 3.5])
        d['Eloignee']    = fuzz.trapmf(d.universe, [3.0, 3.5, 5.0, 5.0])

        # --- angle ---
        t['Gauche'] = fuzz.trimf(t.universe, [-np.pi / 3, -np.pi / 3, 0])
        t['Centre'] = fuzz.trimf(t.universe, [-np.pi / 6,  0,          np.pi / 6])
        t['Droite'] = fuzz.trimf(t.universe, [ 0,           np.pi / 3,  np.pi / 3])

        # --- dérivée distance ---
        dd['S_approche'] = fuzz.trimf(dd.universe, [-0.4, -0.4,  0.0])
        dd['Stable']     = fuzz.trimf(dd.universe, [-0.15, 0.0,  0.15])
        dd['S_eloigne']  = fuzz.trimf(dd.universe, [ 0.0,  0.4,  0.4])

        # --- vitesse linéaire ---
        v['Stop']   = fuzz.trapmf(v.universe, [0.0,  0.0,  0.05, 0.10])
        v['Lent']   = fuzz.trimf(v.universe,  [0.05, 0.15, 0.35])
        v['Moyen']  = fuzz.trimf(v.universe,  [0.25, 0.45, 0.65])
        v['Rapide'] = fuzz.trapmf(v.universe, [0.55, 0.80, 1.00, 1.00])

        # --- vitesse angulaire ---
        w['T_G']    = fuzz.trapmf(w.universe, [-1.2, -1.2, -0.6, -0.3])
        w['L_G']    = fuzz.trimf(w.universe,  [-0.5, -0.25, 0.0])
        w['Neutre'] = fuzz.trimf(w.universe,  [-0.2,  0.0,  0.2])
        w['L_D']    = fuzz.trimf(w.universe,  [ 0.0,  0.25, 0.5])
        w['T_D']    = fuzz.trapmf(w.universe, [ 0.3,  0.60, 1.2, 1.2])

    # ------------------------------------------------------------------
    def _build_rules(self):
        d, t, dd = self.d_rel, self.theta_rel, self.delta_d
        v, w     = self.v_cmd, self.omega_cmd
        R = [
            # Très proche → STOP + correction cap
            ctrl.Rule(d['Tres_Proche'] & t['Gauche'],  (v['Stop'],   w['T_D'])),
            ctrl.Rule(d['Tres_Proche'] & t['Centre'],  (v['Stop'],   w['Neutre'])),
            ctrl.Rule(d['Tres_Proche'] & t['Droite'],  (v['Stop'],   w['T_G'])),
            # Proche
            ctrl.Rule(d['Proche'] & t['Centre'] & dd['S_approche'],           (v['Lent'],  w['Neutre'])),
            ctrl.Rule(d['Proche'] & t['Centre'] & (dd['Stable'] | dd['S_eloigne']), (v['Lent'],  w['Neutre'])),
            ctrl.Rule(d['Proche'] & t['Gauche'], (v['Lent'],  w['L_D'])),
            ctrl.Rule(d['Proche'] & t['Droite'], (v['Lent'],  w['L_G'])),
            # Moyenne
            ctrl.Rule(d['Moyenne'] & t['Centre'] & dd['S_approche'], (v['Moyen'], w['Neutre'])),
            ctrl.Rule(d['Moyenne'] & t['Centre'] & dd['Stable'],     (v['Moyen'], w['Neutre'])),
            ctrl.Rule(d['Moyenne'] & t['Centre'] & dd['S_eloigne'],  (v['Moyen'], w['Neutre'])),
            ctrl.Rule(d['Moyenne'] & t['Gauche'], (v['Moyen'], w['L_G'])),
            ctrl.Rule(d['Moyenne'] & t['Droite'], (v['Moyen'], w['L_D'])),
            # Éloignée
            ctrl.Rule(d['Eloignee'] & dd['S_eloigne'] & t['Gauche'],  (v['Rapide'], w['L_G'])),
            ctrl.Rule(d['Eloignee'] & dd['S_eloigne'] & t['Centre'],  (v['Rapide'], w['Neutre'])),
            ctrl.Rule(d['Eloignee'] & dd['S_eloigne'] & t['Droite'],  (v['Rapide'], w['L_D'])),
            ctrl.Rule(d['Eloignee'] & dd['Stable']    & t['Gauche'],  (v['Moyen'],  w['L_G'])),
            ctrl.Rule(d['Eloignee'] & dd['Stable']    & t['Centre'],  (v['Moyen'],  w['Neutre'])),
            ctrl.Rule(d['Eloignee'] & dd['Stable']    & t['Droite'],  (v['Moyen'],  w['L_D'])),
            ctrl.Rule(d['Eloignee'] & dd['S_approche'],               (v['Lent'],   w['Neutre'])),
            # Règles par défaut (faible activation)
            ctrl.Rule(t['Centre'],                                       (w['Neutre'],)),
            ctrl.Rule(~d['Tres_Proche'] & ~t['Centre'],                  (v['Lent'],   w['Neutre'])),
        ]
        self.rules = R

    # ------------------------------------------------------------------
    def compute(
        self,
        d_val: float,
        theta_val: float,
        dd_val: float,
        d_brut: float | None = None,
    ) -> tuple[float, float]:
        """
        Calcule v_cmd et omega_cmd.

        Parameters
        ----------
        d_val     : distance filtrée (EMA)
        theta_val : angle filtré (EMA)
        dd_val    : dérivée de distance
        d_brut    : distance brute (non filtrée) pour le garde de sécurité

        Returns
        -------
        (v_cmd, omega_cmd)
        """
        d_val     = float(np.clip(d_val,     0.5,        5.0))
        theta_val = float(np.clip(theta_val, -np.pi / 3, np.pi / 3))
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

        # Garde de sécurité sur valeur BRUTE (non filtrée)
        if d_brut is not None and d_brut < self.safety_distance:
            v, w = 0.0, 0.0

        return v, w

    # ------------------------------------------------------------------
    def membership_degrees(
        self,
        d_val: float,
        theta_val: float,
        dd_val: float,
    ) -> dict:
        """Retourne les degrés d'appartenance pour affichage de débogage."""
        def _deg(antecedent, val):
            return {
                name: float(fuzz.interp_membership(
                    antecedent.universe,
                    antecedent[name].mf,
                    val,
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
    """
    Nœud ROS 2 de suivi de personne par fusion LiDAR + Caméra.

    Topics abonnés
    --------------
    /rgb       (sensor_msgs/Image)    — flux caméra RGB
    /scan      (sensor_msgs/LaserScan) — scan RPLidar A1

    Topics publiés
    --------------
    /diff_drive_controller/cmd_vel_unstamped (geometry_msgs/Twist)
    /yolo/fuzzy_output (sensor_msgs/Image) — image annotée
    """

    # ------------------------------------------------------------------ init
    def __init__(self):
        super().__init__('fuzzy_follower_node')

        # ── Paramètres ROS 2 ──────────────────────────────────────────
        self.declare_parameter('image_topic',        '/rgb')
        self.declare_parameter('scan_topic',         '/scan')
        self.declare_parameter('focal_length_px',    600.0)
        self.declare_parameter('real_person_height', 1.77)
        self.declare_parameter('safety_distance',    0.5)
        self.declare_parameter('ema_alpha',          0.25)
        self.declare_parameter('tracker_config',     'bytetrack.yaml')
        self.declare_parameter('yolo_model',         '/home/sidibe/yolo_project/yolo11n.pt')
        self.declare_parameter('conf_threshold',     0.5)
        self.declare_parameter('img_width',          640)
        self.declare_parameter('img_height',         480)
        # Offset de calibration angulaire caméra ↔ LiDAR (en degrés)
        # Mesurez-le avec une mire verticale à 1–3 m, puis renseignez-le ici.
        self.declare_parameter('lidar_angle_offset_deg', 0.0)
        # Âge maximum d'un scan LiDAR pour être considéré comme valide (s)
        self.declare_parameter('lidar_max_age_s',    0.5)

        ip   = self.get_parameter
        self.image_topic        = ip('image_topic').value
        self.scan_topic         = ip('scan_topic').value
        self.focal_length       = ip('focal_length_px').value
        self.real_height        = ip('real_person_height').value
        self.safety_dist        = ip('safety_distance').value
        self.ema_alpha          = ip('ema_alpha').value
        self.tracker_cfg        = ip('tracker_config').value
        self.yolo_path          = ip('yolo_model').value
        self.conf_thr           = ip('conf_threshold').value
        self.img_w              = ip('img_width').value
        self.img_h              = ip('img_height').value
        self.lidar_offset_rad   = np.radians(ip('lidar_angle_offset_deg').value)
        self.lidar_max_age_s    = ip('lidar_max_age_s').value

        # ── Bridge ROS ↔ OpenCV ───────────────────────────────────────
        self.bridge = CvBridge()

        # ── Chargement YOLO ───────────────────────────────────────────
        self.get_logger().info(f"Chargement YOLO : {self.yolo_path}")
        self.model = YOLO(self.yolo_path)

        # ── Contrôleur flou ───────────────────────────────────────────
        self.fuzzy = FuzzyPersonFollower(safety_distance=self.safety_dist)

        # ── État interne ──────────────────────────────────────────────
        self.d_filt      = 2.0
        self.theta_filt  = 0.0
        self.prev_d_filt = 2.0
        self.prev_time   = time.time()
        self.tracked_id  = None

        # ── Buffer LiDAR thread-safe ──────────────────────────────────
        self._scan_lock       = threading.Lock()
        self._last_scan: LaserScan | None = None
        self._last_scan_stamp: float      = 0.0   # temps wall-clock à la réception

        # ── Subscribers ───────────────────────────────────────────────
        self.img_sub  = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._scan_callback, 10)

        # ── Publishers ────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Twist, '/diff_drive_controller/cmd_vel_unstamped', 10)
        self.img_pub = self.create_publisher(
            Image, '/yolo/fuzzy_output', 10)

        self.get_logger().info(
            f"Nœud démarré — image : {self.image_topic}  scan : {self.scan_topic}"
        )

    # ------------------------------------------------------------------ LiDAR callback (Thread 2 — 10 Hz)
    def _scan_callback(self, msg: LaserScan) -> None:
        """
        Stocke le dernier scan de manière thread-safe.
        Appelé à 10 Hz par le thread de ROS executor.
        """
        with self._scan_lock:
            self._last_scan       = msg
            self._last_scan_stamp = time.time()

    # ------------------------------------------------------------------ Helpers LiDAR
    def _get_lidar_distance(
        self,
        angle_camera_rad: float,
        window_deg: float = 10.0,
    ) -> float | None:
        """
        Retourne la distance médiane dans une fenêtre angulaire autour
        de l'angle caméra (converti dans le repère LiDAR via l'offset
        de calibration).

        Parameters
        ----------
        angle_camera_rad : angle horizontal depuis le bas de la bbox
        window_deg       : demi-largeur de la fenêtre en degrés

        Returns
        -------
        distance médiane (m) ou None si aucune mesure valide / scan trop vieux
        """
        with self._scan_lock:
            scan       = self._last_scan
            stamp_wall = self._last_scan_stamp

        # ── Garde de fraîcheur ────────────────────────────────────────
        if scan is None:
            return None
        age = time.time() - stamp_wall
        if age > self.lidar_max_age_s:
            self.get_logger().warning(
                f"Scan LiDAR trop vieux ({age:.2f}s > {self.lidar_max_age_s}s) — fallback visuel"
            )
            return None

        # ── Conversion dans le repère LiDAR ───────────────────────────
        angle_lidar_rad = angle_camera_rad + self.lidar_offset_rad
        angle_lidar_deg = np.degrees(angle_lidar_rad)
        half_win        = window_deg / 2.0

        # ── Collecte des distances valides dans la fenêtre ────────────
        distances = []
        for i, r in enumerate(scan.ranges):
            # Filtre valeurs aberrantes
            if r == 0.0 or not np.isfinite(r):
                continue
            # Hors portée nominale RPLidar A1 (0.15 m – 12 m)
            if r < 0.15 or r > 12.0:
                continue
            ang_deg = np.degrees(scan.angle_min + i * scan.angle_increment)
            if abs(ang_deg - angle_lidar_deg) <= half_win:
                distances.append(r)

        if not distances:
            return None

        return float(np.median(distances))

    # ------------------------------------------------------------------ Fusion distance
    def _fuse_distance(
        self,
        angle_camera_rad: float,
        bbox_h_px: float,
        d_filt_current: float,
    ) -> tuple[float, str]:
        """
        Calcule la distance brute fusionnée (LiDAR pondéré + visuel fallback).

        La fenêtre angulaire est adaptative : plus la personne est loin,
        plus l'incertitude angulaire de YOLO est grande.

        Returns
        -------
        (d_raw, source) où source est 'lidar', 'weighted' ou 'visual'
        """
        # Fenêtre adaptative : [5°, 15°]
        window_deg = float(np.clip(2.0 + 1.5 * d_filt_current, 5.0, 15.0))

        d_lidar  = self._get_lidar_distance(angle_camera_rad, window_deg)
        d_visual = self._bbox_to_distance(bbox_h_px)

        if d_lidar is None:
            return d_visual, 'visual'

        # ── Fusion pondérée par l'âge du scan ─────────────────────────
        with self._scan_lock:
            stamp_wall = self._last_scan_stamp
        age          = time.time() - stamp_wall
        lidar_weight = float(np.clip(1.0 - age / self.lidar_max_age_s, 0.0, 1.0))

        if lidar_weight >= 0.999:
            return d_lidar, 'lidar'

        d_fused = lidar_weight * d_lidar + (1.0 - lidar_weight) * d_visual
        return d_fused, 'weighted'

    # ------------------------------------------------------------------ Helpers caméra
    def _bbox_to_distance(self, bbox_height_px: float) -> float:
        """Estimation de distance par triangulation visuelle (fallback)."""
        if bbox_height_px <= 0:
            return 5.0
        return (self.focal_length * self.real_height) / bbox_height_px

    def _bottom_center_to_angle(self, cx: float) -> float:
        """
        Calcule l'angle horizontal depuis le centre bas (pieds) de la bbox.
        Stable même en cas d'oscillation du haut du corps.
        """
        dx = cx - self.img_w / 2.0
        return float(np.arctan2(dx, self.focal_length))

    # ------------------------------------------------------------------ Image callback (Thread 1 — 30 Hz)
    def image_callback(self, msg: Image) -> None:
        # ── Conversion ROS → OpenCV ───────────────────────────────────
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Erreur conversion image : {e}")
            return

        frame = cv2.resize(frame, (self.img_w, self.img_h))

        # ── Détection + Tracking YOLO / ByteTrack ─────────────────────
        try:
            results = self.model.track(
                frame, persist=True, tracker=self.tracker_cfg,
                classes=[0], conf=self.conf_thr, verbose=False,
            )
        except Exception:
            results = self.model.track(
                frame, persist=True,
                classes=[0], conf=self.conf_thr, verbose=False,
            )

        v_cmd, w_cmd = 0.0, 0.0
        d_raw        = 5.0
        theta_raw    = 0.0
        delta_d      = 0.0
        source       = 'none'
        annotated    = frame.copy()

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # ── Sélection de la cible ─────────────────────────────────
            track_ids = (
                boxes.id.cpu().numpy().astype(int)
                if (hasattr(boxes, 'id') and boxes.id is not None)
                else None
            )
            areas = (
                (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) *
                (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            )

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

            # ── Extraction de la bounding box ─────────────────────────
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[idx].tolist()]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if self.tracked_id is not None:
                cv2.putText(annotated, f"ID:{self.tracked_id}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            bbox_h = float(y2 - y1)

            # ── Angle depuis Bottom Center (pieds) ────────────────────
            # Plus stable que le centre géométrique de la bbox
            bottom_cx  = (x1 + x2) / 2.0
            theta_raw  = self._bottom_center_to_angle(bottom_cx)

            # Marqueur visuel Bottom Center
            cv2.circle(annotated, (int(bottom_cx), y2), 5, (0, 128, 255), -1)

            # ── Fusion LiDAR + Caméra ─────────────────────────────────
            d_raw, source = self._fuse_distance(theta_raw, bbox_h, self.d_filt)

            # ── Filtre EMA sur distance et angle ──────────────────────
            self.d_filt     = self.ema_alpha * d_raw     + (1 - self.ema_alpha) * self.d_filt
            self.theta_filt = self.ema_alpha * theta_raw + (1 - self.ema_alpha) * self.theta_filt

            # ── Dérivée temporelle de la distance ─────────────────────
            now     = time.time()
            dt      = max(now - self.prev_time, 1e-6)
            delta_d = float(np.clip(
                (self.d_filt - self.prev_d_filt) / dt, -0.4, 0.4
            ))
            self.prev_time, self.prev_d_filt = now, self.d_filt

            # ── Contrôleur flou ───────────────────────────────────────
            # Le garde de sécurité dans fuzzy.compute() vérifie d_raw (non filtré)
            v_cmd, w_cmd = self.fuzzy.compute(
                self.d_filt, self.theta_filt, delta_d, d_brut=d_raw
            )

            # ── Affichage degrés d'activation (débogage) ──────────────
            deg = self.fuzzy.membership_degrees(self.d_filt, self.theta_filt, delta_d)
            dr  = deg['d_rel']
            dt_ = deg['theta_rel']
            dd_ = deg['delta_d']
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

        else:
            # Aucune personne détectée → réinitialisation progressive
            self.d_filt     = min(5.0, self.d_filt + 0.1)
            self.theta_filt *= 0.9
            self.tracked_id  = None
            source           = 'none'
            cv2.putText(annotated, "Aucune personne detectee", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ── Publication commande Twist ─────────────────────────────────
        twist             = Twist()
        twist.linear.x    = v_cmd
        twist.angular.z   = w_cmd
        self.cmd_pub.publish(twist)

        # ── Métriques visuelles ────────────────────────────────────────
        safe  = d_raw >= self.safety_dist
        color = (0, 255, 255) if safe else (0, 0, 255)
        label = "STOP - TROP PROCHE!" if not safe else ""

        if label:
            cv2.putText(annotated, label, (self.img_w // 2 - 100, self.img_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        metrics = [
            f"Dist: {d_raw:.2f}m (filt:{self.d_filt:.2f}) [{source}]",
            f"Angle: {np.degrees(theta_raw):.1f} deg",
            f"Delta_d: {delta_d:+.2f} m/s",
            f"v={v_cmd:.2f} m/s  w={w_cmd:.2f} rad/s",
        ]
        for i, txt in enumerate(metrics):
            cv2.putText(annotated, txt, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

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