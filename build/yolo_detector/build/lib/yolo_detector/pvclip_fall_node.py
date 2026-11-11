#!/usr/bin/env python3
"""
PV-CLIP — Détection de Chute Zero-Shot (MONO-PERSONNE)
Adapté pour ROS2 avec Isaac Sim (topic /rgb)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import collections
import math
from typing import Optional

import torch
from PIL import Image as PILImage

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import clip

# ============================================================================
#  CONFIGURATION (identique à l'original)
# ============================================================================
CFG = {
    "yolo_model": "yolo11m-pose.pt",
    "yolo_conf": 0.35,
    "single_person_mode": True,
    "ar_collapse_thresh": 0.9,
    "torso_angle_thresh": 35.0,
    "velocity_thresh": 2.5,
    "velocity_history": 12,
    "static_fall_frames": 8,
    "clip_model": "ViT-B/32",
    "clip_fall_threshold": 0.42,
    "clip_every_n_frames": 2,
    "clip_positive": [
        "a person falling down",
        "a person lying on the floor",
        "a person collapsed on the ground",
        "an elderly person who fell",
        "someone lying flat on the ground",
        "a person on the ground unable to get up",
        "a person who has just fallen",
        "a human body lying horizontally",
    ],
    "clip_negative": [
        "a person standing upright",
        "a person walking normally",
        "a person sitting in a chair",
        "a person bending over to pick something up",
        "daily human activity",
    ],
    "alert_window": 4,
    "alert_ratio": 0.40,
    "min_stages_for_suspicion": 1,
    "standing_ar_thresh": 0.6,
    "standing_torso_thresh": 30.0,
    "recovery_frames": 20,
    "min_person_height_px": 80,
    "min_person_area_ratio": 0.01,
}

# Indices keypoints COCO
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_HIP = 11
KP_R_HIP = 12


# ============================================================================
#  CLASSES DU PIPELINE (adaptées de l'original)
# ============================================================================

class CLIPVerifier:
    def __init__(self, clip_model, preprocess, device: str):
        self.model = clip_model
        self.preprocess = preprocess
        self.device = device
        self.n_pos = len(CFG["clip_positive"])
        all_texts = CFG["clip_positive"] + CFG["clip_negative"]
        tokens = clip.tokenize(all_texts).to(device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            self.text_features = feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def verify(self, frame_bgr: np.ndarray, bbox: np.ndarray) -> dict:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = 15
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(frame_bgr.shape[1], x2 + pad)
        y2p = min(frame_bgr.shape[0], y2 + pad)
        crop = frame_bgr[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            return {"fall_prob": 0.0, "is_fall": False}
        pil_img = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sims = (100.0 * img_feat @ self.text_features.T).softmax(dim=-1)
        sims_np = sims[0].cpu().numpy()
        fall_prob = float(sims_np[:self.n_pos].sum())
        return {"fall_prob": fall_prob, "is_fall": fall_prob >= CFG["clip_fall_threshold"]}


class VelocityTracker:
    def __init__(self):
        self.com_y_hist = {}
        self.low_pos_count = {}

    def update(self, track_id: int, com_y: float, ar: float = 0.0) -> tuple:
        if track_id not in self.com_y_hist:
            self.com_y_hist[track_id] = collections.deque(maxlen=CFG["velocity_history"])
            self.low_pos_count[track_id] = 0
        self.com_y_hist[track_id].append(com_y)
        hist = list(self.com_y_hist[track_id])
        if len(hist) < 2:
            return 0.0, False
        xs = np.arange(len(hist), dtype=float)
        ys = np.array(hist)
        velocity = float(np.polyfit(xs, ys, 1)[0])
        in_low_position = ar >= CFG["ar_collapse_thresh"]
        if in_low_position:
            self.low_pos_count[track_id] += 1
        else:
            self.low_pos_count[track_id] = 0
        static_fallen = self.low_pos_count[track_id] >= CFG["static_fall_frames"]
        return velocity, static_fallen

    def cleanup(self, active_ids: set):
        for tid in list(self.com_y_hist.keys()):
            if tid not in active_ids:
                del self.com_y_hist[tid]
                self.low_pos_count.pop(tid, None)


class FallDecider:
    def __init__(self):
        self.history = {}
        self.alert_state = {}
        self.recovery_ctr = {}

    def _is_standing(self, ar: float, torso_angle: float) -> bool:
        return ar < CFG["standing_ar_thresh"] and torso_angle < CFG["standing_torso_thresh"]

    def update(self, track_id: int, e1: bool, e2: bool, e3: bool,
               static_fallen: bool, ar: float, torso_angle: float) -> bool:
        if track_id not in self.alert_state:
            self.history[track_id] = collections.deque(maxlen=CFG["alert_window"])
            self.alert_state[track_id] = False
            self.recovery_ctr[track_id] = 0
        if self.alert_state[track_id]:
            standing_now = self._is_standing(ar, torso_angle)
            if standing_now:
                self.recovery_ctr[track_id] += 1
                if self.recovery_ctr[track_id] >= CFG["recovery_frames"]:
                    self.alert_state[track_id] = False
                    self.recovery_ctr[track_id] = 0
                    self.history[track_id].clear()
            else:
                self.recovery_ctr[track_id] = 0
            return self.alert_state[track_id]
        n = CFG["min_stages_for_suspicion"]
        if static_fallen:
            suspicious = 1
        else:
            suspicious = int((int(e1) + int(e2) + int(e3)) >= n)
        self.history[track_id].append(suspicious)
        hist = self.history[track_id]
        if len(hist) < CFG["alert_window"]:
            return False
        if (sum(hist) / len(hist)) >= CFG["alert_ratio"]:
            self.alert_state[track_id] = True
            self.recovery_ctr[track_id] = 0
            return True
        return False

    def cleanup(self, active_ids: set):
        for tid in list(self.alert_state.keys()):
            if tid not in active_ids:
                self.alert_state.pop(tid, None)
                self.history.pop(tid, None)
                self.recovery_ctr.pop(tid, None)


def stage1_geometry(keypoints: np.ndarray, bbox: np.ndarray) -> dict:
    x1, y1, x2, y2 = bbox
    bb_w = max(x2 - x1, 1.0)
    bb_h = max(y2 - y1, 1.0)
    ar = bb_w / bb_h
    vis = keypoints[:, 2]
    def get_pt(idx):
        return keypoints[idx, :2] if vis[idx] > 0.3 else None
    torso_angle = 90.0
    ls = get_pt(KP_L_SHOULDER)
    rs = get_pt(KP_R_SHOULDER)
    lh = get_pt(KP_L_HIP)
    rh = get_pt(KP_R_HIP)
    if ls is not None and rs is not None and lh is not None and rh is not None:
        sho_mid = (ls + rs) / 2.0
        hip_mid = (lh + rh) / 2.0
        dx = hip_mid[0] - sho_mid[0]
        dy = hip_mid[1] - sho_mid[1]
        torso_angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))
    visible_pts = keypoints[vis > 0.3, :2]
    com = visible_pts.mean(axis=0) if len(visible_pts) > 0 else np.array([(x1+x2)/2, (y1+y2)/2])
    ar_suspicious = ar >= CFG["ar_collapse_thresh"]
    torso_suspicious = torso_angle >= (90.0 - CFG["torso_angle_thresh"])
    return {
        "ar": ar,
        "torso_angle": torso_angle,
        "com": com,
        "ar_suspicious": ar_suspicious,
        "torso_suspicious": torso_suspicious,
        "is_suspicious": ar_suspicious or torso_suspicious,
    }


def is_valid_person_bbox(bbox: np.ndarray, frame_w: int, frame_h: int) -> bool:
    x1, y1, x2, y2 = bbox
    bb_w = x2 - x1
    bb_h = y2 - y1
    if bb_h < CFG["min_person_height_px"]:
        return False
    area_ratio = (bb_w * bb_h) / (frame_w * frame_h)
    if area_ratio < CFG["min_person_area_ratio"]:
        return False
    ar = bb_w / max(bb_h, 1.0)
    if ar > 6.0:
        return False
    return True


def select_primary_track(confirmed_tracks: list, frame_w: int, frame_h: int):
    if not confirmed_tracks:
        return None
    frame_cx = frame_w / 2.0
    frame_cy = frame_h / 2.0
    def center_distance(track):
        ltrb = track.to_ltrb()
        cx = (ltrb[0] + ltrb[2]) / 2.0
        cy = (ltrb[1] + ltrb[3]) / 2.0
        return abs(cx - frame_cx) + abs(cy - frame_cy)
    return min(confirmed_tracks, key=center_distance)


# ============================================================================
#  NŒUD ROS2
# ============================================================================
class PvclipFallNode(Node):
    def __init__(self):
        super().__init__('pvclip_fall_node')
        self.bridge = CvBridge()

        # Paramètres ROS2
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('yolo_model', '/home/sidibe/yolo_project/yolo11m-pose.pt')
        self.declare_parameter('yolo_conf', 0.35)

        image_topic = self.get_parameter('image_topic').value
        yolo_path = self.get_parameter('yolo_model').value
        yolo_conf = self.get_parameter('yolo_conf').value

        # Initialisation des modèles
        self.get_logger().info("Chargement YOLO...")
        self.yolo = YOLO(yolo_path)
        self.get_logger().info("Chargement CLIP...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(CFG["clip_model"], device=self.device)
        self.clip_model.eval()

        # DeepSORT tracker
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=0.7,
                                embedder="mobilenet", half=True,
                                embedder_gpu=(self.device == "cuda"))

        self.verifier = CLIPVerifier(self.clip_model, self.clip_preprocess, self.device)
        self.vel_tracker = VelocityTracker()
        self.decider = FallDecider()

        self.frame_idx = 0
        self.primary_track_id = None

        # Publishers / Subscriber
        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.alert_pub = self.create_publisher(Bool, '/pvclip/fall_alert', 10)
        self.img_pub = self.create_publisher(Image, '/pvclip/output', 10)

        self.get_logger().info(f"PV-CLIP node démarré sur {image_topic}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Erreur conversion: {e}")
            return

        H, W = frame.shape[:2]
        self.frame_idx += 1

        # Détection YOLO
        results = self.yolo(frame, verbose=False, conf=CFG["yolo_conf"])

        deepsort_dets = []
        yolo_hits = []
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                bb = r.boxes.xyxy[i].cpu().numpy()
                conf = float(r.boxes.conf[i].cpu())
                kp = r.keypoints.data[i].cpu().numpy()
                if not is_valid_person_bbox(bb, W, H):
                    continue
                x1, y1, x2, y2 = bb
                deepsort_dets.append(([x1, y1, x2-x1, y2-y1], conf, 0))
                yolo_hits.append({"bbox": bb, "kp": kp})

        # DeepSORT update
        tracks = self.tracker.update_tracks(deepsort_dets, frame=frame)
        confirmed = [t for t in tracks if t.is_confirmed()]
        active_ids = {t.track_id for t in confirmed}
        primary = select_primary_track(confirmed, W, H)

        alert_this_frame = False

        # Traitement du track principal
        if primary is not None:
            tid = primary.track_id
            tlbr = primary.to_ltrb()
            self.primary_track_id = tid

            matched = None
            if yolo_hits:
                tc_x = (tlbr[0] + tlbr[2]) / 2.0
                tc_y = (tlbr[1] + tlbr[3]) / 2.0
                dists = [math.hypot(tc_x - (h["bbox"][0]+h["bbox"][2])/2.0,
                                    tc_y - (h["bbox"][1]+h["bbox"][3])/2.0)
                         for h in yolo_hits]
                matched = yolo_hits[int(np.argmin(dists))]

            if matched:
                geo = stage1_geometry(matched["kp"], matched["bbox"])
                e1 = geo["is_suspicious"]
                com_y = float(geo["com"][1])
                ar_val = geo["ar"]
                bbox_clip = matched["bbox"]
            else:
                e1 = False
                com_y = (tlbr[1] + tlbr[3]) / 2.0
                ar_val = 0.0
                geo = {"ar": 0, "torso_angle": 90}
                bbox_clip = tlbr

            velocity, static_fallen = self.vel_tracker.update(tid, com_y, ar_val)
            e2 = velocity >= CFG["velocity_thresh"]

            e3 = False
            clip_prob = 0.0
            if (e1 or e2 or static_fallen) and (self.frame_idx % CFG["clip_every_n_frames"] == 0):
                clip_res = self.verifier.verify(frame, bbox_clip)
                e3 = clip_res["is_fall"]
                clip_prob = clip_res["fall_prob"]

            alert = self.decider.update(tid, e1, e2, e3, static_fallen,
                                        ar_val, geo["torso_angle"])
            alert_this_frame = alert

            # Annotation visuelle
            x1, y1, x2, y2 = [int(v) for v in tlbr]
            color = (0, 0, 255) if alert else (0, 210, 60)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            status = "CHUTE !" if alert else "Suivi"
            cv2.putText(frame, f"#{tid} {status}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Affichage des métriques
            cv2.putText(frame, f"AR={geo['ar']:.2f} Torso={geo['torso_angle']:.0f}",
                        (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,0), 1)
            cv2.putText(frame, f"Vel={velocity:+.1f} CLIP={clip_prob*100:.0f}%",
                        (x1, y2+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,0), 1)

            if matched and matched["kp"] is not None:
                for kpt in matched["kp"]:
                    if kpt[2] > 0.3:
                        cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 4, (0,255,255), -1)

        else:
            cv2.putText(frame, "En attente de personne...", (W//2-140, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,255), 2)

        # Nettoyage
        self.vel_tracker.cleanup(active_ids)
        self.decider.cleanup(active_ids)

        # Publication de l'alerte
        alert_msg = Bool()
        alert_msg.data = alert_this_frame
        self.alert_pub.publish(alert_msg)

        # Publication de l'image annotée
        out_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        out_msg.header = msg.header
        self.img_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PvclipFallNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()