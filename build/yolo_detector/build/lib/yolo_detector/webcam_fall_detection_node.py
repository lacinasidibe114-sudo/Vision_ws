#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Détection de chute sur webcam USB (Méthode 1)
Publie l'image annotée sur /yolo/fall_detection et affiche une fenêtre OpenCV.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# ----------------------------------------------------------------------
# Utilitaires MediaPipe simplifiés (identique à fall_detection_node.py)
# ----------------------------------------------------------------------
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

LANDMARK_NAMES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24,
    'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
}

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, image_rgb):
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            return results.pose_landmarks
        return None

    def close(self):
        self.pose.close()

def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=2):
    h, w = image.shape[:2]
    for lm in landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), radius, color, -1)

# ----------------------------------------------------------------------
# Détecteur de chute (Méthode 1)
# ----------------------------------------------------------------------
class FallDetectorMethod1:
    def __init__(self, yolo_model_path, conf_threshold=0.80):
        self.yolo = YOLO(yolo_model_path)
        self.conf_threshold = conf_threshold
        self.pose = PoseDetector()
        self.aspect_ratio_threshold = 0.9
        self.fall_confirm_frames = 3
        self.fall_counter = 0

    def detect_activity(self, landmarks, h, w):
        lm = landmarks.landmark
        def get_xy(part):
            return (lm[LANDMARK_NAMES[part]].x * w,
                    lm[LANDMARK_NAMES[part]].y * h)
        l_sh = get_xy('LEFT_SHOULDER')
        r_sh = get_xy('RIGHT_SHOULDER')
        l_hip = get_xy('LEFT_HIP')
        r_hip = get_xy('RIGHT_HIP')
        l_knee = get_xy('LEFT_KNEE')
        r_knee = get_xy('RIGHT_KNEE')

        h_upper = (calculate_distance(l_sh, l_hip) + calculate_distance(r_sh, r_hip)) / 2.0
        h_leg = (calculate_distance(l_hip, l_knee) + calculate_distance(r_hip, r_knee)) / 2.0

        if h_upper < 1e-5:
            return "Inconnu"
        ratio = h_leg / h_upper
        if ratio < 0.5:
            return "Assis"
        elif ratio > 1.2:
            return "Marche"
        else:
            shoulder_y = (lm[LANDMARK_NAMES['LEFT_SHOULDER']].y + lm[LANDMARK_NAMES['RIGHT_SHOULDER']].y) / 2.0
            hip_y = (lm[LANDMARK_NAMES['LEFT_HIP']].y + lm[LANDMARK_NAMES['RIGHT_HIP']].y) / 2.0
            if abs(shoulder_y - hip_y) < 0.05:
                return "Allonge"
            return "Debout"

    def check_fall_aspect_ratio(self, bbox):
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1
        if w < 1:
            return False, 0.0
        aspect_ratio = h / w
        return (aspect_ratio < self.aspect_ratio_threshold), aspect_ratio

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        fall_detected = False
        activity = "Aucune"
        aspect_ratio_val = 0.0

        results = self.yolo(frame, verbose=False, classes=[0])
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            is_fall, aspect_ratio_val = self.check_fall_aspect_ratio((x1, y1, x2, y2))

            roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if roi.size > 0:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pose_result = self.pose.process(roi_rgb)
                if pose_result:
                    draw_landmarks(roi, pose_result)
                    roi_h, roi_w = roi.shape[:2]
                    activity = self.detect_activity(pose_result, roi_h, roi_w)

            if is_fall:
                self.fall_counter += 1
                color = (0, 0, 255)
            else:
                self.fall_counter = max(0, self.fall_counter - 1)
                color = (0, 255, 0)

            if self.fall_counter >= self.fall_confirm_frames:
                fall_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Personne {conf:.2f} AR:{aspect_ratio_val:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, fall_detected, f"Act:{activity} AR:{aspect_ratio_val:.2f}"

    def release(self):
        self.pose.close()

# ----------------------------------------------------------------------
# Nœud ROS2 pour webcam
# ----------------------------------------------------------------------
class WebcamFallNode(Node):
    def __init__(self):
        super().__init__('webcam_fall_node')
        self.bridge = CvBridge()
        self.declare_parameter('camera_id', 0)          # 0 = webcam intégrée, 2 = externe
        self.declare_parameter('yolo_model', '/home/sidibe/yolo_project/yolo11n.pt')
        self.declare_parameter('conf_threshold', 0.80)

        camera_id = self.get_parameter('camera_id').value
        yolo_path = self.get_parameter('yolo_model').value
        conf = self.get_parameter('conf_threshold').value

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            self.get_logger().error(f"Impossible d'ouvrir la caméra {camera_id}")
            raise RuntimeError("Camera not available")

        self.detector = FallDetectorMethod1(yolo_path, conf)
        self.pub_img = self.create_publisher(Image, '/yolo/fall_detection', 10)

        self.timer = self.create_timer(0.033, self.timer_callback)  # ~30 fps
        self.get_logger().info(f"Webcam fall detection démarrée (caméra {camera_id})")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Frame vide")
            return

        frame = cv2.resize(frame, (980, 740))
        annotated, fall_detected, info = self.detector.process_frame(frame)

        # Ajout texte info
        cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if fall_detected:
            cv2.putText(annotated, "FALL ALERT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            self.get_logger().warn("⚠️ CHUTE DÉTECTÉE !")

        # Affichage local
        cv2.imshow("Fall Detection - Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

        # Publication ROS2
        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = "webcam"
        self.pub_img.publish(out_msg)

    def destroy_node(self):
        self.cap.release()
        self.detector.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WebcamFallNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()