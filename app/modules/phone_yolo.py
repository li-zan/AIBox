import math
from typing import Any, Dict, Iterator

import numpy as np
import cv2
from ultralytics.engine.results import Results

from app.modules.base import BaseModule


class PhoneYoloModule(BaseModule):
    """打电话检测"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model_det = None
        self.model_pose = None
        self.conf_threshold_det = float(self.config.get('threshold_det', self.config.get('conf_threshold_det', 0.15)))
        self.conf_threshold_pose = float(self.config.get('threshold_pose', self.config.get('conf_threshold_pose', 0.5)))

    def load(self) -> None:
        from ultralytics import YOLO
        model_det_path = self.config.get('model_det')
        model_pose_path = self.config.get('model_pose')
        print(f"Loading phone_det model: {model_det_path}")
        print(f"Loading phone_pose model: {model_pose_path}")
        self.model_det = YOLO(model_det_path)
        self.model_pose = YOLO(model_pose_path)
        self.loaded = True
        print(f"phone model ready")

    def unload(self) -> None:
        del self.model_det
        del self.model_pose
        self.model_det = None
        self.model_pose = None
        super().unload()

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model_det is None or self.model_pose is None:
            raise RuntimeError("PhoneYoloModule not loaded")
        # Inference
        phone_results: Iterator[Results] = self.model_det(frame, conf=self.conf_threshold_det)
        pose_results: Iterator[Results] = self.model_pose(frame, conf=self.conf_threshold_pose)

        # 整理手机
        phones = []
        for r in phone_results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                phones.append({
                    'center': (cx, cy),
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'assigned': False,
                    'status': 'Unknown',
                    'color': (128, 128, 128)
                })
        # 3. 逻辑判断
        for r in pose_results:
            if r.keypoints is None: continue
            kps_data = r.keypoints.data

            for i, kps in enumerate(kps_data):
                # 0:鼻 1:左眼 2:右眼 3:左耳 4:右耳 5:左肩 6:右肩
                nose = kps[0][:2].tolist()
                l_eye = kps[1][:2].tolist()
                r_eye = kps[2][:2].tolist()
                l_ear = kps[3][:2].tolist()
                r_ear = kps[4][:2].tolist()
                l_shoulder = kps[5][:2].tolist()
                r_shoulder = kps[6][:2].tolist()

                # 计算肩宽作为参考
                shoulder_width = self.calculate_distance(l_shoulder, r_shoulder)
                if shoulder_width < 10: shoulder_width = 100

                # --- 寻找匹配手机 ---
                my_phone = None
                min_dist = float('inf')
                match_radius = shoulder_width * 2.5

                for p in phones:
                    if p['assigned']: continue
                    d = self.calculate_distance(p['center'], nose)
                    if d < match_radius and d < min_dist:
                        min_dist = d
                        my_phone = p

                # --- 行为判定核心逻辑 ---
                if my_phone:
                    my_phone['assigned'] = True
                    p_center = my_phone['center']

                    # 1. 计算贴耳距离 (用于判断语音)
                    dist_l_ear = self.calculate_distance(p_center, l_ear)
                    dist_r_ear = self.calculate_distance(p_center, r_ear)
                    min_ear_dist = min(dist_l_ear, dist_r_ear)

                    # 2. 计算垂直高度差 (用于区分视频和玩手机)
                    # 取两眼中心高度
                    avg_eye_y = (l_eye[1] + r_eye[1]) / 2
                    # 取两肩中心高度
                    avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2

                    # 手机Y - 眼睛Y (结果为正数，表示手机在眼睛下方；数值越大，离得越远)
                    vertical_dist_from_eye = p_center[1] - avg_eye_y

                    # --- 阈值设定 ---
                    # A. 语音阈值：非常严格，必须紧贴耳朵 (0.4倍肩宽)
                    thresh_voice = shoulder_width * 0.40

                    # B. 玩手机垂直阈值：
                    # 如果手机在眼睛下方超过 0.6 倍肩宽的距离，说明手机放得很低 -> 玩手机
                    # 或者手机直接位于肩膀下方 (p_center[1] > avg_shoulder_y) -> 玩手机
                    thresh_playing_dist = shoulder_width * 0.6

                    # ---------------- 判定树 ----------------

                    # 情况一：语音通话 (优先级最高)
                    if min_ear_dist < thresh_voice:
                        my_phone['status'] = "Voice Call"
                        my_phone['color'] = (139, 0, 0)  # 深蓝色

                    # 情况二：玩手机 (优先级次之)
                    # 满足以下任一条件即判定为玩手机：
                    # 1. 手机中心点 位于 肩膀连线下方 (位置很低)
                    # 2. 手机虽然在肩膀上方，但距离眼睛垂直距离很远 (符合低头玩手机特征)
                    elif (p_center[1] > avg_shoulder_y) or (vertical_dist_from_eye > thresh_playing_dist):
                        my_phone['status'] = "Playing"
                        my_phone['color'] = (0, 0, 255)  # 红色

                    # 情况三：视频通话 (剩余情况)
                    # 既没有贴耳，也没有放得很低(在脸附近)，那就判定为视频
                    else:
                        my_phone['status'] = "Video Call"
                        my_phone['color'] = (0, 255, 0)  # 绿色

                    # 画连接线
                    cv2.line(frame_bgr, (int(nose[0]), int(nose[1])), (int(p_center[0]), int(p_center[1])),
                             (200, 200, 200),
                             1)

                # 4. 绘制
                for p in phones:
                    x1, y1, x2, y2 = p['box']
                    color = p['color']
                    text = p['status']
                    if not p['assigned']: text = "Phone"

                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame_bgr, (x1, y1 - 20), (x1 + w, y1), color, -1)
                    cv2.putText(frame_bgr, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def calculate_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
