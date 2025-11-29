from typing import Any, Dict
from collections import defaultdict, deque

import numpy as np
import cv2

from app.modules.base import BaseModule


class RunYoloModule(BaseModule):
    """奔跑检测"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.25)))
        self.cfg = {
            # 1. 参数配置
            "FRAME_BOUNCE_THRESHOLD": self.config.get("frame_bounce_threshold", 10.5),  # 进门门槛
            "CONSISTENCY_THRESHOLD_ENTRY": self.config.get("consistency_threshold_entry", 0.55),
            "CONSISTENCY_THRESHOLD_KEEP": 0.5,  # 保级门槛 (惯性)
            # 异常震荡判定 (High Bounce Low Consistency)
            # 如果震荡值 B > 12.0 (看似剧烈跑)
            # 但是一致性 C < 0.4 (历史记录不支持)
            # 判定为：骑车颠簸/干扰 -> 扣分
            "ANOMALY_B_THRESHOLD": self.config.get("anomaly_b_threshold", 12.0),
            "ANOMALY_C_THRESHOLD": 0.4,
            "WINDOW_SIZE": 20,
            "TARGET_SCORE": 60,
            "SCORE_EXIT": 25,
            "SCORE_ADD": 3,
            "SCORE_DECAY": 2,
            "SCORE_PENALTY": 5,  # 异常震荡的扣分力度
            "STRIDE_THRESHOLD": self.config.get("stride_threshold", 0.30),
            "HANDS_LOW_TOLERANCE": 0.05,
        }
        self.history_data = defaultdict(lambda: {
            'cy': deque(maxlen=15),
            'height': deque(maxlen=15),
            'state_window': deque(maxlen=self.cfg["WINDOW_SIZE"]),
            'score': 0,
            'hands_low_count': 0,
            'is_running': False
        })

    def load(self) -> None:
        from ultralytics import YOLO
        model_path = self.config.get('model')
        print(f"Loading run model: {model_path}")
        self.model = YOLO(model_path)
        self.loaded = True
        print(f"run model ready")

    def unload(self) -> None:
        del self.model
        self.model = None
        super().unload()

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model is None:
            raise RuntimeError("RunYoloModule not loaded")
        _, w_img = frame_bgr.shape[:2]
        # Inference
        results = self.model.track(frame, classes=[0], conf=self.conf_threshold, persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else []

            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = map(int, boxes[i])

                if x1 < 10 or x2 > w_img - 10: continue

                box_h = y2 - y1
                track = self.history_data[track_id]
                track['height'].append(box_h)

                #  1. 数据提取
                cy = (y1 + y2) / 2
                hands_low = False
                stride_val = 0.0

                if len(keypoints) > i:
                    kps = keypoints[i]
                    if kps[11][2] > 0.5 and kps[12][2] > 0.5:
                        cy = (kps[11][1] + kps[12][1]) / 2

                    l_wrist, r_wrist = kps[9], kps[10]
                    l_hip, r_hip = kps[11], kps[12]
                    if l_wrist[2] > 0.5 and r_wrist[2] > 0.5 and l_hip[2] > 0.5 and r_hip[2] > 0.5:
                        avg_wrist_y = (l_wrist[1] + r_wrist[1]) / 2
                        avg_hip_y = (l_hip[1] + r_hip[1]) / 2
                        if avg_wrist_y > (avg_hip_y - box_h * self.cfg["HANDS_LOW_TOLERANCE"]):
                            hands_low = True

                    l_ankle, r_ankle = kps[15], kps[16]
                    if l_ankle[2] > 0.5 and r_ankle[2] > 0.5:
                        dist_x = abs(l_ankle[0] - r_ankle[0])
                        stride_val = dist_x / box_h

                track['cy'].append(cy)

                if hands_low:
                    track['hands_low_count'] = min(10, track['hands_low_count'] + 2)
                else:
                    track['hands_low_count'] = max(0, track['hands_low_count'] - 1)
                is_hands_low = track['hands_low_count'] > 3

                #  2. 计算B值
                b_val_current = 0.0
                if len(track['cy']) >= 2:
                    avg_h = np.mean(track['height'])
                    y_std = np.std(track['cy'])
                    b_val_current = (y_std / avg_h) * 1000

                #  3. 单帧判定
                is_frame_running = 0
                if not is_hands_low:
                    if (b_val_current > self.cfg["FRAME_BOUNCE_THRESHOLD"]) or (
                            stride_val > self.cfg["STRIDE_THRESHOLD"]):
                        is_frame_running = 1

                track['state_window'].append(is_frame_running)

                #  4. 一致性计算
                consistency_score = 0.0
                if len(track['state_window']) > 5:
                    consistency_score = sum(track['state_window']) / len(track['state_window'])

                # --- 5. 积分逻辑 (异常震荡防御) ---

                # 异常检测：震荡很大，但一致性很低 -> 判定为颠簸
                # 例如：B=16.0 (看似飞奔), 但 C=0.2 (前几帧都没跑)
                is_anomaly_bump = (b_val_current > self.cfg["ANOMALY_B_THRESHOLD"]) and (
                        consistency_score < self.cfg["ANOMALY_C_THRESHOLD"])

                if is_anomaly_bump:
                    # 如果检测到颠簸，重重扣分，防止瞬间变红
                    track['score'] -= self.cfg["SCORE_PENALTY"]

                # A. 处于“RUNNING”状态 (保级)
                elif track['is_running']:
                    # 必须满足保级门槛 C > 0.3
                    if consistency_score > self.cfg["CONSISTENCY_THRESHOLD_KEEP"]:
                        track['score'] += 1
                    else:
                        # 如果一致性掉下来了，快速扣分
                        track['score'] -= self.cfg["SCORE_DECAY"] * 2

                        # B. 处于“Walk”状态 (进门)
                else:
                    if consistency_score > self.cfg["CONSISTENCY_THRESHOLD_ENTRY"]:
                        track['score'] += self.cfg["SCORE_ADD"]
                    else:
                        track['score'] -= self.cfg["SCORE_DECAY"]

                # 分数限制
                track['score'] = max(0, min(track['score'], self.cfg["TARGET_SCORE"] + 20))

                #  6. 状态锁定
                if track['is_running']:
                    if track['score'] < self.cfg["SCORE_EXIT"]:
                        track['is_running'] = False
                else:
                    if track['score'] >= self.cfg["TARGET_SCORE"]:
                        track['is_running'] = True

                #  7. 绘图
                should_draw = False
                if track['is_running']:
                    should_draw = True

                if should_draw:
                    color = (0, 0, 255)
                    thickness = 3
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)

                    # 调试信息: 如果触发了颠簸防御，文字会变色提示(如果有画框的话)
                    # B值后面加 * 表示这帧震荡很大
                    b_mark = "*" if b_val_current > self.cfg["ANOMALY_B_THRESHOLD"] else ""

                    info = f"RUNNING C:{consistency_score:.2f} B:{b_val_current:.1f}{b_mark}"

                    cv2.rectangle(frame_bgr, (x1, y1 - 25), (x1 + 280, y1), color, -1)
                    cv2.putText(frame_bgr, info, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
