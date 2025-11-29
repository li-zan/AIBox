from typing import Any, Dict, Iterator

import numpy as np
import cv2
from ultralytics.engine.results import Results

from app.modules.base import BaseModule


class FatigueYoloModule(BaseModule):
    """人员疲劳检测"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.3)))
        self.eye_closed_threshold = int(self.config.get('eye_closed_threshold', 30))
        self.mouth_open_threshold = int(self.config.get('mouth_open_threshold', 20))
        self.eye_closed_frames = 0
        self.mouth_open_frames = 0
        self.fatigue_count = 0  # 疲劳事件次数
        self.prev_fatigue_state = False  # 上一帧状态

    def load(self) -> None:
        from ultralytics import YOLO
        model_path = self.config.get('model')
        print(f"Loading fatigue model: {model_path}")
        self.model = YOLO(model_path)
        self.loaded = True
        print(f"fatigue model ready")

    def unload(self) -> None:
        del self.model
        self.model = None
        super().unload()

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model is None:
            raise RuntimeError("FatigueYoloModule not loaded")
        # Inference
        results: Iterator[Results] = self.model(frame, conf=self.conf_threshold)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                name = r.names[cls]  # 获取当前框的标签
                color = (0, 255, 255)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            names = [self.model.names[int(cls)] for cls in classes]
            # 判断眼睛状态
            if 'closed_eye' in names:
                self.eye_closed_frames += 1
            else:
                self.eye_closed_frames = 0

            # 判断嘴巴状态
            if 'open_mouth' in names:
                self.mouth_open_frames += 1
            else:
                self.mouth_open_frames = 0

            # 疲劳判定逻辑
            fatigue_alert = (
                    self.eye_closed_frames > self.eye_closed_threshold or
                    self.mouth_open_frames > self.mouth_open_threshold
            )

            # === 疲劳计数，当从正常 -> 疲劳 时累加 ===
            if fatigue_alert and not self.prev_fatigue_state:
                self.fatigue_count += 1

            self.prev_fatigue_state = fatigue_alert

            # === 显示提示信息 ===
            if fatigue_alert:
                cv2.putText(frame_bgr, "FATIGUE ALERT!", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            else:
                cv2.putText(frame_bgr, "Normal", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # 显示疲劳事件次数
            cv2.putText(frame_bgr, f"Fatigue Events: {self.fatigue_count}", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
