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

        # 定义类别颜色（BGR）
        color_map = {
            "closed_eye": (0, 0, 255),  # 红色：闭眼
            "closed_mouth": (0, 255, 0),  # 绿色：闭嘴
            "open_eye": (255, 255, 0),  # 蓝色：睁眼
            "open_mouth": (255, 0, 0),  # 黄色：开嘴
        }

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                label = r.names[cls]  # 获取当前框的标签
                color = color_map.get(label, (255, 255, 255))  # 获取颜色
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
