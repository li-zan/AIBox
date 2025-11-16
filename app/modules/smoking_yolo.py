from typing import Any, Dict, Iterator

import numpy as np
import cv2
from ultralytics.engine.results import Results

from app.modules.base import BaseModule


class SmokingYoloModule(BaseModule):
    """抽烟检测"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.4)))
        self.nms_iou = float(self.config.get('nms_iou', 0.45))

    def load(self) -> None:
        from ultralytics import YOLO
        model_path = self.config.get('model')
        print(f"Loading smoking model: {model_path}")
        self.model = YOLO(model_path)
        self.loaded = True
        print(f"smoking model ready")

    def unload(self) -> None:
        del self.model
        self.model = None
        super().unload()

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model is None:
            raise RuntimeError("SmokingYoloModule not loaded")
        # Inference
        results: Iterator[Results] = self.model(frame, iou=self.nms_iou, conf=self.conf_threshold)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                cls_name = self.model.names[cls]
                # 绘制边界框（红色，线宽2）
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 绘制标签背景（黑色半透明）
                label = f"{cls_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y1 = max(y1 - 10, 0)
                label_y2 = label_y1 + label_size[1] + 5
                cv2.rectangle(frame_bgr, (x1, label_y1), (x1 + label_size[0], label_y2), (0, 0, 0), -1)
                # 绘制标签文字（白色）
                cv2.putText(frame_bgr, label, (x1, label_y1 + label_size[1] + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
