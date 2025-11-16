from typing import Any, Dict, Iterator

import numpy as np
import cv2
from ultralytics.engine.results import Results

from app.modules.base import BaseModule


class FightingYoloModule(BaseModule):
    """打架检测"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.5)))

    def load(self) -> None:
        from ultralytics import YOLO
        model_path = self.config.get('model')
        print(f"Loading fighting model: {model_path}")
        self.model = YOLO(model_path)
        self.loaded = True
        print(f"fighting model ready")

    def unload(self) -> None:
        del self.model
        self.model = None
        super().unload()

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model is None:
            raise RuntimeError("FightingYoloModule not loaded")
        # Inference
        results: Iterator[Results] = self.model(frame, conf=self.conf_threshold)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                label = f"fighting {conf:.2f}"

                # 绘制红色边界框
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 绘制标签（黑色背景+白色文字）
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                label_y = max(y1 - 10, label_height)
                cv2.rectangle(
                    frame_bgr,
                    (x1, label_y - label_height - 5),
                    (x1 + label_width, label_y + 5),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(
                    frame_bgr, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
