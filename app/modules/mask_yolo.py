from typing import Any, Dict, Iterator

import numpy as np
import cv2
from ultralytics.engine.results import Results

from app.modules.base import BaseModule


class MaskYoloModule(BaseModule):
    """口罩检测"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.5)))

    def load(self) -> None:
        from ultralytics import YOLO
        model_path = self.config.get('model')
        print(f"Loading mask model: {model_path}")
        self.model = YOLO(model_path)
        self.loaded = True
        print(f"mask model ready")

    def unload(self) -> None:
        del self.model
        self.model = None
        super().unload()

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model is None:
            raise RuntimeError("MaskYoloModule not loaded")
        # Inference
        results: Iterator[Results] = self.model(frame, conf=self.conf_threshold)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.model.names[cls]
                color = self.color_map.get(cls, (255, 0, 0))

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

                label = f'{class_name} {conf:.2f}'
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                # 绘制一个实心矩形作为文本背景
                cv2.rectangle(frame_bgr, (x1, y1 - text_height - 10), (x1 + text_width, y1 - 5), color, -1)

                # 在背景上绘制文本（白色字体更清晰）
                cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    @property
    def color_map(self):
        return {
            0: (0, 255, 0),  # 'mask' -> 绿色
            1: (0, 0, 255)  # 'no_mask' -> 红色
        }
