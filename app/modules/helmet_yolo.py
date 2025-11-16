from typing import Any, Dict, Iterator

import numpy as np
import cv2
from ultralytics.engine.results import Results

from app.modules.base import BaseModule


class HelmetYoloModule(BaseModule):
    """安全帽检测"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.3)))

    def load(self) -> None:
        from ultralytics import YOLO
        model_path = self.config.get('model')
        print(f"Loading helmet model: {model_path}")
        self.model = YOLO(model_path)
        self.loaded = True
        print(f"helmet model ready")

    def unload(self) -> None:
        del self.model
        self.model = None
        super().unload()

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model is None:
            raise RuntimeError("HelmetYoloModule not loaded")
        # Inference
        results: Iterator[Results] = self.model(frame, classes=[0], conf=self.conf_threshold)  # 只检测安全帽类别，对应ID 0

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = map(int, box)

                # 计算检测框的宽度和高度
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height

                # 过滤条件1：排除太小的检测框
                min_area = self.config.get('min_area', 500)
                if box_area < min_area:
                    continue

                # 过滤条件2：排除长宽比异常的检测框
                aspect_ratio = box_width / box_height
                min_aspect_ratio = self.config.get('min_aspect_ratio', 0.3)
                max_aspect_ratio = self.config.get('max_aspect_ratio', 3.0)
                if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                    continue

                # 过滤条件3：排除在图像边缘的检测框
                edge_threshold = self.config.get('edge_threshold', 20)
                height, width = frame_bgr.shape[:2]
                if (x1 < edge_threshold or y1 < edge_threshold or
                        x2 > width - edge_threshold or y2 > height - edge_threshold):
                    continue

                box_color = (0, 0, 255)  # 红色 (BGR)
                text_color = (255, 255, 255)  # 白色
                box_thickness = 2
                font_scale = 0.6
                # 绘制红色边界框
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), box_color, box_thickness)
                # 准备标签文本
                label = f"no_helmet {conf:.2f}"
                # 绘制文本背景
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                )

                cv2.rectangle(
                    frame_bgr,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    box_color,
                    -1  # 填充
                )

                # 绘制文本
                cv2.putText(
                    frame_bgr,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    2
                )
