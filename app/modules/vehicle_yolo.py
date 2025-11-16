from typing import Any, Dict, Iterator

import numpy as np
import cv2
from ultralytics.engine.results import Results

from app.modules.base import BaseModule


class VehicleYoloModule(BaseModule):
    """车辆型号检测"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.show_stats = bool(self.config.get('show_stats', True))
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.5)))

    def load(self) -> None:
        from ultralytics import YOLO
        model_path = self.config.get('model')
        print(f"Loading vehicle model: {model_path}")
        self.model = YOLO(model_path)
        self.loaded = True
        print(f"vehicle model ready")

    def unload(self) -> None:
        del self.model
        self.model = None
        super().unload()

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model is None:
            raise RuntimeError("VehicleYoloModule not loaded")
        # Inference
        results: Iterator[Results] = self.model(frame, conf=self.conf_threshold)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            stats = {"total_detections": 0, "detections": []}
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.model.names[cls]
                color = self.get_color_for_class(cls)

                stats["total_detections"] += 1
                stats["detections"].append({
                    "id": stats["total_detections"],
                    "class": class_name,
                    "class_id": cls,
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "area": float((x2 - x1) * (y2 - y1))
                })

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)

                label = f"{class_name} {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )

                cv2.rectangle(frame_bgr,
                              (x1, y1 - text_height - baseline - 10),
                              (x1 + text_width, y1),
                              color, -1)

                cv2.putText(frame_bgr, label,
                            (x1, y1 - baseline - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if self.show_stats and stats["total_detections"] > 0:
                stats_text = f"Detected: {stats['total_detections']} car logos"
                cv2.putText(frame_bgr, stats_text,
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def get_color_for_class(self, class_id):
        """为不同类别的车标分配不同的颜色"""
        colors = [
            (255, 0, 0),  # 红色 - 类别0
            (0, 255, 0),  # 绿色 - 类别1
            (0, 0, 255),  # 蓝色 - 类别2
            (255, 255, 0),  # 青色 - 类别3
            (255, 0, 255),  # 紫色 - 类别4
            (0, 255, 255),  # 黄色 - 类别5
            (255, 165, 0),  # 橙色 - 类别6
            (128, 0, 128),  # 深紫色 - 类别7
            (255, 192, 203)  # 粉色 - 类别8
        ]
        return colors[class_id % len(colors)]
