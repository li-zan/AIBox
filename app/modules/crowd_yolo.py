from typing import Any, Dict, Iterator

import numpy as np
import cv2
from ultralytics.engine.results import Results

from app.modules.base import BaseModule


class CrowdYoloModule(BaseModule):
    """人群聚集"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.25)))
        self.person_threshold = int(self.config.get('person_threshold', 5))
        self.distance_threshold_ratio = float(self.config.get('distance_threshold_ratio', 0.3))

    def load(self) -> None:
        from ultralytics import YOLO
        model_path = self.config.get('model')
        print(f"Loading people model: {model_path}")
        self.model = YOLO(model_path)
        self.loaded = True
        print(f"people model ready")

    def unload(self) -> None:
        del self.model
        self.model = None
        super().unload()

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model is None:
            raise RuntimeError("CrowdYoloModule not loaded")
        # Inference
        results: Iterator[Results] = self.model(frame, classes=[0], conf=self.conf_threshold)
        height, width = frame_bgr.shape[:2]
        distance_threshold = int(width * self.distance_threshold_ratio)

        for r in results:
            person_boxes = []
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                person_boxes.append(box)
                x1, y1, x2, y2 = map(int, box)
                # 画框
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # 标签文字
                label = f"person {conf:.2f}"
                # 绘制标签背景
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame_bgr, (x1, y1 - th - 5), (x1 + tw, y1), (0, 255, 255), -1)
                # 绘制标签文字
                cv2.putText(frame_bgr, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 2)
            person_count = len(person_boxes)
            # 判断聚集
            crowded = self.is_crowded(person_boxes, self.person_threshold, distance_threshold)
            # 左上角显示人数
            cv2.putText(frame_bgr,
                        f"Persons: {person_count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2)
            # 左上角显示聚集状态
            status_text = "Crowded" if crowded else "Normal"
            status_color = (0, 0, 255) if crowded else (0, 255, 0)
            cv2.putText(frame_bgr,
                        f"Status: {status_text}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        status_color,
                        2)

    def is_crowded(self, boxes, person_threshold, distance_threshold):
        """ 判断是否人群聚集 """
        num_person = len(boxes)
        if num_person < person_threshold:
            return False

        # 计算框中心
        centers = np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in boxes])
        if len(centers) < 2:
            return False

        # 两两距离
        dists = np.sqrt(np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=-1))
        avg_dist = np.mean(dists[np.triu_indices(num_person, 1)])

        return avg_dist < distance_threshold
