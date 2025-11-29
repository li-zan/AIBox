from typing import Any, Dict, Iterator

import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics.engine.results import Results

from app.modules.base import BaseModule


class PeopleCountYoloModule(BaseModule):
    """人员计数"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.tracker = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.65)))

    def load(self) -> None:
        from ultralytics import YOLO
        model_path = self.config.get('model')
        print(f"Loading people model: {model_path}")
        self.model = YOLO(model_path)
        self.tracker = DeepSort(
            embedder="mobilenet",
            max_age=40,
            n_init=3,
            max_iou_distance=0.7,
            max_cosine_distance=0.2
        )
        self.loaded = True
        print(f"people model ready")

    def unload(self) -> None:
        del self.model
        self.model = None
        super().unload()

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model is None:
            raise RuntimeError("PeopleCountYoloModule not loaded")
        # Inference
        results: Iterator[Results] = self.model(frame, classes=[0], conf=self.conf_threshold)
        detections = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                w_box, h_box = x2 - x1, y2 - y1
                # 过滤掉过小的框
                if w_box * h_box > self.config.get("min_box_area", 300):
                    # DeepSort 格式: ([x, y, w, h], conf, class_name)
                    detections.append(([x1, y1, w_box, h_box], conf, 'person'))
        # DeepSORT 更新
        tracks = self.tracker.update_tracks(detections, frame=frame)

        # 结果分析与绘制
        current_ids = []
        for t in tracks:
            if not t.is_confirmed():
                continue

            track_id = t.track_id
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_bgr, f'ID {track_id}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            current_ids.append(track_id)

        # 统计当前人数
        current_count = len(set(current_ids))
        cv2.putText(frame_bgr, f'Current Persons: {current_count}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
