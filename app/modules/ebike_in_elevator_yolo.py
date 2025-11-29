from typing import Any, Dict, Iterator

import numpy as np
import cv2
from ultralytics.engine.results import Results

from app.modules.base import BaseModule


class EbikeInElevatorYoloModule(BaseModule):
    """电瓶车进电梯检测"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.25)))
        self.roi = np.array(self.config.get('roi', None), dtype=np.int32)

    def load(self) -> None:
        from ultralytics import YOLO
        model_path = self.config.get('model')
        print(f"Loading ebike model: {model_path}")
        self.model = YOLO(model_path)
        self.loaded = True
        print(f"ebike model ready")

    def unload(self) -> None:
        del self.model
        self.model = None
        super().unload()

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model is None:
            raise RuntimeError("EbikeInElevatorYoloModule not loaded")
        # Inference
        results: Iterator[Results] = self.model(frame, conf=self.conf_threshold)

        inside_ebike = False  # Whether an e-bike is inside the region

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                label = self.model.names[cls] if hasattr(self.model, "names") else str(cls)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                in_zone = cv2.pointPolygonTest(self.roi, (cx, cy), False) >= 0
                # Always show detections inside the zone
                if in_zone:
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_bgr, f"{label} {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    if label == "ebike":
                        inside_ebike = True
                else:
                    # Show outside detections only when enabled
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f"{label} {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw region outline
        cv2.polylines(frame_bgr, [self.roi], True, (0, 0, 255), 2)
        # Label zone
        zone_center = np.mean(self.roi, axis=0).astype(int)
        self.draw_text(frame_bgr, "No Parking Zone", (zone_center[0] - 70, zone_center[1]), (0, 255, 255), 22)
        # Status
        if inside_ebike:
            self.draw_text(frame_bgr, "ALARM", (10, 10), (0, 0, 255), 32)
        else:
            self.draw_text(frame_bgr, "NORMAL", (10, 10), (0, 255, 0), 32)

    def draw_text(self, img, text, pos=(10, 30), color=(0, 255, 0), size=24):
        scale = size / 30
        thickness = max(1, int(size / 20))

        cv2.putText(
            img,  # 直接画在原图！
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,  # BGR
            thickness,
            cv2.LINE_AA
        )
