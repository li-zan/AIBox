from typing import Any, Dict, Iterator

import numpy as np
import cv2
from ultralytics.engine.results import Results

from app.modules.base import BaseModule


class VehicleCountYoloModule(BaseModule):
    """车辆数量计数"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.5)))
        self.nms_iou = float(self.config.get('nms_iou', 0.7))

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
            raise RuntimeError("VehicleCountYoloModule not loaded")
        # Inference
        results: Iterator[Results] = self.model.track(
            frame,
            classes=[2, 3, 5, 7],
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            iou=self.nms_iou,
            persist=True
        )

        for r in results:
            if r.boxes is None or r.boxes.id is None:
                current_vehicle_count = 0
            else:
                track_ids = r.boxes.id.int().cpu().tolist()
                current_vehicle_count = len(set(track_ids))

                boxes = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.int().cpu().numpy()
                classes = r.boxes.cls.int().cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = self.model.names[cls]
                    color = self.get_color_for_class(cls)

                    label = f"ID {track_id} {class_name} {conf:.2f}"

                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame_bgr, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                    )

                    # annotated_frame = r.plot(line_width=1, font_size=0.5)

            # 在画面左上角显示车辆数量
            count_text = f"car number: {current_vehicle_count}"
            cv2.putText(
                frame_bgr,
                count_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

    def get_color_for_class(self, class_id):
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
