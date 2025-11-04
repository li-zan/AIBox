from typing import Any, Dict, Iterator

import numpy as np
import cv2
from shapely.geometry import Point, Polygon
from ultralytics.engine.results import Results

from app.modules.base import BaseModule
from app.utils.plotting import draw_chinese_label_inplace


class IntrusionYoloModule(BaseModule):
    """人员闯入"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.3)))
        self.roi = self.config.get('roi', None)
        # 保存每个track_id是否已进入区域的状态
        self.intrusion_state = {}

    def load(self) -> None:
        from ultralytics import YOLO
        model_path = self.config.get('model')
        print(f"Loading intrusion model: {model_path}")
        self.model = YOLO(model_path)
        self.loaded = True
        print(f"intrusion model ready")

    def unload(self) -> None:
        del self.model
        self.model = None
        super().unload()

    def process(self, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model is None:
            raise RuntimeError("IntrusionYoloModule not loaded")
        # Inference
        results: Iterator[Results] = self.model(frame_bgr, stream=True)

        # 绘制危险区域
        zone = np.array(self.roi)
        cv2.polylines(frame_bgr, [zone], isClosed=True, color=(0, 0, 255), thickness=2)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, classes, confs):
                if r.names[cls] != "person" or conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box)
                # 底边中点作为位置判断依据
                cx, cy = int((x1 + x2) / 2), y2
                # 绘制中心点
                cv2.circle(frame_bgr, (cx, cy), 5, (255, 0, 0), -1)

                # 判断是否在危险区域
                polygon = Polygon(zone)
                in_danger = polygon.contains(Point(cx, cy))

                # 更新状态
                if in_danger:
                    # self.intrusion_state[track_id] = True
                    # label = f"ID {track_id}: 人员闯入"
                    # label = f"人员闯入"
                    label = f"人员闯入 {conf:.2f}"
                    color = (0, 0, 255)
                else:
                    # 如果之前在区域内，现在出了，也可以保留状态（可选）
                    # if track_id in intrusion_state and intrusion_state[track_id]:
                    #     label = f"ID {track_id}: 人员闯入"
                    #     color = (0, 0, 255)
                    # else:
                    #     label = f"ID {track_id}: 安全"
                    #     color = (0, 255, 0)
                    # label = f"安全"
                    label = f"安全 {conf:.2f}"
                    color = (0, 255, 0)

                # 绘制边框
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                # 支持中文
                draw_chinese_label_inplace(frame_bgr, label, x1, y1 - 15, 0.7, 2, color)
                # 不支持中文
                # cv2.putText(frame, label, (x1, y1 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
