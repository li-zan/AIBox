from typing import Any, Dict, Iterator

import numpy as np
import cv2
from ultralytics.engine.results import Results

from app.modules.base import BaseModule


class IllegalParkingYoloModule(BaseModule):
    """机动车违停检测"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.5)))
        self.roi = np.array(self.config.get('roi', None), np.int32)

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
            raise RuntimeError("IllegalParkingYoloModule not loaded")
        # Inference
        results: Iterator[Results] = self.model(frame, classes=[2, 3, 5, 7], conf=self.conf_threshold)

        self.draw_parking_zone(frame_bgr)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                in_parking_zone = self.is_in_parking_zone([x1, y1, x2, y2])
                color = (0, 0, 255) if in_parking_zone else (0, 255, 0)
                # 绘制边界框
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                # 绘制标签
                class_name = self.model.names[cls]
                status = "VIOLATION!" if in_parking_zone else "Normal"
                label = f"{class_name} {status} {conf:.2f}"
                # 计算标签背景大小
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                # 绘制标签背景
                cv2.rectangle(frame_bgr, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), color, -1)
                # 绘制标签文本
                cv2.putText(frame_bgr, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def draw_parking_zone(self, image):
        """在图像上绘制违停区域"""
        # 创建半透明覆盖层
        overlay = image.copy()
        cv2.fillPoly(overlay, [self.roi], (0, 0, 255))  # 红色填充

        # 将半透明区域叠加到原图
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

        # 绘制区域边界
        cv2.polylines(image, [self.roi], True, (0, 0, 255), 2)

        # 添加区域标签
        cv2.putText(image, "NO PARKING ZONE",
                    (self.roi[0][0], self.roi[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def is_in_parking_zone(self, bbox):
        """检测车辆是否在违停区域内"""
        x1, y1, x2, y2 = bbox

        # 计算边界框中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 修复：将坐标转换为浮点数，并确保contour是正确格式
        contour = self.roi.astype(np.float32)
        point = (float(center_x), float(center_y))

        # 检查中心点是否在违停区域内
        return cv2.pointPolygonTest(contour, point, False) >= 0
