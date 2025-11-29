from typing import Any, Dict, Iterator

import numpy as np
import cv2
from ultralytics.engine.results import Results

from app.modules.base import BaseModule
from app.utils.plotting import draw_chinese_label_inplace
from app.nn.lprnetv2 import LPRv2Predictor


class PlateYoloModule(BaseModule):
    """车牌识别"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model_det = None
        self.model_rec = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.4)))
        self.border_threshold = int(self.config.get('border_threshold', 5))

    def load(self) -> None:
        from ultralytics import YOLO
        model_det_path = self.config.get('model_det')
        model_rec_path = self.config.get('model_rec')
        print(f"Loading plate_det model: {model_det_path}")
        print(f"Loading plate_rec model: {model_rec_path}")
        self.model_det = YOLO(model_det_path)
        self.model_rec = LPRv2Predictor(model_rec_path, cuda=True)
        self.loaded = True
        print(f"plate model ready")

    def unload(self) -> None:
        del self.model_det
        del self.model_rec
        self.model_det = None
        self.model_rec = None
        super().unload()

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model_det is None or self.model_rec is None:
            raise RuntimeError("PlateYoloModule not loaded")
        # Inference
        results: Iterator[Results] = self.model_det(frame, conf=self.conf_threshold)
        height, width = frame_bgr.shape[:2]

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0)

                if (
                        x1 <= self.border_threshold or
                        y1 <= self.border_threshold or
                        x2 >= width - self.border_threshold or
                        y2 >= height - self.border_threshold
                ):
                    # 说明车牌正在进入 or 正在离开 or 被切一部分，跳过
                    continue

                # 裁剪车牌区域
                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                # 识别车牌号
                plate_str = self.model_rec(plate_crop)

                # 绘制边框和文字
                # label = f"{plate_str} ({conf:.2f})"
                label = f"{plate_str}"
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                draw_chinese_label_inplace(frame_bgr, label, x1, y1 - 25, 0.7, 2, color)
