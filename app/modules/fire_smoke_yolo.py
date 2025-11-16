from typing import Any, Dict

import numpy as np

from app.modules.base import BaseModule


class FireYoloModule(BaseModule):
    """YOLOv8-based fire/smoke detector.

    Expected config:
    - model: path to .pt model file (e.g., fire_detector.pt)
    - classes: list of class names, default ['fire', 'smoke']
    - conf_threshold: float (default 0.3)
    """

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.class_names = self.config.get('classes', ['fire', 'smoke'])
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.3)))

    def load(self) -> None:
        from ultralytics import YOLO
        model_path = self.config.get('model', 'fire_detector.pt')
        print(f"Loading fire model: {model_path}")
        self.model = YOLO(model_path)
        self.loaded = True
        print(f"Fire model ready")

    def unload(self) -> None:
        del self.model
        self.model = None
        super().unload()

    def infer(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        if not self.loaded or self.model is None:
            raise RuntimeError("FireYoloModule not loaded")
        # Inference
        results = self.model(frame_bgr, stream=True)
        bboxes = []
        for r in results:
            boxes = getattr(r, 'boxes', [])
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # Process both fire (cls==0) and smoke (cls==1)
                if conf >= self.conf_threshold and 0 <= cls < len(self.class_names):
                    class_name = self.class_names[cls]
                    # If this is fire module, only show fire; if smoke module, only show smoke
                    if (self.name == 'fire' and class_name == 'fire') or \
                            (self.name == 'smoke' and class_name == 'smoke') or \
                            (self.name not in ['fire', 'smoke']):  # For generic use
                        bboxes.append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "score": conf,
                            "class": class_name,
                        })
        return {"module": self.name, "bboxes": bboxes}

    def draw(self, frame_bgr: np.ndarray, results: Dict[str, Any]) -> None:
        try:
            import cv2
            import cvzone
        except Exception:
            return
        for box in results.get("bboxes", []):
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            w, h = x2 - x1, y2 - y1
            class_name = box.get('class', '')
            score = round(box.get('score', 0.0), 2)
            label = f"{class_name} {score}"

            # Different colors and styles for fire vs smoke
            if class_name == 'fire':
                color = (0, 0, 255)  # Red for fire
                corner_length = 3
                thickness = 2
            else:  # smoke
                color = (0, 255, 255)  # Yellow for smoke
                corner_length = 30
                thickness = 5

            cvzone.putTextRect(frame_bgr, label, (max(0, x1 + 10), max(35, y1 - 10)),
                               1, 1, (0, 255, 255), colorR=(0, 0, 0))
            cvzone.cornerRect(frame_bgr, (x1, y1, w, h), l=corner_length, colorR=color, t=thickness)

    def process(self, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model is None:
            raise RuntimeError("FireYoloModule not loaded")
        # Inference
        results = self.model(frame_bgr, conf=self.conf_threshold)
        bboxes = []
        for r in results:
            boxes = getattr(r, 'boxes', [])
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # Process both fire (cls==0) and smoke (cls==1)
                if conf >= self.conf_threshold and 0 <= cls < len(self.class_names):
                    class_name = self.class_names[cls]
                    # If this is fire module, only show fire; if smoke module, only show smoke
                    if (self.name == 'fire' and class_name == 'fire') or \
                            (self.name == 'smoke' and class_name == 'smoke') or \
                            (self.name not in ['fire', 'smoke']):  # For generic use
                        bboxes.append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "score": conf,
                            "class": class_name,
                        })
        results = {"module": self.name, "bboxes": bboxes}

        try:
            import cv2
            import cvzone
        except Exception:
            return
        for box in results.get("bboxes", []):
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            w, h = x2 - x1, y2 - y1
            class_name = box.get('class', '')
            score = round(box.get('score', 0.0), 2)
            label = f"{class_name} {score}"

            # Different colors and styles for fire vs smoke
            if class_name == 'fire':
                color = (0, 0, 255)  # Red for fire
                corner_length = 3
                thickness = 2
            else:  # smoke
                color = (0, 255, 255)  # Yellow for smoke
                corner_length = 30
                thickness = 5

            cvzone.putTextRect(frame_bgr, label, (max(0, x1 + 10), max(35, y1 - 10)),
                               1, 1, (0, 255, 255), colorR=(0, 0, 0))
            cvzone.cornerRect(frame_bgr, (x1, y1, w, h), l=corner_length, colorR=color, t=thickness)
