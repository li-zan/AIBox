from typing import Any, Dict
import numpy as np

from app.modules.base import BaseModule
from . import register_module, get_registered_modules


class SleepyStub(BaseModule):
	"""A lightweight stub that returns dummy results without delay."""

	def __init__(self, name: str, delay_ms: int = 0, **kwargs: Any) -> None:
		super().__init__(name, kwargs.get('config'))
		self.delay_ms = delay_ms

	def infer(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
		if not self.loaded:
			raise RuntimeError(f"Module {self.name} not loaded")
		# No sleep - stubs should be instant
		# Return empty results (no dummy bboxes to avoid clutter)
		return {
			"module": self.name,
			"bboxes": []
		}

	def draw(self, frame_bgr: np.ndarray, results: Dict[str, Any]) -> None:
		for box in results.get("bboxes", []):
			x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
			try:
				import cv2
				cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
				cv2.putText(frame_bgr, self.name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
			except Exception:
				pass


# Define 20 stub names mapping to SleepyStub with different delays
MODULE_NAMES = [
	"fire",
	"plate",
	"intrusion",
	"helmet",
	"smoke",
	"fall",
	"fighting",
	"loitering",
	"crowd",
	"vehicle",
	"person",
	"pet",
	"face",
	"mask",
	"phone",
	"defect",
	"spill",
	"door_open",
	"run",
	"crossline",
]


for idx, name in enumerate(MODULE_NAMES):
	cls = type(f"{name.title()}Module", (SleepyStub,), {})
	# Only register if not already registered (e.g., fire module from fire_yolo.py)
	if name not in get_registered_modules():
		register_module(name, cls)
