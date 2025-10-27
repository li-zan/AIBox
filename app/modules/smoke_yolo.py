from app.modules.fire_yolo import FireYoloModule


class SmokeYoloModule(FireYoloModule):
	"""Smoke detector using the same fire_detector.pt model.
	
	This is just an alias that filters smoke detections from the fire model.
	"""
	pass

