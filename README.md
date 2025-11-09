# Jetson Orin Nano Smart Surveillance GUI

A PySide6 GUI application for connecting to Hikvision cameras/NVR via RTSP, selecting up to 20 AI modules, and running inference on RTSP or local video. This is a runnable scaffold with module stubs; replace stubs with real models (TensorRT/DeepStream) as needed.

## Features
- Read `config.json` to discover cameras and default module states
- RTSP and local video input
- 20 pluggable AI module stubs (fire, plate, intrusion, etc.)
- Start/Stop preview, per-frame processing hooks

## Requirements
- Jetson Orin Nano with JetPack (GStreamer preinstalled)
- Python 3.10

Install Python deps:

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m app.main --config config.json
```

## Jetson Notes
- For RTSP hardware decode, ensure GStreamer is available and OpenCV is built with GStreamer. If `cv2.VideoCapture(..., cv2.CAP_GSTREAMER)` fails, install OpenCV with GStreamer support or use system OpenCV.
- Test a camera URL first with `gst-launch-1.0`:

```bash
gst-launch-1.0 rtspsrc location="rtsp://user:pass@IP:554/h264/ch1/sub/av_stream" latency=100 ! rtph264depay ! h264parse ! nvv4l2decoder ! nveglglessink sync=false
```

## Customize Modules
- Replace stubs in `app/modules/stubs.py` with real implementations (TensorRT/DeepStream). Each module must implement `BaseModule` interface.
