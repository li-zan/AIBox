from typing import Callable, Optional
import threading
import time

import cv2


def build_gst_caps(rtsp_url: str) -> str:
    return (
        f'rtspsrc location="{rtsp_url}" latency=100 ! '
        'rtph264depay ! h264parse ! nvv4l2decoder ! '
        'nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink drop=true sync=false'
    )


class VideoWorker:
    """Video capture worker that reads frames in a thread and calls a callback."""

    def __init__(self, source: str, is_rtsp: bool, on_frame: Callable[[Optional[object]], None]) -> None:
        self.source = source
        self.is_rtsp = is_rtsp
        self.on_frame = on_frame
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _open(self) -> bool:
        if self.is_rtsp:
            caps = build_gst_caps(self.source)
            self._cap = cv2.VideoCapture(caps, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(self.source)
        return bool(self._cap and self._cap.isOpened())

    def _run(self) -> None:
        if not self._open():
            self.on_frame(None)
            return
        try:
            while not self._stop_event.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                self.on_frame(frame)
        finally:
            if self._cap:
                self._cap.release()
                self._cap = None
