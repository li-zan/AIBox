from typing import Any, Dict, Optional
import os

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QFileDialog, QScrollArea,
    QCheckBox, QMessageBox
)

from app.video import VideoWorker
from app.modules import get_registered_modules
from app.modules import BaseModule

# Module name translations (English -> Chinese)
MODULE_NAME_CN = {
    "fire": "火焰识别",
    "plate": "车牌识别",
    "intrusion": "人员闯入",
    "helmet": "安全帽检测",
    "smoke": "烟雾检测",
    "fall": "跌倒检测",
    "fighting": "打架检测",
    "loitering": "徘徊检测",
    "crowd": "人群聚集",
    "vehicle": "车辆型号检测",
    "person": "车辆数量计数",
    "pet": "抽烟检测",
    "face": "机动车违停检测",
    "mask": "口罩检测",
    "phone": "打电话检测",
    "defect": "电瓶车进电梯检测",
    "spill": "人员疲劳检测",
    "door_open": "电瓶车违停检测",
    "run": "奔跑检测",
    "crossline": "人员计数",
}


class VideoView(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText("未加载视频")

    def show_frame(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))


class MainWindow(QMainWindow):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.setWindowTitle("Jetson 智能监控系统")
        self.resize(1200, 800)

        self.config = config
        self.active_modules: Dict[str, BaseModule] = {}
        self.video_worker: Optional[VideoWorker] = None
        self.current_source: Optional[str] = None
        self.current_is_rtsp: bool = True

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: sources list
        left = QVBoxLayout()
        self.sources_list = QListWidget()
        for s in self.config.get('sources', []):
            item = QListWidgetItem(f"{s.get('name')} - {s.get('ip')}")
            item.setData(Qt.UserRole, s)
            self.sources_list.addItem(item)
        left.addWidget(QLabel("视频源"))
        left.addWidget(self.sources_list)
        self.sources_list.setCurrentRow(0)
        self.btn_start = QPushButton("开始")
        self.btn_stop = QPushButton("停止")
        self.btn_open_file = QPushButton("打开本地视频")
        btns = QHBoxLayout()
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_open_file)
        left.addLayout(btns)

        root.addLayout(left, 3)

        # Center: video view
        self.view = VideoView()
        root.addWidget(self.view, 7)

        # Right: modules panel
        right = QVBoxLayout()
        right.addWidget(QLabel("智能模块"))

        self.module_checks: Dict[str, QCheckBox] = {}
        mod_container = QWidget()
        mod_layout = QVBoxLayout(mod_container)
        for name in get_registered_modules().keys():
            # Use Chinese name if available, otherwise use English name
            display_name = MODULE_NAME_CN.get(name, name)
            chk = QCheckBox(display_name)
            if self.config.get('modules', {}).get(name, {}).get('enabled', False):
                chk.setChecked(True)
            self.module_checks[name] = chk
            mod_layout.addWidget(chk)
        mod_layout.addStretch(1)  # 调空白空间来控制小组件位置

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(mod_container)
        right.addWidget(scroll)

        root.addLayout(right, 3)

        # Signals
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_open_file.clicked.connect(self.on_open_file)

        # Timer for UI updates if needed
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(1000)
        self._fps_timer.timeout.connect(lambda: None)
        self._fps_timer.start()

    def on_start(self) -> None:
        item = self.sources_list.currentItem()
        if not item:
            QMessageBox.warning(self, "未选择视频源", "请从列表中选择一个视频源。")
            return
        src = item.data(Qt.UserRole)
        rtsp = src.get('rtsp')
        if not rtsp:
            QMessageBox.warning(self, "无效视频源", "所选视频源没有RTSP地址。")
            return
        self.current_source = rtsp
        self.current_is_rtsp = True
        self._start_stream()

    def on_stop(self) -> None:
        if self.video_worker:
            self.video_worker.stop()
            self.video_worker = None
        self.view.setText("已停止")

    def on_open_file(self) -> None:
        last_dir = self.config.get('local_video', {}).get('last_open_dir') or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "打开视频", last_dir, "视频文件 (*.mp4 *.avi *.mkv)")
        if not path:
            return
        self.current_source = path
        self.current_is_rtsp = False
        self._start_stream()

    def _start_stream(self) -> None:
        if self.video_worker:
            self.video_worker.stop()
        self._setup_modules()
        if not self.active_modules:
            return
        self.video_worker = VideoWorker(self.current_source or '', self.current_is_rtsp, self._on_frame)
        self.video_worker.start()
        self.view.setText("正在连接...")

    def _setup_modules(self) -> None:
        # unload existing
        for m in self.active_modules.values():
            try:
                m.unload()
            except Exception:
                pass
        self.active_modules.clear()

        # load selected
        registry = get_registered_modules()
        for name, chk in self.module_checks.items():
            if chk.isChecked() and name in registry:
                cls = registry[name]
                cfg = self.config.get('modules', {}).get(name, {})
                mod = cls(name=name, config=cfg)
                mod.load()
                self.active_modules[name] = mod
                # Only print for real modules (not stubs)
                if cls.__name__ != 'type':
                    print(f"✓ Loaded {name}: {cls.__name__}")
        if self.active_modules:
            print(f"Active modules: {list(self.active_modules.keys())}")
        else:
            print("No active modules.")
            self.view.setText("未启用任何检测模块")

    def _on_frame(self, frame_bgr: Optional[np.ndarray]) -> None:
        if frame_bgr is None:
            self.view.setText("无法打开视频源")
            return
        # run modules in sequence (demo); real impl can batch/async
        results: Dict[str, Any] = {}
        for name, mod in self.active_modules.items():
            try:
                # res = mod.infer(frame_bgr)
                # results[name] = res
                # # Only print if detections found
                # if res.get('bboxes'):
                #     print(f"[{name}] Found {len(res['bboxes'])} detections")
                # mod.draw(frame_bgr, res)
                mod.process(frame_bgr)
            except Exception as e:
                print(f"Error in module {name}: {e}")
                continue
        self.view.show_frame(frame_bgr)
