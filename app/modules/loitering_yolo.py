from typing import Any, Dict, Iterator

import numpy as np
import cv2
from ultralytics.engine.results import Results
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from collections import defaultdict, deque
from scipy.spatial import distance

from app.modules.base import BaseModule


class LoiteringYoloModule(BaseModule):
    """徘徊检测"""

    def __init__(self, name: str, config: Dict[str, Any] | None = None) -> None:
        super().__init__(name, config)
        self.model = None
        self.detector = None
        self.conf_threshold = float(self.config.get('threshold', self.config.get('conf_threshold', 0.25)))
        self.total_distance_threshold = float(self.config.get('total_distance_threshold', 200))
        self.stay_time_threshold = float(self.config.get('stay_time_threshold', 10))
        self.initial_distance_threshold = float(self.config.get('initial_distance_threshold', 100))
        self.max_age = int(self.config.get('max_age', 70))

    def load(self) -> None:
        from ultralytics import YOLO
        model_path = self.config.get('model')
        print(f"Loading loitering model: {model_path}")
        self.model = YOLO(model_path)
        self.detector = self.LoiteringDetector(
            self.model,
            conf_threshold=self.conf_threshold,
            total_distance_threshold=self.total_distance_threshold,
            stay_time_threshold=self.stay_time_threshold,
            initial_distance_threshold=self.initial_distance_threshold,
            max_age=self.max_age
        )
        self.loaded = True
        print(f"loitering model ready")

    def unload(self) -> None:
        del self.model
        self.model = None
        super().unload()

    def process(self, frame: np.ndarray, frame_bgr: np.ndarray) -> None:
        if not self.loaded or self.model is None:
            raise RuntimeError("HelmetYoloModule not loaded")
        _, total, loitering = self.detector.process_and_draw(frame, frame_bgr)
        # 显示全局统计
        cv2.putText(frame_bgr, f"Loitering: {loitering}/{total}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    class LoiteringDetector:
        def __init__(self, model, **kwargs):
            self.model = model

            # 1. DeepSORT 参数调优 (防抖动、防ID切换)
            self.tracker = DeepSort(
                max_age=kwargs['max_age'],  # 目标丢失后的保留帧数
                n_init=3,  # 确认目标所需的帧数
                max_cosine_distance=0.3,  # 外观匹配阈值
                max_iou_distance=0.9,  # IOU阈值 (0.9允许预测框和检测框分离较远，防止急转弯跟丢)
                nn_budget=100,  # 特征库大小
                embedder="mobilenet",
                embedder_gpu=True,
                bgr=True,
                polygon=False
            )

            # 2. 徘徊检测参数
            self.config = {
                'conf_threshold': kwargs['conf_threshold'],  # 置信度阈值
                'total_distance_threshold': kwargs['total_distance_threshold'],  # 判定阈值：移动距离
                'stay_time_threshold': kwargs['stay_time_threshold'],  # 判定阈值：停留时间
                'initial_distance_threshold': kwargs['initial_distance_threshold']  # 判定阈值：活动半径
            }

            # 3. 状态存储
            self.track_history = defaultdict(lambda: {
                'positions': deque(maxlen=300),
                'timestamps': deque(maxlen=300),
                'initial_position': None,
                'first_seen': None,
                'total_distance': 0,
                'last_seen': None,
                'is_loitering': False,
                'bbox_history': deque(maxlen=20),
                'confidence': 1.0,
                'active': False,
                'features': deque(maxlen=10),
                'last_features': None,
                'reappear_count': 0,
                'loitering_start_time': None,
                'confirmed_frames': 0,
                'permanent_id': None,
                'status_label': "Init",
                'trajectory_efficiency': 1.0,
                'birth_time': time.time()
            })

            self.permanent_id_map = {}
            self.next_permanent_id = 1
            self.recently_dead_tracks = {}
            self.max_lost_time = 10.0
            self.frame_count = 0
            self.frame_shape = None

        def calculate_center(self, box):
            return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        def extract_appearance_features(self, frame, box):
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1: return None
            target_roi = frame[y1:y2, x1:x2]
            if target_roi.size == 0: return None

            h_roi = y2 - y1
            mid = h_roi // 2
            top = target_roi[:mid, :]
            bottom = target_roi[mid:, :]

            def get_hist(img):
                if img.size == 0: return np.zeros(256)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
                return cv2.normalize(hist, hist).flatten()

            return {'top': get_hist(top), 'bottom': get_hist(bottom)}

        def calculate_feature_similarity(self, feat1, feat2):
            if feat1 is None or feat2 is None: return 0
            s1 = cv2.compareHist(feat1['top'], feat2['top'], cv2.HISTCMP_CORREL)
            s2 = cv2.compareHist(feat1['bottom'], feat2['bottom'], cv2.HISTCMP_CORREL)
            return 0.5 * s1 + 0.5 * s2

        def get_permanent_id(self, track_id):
            if track_id not in self.permanent_id_map:
                self.permanent_id_map[track_id] = self.next_permanent_id
                self.next_permanent_id += 1
            return self.permanent_id_map[track_id]

        def yolo_to_deepsort_format(self, detections, frame):
            deepsort_detections = []
            for det in detections:
                bbox = det['box']
                confidence = det['confidence']
                x1, y1, x2, y2 = bbox
                det['features'] = self.extract_appearance_features(frame, bbox)
                deepsort_detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, 'person'))
            return deepsort_detections

        def update_track(self, track_id, position, bbox, confidence, features):
            permanent_id = self.get_permanent_id(track_id)
            data = self.track_history[permanent_id]
            current_time = time.time()

            if data['initial_position'] is None:
                data['initial_position'] = position
                data['first_seen'] = current_time
                data['permanent_id'] = permanent_id
                data['birth_time'] = current_time

            # 去抖动逻辑
            if len(data['positions']) > 0:
                last_position = data['positions'][-1]
                move_distance = distance.euclidean(position, last_position)
                if move_distance > 1.5:
                    data['total_distance'] += move_distance

            data['positions'].append(position)
            data['timestamps'].append(current_time)
            data['bbox_history'].append(bbox)
            data['last_seen'] = current_time
            data['confidence'] = confidence
            data['active'] = True
            data['confirmed_frames'] += 1

            if features is not None:
                data['features'].append(features)
                data['last_features'] = features

            if permanent_id in self.recently_dead_tracks:
                del self.recently_dead_tracks[permanent_id]

        def check_loitering(self, permanent_id):
            data = self.track_history[permanent_id]
            if data['is_loitering']: return True
            if len(data['positions']) < 30:
                data['status_label'] = "Init"
                return False

            current_position = data['positions'][-1]
            initial_position = data['initial_position']

            # 1. 边缘豁免 (出画不误判)
            if self.frame_shape is not None:
                h, w = self.frame_shape[:2]
                cx, cy = current_position
                margin = 40
                is_near_border = (cx < margin) or (cx > w - margin) or (cy < margin) or (cy > h - margin)
                if is_near_border and data['total_distance'] > 100:
                    data['status_label'] = "Exiting"
                    return False

            dist_to_init = distance.euclidean(current_position, initial_position)
            time_elapsed = data['last_seen'] - data['first_seen']

            # 2. 轨迹效率 (慢走不误判)
            eff = 0.0
            if data['total_distance'] > 10:
                eff = dist_to_init / data['total_distance']
            data['trajectory_efficiency'] = eff
            is_straight = eff > 0.6

            cond1 = data['total_distance'] > self.config['total_distance_threshold']
            cond2 = time_elapsed > self.config['stay_time_threshold']
            cond3 = dist_to_init < self.config['initial_distance_threshold']

            is_loitering = cond1 and cond2 and cond3 and (not is_straight)

            if dist_to_init > 300: is_loitering = False

            if is_loitering and not data['is_loitering']:
                data['is_loitering'] = True
                data['loitering_start_time'] = time.time()
                data['status_label'] = "LOITERING"
                print(f"目标开始徘徊: ID {permanent_id}")

            return is_loitering

        def cleanup_old_tracks(self, max_age=5):
            current_time = time.time()
            ids_to_remove = []

            edge_margin = 50
            h, w = 0, 0
            if self.frame_shape is not None: h, w = self.frame_shape[:2]

            for permanent_id, data in self.track_history.items():
                if (current_time - data['last_seen']) > max_age and data['active']:
                    # 检查是否边缘消失
                    last_pos = data['positions'][-1] if data['positions'] else (0, 0)
                    lx, ly = last_pos
                    is_edge_lost = False
                    if w > 0:
                        if (lx < edge_margin or lx > w - edge_margin or ly < edge_margin or ly > h - edge_margin):
                            is_edge_lost = True

                    # 存入停尸房
                    self.recently_dead_tracks[permanent_id] = {
                        'data': data,
                        'is_edge_lost': is_edge_lost,
                        'death_time': current_time,
                        'last_pos': last_pos
                    }
                    ids_to_remove.append(permanent_id)
                elif (current_time - data['last_seen']) > max_age:
                    ids_to_remove.append(permanent_id)

            # 清理停尸房
            dead_to_del = [k for k, v in self.recently_dead_tracks.items() if
                           current_time - v['death_time'] > self.max_lost_time]
            for k in dead_to_del: del self.recently_dead_tracks[k]

            for pid in ids_to_remove:
                self.track_history[pid]['active'] = False

        def stitch_broken_tracks(self, current_time):
            """轨迹缝合"""
            young_pids = []
            for pid, data in self.track_history.items():
                if data['active'] and (current_time - data['birth_time'] < 1.0):
                    young_pids.append(pid)

            if not young_pids or not self.recently_dead_tracks: return

            for new_pid in young_pids:
                new_data = self.track_history[new_pid]
                new_pos = new_data['positions'][0]
                new_feat = new_data['last_features']
                best_old_pid, best_score = None, 0

                for old_pid, old_record in self.recently_dead_tracks.items():
                    if old_record['is_edge_lost']: continue
                    dist = distance.euclidean(new_pos, old_record['last_pos'])
                    if dist > 150: continue

                    sim = 0
                    if new_feat is not None and old_record['data']['last_features'] is not None:
                        sim = self.calculate_feature_similarity(new_feat, old_record['data']['last_features'])

                    if sim > 0.6 and sim > best_score:
                        best_score = sim
                        best_old_pid = old_pid

                if best_old_pid:
                    ds_id_to_remap = None
                    for ds, p in self.permanent_id_map.items():
                        if p == new_pid: ds_id_to_remap = ds; break

                    if ds_id_to_remap is not None:
                        self.permanent_id_map[ds_id_to_remap] = best_old_pid

                    old_data = self.track_history[best_old_pid]
                    old_data['positions'].extend(new_data['positions'])
                    old_data['timestamps'].extend(new_data['timestamps'])
                    old_data['bbox_history'].extend(new_data['bbox_history'])
                    old_data['last_seen'] = new_data['last_seen']
                    old_data['active'] = True
                    old_data['last_features'] = new_data['last_features']
                    old_data['total_distance'] += new_data['total_distance']
                    old_data['status_label'] = "Stitched"

                    del self.track_history[new_pid]
                    del self.recently_dead_tracks[best_old_pid]

        def process_and_draw(self, frame, frame_bgr):
            """核心处理并返回画好的帧"""
            self.frame_count += 1
            if self.frame_shape is None: self.frame_shape = frame.shape

            # 1. 状态重置
            for pid in self.track_history: self.track_history[pid]['active'] = False

            # 2. 推理
            results = self.model(frame, conf=self.config['conf_threshold'], verbose=False)
            detections = []
            if len(results) > 0 and results[0].boxes:
                for box, conf, cls in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy(),
                                          results[0].boxes.cls.cpu().numpy()):
                    if cls == 0 and conf > 0.5:
                        detections.append({'box': box, 'center': self.calculate_center(box), 'confidence': conf})

            # 3. 追踪
            ds_dets = self.yolo_to_deepsort_format(detections, frame)
            try:
                tracks = self.tracker.update_tracks(ds_dets, frame=frame)
                matched_det_ids = set()

                for track in tracks:
                    if not track.is_confirmed(): continue
                    det_features = None
                    track_pos = self.calculate_center(track.to_ltrb())
                    for det in detections:
                        if distance.euclidean(track_pos, det['center']) < 30:
                            det_features = det.get('features')
                            matched_det_ids.add(id(det))
                            break
                    self.update_track(track.track_id, track_pos, track.to_ltrb(),
                                      track.det_conf if hasattr(track, 'det_conf') else 0.5,
                                      det_features)
                    pid = self.get_permanent_id(track.track_id)
                    self.check_loitering(pid)

            except Exception as e:
                print(f"DeepSORT Error: {e}")

            # 4. 后处理
            self.cleanup_old_tracks()
            self.stitch_broken_tracks(time.time())

            # 5. 绘图与统计
            active_count, loitering_count = 0, 0
            for pid, data in self.track_history.items():
                if not data['active']: continue
                active_count += 1
                if data['is_loitering']: loitering_count += 1

                bbox = data['bbox_history'][-1]
                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 0, 255) if data['is_loitering'] else (0, 255, 0)

                label = f"ID:{pid} {data.get('status_label', '')}"
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                pts = list(data['positions'])
                for i in range(1, len(pts)):
                    cv2.line(frame_bgr, (int(pts[i - 1][0]), int(pts[i - 1][1])), (int(pts[i][0]), int(pts[i][1])),
                             color,
                             2)

                info = f"D:{data['total_distance']:.0f} T:{(time.time() - data['first_seen']):.0f}s"
                cv2.putText(frame_bgr, info, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            return frame_bgr, active_count, loitering_count
