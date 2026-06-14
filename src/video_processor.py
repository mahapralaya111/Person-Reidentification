"""
抗恶劣环境行人重识别系统 - 视频流处理模块
支持实时视频流处理、多目标跟踪、ReID检索、带播放控制

使用方法：
    # 实时摄像头
    python video_processor.py --source webcam --model_path checkpoints/best_model.pth

    # 视频文件
    python video_processor.py --source video.mp4 --model_path checkpoints/best_model.pth

    # 带底库识别
    python video_processor.py --source video.mp4 --gallery rescue_gallery --model_path checkpoints/best_model.pth

播放控制：
    空格键: 暂停/播放
    左方向键: 后退5秒
    右方向键: 前进5秒
    上方向键: 加速(+0.25x)
    下方向键: 减速(-0.25x)
    R键: 重新开始
    Q键/ESC: 退出
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import threading
import json

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image  # 只用于ReID特征提取的图像转换


# ============== 配置 ==============
class Config:
    # 检测器配置
    DETECTOR_CONFIDENCE = 0.35  # 检测置信度
    DETECTOR_NMS = 0.7  # IoU>0.7才合并，避免误合并靠近的人

    # ReID配置
    FEATURE_DIM = 512
    MATCHING_THRESHOLD = 0.75   # 相似度>=0.75才显示人名（防止误报）
    MIN_GALLERY_SIM = 0.01      # 最低底库匹配相似度

    # 跟踪配置
    MAX_AGE = 30  # 最大丢失帧数
    MIN_HITS = 1  # 降低门槛，立刻显示检测框
    IOU_THRESHOLD = 0.3  # IoU阈值

    # 轨迹配置
    TRAJECTORY_LENGTH = 30  # 轨迹保存帧数

    # 显示配置
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720
    WINDOW_NAME = "ReID Video Player"


# ============== 目标检测 ==============
class PersonDetector:
    """人体检测器（使用YOLOv8）"""

    def __init__(self, model_path=None):
        self.use_yolo = True
        import os as _os
        # 优先使用传入的路径，否则在项目根目录查找 yolov8n.pt
        if model_path is None:
            _script_dir = _os.path.dirname(_os.path.abspath(__file__))
            _project_root = _os.path.dirname(_script_dir)
            _model_default = _os.path.join(_project_root, 'yolov8n.pt')
            model_path = _model_default if _os.path.exists(_model_default) else 'yolov8n.pt'
        print(f"[YOLO] 加载YOLOv8模型: {model_path}")
        if not _os.path.exists(model_path):
            print(f"[YOLO] 模型文件不存在: {model_path}")
            self.use_yolo = False
            self.yolo_model = None
            return
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(model_path)
            print("[YOLO] YOLOv8加载完成")
        except Exception as e:
            print(f"[YOLO] 加载失败: {e}")
            print("[YOLO] 请运行: pip install ultralytics")
            self.use_yolo = False
            self.yolo_model = None

    def detect(self, frame):
        """检测人体区域（含NMS合并重叠框）"""
        if not self.use_yolo:
            return []
        results = self.yolo_model(frame, verbose=False)
        person_boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if int(box.cls) == 0:  # person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    if conf > Config.DETECTOR_CONFIDENCE:
                        person_boxes.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf)
                        })
        # 过滤不合理尺寸/比例的框（非人物误检）
        h, w = frame.shape[:2]
        frame_area = h * w
        filtered = []
        for pb in person_boxes:
            x1, y1, x2, y2 = pb['bbox']
            bw = x2 - x1
            bh = y2 - y1
            if bw < 20 or bh < 40:  # 太小不可能是人
                continue
            if bw > w * 0.9 or bh > h * 0.95:  # 几乎全屏不可能是人
                continue
            area = bw * bh
            if area < frame_area * 0.005 or area > frame_area * 0.8:  # 面积过滤
                continue
            aspect_ratio = bh / max(bw, 1)
            if aspect_ratio < 0.4 or aspect_ratio > 8.0:  # 长宽比过滤（排除方形物体）
                continue
            filtered.append(pb)
        person_boxes = filtered
        
        # NMS：合并高度重叠的框（同一人的多个检测）
        if len(person_boxes) > 1:
            person_boxes = self._nms_merge(person_boxes)
        return person_boxes

    def _nms_merge(self, boxes, iou_threshold=0.5):
        """合并高度重叠的检测框"""
        if len(boxes) <= 1:
            return boxes
        # 按置信度排序
        sorted_boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
        merged = []
        for box in sorted_boxes:
            b1 = box['bbox']
            keep = True
            for m in merged:
                b2 = m['bbox']
                # 计算IoU
                x1 = max(b1[0], b2[0])
                y1 = max(b1[1], b2[1])
                x2 = min(b1[2], b2[2])
                y2 = min(b1[3], b2[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter / (area1 + area2 - inter + 1e-6)
                if iou > iou_threshold:
                    keep = False
                    break
            if keep:
                merged.append(box)
        return merged


# ============== 卡尔曼跟踪器 ==============
class KalmanBoxTracker:
    """基于卡尔曼滤波的跟踪器"""

    count = 0

    def __init__(self, bbox):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32) * 0.01
        self.kalman.measurementNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * 0.1

        # 初始化状态
        self.kalman.statePre = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history = deque(maxlen=Config.TRAJECTORY_LENGTH)
        self.history.append(bbox)

        self.hits = 0
        self.hit_streak = 0

    def update(self, bbox):
        """更新跟踪"""
        self.time_since_update = 0
        self.history.append(bbox)
        self.hits += 1
        self.hit_streak += 1

        self.kalman.correct(np.array([[np.float32(bbox[0])],
                                       [np.float32(bbox[1])]]))

    def predict(self):
        """预测下一位置"""
        prediction = self.kalman.predict()
        x, y = prediction[0, 0], prediction[1, 0]

        # 添加速度偏移
        if len(self.history) >= 2:
            vx = self.history[-1][0] - self.history[-2][0]
            vy = self.history[-1][1] - self.history[-2][1]
            x += vx * 0.5
            y += vy * 0.5

        return [x, y, x + 50, y + 100]  # 估计宽高

    def get_state(self):
        """获取当前状态"""
        return self.history[-1]


# ============== 多目标跟踪器 ==============
class MultiObjectTracker:
    """多目标跟踪器"""

    def __init__(self):
        self.trackers = []
        self.next_id = 1

    def update(self, detections):
        """更新跟踪器"""
        # 预测所有跟踪器位置
        for tracker in self.trackers:
            tracker.predict()

        # 简单匹配：IoU匹配
        matched, unmatched_dets, unmatched_trks = self._match(detections)

        # 更新匹配的跟踪器
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(detections[det_idx]['bbox'])

        # 创建新的跟踪器
        for det_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[det_idx]['bbox']))

        # 删除丢失的跟踪器
        self.trackers = [t for t in self.trackers if t.time_since_update < Config.MAX_AGE]

        # 更新time_since_update
        for t in self.trackers:
            t.time_since_update += 1

        # 合并重叠的track（同一个人的多个track合并为一个）
        self._merge_overlapping_tracks()

        return self._get_active_tracks()

    def _merge_overlapping_tracks(self):
        """合并高度重叠的track（同一人的多个track）"""
        if len(self.trackers) < 2:
            return
        to_remove = set()
        for i in range(len(self.trackers)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(self.trackers)):
                if j in to_remove:
                    continue
                bbox_i = self.trackers[i].get_state()
                bbox_j = self.trackers[j].get_state()
                iou = self._compute_iou(bbox_i, bbox_j)
                if iou > 0.85:  # 只有几乎完全重合的track才合并（同一人的重复track）
                    # 保留hits更多的track，丢弃另一个
                    if self.trackers[i].hits >= self.trackers[j].hits:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
        if to_remove:
            self.trackers = [t for idx, t in enumerate(self.trackers) if idx not in to_remove]

    def _match(self, detections):
        """IoU匹配"""
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []

        # 计算IoU矩阵
        iou_matrix = np.zeros((len(detections), len(self.trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = self._compute_iou(det['bbox'], trk.get_state())

        # 匈牙利算法匹配
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(self.trackers)))

        for d in range(len(detections)):
            for t in range(len(self.trackers)):
                if d in unmatched_detections and t in unmatched_trackers and iou_matrix[d, t] > Config.IOU_THRESHOLD:
                    matched_indices.append([d, t])
                    unmatched_detections.remove(d)
                    unmatched_trackers.remove(t)

        return matched_indices, unmatched_detections, unmatched_trackers

    def _compute_iou(self, box1, box2):
        """计算IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-6)

    def _get_active_tracks(self):
        """获取活跃跟踪（降低门槛，立刻显示）"""
        active = []
        for trk in self.trackers:
            # 降低门槛，只要检测到就显示
            if trk.hits >= Config.MIN_HITS:
                active.append({
                    'id': trk.id,
                    'bbox': trk.get_state(),
                    'trajectory': list(trk.history),
                    'features': []
                })
        return active


# ============== ReID检索 ==============
class ReIDRetriever:
    """ReID特征提取和检索"""

    def __init__(self, model, device, gallery_features=None, gallery_ids=None, gallery_paths=None):
        self.model = model
        self.device = device
        self.feature_db = {}  # 视频内跟踪ID -> 特征列表
        self.gallery_features = gallery_features   # 底库特征矩阵 (N, D)
        self.gallery_ids = gallery_ids             # 底库ID列表 [str]
        self.gallery_paths = gallery_paths or []   # 底库图片路径

        # 图像转换
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_feature(self, frame, bbox):
        """提取单个人体特征（含预处理增强）"""
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

        person = frame[y1:y2, x1:x2]
        if person.size == 0:
            return None

        # 图像增强：CLAHE自适应直方图均衡（提升低对比度帧的特征质量）
        try:
            lab = cv2.cvtColor(person, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        except Exception:
            enhanced = person  # 增强失败时使用原图

        person_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        person_tensor = self.transform(person_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(person_tensor)
            # 模型可能返回 (logits, features) 或直接返回 features
            if isinstance(outputs, tuple):
                if len(outputs) >= 2:
                    outputs = outputs[1]  # 取特征向量，不要logits
                else:
                    outputs = outputs[0]
            # L2归一化
            feat = F.normalize(outputs, p=2, dim=1)
            feat = feat.cpu().numpy()[0]

        return feat

    def search_gallery(self, query_feature):
        """在底库中搜索最相似的人（返回最高分匹配）"""
        if self.gallery_features is None or len(self.gallery_features) == 0:
            return None, 0.0
        best_sim = -1
        best_id = None
        for gf, gid in zip(self.gallery_features, self.gallery_ids):
            sim = self.compute_similarity(query_feature, gf)
            if sim > best_sim:
                best_sim = sim
                best_id = gid
        return best_id, best_sim

    def compute_similarity(self, feat1, feat2):
        """计算余弦相似度"""
        feat1 = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2 = feat2 / (np.linalg.norm(feat2) + 1e-8)
        return np.dot(feat1, feat2)

    def retrieve(self, query_feature, top_k=5):
        """检索相似身份"""
        results = []

        for track_id, features in self.feature_db.items():
            # 使用平均特征
            avg_feat = np.mean(features, axis=0)
            sim = self.compute_similarity(query_feature, avg_feat)
            results.append((track_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


# ============== 视频处理器（带播放控制） ==============
class VideoProcessor:
    """视频流处理器（带播放控制）"""

    def __init__(self, source, model, device, gallery_features=None, gallery_ids=None, gallery_paths=None, debug_mode=True):
        self.model = model
        self.device = device
        self.debug_mode = debug_mode

        # 初始化组件：始终使用YOLO检测+跟踪
        self.detector = PersonDetector()
        self.tracker = MultiObjectTracker()
        self.retriever = ReIDRetriever(model, device,
                                       gallery_features=gallery_features,
                                       gallery_ids=gallery_ids,
                                       gallery_paths=gallery_paths)
        self.match_threshold = Config.MATCHING_THRESHOLD

        # 匹配日志（用于生成时间线报告）
        self.match_log = []
        
        # 打开视频源
        if source.isdigit():
            self.cap = cv2.VideoCapture(int(source))
        elif isinstance(source, str) and source.startswith('rtsp'):
            self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(str(source))
        
        # 获取视频信息
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self.current_frame = 0
        
        # 播放控制状态
        self.playing = True  # 默认播放
        self.speed = 1.0     # 播放速度
        
        # 状态
        self.running = False
        self.frame_count = 0
        self.fps = 0
        
        # 跟踪计数重置
        KalmanBoxTracker.count = 0
        
        # 统计
        self.stats = {
            'tracks': 0,
            'retrievals': 0,
            'fps': 0
        }
        
        # 检测结果统计（用于调试）
        self.detection_stats = {
            'total_detections': 0,
            'frames_with_detections': 0
        }

        sys.stdout.flush()




    def process_frame(self, frame):
        """处理单帧"""
        start_time = time.time()
        self.frame_count += 1

        if frame is None:
            return None, [], 0

        result_frame = frame.copy()
        detections = []

        # 检测：始终使用YOLO检测人物
        if self.detector:
            detections = self.detector.detect(frame)
            if self.debug_mode and detections:
                print(f"[调试] YOLO检测到 {len(detections)} 个人物")

        # 如果YOLO没有检测到人物，尝试滑动窗口ReID检测
        if len(detections) == 0:
            detections = self._sliding_window_detect(frame)
        
        # 精确定位：对每个检测框做轮廓提取，紧贴人物
        for det in detections:
            det['bbox'] = self._refine_bbox(frame, det['bbox'])
        
        # 统计检测结果
        if len(detections) > 0:
            self.detection_stats['total_detections'] += len(detections)
            self.detection_stats['frames_with_detections'] += 1
        
        # 跟踪
        tracks = self.tracker.update(detections)

        # 提取特征并检索（逐帧识别）
        for track in tracks:
            bbox = track['bbox']

            # 逐帧提取特征并匹配
            feat = self.retriever.extract_feature(frame, bbox)
            if feat is not None:
                track['feature'] = feat

                # 底库匹配（使用当前帧的特征，逐帧独立判断）
                gid, gsim = self.retriever.search_gallery(feat)
                track['gallery_match'] = {'id': gid, 'similarity': gsim}
                # 记录匹配日志（所有匹配都记录，不按阈值过滤）
                if gid is not None:
                    self.match_log.append({
                        'frame': self.current_frame,
                        'person': gid,
                        'similarity': round(gsim, 4)
                    })

        # 绘制结果
        result_frame = self._draw_results(result_frame, tracks, len(detections))

        # 更新FPS
        self.fps = 1 / (time.time() - start_time + 1e-6)

        return result_frame, tracks, len(detections)

    def _simple_person_detection(self, frame):
        """简单人物检测（调试模式使用）"""
        # 返回空列表，让跟踪器显示手动添加的"虚拟"检测框
        return []
    
    def _draw_results(self, frame, tracks, num_detections):
        """绘制结果（不包含进度条，进度条在letterbox后绘制）"""
        h, w = frame.shape[:2]

        # 顶部分布信息
        info_text = f"Detect:{num_detections} | Track:{len(tracks)} | Gallery:{len(self.retriever.gallery_ids) if self.retriever.gallery_ids else 0} | FPS:{self.fps:.1f}"
        cv2.rectangle(frame, (5, 5), (370, 28), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # 筛选有效track，按y坐标排序确保标签不重叠
        valid_tracks = []
        for track in tracks:
            bbox = track['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 15 or y2 - y1 < 15:
                continue
            track['_x1'] = x1
            track['_y1'] = y1
            track['_x2'] = x2
            track['_y2'] = y2
            valid_tracks.append(track)

        # 按y坐标排序，从上到下渲染标签
        valid_tracks.sort(key=lambda t: t['_y1'])

        # 记录已占用的标签区域，避免重叠
        used_label_regions = []

        for track in valid_tracks:
            x1, y1, x2, y2 = track['_x1'], track['_y1'], track['_x2'], track['_y2']

            # 绘制边界框
            color = self._get_color(track['id'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 构建标签行
            lines = [f"ID:{track['id']}"]
            gm = track.get('gallery_match', {})
            gid = gm.get('id') if gm else None
            gsim = gm.get('similarity', 0) if gm else 0
            if gid is not None and gsim >= self.match_threshold:
                lines.append(f"{gid} ({gsim:.2f})")
            elif gid is not None:
                lines.append(f"未知 ({gsim:.2f})")

            # 计算标签位置（先在框上方，碰撞则逐步下移，最终在框内左上角）
            line_h = 20
            bg_w = min(210, w - x1 - 4)
            bg_h = len(lines) * line_h + 6
            # 尝试5个位置：框正上方 → 上方偏左 → 框内左上 → 框内右上 → 框正下方
            label_candidates = [
                (x1, y1 - bg_h - 2),                    # 正上方
                (max(0, x1 - 30), y1 - bg_h - 2),        # 偏左上方
                (x1, max(y1 + 4, 30)),                   # 框内左上
                (max(x2 - bg_w, 0), max(y1 + 4, 30)),    # 框内右上
                (x1, min(y2 + 2, h - bg_h - 2)),         # 框正下方
            ]
            label_x, label_y = label_candidates[0]
            placed = False
            for (cx, cy) in label_candidates:
                if cy < 28 or cx + bg_w > w or cy + bg_h > h:
                    continue
                # 检查是否与已有标签碰撞
                collision = False
                for used_region in used_label_regions:
                    ur_x, ur_y, ur_x2, ur_y2 = used_region
                    if (cx < ur_x2 and cx + bg_w > ur_x and
                            cy < ur_y2 and cy + bg_h > ur_y):
                        collision = True
                        break
                if not collision:
                    label_x, label_y = cx, cy
                    placed = True
                    break
            if not placed:
                # 所有位置都冲突时，强制放在框上方（覆盖旧标签）
                label_x, label_y = x1, max(30, y1 - bg_h - 2)

            # 记录当前标签区域
            used_label_regions.append((label_x, label_y, label_x + bg_w, label_y + bg_h))

            # 绘制标签背景（半透明）
            overlay = frame.copy()
            cv2.rectangle(overlay, (label_x, label_y), (label_x + bg_w, label_y + bg_h), color, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            # 边框
            cv2.rectangle(frame, (label_x, label_y), (label_x + bg_w, label_y + bg_h), (255, 255, 255), 1)

            # 绘制文字
            for i, line in enumerate(lines):
                text_y = label_y + line_h * (i + 1) - 5
                cv2.putText(frame, line, (label_x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 轨迹（仅当bbox大小合理时绘制，避免全屏检测框导致线条异常）
            trajectory = track.get('trajectory', [])
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = w * h
            # 只绘制框面积不超过画面50%的轨迹（排除全屏检测）
            if len(trajectory) > 2 and bbox_area < frame_area * 0.5:
                # 取最近15个点
                recent_traj = trajectory[-15:]
                pts = []
                for tb in recent_traj:
                    cx = int((tb[0] + tb[2]) / 2)
                    cy = int((tb[1] + tb[3]) / 2)
                    # 过滤超出画面的异常点
                    if 0 <= cx <= w and 0 <= cy <= h:
                        pts.append((cx, cy))
                if len(pts) > 1:
                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i-1], pts[i], color, 1)

        # 进度条在letterbox缩放后由run()方法绘制，此处不绘制

        return frame

    def _draw_progress_bar(self, frame):
        """绘制视频进度条和播放控制"""
        h, w = frame.shape[:2]
        
        # 进度条参数
        bar_h = 40
        bar_y = h - bar_h
        
        # 半透明黑色背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # 时间信息（左侧）
        current_sec = self.current_frame / self.video_fps if self.video_fps > 0 else 0
        total_sec = self.total_frames / self.video_fps if self.video_fps > 0 else 0
        time_str = f"{int(current_sec//60):02d}:{int(current_sec%60):02d} / {int(total_sec//60):02d}:{int(total_sec%60):02d}"
        cv2.putText(frame, time_str, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 进度条轨道
        progress_x_start = 160
        progress_x_end = w - 10
        progress_width = progress_x_end - progress_x_start
        track_y = h - 18
        track_h = 6
        
        # 背景轨道
        cv2.rectangle(frame, (progress_x_start, track_y), (progress_x_end, track_y + track_h), (60, 60, 60), -1)
        
        # 已播放进度
        if self.total_frames > 0:
            progress = self.current_frame / self.total_frames
            fill_end = progress_x_start + int(progress * progress_width)
            cv2.rectangle(frame, (progress_x_start, track_y), (fill_end, track_y + track_h), (0, 200, 255), -1)

        # 状态信息（右侧-进度条上方，使用ASCII字符避免编码问题）
        status_icon = "[>]" if self.playing else "[||]"
        status_text = f"{status_icon} {self.speed:.1f}x"
        cv2.putText(frame, status_text, (progress_x_end - 100, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 控制提示（进度条下方居中，使用英文避免编码问题）
        controls = "[Space:Pause] [<- ->:Seek5s] [Up-Dn:Speed] [R:Reset] [Q:Quit]"
        tw = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cx = w // 2
        cv2.putText(frame, controls, (cx - tw // 2, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        # 保存进度条区域用于鼠标点击检测（扩大点击范围，覆盖整个状态栏）
        self._progress_bar_region = (progress_x_start, bar_y, progress_x_end, h)

    def _resize_with_aspect_ratio(self, frame, target_w, target_h):
        """保持长宽比缩放，添加黑边（letterbox）"""
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w < 1 or new_h < 1:
            scale = 1.0
            new_w, new_h = w, h
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas, x_offset, y_offset, scale

    def _sliding_window_detect(self, frame):
        """ReID特征匹配检测：中心/象限搜索 + 跟踪预测 + 回落全搜索"""
        h, w = frame.shape[:2]
        if h < 100 or w < 100:
            return []
        if self.retriever.gallery_features is None or len(self.retriever.gallery_ids) == 0:
            return []

        # 检查活跃跟踪，在其预测位置附近优先搜索
        active = self.tracker._get_active_tracks()
        if active:
            last = active[-1]
            bx = last['bbox']
            cx = (bx[0] + bx[2]) // 2
            cy = (bx[1] + bx[3]) // 2
            win_w, win_h = w // 3, h // 3
            x1 = max(0, cx - win_w // 2)
            y1 = max(0, cy - win_h // 2)
            x2 = min(w, x1 + win_w)
            y2 = min(h, y1 + win_h)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0 and crop.shape[0] >= 50 and crop.shape[1] >= 50:
                feat = self.retriever.extract_feature(crop, [x1, y1, x2, y2])
                if feat is not None:
                    gid, gsim = self.retriever.search_gallery(feat)
                    if gid and gsim > Config.MIN_GALLERY_SIM:
                        return [{'bbox': [x1, y1, x2, y2], 'confidence': gsim}]
            # 预测位置搜索失败，回退到全区域搜索
            # 继续执行下面的全区域搜索

        # 全区域搜索（当无跟踪或预测失败时）
        best = self._fullframe_search(frame)
        if best:
            return [best]
        return []

    def _fullframe_search(self, frame):
        """在画面中搜索匹配的人物（先中心、再周边，使用重叠窗口精确定位）"""
        h, w = frame.shape[:2]
        
        # 优先检查中心区域（大多数场景目标在画面中央）
        cx, cy = w // 2, h // 2
        center_box = (cx - w//4, cy - h//3, cx + w//4, cy + h//3)
        x1, y1, x2, y2 = center_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0 and crop.shape[0] >= 60 and crop.shape[1] >= 60:
            feat = self.retriever.extract_feature(crop, [x1, y1, x2, y2])
            if feat is not None:
                gid, gsim = self.retriever.search_gallery(feat)
                if gid and gsim > 0.25:
                    if self.debug_mode:
                        print(f"[检测] 中心匹配: {gid} sim={gsim:.3f}")
                    return {'bbox': [x1, y1, x2, y2], 'confidence': gsim}
        
        # 中心未匹配到，搜索周边区域（纵向分层）
        regions = [
            (w//4, 0, w*3//4, h//2),         # 上半
            (w//4, h//2, w*3//4, h),          # 下半
            (0, h//4, w//2, h*3//4),          # 左半
            (w//2, h//4, w, h*3//4),          # 右半
        ]
        best = None
        best_sim = Config.MIN_GALLERY_SIM
        for (x1, y1, x2, y2) in regions:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 50 or crop.shape[1] < 50:
                continue
            feat = self.retriever.extract_feature(crop, [x1, y1, x2, y2])
            if feat is None:
                continue
            gid, gsim = self.retriever.search_gallery(feat)
            if gid and gsim > best_sim:
                best_sim = gsim
                best = {'bbox': [x1, y1, x2, y2], 'confidence': gsim}
        if best and self.debug_mode:
            print(f"[检测] 周边匹配 sim={best_sim:.3f}")
        return best

    def _refine_bbox(self, frame, bbox):
        """在匹配区域内用颜色+边缘检测精确定位人物（保守策略，不过度扩大）"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 60 or bh < 60:
            return bbox

        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        # Canny边缘检测 + 膨胀合并
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # 颜色差异检测
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_hsv = cv2.mean(hsv)[:3]
        lower = np.array([max(0, int(avg_hsv[0])-30), 30, 30], dtype=np.uint8)
        upper = np.array([min(180, int(avg_hsv[0])+30), 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_not(mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        combined = cv2.bitwise_or(dilated, mask)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return bbox

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < roi_w * roi_h * 0.02:
            return bbox

        cx_, cy_, cw, ch = cv2.boundingRect(largest)
        # 保守边距，不过度扩大
        margin = 3
        rx1 = max(x1, x1 + cx_ - margin)
        ry1 = max(y1, y1 + cy_ - margin)
        rx2 = min(x2, x1 + cx_ + cw + margin)
        ry2 = min(y2, y1 + cy_ + ch + margin)
        # 确保输出比例合理
        if (ry2 - ry1) / max(rx2 - rx1, 1) < 0.3:
            return bbox  # 精确定位后比例不对，使用原始框
        return [rx1, ry1, rx2, ry2]

    def _get_color(self, idx):
        """获取ID对应的颜色"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (255, 128, 0), (128, 0, 255), (0, 128, 128)
        ]
        return colors[idx % len(colors)]

    def seek_relative(self, seconds):
        """相对跳转（秒）"""
        frame_offset = int(seconds * self.video_fps)
        new_frame = max(0, min(self.total_frames - 1, self.current_frame + frame_offset))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.current_frame = new_frame
        print(f"[跳转] 移动到帧 {self.current_frame}")

    def seek_to_frame(self, frame_num):
        """跳转到指定帧"""
        frame_num = max(0, min(self.total_frames - 1, frame_num))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        # 读一帧让解码器同步
        self.cap.read()
        # 重置到目标帧位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.current_frame = frame_num

    def adjust_speed(self, delta):
        """调整播放速度"""
        self.speed = max(0.25, min(4.0, self.speed + delta))
        print(f"[速度] {self.speed:.2f}x")

    def toggle_pause(self):
        """切换暂停状态"""
        self.playing = not self.playing
        print(f"[播放] {'暂停' if not self.playing else '继续'}")

    def reset(self):
        """重置视频"""
        self.seek_to_frame(0)
        KalmanBoxTracker.count = 0
        self.tracker = MultiObjectTracker()
        self.retriever.feature_db = {}
        self.detection_stats = {'total_detections': 0, 'frames_with_detections': 0}
        self.playing = True
        self.frame_count = 0
        print("[重置] 视频已重置")

    def _mouse_seek_callback(self, event, x, y, flags, param):
        """鼠标回调：点击/拖拽进度条跳转"""
        def _do_seek(mx):
            if self.total_frames > 0:
                ratio = max(0.0, min(1.0, mx / max(Config.DISPLAY_WIDTH, 1)))
                tf = min(self.total_frames - 1, int(ratio * self.total_frames))
                self._mouse_seek_pending = tf
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_dragging = True
            _do_seek(x)
        elif event == cv2.EVENT_MOUSEMOVE and self._mouse_dragging:
            _do_seek(x)
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_dragging = False

    def run(self, display=True, output_path=None):
        """运行处理（带播放控制）"""
        self.running = True

        # 检查视频是否成功打开
        if not self.cap.isOpened():
            print("[错误] 无法打开视频!")
            self.stop()
            return

        # 鼠标拖拽状态
        self._mouse_dragging = False
        self._mouse_seek_pending = -1
        video_ended_local = [False]  # 用列表模拟 nonlocal
        speed_counter = 0  # 速度控制计数器

        # 输出视频写入器
        writer = None
        if output_path:
            ext = os.path.splitext(output_path)[1].lower()
            codec = 'XVID' if ext == '.avi' else 'mp4v'
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, 30.0,
                                   (Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT))
            self._writer = writer

        print("\n" + "="*60)
        print("视频ReID播放器控制:")
        print("  Space: Pause/Play")
        print("  Left/Right: Seek +/-5s")
        print("  Up/Down: Speed +/-0.25x")
        print("  R: Reset  Q/ESC: Quit")
        print("  Click on progress bar: Seek")
        print("="*60 + "\n")

        # 在控制台输出后才创建窗口（避免窗口阻塞导致输出不可见）
        if display:
            try:
                cv2.namedWindow(Config.WINDOW_NAME)
                cv2.setMouseCallback(Config.WINDOW_NAME, self._mouse_seek_callback)
                cv2.waitKey(1)
                print("[系统] 窗口已创建")
            except Exception as e:
                print(f"[警告] 窗口创建失败: {e}")
                display = False

        no_display_exit = False

        while self.running:
            # ── 事件处理：固定1ms超时确保windows事件能被及时处理 ──
            key_raw = cv2.waitKey(1) if display else -1

            # 执行鼠标拖拽产生的跳转
            if self._mouse_seek_pending >= 0:
                self.seek_to_frame(self._mouse_seek_pending)
                video_ended_local[0] = False
                self._mouse_seek_pending = -1
                speed_counter = 0
                continue

            # 处理键盘
            if self._handle_key(key_raw):
                video_ended_local[0] = False
                speed_counter = 0
                continue

            # 检测窗口关闭
            if display:
                try:
                    if cv2.getWindowProperty(Config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except:
                    pass

            # 无显示模式：视频结束即退出
            if no_display_exit:
                self.running = False
                break

            # 暂停：显示当前帧，不读帧
            if not self.playing:
                # 暂停时用imshow刷新帧（无新读取）
                if display:
                    cv2.imshow(Config.WINDOW_NAME, self._last_display_frame if hasattr(self, '_last_display_frame') else np.zeros((Config.DISPLAY_HEIGHT, Config.DISPLAY_WIDTH, 3), dtype=np.uint8))
                continue

            # 视频已结束
            if video_ended_local[0]:
                if not display:
                    no_display_exit = True
                else:
                    # 显示结束提示 3 秒后自动退出
                    self._end_frame_counter = getattr(self, '_end_frame_counter', 0) + 1
                    if self._end_frame_counter > 90:  # ~3秒 (30fps × 3)
                        self.running = False
                continue

            # ── 速度控制：用帧跳过代替waitKey超时调整 ──
            # speed=1.0: 每帧都读; speed=2.0: 读一帧跳一帧; speed=4.0: 读一帧跳三帧
            skip_interval = int(round(1.0 / max(self.speed, 0.25)))
            if skip_interval > 0 and speed_counter % (skip_interval + 1) != 0:
                # 跳过这一帧，但需要调用read()来前进
                self.cap.read()
                speed_counter += 1
                continue
            speed_counter += 1

            # 读取帧
            ret, frame = self.cap.read()
            if not ret:
                print("[提示] 视频播放完毕，按 R 重新开始 或 Q 退出")
                if not display:
                    no_display_exit = True
                video_ended_local[0] = True
                self.playing = False
                continue

            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 处理（含异常保护）
            try:
                result, tracks, num_dets = self.process_frame(frame)
                if result is None:
                    continue
            except Exception as e:
                print(f"[警告] 帧处理异常: {e}")
                continue

            # 保持长宽比缩放（letterbox）
            result, self._letterbox_offset_x, self._letterbox_offset_y, self._letterbox_scale = \
                self._resize_with_aspect_ratio(result, Config.DISPLAY_WIDTH, Config.DISPLAY_HEIGHT)

            # 绘制进度条
            self._draw_progress_bar(result)

            # 保存最后一帧用于暂停时显示
            self._last_display_frame = result.copy()

            # 写入
            if writer:
                writer.write(result)

            # 显示
            if display:
                cv2.imshow(Config.WINDOW_NAME, result)

        # === 循环结束 ===
        
        if display:
            cv2.setMouseCallback(Config.WINDOW_NAME, lambda *args: None)
        
        # 打印统计
        print("\n" + "="*60)
        print("--- Processing Stats ---")
        print(f"  Total frames: {self.current_frame}")
        print(f"  Detections: {self.detection_stats['total_detections']}")
        print(f"  Frames with detections: {self.detection_stats['frames_with_detections']}")
        print("="*60)

        # 生成时间线报告（始终生成CSV，match_log为空也生成）
        import csv as csv_module
        from pathlib import Path
        from collections import defaultdict
        report_dir = Path(__file__).resolve().parent.parent / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        csv_path = str(report_dir / f"timeline_report_{int(time.time())}.csv")
        
        if self.match_log:
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            # 按人员合并时间线
            person_timeline = defaultdict(list)
            person_sims = defaultdict(list)
            for m in self.match_log:
                person_timeline[m['person']].append(m['frame'] / fps)
                person_sims[m['person']].append(m['similarity'])
            
            print("\n" + "="*60)
            print("⏱ 人员出现时间线报告")
            print("="*60)
            for person, times in sorted(person_timeline.items()):
                times = sorted(set(times))
                segments = []
                seg_start = times[0]
                seg_end = times[0]
                for t in times[1:]:
                    if t - seg_end > 3:
                        segments.append((seg_start, seg_end))
                        seg_start = t
                    seg_end = t
                segments.append((seg_start, seg_end))
                seg_str = []
                for s, e in segments:
                    s_str = f"{int(s//3600):02d}:{int(s%3600//60):02d}:{int(s%60):02d}"
                    e_str = f"{int(e//3600):02d}:{int(e%3600//60):02d}:{int(e%60):02d}"
                    if s == e:
                        seg_str.append(f"{s_str}")
                    else:
                        seg_str.append(f"{s_str} ~ {e_str} ({int(e-s)}秒)")
                print(f"  {person}: {' → '.join(seg_str)}")
            
            # 保存CSV（合并为连续片段，更简洁）
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv_module.writer(f)
                w.writerow(['人员', '开始时间(秒)', '结束时间(秒)', '持续时长(秒)', '开始时间', '结束时间', '平均相似度', '最大相似度', '最小相似度', '可信度'])
                for person, times in sorted(person_timeline.items()):
                    times = sorted(set(times))
                    sims = person_sims[person]
                    avg_sim = sum(sims) / len(sims)
                    max_sim = max(sims)
                    min_sim = min(sims)
                    confident = "是" if max_sim >= self.match_threshold else "否"
                    
                    segments = []
                    seg_start = times[0]
                    seg_end = times[0]
                    for t in times[1:]:
                        if t - seg_end > 3:
                            segments.append((seg_start, seg_end))
                            seg_start = t
                        seg_end = t
                    segments.append((seg_start, seg_end))
                    
                    for s, e in segments:
                        s_str = f"{int(s//3600):02d}:{int(s%3600//60):02d}:{int(s%60):02d}"
                        e_str = f"{int(e//3600):02d}:{int(e%3600//60):02d}:{int(e%60):02d}"
                        duration = round(e - s, 1)
                        w.writerow([person, f"{s:.1f}", f"{e:.1f}", duration, s_str, e_str, f"{avg_sim:.4f}", f"{max_sim:.4f}", f"{min_sim:.4f}", confident])
            print(f"📁 时间线报告已保存: {csv_path}")
        else:
            # match_log为空，生成空CSV供参考
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv_module.writer(f)
                w.writerow(['人员', '开始时间(秒)', '结束时间(秒)', '持续时长(秒)', '开始时间', '结束时间', '平均相似度', '最大相似度', '最小相似度', '可信度'])
                w.writerow(['（无任何匹配记录）', '', '', '', '', '', '', '', '', ''])
            print(f"\n⚠ 视频中未检测到任何匹配（所有相似度均低于阈值 {self.match_threshold}）")
            print(f"📁 空报告已保存: {csv_path}")
        
        print("="*60)

        self.stop()

    def _handle_key(self, key_raw):
        """处理键盘事件，返回 True 表示已处理（跳过当前帧）"""
        key_low = key_raw & 0xFF

        # Windows 扩展键前缀（方向键）
        if key_low == 224:
            key2 = cv2.waitKey(5) & 0xFF
            if key2 == 75:  # 左箭头 ←
                self.seek_relative(-5)
                return True
            elif key2 == 77:  # 右箭头 →
                self.seek_relative(5)
                return True
            elif key2 == 72:  # 上箭头 ↑
                self.adjust_speed(0.25)
                return True
            elif key2 == 80:  # 下箭头 ↓
                self.adjust_speed(-0.25)
                return True
            # 没有实际方向键按下，不跳过当前帧
            return False

        # 普通键（过滤无键按下情况）
        if key_low == 255 or key_low == 0:
            return False

        # 普通键
        if key_low == ord('q') or key_low == 27:  # Q 或 ESC
            self.running = False
            return True
        elif key_low == ord(' ') or key_low == 32:  # 空格
            self.toggle_pause()
            return True
        elif key_low == ord('r') or key_low == 82:  # R
            self.reset()
            return True
        elif key_low == ord('['):  # 快捷键: 后退1秒
            self.seek_relative(-1)
            return True
        elif key_low == ord(']'):  # 快捷键: 前进1秒
            self.seek_relative(1)
            return True

        return False

    def stop(self):
        """停止处理"""
        self.running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        # 释放VideoWriter
        if hasattr(self, '_writer') and self._writer:
            self._writer.release()
            self._writer = None
        cv2.destroyAllWindows()


# ============== 轨迹拼接 ==============
class TrajectoryTracker:
    """轨迹拼接模块 - 关联不同摄像头的同一目标"""

    def __init__(self, reid_model=None, device=None):
        self.reid_model = reid_model
        self.device = device

        # 轨迹数据库
        self.trajectories = {}  # track_id -> trajectory info

        # 相似度阈值
        self.merge_threshold = 0.6

    def add_trajectory(self, camera_id, track_id, bbox, feature, timestamp=None):
        """添加轨迹"""
        key = f"{camera_id}_{track_id}"

        if key not in self.trajectories:
            self.trajectories[key] = {
                'camera_id': camera_id,
                'original_id': track_id,
                'merged_id': None,
                'bbox': bbox,
                'feature': feature,
                'timestamps': [timestamp] if timestamp else [],
                'confirmed': False
            }

    def merge_trajectories(self):
        """合并相似轨迹"""
        pass  # 简化实现






# ============== 工具函数 ==============
def load_reid_model(model_path, device):
    """加载ReID模型（自动适配 checkpoint 中的 backbone 类型）"""
    from backbone import build_backbone
    
    print(f"加载模型: {model_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 解析checkpoint信息
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        num_classes = checkpoint.get('num_classes', 751)
        backbone_name = checkpoint.get('backbone', 'osnet')
        print(f"  模型信息: backbone={backbone_name}, num_classes={num_classes}")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        num_classes = 751
        backbone_name = 'osnet'
    else:
        if hasattr(checkpoint, 'eval'):
            checkpoint = checkpoint.to(device)
            checkpoint.eval()
            return checkpoint
        raise RuntimeError("无法识别的模型格式")
    
    # 使用统一的骨干网络接口构建模型
    print(f"  构建 {backbone_name} 模型...")
    model = build_backbone(backbone_name, num_classes=num_classes, pretrained=False)
    
    # 加载权重（忽略不匹配层）
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  模型缺少 {len(missing)} 个层（新初始化）")
    if unexpected:
        print(f"  checkpoint有 {len(unexpected)} 个额外层（已忽略）")
    
    model = model.to(device)
    model.eval()
    
    # 保存特征维度
    model.feat_dim = getattr(model, 'feature_dim', 512)
    
    print(f"  模型加载成功! ({backbone_name} + {model.feat_dim}D特征)")
    return model








# ============== 主函数 ==============
def main():
    import argparse
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 从 sys.argv 解析参数
    parser = argparse.ArgumentParser(description='ReID视频流处理（带播放控制）')
    parser.add_argument('--source', type=str, default='0',
                       help='视频源: webcam id, video file, or rtsp url')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出视频路径')
    parser.add_argument('--no_display', action='store_true', help='不显示画面（仅保存输出）')
    parser.add_argument('--gallery', type=str, default=None,
                       help='底库目录（用于在视频中识别人物身份）')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式：显示更多调试信息')

    args = parser.parse_args()

    # 加载模型
    print("加载模型...")
    model = load_reid_model(args.model_path, device)

    # 加载底库（可选）—— 使用与视频处理相同的模型和预处理，保证特征空间一致
    gallery_features, gallery_ids, gallery_paths = None, None, None
    if args.gallery:
        print(f"加载底库: {args.gallery}")
        try:
            # 使用与视频处理相同的模型提取底库特征
            temp_retriever = ReIDRetriever(model, device)
            gallery_dir = Path(args.gallery)
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            # 只收集子目录中的图片（每个子目录=一个人物）
            img_paths = []
            if any(d.is_dir() and not d.name.startswith('__') for d in gallery_dir.iterdir()):
                for person_dir in gallery_dir.iterdir():
                    if person_dir.is_dir() and not person_dir.name.startswith('__'):
                        for ext in extensions:
                            img_paths.extend(list(person_dir.glob(ext)))
            else:
                for ext in extensions:
                    img_paths.extend(list(gallery_dir.glob(ext)))
            # 过滤非人物图片
            img_paths = sorted([p for p in img_paths if not any(
                kw in p.stem.lower() for kw in ['search_result', 'query', 'temp', 'thumb', 'debug', '._']
            )])
            if not img_paths:
                print(f"[警告] 底库目录无有效人物图片: {args.gallery}")
            else:
                features_list = []
                ids_list = []
                paths_list = []
                for p in img_paths:
                    person_dir = p.parent
                    person_id = person_dir.name if person_dir.name != gallery_dir.name else p.stem
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    hh, ww = img.shape[:2]
                    feat = temp_retriever.extract_feature(img, [0, 0, ww, hh])
                    if feat is not None:
                        features_list.append(feat)
                        ids_list.append(person_id)
                        paths_list.append(str(p))
                if features_list:
                    # 保持所有图片为独立样本（不按人平均，多角度照片提供更丰富的表征）
                    gallery_features = np.array(features_list)
                    gallery_ids = ids_list
                    gallery_paths = paths_list
                    # 统计
                    from collections import Counter
                    cnt = Counter(gallery_ids)
                    print(f"[底库] 总样本: {len(gallery_ids)}, 人员数: {len(cnt)}")
                    for pid, n in sorted(cnt.items()):
                        print(f"    - {pid}: {n} 张照片")
        except Exception as e:
            print(f"[警告] 底库加载失败: {e}")
            import traceback
            traceback.print_exc()

    # 创建处理器
    print(f"打开视频源: {args.source}")
    try:
        processor = VideoProcessor(
            source=args.source,
            model=model,
            device=device,
            gallery_features=gallery_features,
            gallery_ids=gallery_ids,
            gallery_paths=gallery_paths,
            debug_mode=args.debug
        )
    except Exception as e:
        print(f"[错误] 创建处理器失败: {e}")
        return

    # 运行
    print("[系统] 准备进入播放循环...")
    try:
        processor.run(
            display=not args.no_display,
            output_path=args.output
        )
    except Exception as e:
        print(f"[错误] 运行异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
