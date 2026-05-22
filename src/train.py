"""
抗恶劣环境行人重识别系统 - 完整训练脚本
支持多数据集训练、模型保存、特征库构建

使用方法：
1. 快速演示训练（无需下载数据集）: python train.py --demo
2. 使用Market-1501数据集训练: python train.py --download --epochs 60
3. 仅评估模型: python train.py --eval_only --model_path checkpoints/best_model.pth
"""

import sys
import argparse
import zipfile
import urllib.request
from pathlib import Path
import time
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

# 导入骨干网络模块
# 兼容两种导入方式：作为脚本运行时用绝对路径，作为包导入时用相对路径
try:
    from .backbone import build_backbone, SUPPORTED_BACKBONES
    from .augmentation import WeatherAugmentation
except ImportError:
    from backbone import build_backbone, SUPPORTED_BACKBONES
    from augmentation import WeatherAugmentation


# ============== Circle Loss ==============
class CircleLoss(nn.Module):
    """Circle Loss: A Unified Perspective of Pair Similarity Optimization
    
    参考: https://arxiv.org/abs/2002.10857
    适用于开放集ReID任务，比CrossEntropy更适合度量学习
    
    稳定版：带 warmup 的渐进式 margin
    """
    
    def __init__(self, num_classes, feat_dim=512, margin=0.15, scale=32):
        super(CircleLoss, self).__init__()
        self.margin = margin  # 降低margin从0.25到0.15，减少损失波动
        self.scale = scale     # 降低scale从64到32，使训练更稳定
        self.num_classes = num_classes
        
        # 可学习的分类器权重
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_normal_(self.weight)
        
        # 当前 margin（用于 warmup）
        self.current_margin = 0.0
    
    def set_margin(self, margin):
        """动态设置 margin（用于 warmup）"""
        self.current_margin = margin
    
    def forward(self, features, labels):
        """
        Args:
            features: [B, D] 特征向量 (已归一化)
            labels: [B] 类别标签
        """
        # 归一化特征和权重
        features = F.normalize(features, p=2, dim=1)
        w = F.normalize(self.weight, dim=1).to(features.device)
        
        # 计算余弦相似度作为 logits
        cosine_logits = F.linear(features, w)  # [B, C]
        
        # 使用当前 margin（支持 warmup）
        margin = self.current_margin
        
        # 对目标类别应用 margin
        target_logits = cosine_logits.gather(1, labels.view(-1, 1))  # [B, 1]
        target_logits = target_logits - margin  # 降低目标类别的分数
        
        # 重新合并 logits
        final_logits = cosine_logits.clone()
        final_logits.scatter_(1, labels.view(-1, 1), target_logits)
        
        # 使用 scale 缩放并计算交叉熵
        scaled_logits = final_logits * self.scale
        loss = F.cross_entropy(scaled_logits, labels)
        
        # 返回原始 cosine_logits 用于计算准确率
        return loss, cosine_logits


# ============== Warmup 学习率调度器 ==============
class WarmupMultiStepLR(optim.lr_scheduler._LRScheduler):
    """带Warmup的学习率调度器
    
    前 warmup_epochs 个 epoch 线性增加学习率，
    之后按预设策略衰减
    """
    
    def __init__(self, optimizer, warmup_epochs, milestones, gamma=0.1, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段: 线性增长
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 正常衰减
            # 计算在第几个milestone区间
            decay_count = sum([self.last_epoch >= m for m in self.milestones])
            return [base_lr * (self.gamma ** decay_count) for base_lr in self.base_lrs]


class WarmupCosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    """带Warmup的余弦退火学习率"""
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [base_lr * (self.min_lr / base_lr + (1 - self.min_lr / base_lr) * cosine_decay) 
                    for base_lr in self.base_lrs]


# ============== 进度显示 ==============
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """显示进度条"""
    if total <= 0:
        print(f'\r{prefix} |{"-" * length}| 0% {suffix}', end='', flush=True)
        return
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============== 数据集下载 ==============
def download_file(url, destination, desc="下载"):
    """带进度显示的文件下载"""
    print(f"\n正在{desc}...")
    print(f"  链接: {url}")
    print(f"  目标: {destination}")
    
    def report_progress(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            print_progress_bar(
                min(downloaded, total_size), total_size,
                prefix=f'{desc}进度:', suffix='完成'
            )
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print(f"\n{desc}完成！")
        return True
    except Exception as e:
        print(f"\n{desc}失败: {e}")
        return False


def download_market1501(data_dir):
    """下载Market-1501数据集"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_dir / 'market1501.zip'
    
    # 尝试多个下载源
    urls = [
        'https://rdm.ding.pku.edu.cn/dataset/market1501/Market-1501-v15.09.15.zip',
        'http://isee.ustc.edu.cn/~sbhuang/resources/market1501/Market-1501-v15.09.15.zip',
    ]
    
    # 下载
    success = False
    for url in urls:
        if download_file(url, str(zip_path), "下载Market-1501数据集"):
            success = True
            break
    
    if not success:
        print("\n" + "!" * 50)
        print("自动下载失败，请手动下载：")
        print("  1. 访问: https://rdm.ding.pku.edu.cn/dataset/market1501")
        print("  2. 或访问: https://www.kaggle.com/datasets/pengcw/market1501")
        print("  3. 下载 Market-1501-v15.09.15.zip")
        print("  4. 解压到 ./data/Market-1501 目录")
        print("!" * 50)
        return None
    
    # 解压
    print("\n正在解压...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # 重命名目录
    extracted_dir = data_dir / 'Market-1501-v15.09.15'
    target_dir = data_dir / 'Market-1501'
    
    if extracted_dir.exists() and not target_dir.exists():
        extracted_dir.rename(target_dir)
    
    # 删除zip
    zip_path.unlink()
    
    print(f"\n数据集已准备好: {target_dir}")
    return target_dir


def create_demo_dataset(data_dir, num_ids=150, samples_per_id=4):
    """创建演示数据集（当没有真实数据时）"""
    print_section("创建演示数据集")
    
    train_dir = Path(data_dir) / 'Market-1501' / 'bounding_box_train'
    test_dir = Path(data_dir) / 'Market-1501' / 'bounding_box_test'
    query_dir = Path(data_dir) / 'Market-1501' / 'query'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"创建演示数据集...")
    print(f"  身份数量: {num_ids}")
    print(f"  每身份训练样本: {samples_per_id}")
    
    np.random.seed(42)
    
    for pid in range(1, num_ids + 1):
        # 为每个身份生成独特颜色（模拟不同衣服）
        base_color = np.random.randint(50, 200, 3).tolist()
        
        for i in range(samples_per_id):
            img = np.random.randint(20, 80, (256, 128, 3), dtype=np.uint8)
            # 人形区域
            cv2.rectangle(img, (25, 60), (103, 235), base_color, -1)
            # 头部
            cv2.circle(img, (64, 40), 28, (220, 180, 160), -1)
            # 随机噪声模拟不同光照
            noise = np.random.randint(-30, 30, (256, 128, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            filename = f"{pid:04d}_c{1}s1_001101_{i:02d}.jpg"
            cv2.imwrite(str(train_dir / filename), img)
        
        # 测试样本（查询集：cam1，测试集：cam2）
        for i in range(2):
            img = np.random.randint(20, 80, (256, 128, 3), dtype=np.uint8)
            cv2.rectangle(img, (25, 60), (103, 235), base_color, -1)
            cv2.circle(img, (64, 40), 28, (220, 180, 160), -1)
            noise = np.random.randint(-40, 40, (256, 128, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # 查询集用cam1
            filename = f"{pid:04d}_c{1}s1_001102_{i:02d}.jpg"
            cv2.imwrite(str(query_dir / filename), img)
            
            # 测试集用cam2
            filename = f"{pid:04d}_c{2}s1_001102_{i:02d}.jpg"
            cv2.imwrite(str(test_dir / filename), img)
    
    print("演示数据集创建完成！")


# ============== 数据集类 ==============
class Market1501Dataset(Dataset):
    """Market-1501数据集加载器 - 支持标签重映射
    
    mode参数用于指定加载模式:
    - 'train': 加载训练集 (bounding_box_train)
    - 'query': 仅加载查询集 (query目录)
    - 'gallery': 仅加载测试集 (bounding_box_test目录)
    - None/其他: 同时加载query和gallery (向后兼容)
    
    重要: labels存储用于分类的映射标签, pids存储用于评估的原始PID
    """
    
    def __init__(self, data_dir, transform=None, train=True, label_mapping=None, mode=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.train = train
        self.mode = mode  # 'train', 'query', 'gallery', 或 None
        self.images = []
        self.labels = []  # 分类标签 (映射后的ID)
        self.pids = []   # 原始PID (用于评估时的身份匹配)
        self.cam_ids = []
        
        # 标签映射：原始ID -> 连续ID (0, 1, 2, ...)
        self.label_mapping = label_mapping
        
        self._load_dataset()
    
    def _load_dataset(self):
        """加载数据集"""
        if self.mode == 'train' or (self.mode is None and self.train):
            # 训练模式
            bbox_dir = self.data_dir / 'bounding_box_train'
            if not bbox_dir.exists():
                print(f"[错误] 训练目录不存在: {bbox_dir}")
                return
            
            raw_labels = []  # 保存原始标签用于创建映射
            
            for img_file in bbox_dir.glob('*.jpg'):
                name = img_file.stem
                parts = name.split('_')
                if len(parts) >= 2 and parts[0].isdigit():
                    pid = int(parts[0])
                    cam_id = int(parts[1][1])
                    self.images.append(str(img_file))
                    raw_labels.append(pid)
                    self.cam_ids.append(cam_id)
            
            # 创建标签映射
            unique_labels = sorted(set(raw_labels))
            self.label_mapping = {orig: new for new, orig in enumerate(unique_labels)}
            
            # 应用映射
            self.labels = [self.label_mapping[pid] for pid in raw_labels]
            self.pids = raw_labels  # 原始PID
            
            print(f"[Market-1501] 训练集: {len(self.images)} 张图片, {len(self.label_mapping)} 个身份")
        
        elif self.mode == 'query':
            # 仅加载查询集
            query_dir = self.data_dir
            for img_file in query_dir.glob('*.jpg'):
                name = img_file.stem
                parts = name.split('_')
                if len(parts) >= 2 and parts[0].isdigit():
                    pid = int(parts[0])
                    cam_id = int(parts[1][1])
                    self.images.append(str(img_file))
                    raw_pid = pid
                    self.pids.append(raw_pid)  # 存储原始PID
                    if self.label_mapping and raw_pid in self.label_mapping:
                        self.labels.append(self.label_mapping[raw_pid])
                    elif self.label_mapping:
                        self.labels.append(-1)  # 未在训练集中出现的ID
                    else:
                        self.labels.append(raw_pid)
                    self.cam_ids.append(cam_id)
            
            print(f"[Market-1501] 查询集: {len(self.images)} 张图片")
        
        elif self.mode == 'gallery':
            # 仅加载gallery测试集
            gallery_dir = self.data_dir
            for img_file in gallery_dir.glob('*.jpg'):
                name = img_file.stem
                parts = name.split('_')
                if len(parts) >= 2 and parts[0].isdigit():
                    pid = int(parts[0])
                    cam_id = int(parts[1][1])
                    self.images.append(str(img_file))
                    raw_pid = pid
                    self.pids.append(raw_pid)  # 存储原始PID
                    if self.label_mapping and raw_pid in self.label_mapping:
                        self.labels.append(self.label_mapping[raw_pid])
                    elif self.label_mapping:
                        self.labels.append(-1)  # 未在训练集中出现的ID（如distractor -1）
                    else:
                        self.labels.append(raw_pid)
                    self.cam_ids.append(cam_id)
            
            print(f"[Market-1501] 测试集(gallery): {len(self.images)} 张图片")
        
        else:
            # 向后兼容：同时加载query和gallery
            query_dir = self.data_dir / 'query'
            for img_file in query_dir.glob('*.jpg'):
                name = img_file.stem
                parts = name.split('_')
                if len(parts) >= 2 and parts[0].isdigit():
                    pid = int(parts[0])
                    cam_id = int(parts[1][1])
                    self.images.append(str(img_file))
                    self.labels.append(pid)
                    self.pids.append(pid)
                    self.cam_ids.append(cam_id)
            
            gallery_dir = self.data_dir / 'bounding_box_test'
            for img_file in gallery_dir.glob('*.jpg'):
                name = img_file.stem
                parts = name.split('_')
                if len(parts) >= 2 and parts[0].isdigit():
                    pid = int(parts[0])
                    cam_id = int(parts[1][1])
                    self.images.append(str(img_file))
                    self.labels.append(pid)
                    self.pids.append(pid)
                    self.cam_ids.append(cam_id)
            
            print(f"[Market-1501] 查询+测试集: {len(self.images)} 张图片")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        raw_label = self.labels[idx]
        
        # 训练集：_load_dataset 已经应用了映射，self.labels 已经是映射后的值
        # 测试集：self.labels 是原始 pid，需要应用映射
        if self.train:
            label = raw_label  # 直接使用（已映射）
        elif self.label_mapping and raw_label in self.label_mapping:
            label = self.label_mapping[raw_label]
        else:
            label = -1  # 未知的标签（如distractor -1）
        
        with Image.open(img_path) as img_file:
            img = img_file.convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, label


# ============== 模型定义 ==============
class ReidModel(nn.Module):
    """ReID模型 - 使用预训练ResNet50"""
    
    def __init__(self, num_classes=1000, feat_dim=512, pretrained=True):
        super(ReidModel, self).__init__()
        
        # 加载ResNet50（支持关闭预训练，用于单元测试）
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # 移除最后两层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征维度压缩
        self.feat_bn = nn.BatchNorm1d(2048)
        self.feat_fc = nn.Linear(2048, feat_dim)
        
        # 分类头
        self.classifier = nn.Linear(feat_dim, num_classes)
        
        self.feat_dim = feat_dim
        
        # 初始化特征层
        nn.init.kaiming_normal_(self.feat_fc.weight, mode='fan_out')
        nn.init.constant_(self.feat_bn.weight, 1)
        nn.init.constant_(self.feat_bn.bias, 0)
    
    def forward(self, x, return_feat=True):
        """
        前向传播
        Args:
            x: 输入图像 [B, 3, H, W]
            return_feat: 是否返回特征向量
        """
        # 特征提取
        x = self.backbone(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 特征归一化和压缩
        feat = self.feat_bn(x)
        feat = self.feat_fc(feat)
        
        # L2归一化
        feat = nn.functional.normalize(feat, p=2, dim=1)
        
        if return_feat:
            return feat
        else:
            logits = self.classifier(feat)
            return logits, feat


class TripletLoss(nn.Module):
    """三元组损失 - 用于度量学习"""
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
        losses = torch.relu(dist_pos - dist_neg + self.margin)
        return losses.mean()


# ============== 评估函数 ==============
def extract_features(model, dataloader, device):
    """提取特征向量 - 适配不同backbone的输出格式"""
    model.eval()
    features = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            # 适配不同输出格式
            if isinstance(outputs, tuple):
                if len(outputs) >= 4:  # NNLRNet 三分支: (logits, l1, l2, feat)
                    feat = outputs[3]
                else:  # 标准: (logits, features)
                    feat = outputs[1]
            else:
                feat = outputs
            
            # 确保特征是numpy数组
            if torch.is_tensor(feat):
                feat = feat.cpu().numpy()
            
            features.append(feat)
    
    return np.vstack(features)


def evaluate_model(model, query_loader, gallery_loader, query_pids, gallery_pids, device):
    """评估模型 - 计算Rank-1, Rank-5, mAP
    
    Args:
        model: ReID模型
        query_loader: 查询集数据加载器
        gallery_loader: 测试集数据加载器
        query_pids: 查询集原始PID列表 (用于身份匹配)
        gallery_pids: 测试集原始PID列表 (用于身份匹配)
        device: 计算设备
    """
    model.eval()
    
    # 检查 gallery 是否为空
    if len(gallery_pids) == 0:
        print("[错误] 测试集为空，无法评估！")
        return 0.0, 0.0, 0.0
    
    print("\n正在提取查询集特征...")
    query_features = extract_features(model, query_loader, device)
    
    print("正在提取底库特征...")
    gallery_features = extract_features(model, gallery_loader, device)
    
    print("正在计算相似度...")
    # 计算余弦相似度
    similarities = np.dot(query_features, gallery_features.T)
    
    # CMC和mAP计算
    CMC = np.zeros(len(gallery_pids))
    APs = []
    matched_queries = 0
    
    # 使用原始PID进行匹配
    query_size = len(query_pids)
    for query_idx in range(query_size):
        q_id = query_pids[query_idx]
        
        # 找到该查询的所有匹配项（同一ID在gallery中的索引）
        gallery_indices = [j for j, pid in enumerate(gallery_pids) if pid == q_id]
        
        if len(gallery_indices) == 0:
            # 查询ID在gallery中不存在，跳过（可能是distractor）
            continue
        
        matched_queries += 1
        
        # 获取相似度排名
        sim = similarities[query_idx]
        sorted_indices = np.argsort(-sim)
        
        # 找到第一个正确匹配
        found = False
        for rank, idx in enumerate(sorted_indices):
            if idx in gallery_indices:
                CMC[rank:] += 1
                APs.append(1.0 / (rank + 1))
                found = True
                break
        
        if not found:
            # 该查询在gallery中没有匹配（rank超过gallery大小）
            pass
    
    if matched_queries == 0:
        print("[警告] 没有匹配的查询，无法计算评估指标！")
        return 0.0, 0.0, 0.0
    
    CMC = CMC / matched_queries
    mAP = np.mean(APs) * 100 if APs else 0.0
    
    return CMC[0] * 100, CMC[4] * 100 if len(CMC) > 4 else 0, mAP


# ============== 训练函数 ==============
def train_one_epoch(model, dataloader, criterion_ce, criterion_triplet, optimizer, device, epoch, use_circle_loss=False):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    batch_count = len(dataloader)
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播 - 适配不同backbone的输出格式
        outputs = model(images)
        
        # 检查 NNLRNet 三分支输出: (logits_g, logits_l1, logits_l2, feat_concat)
        is_nnlrnet = isinstance(outputs, tuple) and len(outputs) >= 4
        
        if is_nnlrnet:
            logits_g, logits_l1, logits_l2, features = outputs[:4]
            
            if use_circle_loss:
                # Circle Loss 模式：传特征向量（768D），不要传 logits
                features = F.normalize(features, p=2, dim=1)
                loss_ce, cosine_logits = criterion_ce(features, labels)
            else:
                # 标准 CrossEntropy：三个分支的 logits 分别计算损失
                loss_ce = (criterion_ce(logits_g, labels) +
                           criterion_ce(logits_l1, labels) +
                           criterion_ce(logits_l2, labels)) / 3.0
                cosine_logits = None
                logits = logits_g  # 用于准确率计算
        elif use_circle_loss:
            # Circle Loss 模式：返回归一化特征
            if isinstance(outputs, tuple):
                _, features = outputs
            else:
                features = outputs
            
            # 特征归一化 (Circle Loss 需要)
            features = F.normalize(features, p=2, dim=1)
            
            # Circle Loss - 返回 (loss, cosine_logits 用于准确率)
            loss_ce, cosine_logits = criterion_ce(features, labels)
        else:
            # 标准模式
            if isinstance(outputs, tuple):
                logits, features = outputs
            else:
                features = outputs
                logits = None
            
            # 分类损失
            loss_ce = criterion_ce(logits, labels)
            cosine_logits = None
        
        # 验证损失值：检查是否为有效值，防止 LOSS=0 或负数问题
        if not torch.isfinite(loss_ce):
            print(f"  [警告] 检测到无效 loss_ce: {loss_ce.item()}, 使用默认损失")
            loss_ce = torch.tensor(1.0, device=loss_ce.device, requires_grad=True)
        
        # 三元组损失：使用同一个batch内的正负样本
        batch_size = images.size(0)
        if batch_size >= 3:  # 移除 loss_ce.item() > 0 检查，Circle Loss 可能为小值
            # 按标签分组构建三元组
            unique_labels = labels.unique()
            if len(unique_labels) >= 2:
                anchor_list = []
                positive_list = []
                negative_list = []
                
                for lbl in unique_labels:
                    mask = (labels == lbl).nonzero(as_tuple=True)[0]
                    if len(mask) >= 2:
                        anchor_idx = mask[0]
                        positive_idx = mask[1]
                        anchor_list.append(features[anchor_idx])
                        positive_list.append(features[positive_idx])
                        
                        # 为这个anchor选择negative
                        neg_labels = unique_labels[unique_labels != lbl]
                        if len(neg_labels) > 0:
                            neg_label = neg_labels[0]
                            neg_mask = (labels == neg_label).nonzero(as_tuple=True)[0]
                            if len(neg_mask) > 0:
                                negative_list.append(features[neg_mask[0]])
                
                # 确保长度一致
                if len(anchor_list) >= 1 and len(negative_list) >= 1:
                    min_len = min(len(anchor_list), len(positive_list), len(negative_list))
                    anchors = torch.stack(anchor_list[:min_len])
                    positives = torch.stack(positive_list[:min_len])
                    negatives = torch.stack(negative_list[:min_len])
                    loss_triplet = criterion_triplet(anchors, positives, negatives)
                else:
                    # 修复：使用 torch.tensor 确保类型一致，避免 LOSS=0 问题
                    loss_triplet = torch.tensor(0.0, device=features.device)
            else:
                # 修复：使用 torch.tensor 确保类型一致
                loss_triplet = torch.tensor(0.0, device=features.device)
        else:
            # 修复：使用 torch.tensor 确保类型一致
            loss_triplet = torch.tensor(0.0, device=features.device)
        
        # 总损失
        loss = loss_ce + 0.1 * loss_triplet
        
        # 损失验证：检查是否为有效值，防止 LOSS=0 或负数
        if not torch.isfinite(loss):
            print(f"  [警告] 检测到无效损失值: {loss.item()}, 跳过此批次")
            continue
        
        # 梯度裁剪：防止梯度爆炸
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        # 准确率计算：Circle Loss 用 cosine_logits，普通模式用 logits
        acc_logits = cosine_logits if cosine_logits is not None else logits
        if acc_logits is not None:
            _, predicted = acc_logits.max(1)
            correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # 显示进度
        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx+1}/{batch_count}, Loss: {loss.item():.4f}")
    
    acc = 100. * correct / total if total > 0 else 0
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return avg_loss, acc


def train_model(args):
    """完整训练流程"""
    print_section("ReID模型训练")
    
    # 设备
    # 强制使用 GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    
    # 数据目录：如果目录下直接有 bounding_box_train 就用它；否则兼容旧的 Market-1501 子目录
    data_dir = Path(args.data_dir)
    if (data_dir / 'bounding_box_train').exists():
        train_data_dir = data_dir
    else:
        train_data_dir = data_dir / 'Market-1501'

    if not (train_data_dir / 'bounding_box_train').exists():
        if args.download or args.demo:
            download_market1501(data_dir)
            train_data_dir = data_dir / 'Market-1501'
        else:
            print("\n[警告] 数据集不存在！")
            print(f"  位置: {train_data_dir}")
            print("\n选项:")
            print("  1. 使用 --download 自动下载")
            print("  2. 使用 --demo 创建演示数据集")
            print("  3. 手动下载并放置到指定目录")
            print("  4. 先运行 python tools/prepare_train_data.py 整理底库数据")
            
            if args.auto_demo:
                create_demo_dataset(data_dir)
                train_data_dir = data_dir / 'Market-1501'
            else:
                response = input("\n是否创建演示数据集? (y/n): ").strip().lower()
                if response == 'y':
                    create_demo_dataset(data_dir)
                    train_data_dir = data_dir / 'Market-1501'
                else:
                    return None

    
    # 数据增强
    train_transform_list = [
        transforms.Resize((256, 128)),
        transforms.RandomCrop((224, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    ]
    
    # 添加天气增强 (抗恶劣环境)
    if args.use_weather_aug:
        print("使用天气增强 (模拟恶劣环境)")
        # 默认包含：烟雾、雾霾、雨天、运动模糊、低光照、噪声
        # 新增：红外热成像、姿态畸变（搜救场景）
        weather_types = ['smoke', 'fog', 'rain', 'motion_blur', 'low_light', 'noise']
        if hasattr(args, 'use_infrared_aug') and args.use_infrared_aug:
            weather_types.append('infrared')
            print("  - 红外/热成像增强")
        if hasattr(args, 'use_pose_aug') and args.use_pose_aug:
            weather_types.append('pose_distortion')
            print("  - 搜救姿态增强 (倒地/匍匐/蜷缩)")
        if hasattr(args, 'use_viewpoint_aug') and args.use_viewpoint_aug:
            weather_types.append('viewpoint')
            print("  - 视角变换增强 (模拟不同拍摄角度)")
        train_transform_list.append(
            WeatherAugmentation(p=0.5, weather_types=weather_types)
        )
    
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 添加 RandomErasing (模拟遮挡)
    if args.use_random_erasing:
        train_transform_list.append(
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3))
        )
    
    train_transform = transforms.Compose(train_transform_list)
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.CenterCrop((224, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print_section("加载数据集")
    train_dataset = Market1501Dataset(str(train_data_dir), transform=train_transform, train=True)
    
    # Windows 兼容性：num_workers=0 更稳定
    num_workers = 0 if sys.platform == 'win32' else 4
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=(device.type == 'cuda')
    )
    
    # 获取标签映射和类别数量
    num_classes = len(train_dataset.label_mapping)
    print(f"身份数量: {num_classes}")
    
    # 查询和测试集 - 需要正确分离query和gallery
    # query来自query目录，gallery来自bounding_box_test目录
    query_dataset = Market1501Dataset(str(train_data_dir / 'query'), transform=test_transform, 
                                       label_mapping=train_dataset.label_mapping, mode='query')
    gallery_dataset = Market1501Dataset(str(train_data_dir / 'bounding_box_test'), transform=test_transform, 
                                         label_mapping=train_dataset.label_mapping, mode='gallery')
    
    query_size = len(query_dataset)
    gallery_size = len(gallery_dataset)
    
    # 使用原始PID用于评估（因为训练集和测试集身份ID不重叠）
    query_pids = query_dataset.pids
    gallery_pids = gallery_dataset.pids
    
    # 检查有多少查询的PID在gallery中存在
    query_unique_pids = set(query_pids)
    gallery_unique_pids = set(gallery_pids)
    valid_queries = sum(1 for pid in query_unique_pids if pid in gallery_unique_pids and pid >= 0)
    
    print(f"查询集大小: {query_size}")
    print(f"测试集大小: {gallery_size}")
    print(f"有效查询身份数量: {valid_queries}")
    
    # 创建查询和测试loader
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size, shuffle=False
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size, shuffle=False
    )
    
    # 创建模型
    print_section("初始化模型")
    backbone_name = args.backbone
    print(f"使用 {backbone_name} 作为backbone...")
    
    # 使用统一的骨干网络接口
    model = build_backbone(backbone_name, num_classes=num_classes)
    model = model.to(device)
    
    # 断点续训：加载已有checkpoint
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n[恢复训练] 从 {resume_path} 加载权重...")
            try:
                checkpoint = torch.load(str(resume_path), map_location=device, weights_only=False)
                state_dict = checkpoint['state_dict']
                # 兼容旧版本键名映射（_branch_N → branch_g/l1/l2）
                key_map = {'_branch_1': 'branch_g', '_branch_2': 'branch_l1', '_branch_3': 'branch_l2'}
                renamed = {}
                for k, v in state_dict.items():
                    new_k = k
                    for old_prefix, new_prefix in key_map.items():
                        if k.startswith(old_prefix):
                            new_k = k.replace(old_prefix, new_prefix, 1)
                            break
                    renamed[new_k] = v
                # strict=False 允许不完整加载
                missing, unexpected = model.load_state_dict(renamed, strict=False)
                if missing:
                    print(f"  模型缺少 {len(missing)} 个key（新初始化）")
                if unexpected:
                    print(f"  checkpoint有 {len(unexpected)} 个额外key（已忽略）")
                start_epoch = checkpoint.get('epoch', 0)
                print(f"  已加载 epoch {start_epoch} 的模型权重")
                if 'rank1' in checkpoint:
                    print(f"  该模型之前 Rank-1: {checkpoint['rank1']:.2f}%")
            except Exception as e:
                print(f"\n[错误] 加载 checkpoint 失败: {e}")
                print(f"  文件可能已损坏: {resume_path}")
                print(f"  建议删除该文件后重试，或换用其他 checkpoint。")
                print(f"  将从头开始训练。")
                start_epoch = 0
        else:
            print(f"[警告] 恢复路径不存在: {resume_path}，从头开始训练")
    
    # 微调：加载预训练权重（忽略分类器层维度不匹配）
    if args.finetune:
        finetune_path = Path(args.finetune)
        if finetune_path.exists():
            print(f"\n[微调模式] 从 {finetune_path} 加载骨干网络权重...")
            try:
                checkpoint = torch.load(str(finetune_path), map_location=device, weights_only=False)
                state_dict = checkpoint['state_dict']
                # 加载匹配的层，跳过维度不匹配的分类器
                model_dict = model.state_dict()
                loaded = 0
                skipped = 0
                for k, v in state_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        model_dict[k] = v
                        loaded += 1
                    elif k in model_dict:
                        skipped += 1
                        print(f"  跳过 {k}: checkpoint形状{v.shape} vs 模型形状{model_dict[k].shape}")
                model.load_state_dict(model_dict)
                print(f"  加载 {loaded} 个权重层，跳过 {skipped} 个不匹配层")
                if loaded > 0:
                    print(f"  预训练基础加载成功！")
                else:
                    print(f"  [警告] 没有成功加载任何权重层，将使用随机初始化。")
            except Exception as e:
                print(f"\n[错误] 加载微调权重失败: {e}")
                print(f"  文件可能已损坏: {finetune_path}")
                print(f"  建议删除该文件后重新训练生成，或关闭 --finetune 使用 ImageNet 预训练。")
                print(f"  将回退到默认预训练/随机初始化并继续训练。")
        else:
            print(f"[警告] 微调权重路径不存在: {finetune_path}，从头开始训练")
    
    # 获取特征维度
    if hasattr(model, 'feature_dim'):
        actual_feat_dim = model.feature_dim
    else:
        actual_feat_dim = args.feat_dim
        print(f"[警告] 模型未定义feature_dim，使用参数值: {actual_feat_dim}")
    
    print(f"特征维度: {args.feat_dim}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数
    if args.use_circle_loss:
        print("使用 Circle Loss")
        criterion_ce = CircleLoss(num_classes=num_classes, feat_dim=actual_feat_dim)
        criterion_triplet = TripletLoss(margin=args.margin)
    else:
        criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)  # 添加标签平滑
        criterion_triplet = TripletLoss(margin=args.margin)
    
    # 将损失函数移动到GPU
    criterion_ce = criterion_ce.to(device)
    criterion_triplet = criterion_triplet.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
    
    # 学习率调度器 - 使用Warmup
    warmup_epochs = min(5, args.epochs // 10)  # 最多5个epoch warmup
    if args.use_cosine_annealing:
        print(f"使用余弦退火学习率 (warmup: {warmup_epochs} epochs)")
        scheduler = WarmupCosineAnnealingLR(
            optimizer, warmup_epochs=warmup_epochs, max_epochs=args.epochs, min_lr=1e-6
        )
    else:
        print(f"使用阶梯衰减学习率 (warmup: {warmup_epochs} epochs, step=20)")
        scheduler = WarmupMultiStepLR(
            optimizer, warmup_epochs=warmup_epochs, milestones=[20, 40, 60], gamma=0.1
        )
    
    # 保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    print_section("开始训练")
    best_rank1 = 0
    start_time = time.time()
    
    total_epochs = args.epochs
    if args.resume and start_epoch > 0:
        total_epochs = start_epoch + args.epochs  # resume时epochs=继续训练的轮数
        print(f"断点续训: 已训练 {start_epoch} 轮，再续训 {args.epochs} 轮，总计 {total_epochs} 轮")
    
    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        print("-" * 40)
        
        # Circle Loss margin warmup: 前 20 个 epoch 线性增加 margin（更缓慢的warmup）
        if args.use_circle_loss and hasattr(criterion_ce, 'set_margin'):
            warmup_epochs = 20  # 从10增加到20，margin增加更缓慢
            if epoch < warmup_epochs:
                current_margin = (criterion_ce.margin * epoch) / warmup_epochs
                criterion_ce.set_margin(current_margin)
            else:
                criterion_ce.set_margin(criterion_ce.margin)
        
        # 训练
        epoch_start = time.time()
        loss, acc = train_one_epoch(
            model, train_loader, criterion_ce, criterion_triplet, 
            optimizer, device, epoch, use_circle_loss=args.use_circle_loss
        )
        epoch_time = time.time() - epoch_start
        
        print(f"  Loss: {loss:.4f}, Acc: {acc:.2f}%, Time: {epoch_time:.1f}s")
        
        # 学习率调整
        scheduler.step()
        
        # 每5个epoch评估一次
        if (epoch + 1) % 5 == 0:
            Rank1, Rank5, mAP = evaluate_model(
                model, query_loader, gallery_loader, query_pids, gallery_pids, device
            )
            print(f"\n  评估结果:")
            print(f"    Rank-1: {Rank1:.2f}%")
            print(f"    Rank-5: {Rank5:.2f}%")
            print(f"    mAP: {mAP:.2f}%")
            
            # 保存最佳模型
            if Rank1 > best_rank1:
                best_rank1 = Rank1
                save_path = save_dir / 'best_model.pth'
                torch.save({
                    'state_dict': model.state_dict(),
                    'rank1': Rank1,
                    'rank5': Rank5,
                    'mAP': mAP,
                    'epoch': epoch + 1,
                    'num_classes': num_classes,
                    'feat_dim': actual_feat_dim,
                    'backbone': backbone_name,
                    'label_mapping': train_dataset.label_mapping  # 保存标签映射
                }, save_path)
                print(f"  [保存] 最佳模型: {save_path}")
    
    # 保存最终模型
    total_time = time.time() - start_time
    final_path = save_dir / 'final_model.pth'
    torch.save({
        'state_dict': model.state_dict(),
        'rank1': best_rank1,
        'epoch': args.epochs,
        'num_classes': num_classes,
        'feat_dim': actual_feat_dim,
        'backbone': backbone_name,
        'train_time': total_time,
        'label_mapping': train_dataset.label_mapping  # 保存标签映射
    }, final_path)
    
    print_section("训练完成")
    print(f"总用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"最佳Rank-1: {best_rank1:.2f}%")
    print(f"模型保存: {save_dir}")
    
    return model, best_rank1


def evaluate_saved_model(model_path, data_dir):
    """评估已保存的模型"""
    print_section("评估模型")
    
    # 强制使用 GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # 加载模型 (weights_only=False 兼容 PyTorch 2.6+)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = checkpoint['num_classes']
    feat_dim = checkpoint['feat_dim']
    backbone_name = checkpoint.get('backbone', 'resnet50')  # 默认resnet50
    saved_label_mapping = checkpoint.get('label_mapping', None)  # 使用模型保存的标签映射
    
    print(f"模型信息: {num_classes} 身份, {feat_dim} 维特征, backbone: {backbone_name}")
    
    # 使用统一的骨干网络接口
    model = build_backbone(backbone_name, num_classes=num_classes)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    # 加载测试数据
    test_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.CenterCrop((224, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 使用模型保存的标签映射（如果可用）
    if saved_label_mapping is None:
        # 向后兼容：没有保存label_mapping时，从训练集重新创建
        print("[警告] 模型未保存标签映射，将重新创建...")
        train_dataset = Market1501Dataset(data_dir, transform=test_transform, train=True)
        label_mapping = train_dataset.label_mapping
    else:
        label_mapping = saved_label_mapping
    
    # 创建独立的查询和测试集
    query_dataset = Market1501Dataset(str(Path(data_dir) / 'query'), transform=test_transform, 
                                     label_mapping=label_mapping, mode='query')
    gallery_dataset = Market1501Dataset(str(Path(data_dir) / 'bounding_box_test'), transform=test_transform, 
                                        label_mapping=label_mapping, mode='gallery')
    
    # 使用原始PID用于评估
    query_pids = query_dataset.pids
    gallery_pids = gallery_dataset.pids
    
    # 创建查询和测试loader
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)
    
    Rank1, Rank5, mAP = evaluate_model(
        model, query_loader, gallery_loader, query_pids, gallery_pids, device
    )
    
    print(f"\n评估结果:")
    print(f"  Rank-1: {Rank1:.2f}%")
    print(f"  Rank-5: {Rank5:.2f}%")
    print(f"  mAP: {mAP:.2f}%")
    
    return Rank1, Rank5, mAP


# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(
        description='ReID模型训练 - 优化版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的骨干网络:
  osnet       - 专用轻量级ReID网络 (推荐)
  resnet50    - ResNet50 (ImageNet预训练)
  resnet34    - ResNet34 (轻量版)
  agw         - 带Non-local注意力
  resnet_ibn  - 带Instance Normalization
  pcb         - Part-based卷积基线

示例:
  # 使用OSNet骨干网络训练
  python train.py --backbone osnet --epochs 80
  
  # 使用OSNet + Circle Loss + RandomErasing
  python train.py --backbone osnet --use_circle_loss --use_random_erasing --epochs 80
  
  # 使用天气增强 (抗恶劣环境)
  python train.py --backbone osnet --use_weather_aug --epochs 80
  
  # 完整优化配置 (最佳效果)
  python train.py --backbone osnet --use_circle_loss --use_random_erasing --use_weather_aug --use_cosine_annealing --epochs 100
  
  # 评估已有模型
  python train.py --eval_only --model_path checkpoints/best_model.pth
        """
    )
    
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据集目录')
    parser.add_argument('--download', action='store_true',
                       help='下载Market-1501数据集')
    parser.add_argument('--demo', action='store_true',
                       help='使用演示数据集训练')
    parser.add_argument('--auto_demo', action='store_true',
                       help='自动创建演示数据集（无需确认）')
    parser.add_argument('--epochs', type=int, default=60,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小 (建议64或更大，32太小导致训练不稳定)')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='学习率 (建议0.0001，0.0003太高会导致Loss波动大)')
    parser.add_argument('--margin', type=float, default=0.3,
                       help='三元组损失margin')
    parser.add_argument('--feat_dim', type=int, default=512,
                       help='特征维度')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    
    # 骨干网络选择
    parser.add_argument('--backbone', type=str, default='osnet',
                       choices=list(SUPPORTED_BACKBONES.keys()),
                       help='骨干网络类型 (默认: osnet)')
    
    # 高级优化选项
    parser.add_argument('--use_circle_loss', action='store_true',
                       help='使用Circle Loss替代CrossEntropy (推荐用于ReID)')
    parser.add_argument('--use_random_erasing', action='store_true',
                       help='使用RandomErasing数据增强 (模拟遮挡)')
    parser.add_argument('--use_weather_aug', action='store_true',
                       help='使用天气增强 (模拟烟雾、雾霾、雨天、低光照等恶劣环境)')
    parser.add_argument('--use_infrared_aug', action='store_true',
                       help='使用红外/热成像增强 (夜间/浓烟搜救场景)')
    parser.add_argument('--use_pose_aug', action='store_true',
                       help='使用搜救姿态增强 (倒地/匍匐/蜷缩/残缺)')
    parser.add_argument('--use_viewpoint_aug', action='store_true',
                       help='使用视角变换增强 (模拟不同拍摄角度)')
    parser.add_argument('--use_cosine_annealing', action='store_true',
                       help='使用余弦退火学习率衰减 (替代阶梯衰减)')
    
    # 评估选项
    parser.add_argument('--eval_only', action='store_true',
                       help='仅评估模型')
    parser.add_argument('--model_path', type=str, default='',
                       help='模型路径（评估用）')
    parser.add_argument('--resume', type=str, default='',
                       help='断点续训：加载checkpoint继续训练，如 checkpoints/best_model.pth')
    parser.add_argument('--finetune', type=str, default='',
                       help='微调模式：加载预训练模型骨干权重（忽略分类器维度不匹配），如 checkpoints/best_model.pth')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  抗恶劣环境行人重识别系统 - 模型训练")
    print("=" * 70)
    
    # 评估模式
    if args.eval_only:
        if not args.model_path:
            print("[错误] 请指定模型路径: --model_path")
            return
        # 评估模式：兼容直接包含 bounding_box_train 的新目录结构
        data_path = Path(args.data_dir)
        if not (data_path / 'bounding_box_train').exists():
            data_path = data_path / 'Market-1501'
        evaluate_saved_model(args.model_path, str(data_path))
        return
    
    # 训练模式
    train_model(args)


if __name__ == '__main__':
    main()
