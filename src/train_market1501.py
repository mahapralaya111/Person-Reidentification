"""
Market-1501 数据集训练脚本
用于行人重识别模型训练

使用方法:
1. 自动下载: python train_market1501.py --download
2. 使用本地数据: python train_market1501.py --data_dir <path>
3. 仅下载数据集: python train_market1501.py --download_only
"""

import os
import sys
import argparse
import zipfile
import urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
import time

# 进度显示
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()


class Market1501Dataset(Dataset):
    """Market-1501 数据集加载器"""
    
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.train = train
        self.images = []
        self.labels = []
        self.camera_ids = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """加载数据集"""
        if self.train:
            # 训练集
            bbox_dir = self.data_dir / 'bounding_box_train'
            if not bbox_dir.exists():
                print(f"[错误] 训练目录不存在: {bbox_dir}")
                return
            
            # 解析文件名格式: 0001_c1s1_001101_00.jpg
            # 格式: [id]_c[camera]s[sequence]_[frame]_[track].jpg
            for img_file in bbox_dir.glob('*.jpg'):
                name = img_file.stem
                parts = name.split('_')
                if len(parts) >= 2:
                    try:
                        pid = int(parts[0])
                        cam_id = int(parts[1][1])  # c1 -> 1
                        self.images.append(str(img_file))
                        self.labels.append(pid)
                        self.camera_ids.append(cam_id)
                    except Exception:
                        continue
            
            print(f"[Market-1501] 训练集: {len(self.images)} 张图片, {len(set(self.labels))} 个身份")
        
        else:
            # 查询集和测试集
            query_dir = self.data_dir / 'query'
            gallery_dir = self.data_dir / 'bounding_box_test'
            
            for img_file in query_dir.glob('*.jpg'):
                name = img_file.stem
                parts = name.split('_')
                if len(parts) >= 2:
                    try:
                        pid = int(parts[0])
                        cam_id = int(parts[1][1])
                        self.images.append(str(img_file))
                        self.labels.append(pid)
                        self.camera_ids.append(cam_id)
                    except Exception:
                        continue
            
            for img_file in gallery_dir.glob('*.jpg'):
                name = img_file.stem
                parts = name.split('_')
                if len(parts) >= 2:
                    try:
                        pid = int(parts[0])
                        cam_id = int(parts[1][1])
                        self.images.append(str(img_file))
                        self.labels.append(pid)
                        self.camera_ids.append(cam_id)
                    except Exception:
                        continue
            
            print(f"[Market-1501] 查询+测试集: {len(self.images)} 张图片")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        with Image.open(img_path) as img_file:
            img = img_file.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def download_market1501(data_dir):
    """下载 Market-1501 数据集"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Market-1501 下载链接
    urls = [
        ('https://rdm.ding.pku.edu.cn/dataset/market1501/Market-1501-v15.09.15.zip', 
         'market1501.zip'),
        # 备用链接 (如果主链接不可用)
        # ('https://drive.google.com/file/d/0B0eAyBuvSiC5fkExRHJaSXdIZWc/view', ...)
    ]
    
    zip_path = data_dir / 'market1501.zip'
    
    print("=" * 60)
    print("Market-1501 数据集下载")
    print("=" * 60)
    print(f"下载地址: {urls[0][0]}")
    print(f"保存位置: {zip_path}")
    print(f"数据集大小: 约 250MB")
    print("=" * 60)
    
    # 尝试下载
    try:
        print("\n正在下载... (这可能需要几分钟)")
        
        # 使用 urllib 下载 (带进度显示)
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                print_progress_bar(
                    min(downloaded, total_size), total_size,
                    prefix='下载进度:', suffix='完成'
                )
        
        urllib.request.urlretrieve(urls[0][0], zip_path, reporthook=report_progress)
        
        print("\n下载完成！正在解压...")
        
        # 解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # 重命名文件夹
        extracted_dir = data_dir / 'Market-1501-v15.09.15'
        target_dir = data_dir / 'Market-1501'
        
        if extracted_dir.exists() and not target_dir.exists():
            extracted_dir.rename(target_dir)
        
        # 删除zip文件
        zip_path.unlink()
        
        print(f"\n数据集已解压到: {target_dir}")
        print("\n目录结构:")
        for item in target_dir.iterdir():
            print(f"  - {item.name}")
        
        return target_dir
        
    except Exception as e:
        print(f"\n[错误] 下载失败: {e}")
        print("\n请手动下载数据集:")
        print("1. 访问: https://rdm.ding.pku.edu.cn/dataset/market1501")
        print("2. 下载 Market-1501-v15.09.15.zip")
        print("3. 解压到当前目录")
        return None


def create_simple_dataset(data_dir, num_ids=150, samples_per_id=4):
    """
    创建简化数据集用于快速测试
    当没有真实数据集时使用
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'bounding_box_train'
    test_dir = data_dir / 'bounding_box_test'
    query_dir = data_dir / 'query'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n创建简化数据集...")
    print(f"  身份数量: {num_ids}")
    print(f"  每身份训练样本: {samples_per_id}")
    print(f"  每身份测试样本: 2")
    
    np.random.seed(42)
    
    for pid in range(1, num_ids + 1):
        # 为每个身份生成独特的衣服颜色
        color = np.random.randint(50, 200, 3).tolist()
        
        # 训练样本
        for i in range(samples_per_id):
            img = np.random.randint(20, 80, (256, 128, 3), dtype=np.uint8)
            # 添加人形区域
            cv2.rectangle(img, (30, 50), (98, 230), color, -1)
            # 头部
            cv2.circle(img, (64, 35), 25, (220, 180, 160), -1)
            # 添加噪声模拟不同光照
            noise = np.random.randint(-30, 30, (256, 128, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            filename = f"{pid:04d}_c{1}s1_001101_{i:02d}.jpg"
            cv2.imwrite(str(train_dir / filename), img)
        
        # 测试样本
        for i in range(2):
            img = np.random.randint(20, 80, (256, 128, 3), dtype=np.uint8)
            cv2.rectangle(img, (30, 50), (98, 230), color, -1)
            cv2.circle(img, (64, 35), 25, (220, 180, 160), -1)
            noise = np.random.randint(-40, 40, (256, 128, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            filename = f"{pid:04d}_c{1}s1_001102_{i:02d}.jpg"
            cv2.imwrite(str(test_dir / filename), img)
            cv2.imwrite(str(query_dir / filename), img)
    
    print("简化数据集创建完成！")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, features = model(images, return_feat=False)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, query_loader, gallery_loader, device, query_labels, gallery_labels):
    """评估模型 (计算 CMC 和 mAP)"""
    model.eval()
    
    print("\n正在提取查询集特征...")
    query_features = []
    with torch.no_grad():
        for images, _ in query_loader:
            images = images.to(device)
            feat = model(x_visible=images)
            query_features.append(feat.cpu().numpy())
    query_features = np.vstack(query_features)
    
    print("正在提取底库特征...")
    gallery_features = []
    with torch.no_grad():
        for images, _ in gallery_loader:
            images = images.to(device)
            feat = model(x_visible=images)
            gallery_features.append(feat.cpu().numpy())
    gallery_features = np.vstack(gallery_features)
    
    print("正在计算相似度...")
    # 计算余弦相似度
    similarities = np.dot(query_features, gallery_features.T)
    
    # CMC 和 mAP 计算
    CMC = np.zeros(len(gallery_labels))
    AP = []
    
    for i, query_label in enumerate(set(query_labels)):
        # 找到该查询的所有匹配项
        query_indices = [j for j, l in enumerate(query_labels) if l == query_label]
        gallery_indices = [j for j, l in enumerate(gallery_labels) if l == query_label]
        
        if len(gallery_indices) == 0:
            continue
        
        # 获取相似度排名
        sim = similarities[i]
        sorted_indices = np.argsort(-sim)
        
        # 找到第一个正确匹配的位置
        matches = 0
        precisions = []
        for rank, idx in enumerate(sorted_indices):
            if idx in gallery_indices:
                matches += 1
                precisions.append(matches / (rank + 1))
                CMC[rank] += 1
                break
        
        if precisions:
            AP.append(np.mean(precisions))
    
    CMC = CMC / len(set(query_labels))
    mAP = np.mean(AP)
    Rank1 = CMC[0] * 100
    Rank5 = CMC[4] * 100 if len(CMC) > 4 else 0
    
    return Rank1, Rank5, mAP * 100


def main():
    parser = argparse.ArgumentParser(description='Market-1501 ReID Training')
    parser.add_argument('--data_dir', type=str, default='./market1501_data',
                       help='数据集目录')
    parser.add_argument('--download', action='store_true',
                       help='下载并解压数据集')
    parser.add_argument('--download_only', action='store_true',
                       help='仅下载数据集')
    parser.add_argument('--create_demo', action='store_true',
                       help='创建演示数据集')
    parser.add_argument('--epochs', type=int, default=60,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='学习率')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  面向灾害应急搜救的抗恶劣环境行人重识别系统")
    print("  Market-1501 数据集训练")
    print("=" * 60)
    
    # 检查 OpenCV
    try:
        import cv2
    except ImportError:
        print("[错误] 请先安装 OpenCV: pip install opencv-python")
        return
    
    global cv2
    import cv2
    
    # 下载数据集
    if args.download or args.download_only:
        data_dir = download_market1501(args.data_dir)
        if args.download_only:
            return
        if data_dir:
            args.data_dir = str(data_dir)
    
    # 检查数据集是否存在
    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'bounding_box_train'
    
    if not train_dir.exists():
        print("\n[警告] 数据集不存在！")
        print(f"  预期位置: {train_dir}")
        print("\n请选择以下选项之一:")
        print("  1. 使用 --download 下载数据集")
        print("  2. 使用 --create_demo 创建演示数据集")
        print("  3. 手动下载并放置到指定目录")
        
        # 创建演示数据集
        print("\n是否创建演示数据集进行快速测试? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            create_simple_dataset(args.data_dir, num_ids=150, samples_per_id=4)
        else:
            return
    
    # 创建设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomCrop((224, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.CenterCrop((224, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print("\n加载数据集...")
    train_dataset = Market1501Dataset(args.data_dir, transform=train_transform, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    num_classes = len(set(train_dataset.labels))
    print(f"身份数量: {num_classes}")
    
    # 加载测试数据
    query_dataset = Market1501Dataset(args.data_dir, transform=test_transform, train=False)
    gallery_dataset = Market1501Dataset(args.data_dir, transform=test_transform, train=False)
    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=args.num_workers)
    
    # 创建模型
    print("\n初始化模型...")
    from reid_model import ReIDModel
    model = ReIDModel(num_classes=num_classes, feat_dim=512, modality='visible')
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # 保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    
    best_rank1 = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        start_time = time.time()
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
        epoch_time = time.time() - start_time
        
        print(f"  损失: {loss:.4f}, 准确率: {acc:.2f}%, 用时: {epoch_time:.1f}s")
        
        # 学习率调整
        scheduler.step()
        
        # 每10个epoch评估一次
        if (epoch + 1) % 10 == 0:
            Rank1, Rank5, mAP = evaluate(model, query_loader, gallery_loader, device,
                                        query_dataset.labels, gallery_dataset.labels)
            print(f"\n  评估结果:")
            print(f"    Rank-1: {Rank1:.2f}%")
            print(f"    Rank-5: {Rank5:.2f}%")
            print(f"    mAP: {mAP:.2f}%")
            
            # 保存最佳模型
            if Rank1 > best_rank1:
                best_rank1 = Rank1
                save_path = save_dir / 'best_model.pth'
                torch.save(model.state_dict(), save_path)
                print(f"  [保存] 最佳模型: {save_path}")
    
    # 保存最终模型
    final_path = save_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\n训练完成！模型保存至: {final_path}")
    
    # 保存特征库
    print("\n正在保存特征库...")
    gallery_features = []
    model.eval()
    with torch.no_grad():
        for images, _ in gallery_loader:
            images = images.to(device)
            feat = model(x_visible=images)
            gallery_features.append(feat.cpu().numpy())
    
    gallery_features = np.vstack(gallery_features)
    feature_path = save_dir / 'gallery_features.npz'
    np.savez(feature_path, 
             features=gallery_features,
             labels=np.array(gallery_dataset.labels),
             image_paths=np.array(gallery_dataset.images, dtype=object))
    print(f"特征库保存至: {feature_path}")


if __name__ == '__main__':
    main()
