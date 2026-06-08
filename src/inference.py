# -*- coding: utf-8 -*-
"""
ReID模型 推理工具
提供特征提取、行人检索、批量处理等功能
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from backbone import build_backbone, SUPPORTED_BACKBONES
from eval import rerank_with_distmat, cosine_distance


class ReidInferencer:
    """ReID推理器"""
    
    def __init__(self, model_path='checkpoints/best_model.pth', backbone='resnet50', num_classes=751, device=None):
        """
        初始化推理器
        
        Args:
            model_path: 模型路径
            backbone: 骨干网络类型
            num_classes: 类别数量
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ReID] 使用设备: {self.device}")
        
        # 验证骨干网络
        if backbone not in SUPPORTED_BACKBONES:
            print(f"[警告] 不支持的骨干网络: {backbone}，默认使用 resnet50")
            backbone = 'resnet50'
        
        # 加载模型
        self.model = build_backbone(backbone, num_classes=num_classes, loss='softmax', pretrained=False).to(self.device)
        self.model.eval()
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            # 尝试多种可能的key格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint  # 直接是state_dict
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[ReID] 模型加载成功: {model_path}")
            # 显示训练指标
            if 'rank1' in checkpoint:
                print(f"    Rank-1: {checkpoint.get('rank1', 'N/A'):.2f}%")
            if 'rank5' in checkpoint:
                print(f"    Rank-5: {checkpoint.get('rank5', 'N/A'):.2f}%")
            if 'mAP' in checkpoint:
                print(f"    mAP: {checkpoint.get('mAP', 'N/A'):.2f}%")
        else:
            print(f"[警告] 模型文件不存在: {model_path}")
        
        # 图像预处理 — 对齐训练时的 test_transform（含 CenterCrop）
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.CenterCrop((224, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_feature(self, image_path):
        """
        提取单张图片的特征
        
        Args:
            image_path: 图片路径
            
        Returns:
            特征向量 (512维 or 2048维)
        """
        with Image.open(image_path) as img_obj:
            img = img_obj.convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            if isinstance(outputs, tuple):
                _, feature = outputs
            else:
                feature = outputs
        
        return feature.cpu().numpy().flatten()
    
    def extract_features_batch(self, image_paths, batch_size=32):
        """
        批量提取图片特征
        
        Args:
            image_paths: 图片路径列表
            batch_size: 批次大小
            
        Returns:
            特征矩阵 (N, D)
        """
        features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                try:
                    with Image.open(path) as img_obj:
                        img = img_obj.convert('RGB')
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                except Exception as e:
                    print(f"[错误] 无法加载图片: {path}, {e}")
                    continue
            
            if batch_tensors:
                batch = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(batch)
                    if isinstance(outputs, tuple):
                        _, feat = outputs
                    else:
                        feat = outputs
                
                features.append(feat.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"[ReID] 已处理: {min(i+batch_size, len(image_paths))}/{len(image_paths)}")
        
        return np.vstack(features) if features else np.array([])
    
    def compute_similarity(self, feat1, feat2, method='cosine'):
        """
        计算两个特征向量的相似度
        
        Args:
            feat1: 特征向量1
            feat2: 特征向量2
            method: 'cosine' or 'euclidean'
            
        Returns:
            相似度分数
        """
        if method == 'cosine':
            norm1 = np.linalg.norm(feat1)
            norm2 = np.linalg.norm(feat2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(feat1, feat2) / (norm1 * norm2)
        elif method == 'euclidean':
            return -np.linalg.norm(feat1 - feat2)  # 负的距离，越大越相似
        return 0
    
    def search(self, query_feature, gallery_features, gallery_ids=None, top_k=10, method='cosine', rerank=False):
        """
        在底库中搜索与查询最相似的行人
        
        Args:
            query_feature: 查询特征
            gallery_features: 底库特征矩阵 (N, D)
            gallery_ids: 底库ID列表
            top_k: 返回前k个结果
            method: 相似度计算方法
            rerank: 是否使用重排序
            
        Returns:
            [(id, similarity), ...] 按相似度降序排列
        """
        if len(gallery_features) == 0:
            return []
        
        if gallery_ids is None:
            gallery_ids = list(range(len(gallery_features)))
        
        # 特征中心化: 减去底库均值，去除共有"基底"相似度，增强不同人之间的区分度
        gallery_mean = np.mean(gallery_features, axis=0)
        query_centered = query_feature - gallery_mean
        gallery_centered = gallery_features - gallery_mean

        # 计算距离矩阵
        if rerank:
            # 对于重排序，需要计算完整的距离矩阵
            query_features = query_centered.reshape(1, -1)
            distmat = cosine_distance(query_features, gallery_centered)
            
            # 应用重排序
            reranked_dist = rerank_with_distmat(distmat, k1=20, k2=6, lambda_value=0.3)
            similarities = [(gallery_ids[i], 1.0 - reranked_dist[0, i]) for i in range(len(gallery_ids))]
        else:
            # 原始方法
            similarities = []
            for i, gallery_feat in enumerate(gallery_centered):
                if method == 'cosine':
                    # 对已 L2 归一化的特征，点积 = 余弦相似度
                    sim = max(0.0, float(np.dot(query_centered, gallery_feat)))
                else:
                    sim = -np.linalg.norm(query_centered - gallery_feat)
                similarities.append((gallery_ids[i], sim))
        
        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def build_gallery(self, gallery_dir, recursive=True):
        """
        构建底库特征库
        
        支持两种模式:
        1. 子文件夹模式(推荐): rescue_gallery/张三/img1.jpg, 张三/img2.jpg → ID都是"张三"
        2. 扁平模式: rescue_gallery/张三_01.jpg → ID是"张三_01"
        
        Args:
            gallery_dir: 底库图片目录
            recursive: 是否递归搜索子目录
            
        Returns:
            (features, ids, paths)
        """
        # 收集所有图片
        image_paths = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.gif']
        if recursive:
            for ext in extensions:
                image_paths.extend(Path(gallery_dir).rglob(ext))
        else:
            for ext in extensions:
                image_paths.extend(Path(gallery_dir).glob(ext))
        
        image_paths = [str(p) for p in image_paths]
        
        # 过滤掉非人员图片（演示生成的临时文件、搜索结果图等）
        def _is_person_image(path):
            name = os.path.basename(path).lower()
            return not any(keyword in name for keyword in [
                '_query_temp', 'search_results', 'temp_', 'thumbnail', '._'
            ])
        image_paths = [p for p in image_paths if _is_person_image(p)]
        
        if not image_paths:
            print(f"[警告] 底库目录为空: {gallery_dir}")
            return np.array([]), [], []
        
        print(f"[ReID] 底库图片数量: {len(image_paths)}")
        
        # 提取特征
        features = self.extract_features_batch(image_paths)
        
        # 生成ID: 如果有子文件夹，用子文件夹名作为ID（支持同一人多张照片）
        ids = []
        person_count = {}
        for p in image_paths:
            parent = Path(p).parent
            # 如果父文件夹名不同于底库根目录，说明是子文件夹模式
            if parent.name != Path(gallery_dir).name:
                person_id = parent.name  # 用子文件夹名作为ID
            else:
                person_id = Path(p).stem  # 否则用文件名作为ID
            
            # 统计每个人员照片数量
            if person_id not in person_count:
                person_count[person_id] = 0
            person_count[person_id] += 1
            
            ids.append(person_id)
        
        # 打印人员统计
        unique_persons = set(ids)
        print(f"[ReID] 人员数量: {len(unique_persons)}")
        print(f"[ReID] 人员照片分布:")
        for person, count in person_count.items():
            print(f"    - {person}: {count} 张照片")
        
        return features, ids, image_paths


def main():
    """命令行工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ReID模型推理工具')
    parser.add_argument('--mode', type=str, default='feature', choices=['feature', 'search', 'gallery'],
                       help='运行模式: feature=提取特征, search=搜索, gallery=构建底库')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', help='模型路径')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--images', type=str, help='图片目录')
    parser.add_argument('--query', type=str, help='查询图片')
    parser.add_argument('--top_k', type=int, default=10, help='返回前k个结果')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--rerank', action='store_true', help='使用重排序')
    
    args = parser.parse_args()
    
    # 创建推理器
    infer = ReidInferencer(model_path=args.model)
    
    def get_first_image(directory):
        """获取目录中的第一张图片"""
        from pathlib import Path
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp', '*.gif']
        for ext in extensions:
            files = sorted(Path(directory).glob(ext)) + sorted(Path(directory).rglob(ext))
            if files:
                return str(files[0])
        return None
    
    if args.mode == 'feature':
        if args.image or args.images:
            # 自动检测是文件还是目录
            input_path = args.image or args.images
            if os.path.isdir(input_path):
                # 目录：自动选取第一张图片（仅用于显示）
                first_img = get_first_image(input_path)
                if first_img:
                    print(f"[INFO] 检测到目录，自动选取第一张图片: {first_img}")
                else:
                    print(f"[错误] 目录为空或不包含图片: {input_path}")
                    return
                # 目录：批量提取（使用原始目录路径）
                print(f"[INFO] 检测到目录，自动使用批量模式...")
                original_dir = input_path  # 保留原始目录路径
                paths = [str(p) for p in Path(original_dir).rglob('*.jpg')]
                paths += [str(p) for p in Path(original_dir).rglob('*.jpeg')]
                paths += [str(p) for p in Path(original_dir).rglob('*.png')]
                paths += [str(p) for p in Path(original_dir).rglob('*.webp')]
                paths += [str(p) for p in Path(original_dir).rglob('*.bmp')]
                paths += [str(p) for p in Path(original_dir).rglob('*.gif')]
                if not paths:
                    print(f"[错误] 目录为空或不包含图片: {original_dir}")
                else:
                    print(f"[INFO] 发现 {len(paths)} 张图片")
                    feats = infer.extract_features_batch(paths)
                    print(f"\n提取完成: {len(paths)} 张图片, 特征维度: {feats.shape[1]}")
                    if args.output:
                        out_path = args.output
                        # 如果输出路径指向一个已存在的目录，在里面创建 features.npy
                        if os.path.isdir(out_path):
                            out_path = os.path.join(out_path, 'features.npy')
                        np.save(out_path, feats)
                        print(f"特征矩阵已保存: {out_path}, shape: {feats.shape}")
            else:
                # 单文件：单张提取
                feat = infer.extract_feature(input_path)
                print(f"特征维度: {len(feat)}")
                print(f"特征范数: {np.linalg.norm(feat):.4f}")
                if args.output:
                    np.save(args.output, feat)
                    print(f"特征已保存: {args.output}")
    
    elif args.mode == 'gallery':
        if args.images:
            features, ids, paths = infer.build_gallery(args.images)
            if args.output:
                np.savez(args.output, features=features, ids=np.array(ids))
                print(f"底库已保存: {args.output}")
    
    elif args.mode == 'search':
        if args.query and args.images:
            # 如果query是目录，自动选取第一张图片
            if os.path.isdir(args.query):
                first_img = get_first_image(args.query)
                if first_img:
                    print(f"[INFO] 检测到查询目录，自动选取第一张图片: {first_img}")
                    args.query = first_img
                else:
                    print(f"[错误] 查询目录为空: {args.query}")
                    return
            
            # 构建底库
            features, ids, _ = infer.build_gallery(args.images)
            if len(features) == 0:
                print("[错误] 底库为空")
                return
            
            # 查询
            query_feat = infer.extract_feature(args.query)
            
            # 获取更多结果用于聚合（因为同一个人可能有多张照片）
            results_all = infer.search(query_feat, features, ids, top_k=len(ids), rerank=args.rerank)
            
            # 按ID聚合：同一个人取最高相似度
            person_scores = {}
            for pid, sim in results_all:
                if pid not in person_scores or sim > person_scores[pid]:
                    person_scores[pid] = sim
            
            # 按相似度排序
            aggregated = sorted(person_scores.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n查询图片: {args.query}")
            print(f"底库: {len(ids)} 张照片, {len(person_scores)} 位不同人员")
            print(f"\n【人员识别结果】")
            print("-" * 50)
            for i, (pid, sim) in enumerate(aggregated[:args.top_k]):
                # 判断置信度（针对现实应用调整阈值）
                # 现实应用中，由于视角、光照、姿态变化，相似度普遍较低
                # 建议根据实际场景调整阈值：搜救场景可适当放宽
                if sim > 0.6:
                    confidence = "高置信度"
                elif sim > 0.4:
                    confidence = "中置信度"
                elif sim > 0.2:
                    confidence = "低置信度"
                else:
                    confidence = "极低置信度"
                print(f" {i+1}. {pid:15s}  相似度: {sim:.4f}  [{confidence}]")


if __name__ == '__main__':
    main()
