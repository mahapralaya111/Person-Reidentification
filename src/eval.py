"""
抗恶劣环境行人重识别系统 - 评估模块
计算 CMC、mAP、mINP 等评估指标

使用方法：
    python eval.py --model_path checkpoints/best_model.pth --data_dir ./data
    python eval.py --model_path checkpoints/best_model.pth --dataset market1501 --rerank
"""

import os
import argparse
import numpy as np
from pathlib import Path
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


# ============== 评估指标计算 ==============
def compute_cmc_map(query_features, gallery_features, query_pids, gallery_pids, query_camids=None, gallery_camids=None):
    """计算 CMC 和 mAP
    
    Args:
        query_features: 查询集特征 [N, D]
        gallery_features: 底库特征 [M, D]
        query_pids: 查询集行人ID [N]
        gallery_pids: 底库行人ID [M]
        query_camids: 查询集摄像头ID [N]（可选）
        gallery_camids: 底库摄像头ID [M]（可选）
    
    Returns:
        dict: 包含 cmc 和 mAP 的字典
    """
    # 计算相似度矩阵
    distmat = cosine_distance(query_features, gallery_features)
    
    # 处理数据格式（确保是 numpy 数组，支持列表索引）
    query_pids = np.asarray(query_pids)
    gallery_pids = np.asarray(gallery_pids)
    
    num_query = len(query_pids)
    
    # CMC @ k
    cmc = np.zeros((num_query, 100))
    ap = np.zeros(num_query)
    
    for i in range(num_query):
        # 获取相似度排序
        sorted_indices = np.argsort(distmat[i])
        
        # 获取查询的真实身份
        query_pid = query_pids[i]
        
        # 找出底库中相同身份的样本
        matches = (gallery_pids[sorted_indices] == query_pid)
        
        # 计算CMC
        match_positions = np.where(matches)[0]
        if len(match_positions) > 0:
            first_match = match_positions[0]
            cmc[i, first_match:] = 1
            
            # 计算 AP = 每个正确位置的 precision 的平均值
            precisions = []
            correct_count = 0
            for rank in range(len(sorted_indices)):
                if matches[rank]:
                    correct_count += 1
                    precisions.append(correct_count / (rank + 1))
            ap[i] = np.mean(precisions) if precisions else 0
    
    # CMC曲线
    cmc_scores = cmc.mean(axis=0)
    
    # mAP
    mAP = ap.mean()
    
    return {
        'cmc': cmc_scores,
        'mAP': mAP * 100,
        'Rank-1': cmc_scores[0] * 100,
        'Rank-5': cmc_scores[4] * 100,
        'Rank-10': cmc_scores[9] * 100,
        'Rank-20': cmc_scores[19] * 100,
    }


def compute_mINP(query_pids, gallery_pids, distmat):
    """计算 mINP (Mean Inverse Negative Penalty)
    
    INP = 1 / (max_rank_of_true_match - min_rank_of_true_match + 1)
    mINP = mean(INP) across all queries
    """
    query_pids = np.asarray(query_pids)
    gallery_pids = np.asarray(gallery_pids)
    num_query = len(query_pids)
    inp = np.zeros(num_query)
    
    for i in range(num_query):
        query_pid = query_pids[i]
        sorted_indices = np.argsort(distmat[i])
        
        # 找到所有匹配的位置
        match_ranks = np.where(gallery_pids[sorted_indices] == query_pid)[0]
        
        if len(match_ranks) > 0:
            max_rank = match_ranks.max() + 1
            min_rank = match_ranks.min() + 1
            inp[i] = 1.0 / (max_rank - min_rank + 1)
    
    mINP = inp.mean() * 100
    return mINP


def cosine_distance(qf, gf):
    """计算余弦距离矩阵"""
    # 归一化
    qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-8)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    
    # 余弦相似度
    similarities = np.dot(qf, gf.T)
    
    # 转为距离
    distmat = 1 - similarities
    
    return distmat


def euclidean_distance(qf, gf):
    """计算欧式距离矩阵"""
    distmat = np.zeros((len(qf), len(gf)))
    
    for i in range(len(qf)):
        diff = qf[i] - gf
        distmat[i] = np.sqrt(np.sum(diff ** 2, axis=1))
    
    return distmat


# ============== Re-ranking ==============
def k_reciprocal_reranking(qf, gf, k1=20, k2=6, lambda_value=0.3):
    """K-reciprocal encoding re-ranking
    
    基于 K-reciprocal 编码的重排序方法
    来自: 'Re-ranking Person Re-identification with k-reciprocal Encoding'
    
    Args:
        qf: 查询特征 [N, D]
        gf: 底库特征 [M, D]
        k1: 初始 k 值
        k2: 扩增 k 值
        lambda_value: Jaccard 距离权重
    
    Returns:
        重新排序后的距离矩阵
    """
    # 初始距离
    original_dist = cosine_distance(qf, gf)
    
    num_q, num_g = original_dist.shape
    
    # 处理 query 和 gallery 集合
    if num_q <= num_g:
        # Q <= G
        all_features = np.concatenate([qf, gf], axis=0)
    else:
        # Q > G
        all_features = np.concatenate([qf, gf], axis=0)
    
    all_features = all_features / (np.linalg.norm(all_features, axis=1, keepdims=True) + 1e-8)
    
    # 计算欧式距离
    euclidean_dist = np.zeros((len(all_features), len(all_features)))
    
    for i in range(len(all_features)):
        diff = all_features[i] - all_features
        euclidean_dist[i] = np.sqrt(np.sum(diff ** 2, axis=1))
    
    # 原始距离矩阵
    euclidean_dist_qq = euclidean_dist[:num_q, :num_q]
    euclidean_dist_qg = euclidean_dist[:num_q, num_q:]
    euclidean_dist_gg = euclidean_dist[num_q:, num_q:]
    
    # Jaccard距离
    def jaccard_distance(features):
        """计算Jaccard距离（基于K-reciprocal）"""
        n = len(features)
        Z = np.zeros((n, n), dtype=np.float32)
        
        # 计算初始排序
        initial_rank = np.argsort(euclidean_dist, axis=1)
        
        for i in range(n):
            # K-reciprocal set
            forward_k_neigh_index = initial_rank[i, :k1 + 1]
            backward_k_neigh_index = initial_rank[forward_k_neigh_index, :int(np.round(k1))]
            
            # 使用 isin 替代错误的 geterprop1d
            fi = forward_k_neigh_index[np.isin(forward_k_neigh_index, backward_k_neigh_index)]
            fi = np.append(fi, i)
            
            # 扩增 k 值
            k2_reprocal = np.intersect1d(forward_k_neigh_index, backward_k_neigh_index)
            
            # 扩增 reciprocal set
            k_reprocal_expanded = k2_reprocal
            for candidate in k2_reprocal:
                forward_candidate = initial_rank[candidate, :int(np.round(k1))]
                backward_candidate = initial_rank[forward_candidate, :int(np.round(k1))]
                if len(np.intersect1d(forward_k_neigh_index, backward_candidate)) > k1 / 2:
                    k_reprocal_expanded = np.union1d(k_reprocal_expanded, forward_candidate)
            
            k_reprocal_expanded = np.unique(k_reprocal_expanded)
            
            # 权重
            weight = np.exp(-euclidean_dist[i, k_reprocal_expanded])
            Z[i, k_reprocal_expanded] = weight / weight.sum()
        
        # Jaccard 距离
        Z = Z / (Z.sum(axis=1, keepdims=True) + 1e-8)
        Z = Z + Z.T
        return Z
    
    Z = jaccard_distance(all_features)
    
    # 最终距离
    final_dist = euclidean_dist_qg * (1 - lambda_value) + np.dot(euclidean_dist_qg, Z[num_q:, :]) * lambda_value
    
    return final_dist


def rerank_with_distmat(original_dist, k1=20, k2=6, lambda_value=0.3):
    """使用原始距离矩阵进行重排序
    
    简化版本的重排序
    """
    num_q, num_g = original_dist.shape
    
    # 合并 Q 和 G
    all_dist = np.zeros((num_q + num_g, num_q + num_g))
    all_dist[:num_q, num_q:] = original_dist
    all_dist[num_q:, :num_q] = original_dist.T
    
    # 获取 k 近邻
    initial_rank = np.argsort(all_dist, axis=1)
    
    # K-reciprocal encoding
    V = np.zeros_like(all_dist)
    
    for i in range(num_q + num_g):
        forward_k = initial_rank[i, :k1 + 1]
        backward_k = initial_rank[forward_k, :k1 // 2]
        
        k_reciprocal = np.intersect1d(forward_k, backward_k)
        
        if len(k_reciprocal) > 0:
            V[i, k_reciprocal] = 1.0 / len(k_reciprocal)
    
    # 局部匹配
    re_ranking_dist = np.zeros_like(original_dist)
    
    for i in range(num_q):
        V_i = V[i, num_q:]
        S_i = original_dist[i]
        
        # 获取 k1 近邻
        k2_neighbors = np.argsort(S_i)[:k2]
        
        # 局部匹配
        for j in k2_neighbors:
            V_j = V[num_q + j, :]
            S_j = original_dist[:, j]
            
            # 共同近邻
            common = np.intersect1d(np.where(V_i > 0)[0], np.where(V_j > 0)[0])
            
            if len(common) > 0:
                re_ranking_dist[i, j] = S_i[j] - lambda_value * len(common) / k1
    
    return re_ranking_dist


# ============== 评估函数 ==============
def evaluate_model(model, query_loader, gallery_loader, query_pids, gallery_pids, device, use_rerank=False):
    """评估模型
    
    Args:
        model: ReID模型
        query_loader: 查询集数据加载器
        gallery_loader: 底库数据加载器
        query_pids: 查询集行人ID
        gallery_pids: 底库行人ID
        device: 计算设备
        use_rerank: 是否使用重排序
    
    Returns:
        dict: 评估结果
    """
    print("正在提取查询集特征...")
    query_features, _ = extract_features(model, query_loader, device)
    
    print("正在提取底库特征...")
    gallery_features, _ = extract_features(model, gallery_loader, device)
    
    print("正在计算评估指标...")
    
    if use_rerank:
        print("正在应用重排序...")
        distmat = rerank_with_distmat(
            cosine_distance(query_features, gallery_features),
            k1=20, k2=6, lambda_value=0.3
        )
    else:
        distmat = cosine_distance(query_features, gallery_features)
    
    # 计算 CMC 和 mAP
    results = compute_cmc_map(query_features, gallery_features, query_pids, gallery_pids)
    
    # 计算 mINP
    results['mINP'] = compute_mINP(query_pids, gallery_pids, distmat)
    
    return results


def extract_features(model, dataloader, device):
    """提取特征"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            # 骨干网络返回格式: (logits, feat)
            outputs = model(images)
            if isinstance(outputs, tuple):
                feats = outputs[1]
            else:
                feats = outputs
            features.append(feats.cpu().numpy())
            labels.extend(targets.numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    
    return features, labels


# ============== 逐摄像头评估 ==============
def evaluate_by_camera(query_features, gallery_features, query_pids, gallery_pids, query_camids, gallery_camids):
    """按摄像头分组评估
    
    评估跨摄像头检索性能
    """
    results = {}
    
    # 获取所有摄像头ID
    all_cams = sorted(set(query_camids) | set(gallery_camids))
    
    for cam in all_cams:
        # 查询来自cam，底库来自其他摄像头
        query_mask = np.array(query_camids) == cam
        gallery_mask = np.array(gallery_camids) != cam
        
        if query_mask.sum() == 0 or gallery_mask.sum() == 0:
            continue
        
        q_feats = query_features[query_mask]
        g_feats = gallery_features[gallery_mask]
        q_pids = np.array(query_pids)[query_mask]
        g_pids = np.array(gallery_pids)[gallery_mask]
        
        result = compute_cmc_map(q_feats, g_feats, q_pids, g_pids)
        
        results[f'cam{cam}'] = {
            'Rank-1': result['Rank-1'],
            'mAP': result['mAP']
        }
    
    return results


# ============== 恶劣环境分组评估 ==============
def evaluate_by_environment(query_features, gallery_features, query_pids, gallery_pids, query_conditions=None):
    """按环境条件分组评估
    
    评估模型在不同恶劣环境下的性能
    """
    # 定义环境条件
    if query_conditions is None:
        query_conditions = ['normal'] * len(query_pids)
    
    results = {}
    
    for condition in set(query_conditions):
        mask = np.array([c == condition for c in query_conditions])
        
        if mask.sum() == 0:
            continue
        
        q_feats = query_features[mask]
        g_feats = gallery_features
        q_pids = np.array(query_pids)[mask]
        g_pids = np.array(gallery_pids)
        
        result = compute_cmc_map(q_feats, g_feats, q_pids, g_pids)
        
        results[condition] = {
            'Rank-1': result['Rank-1'],
            'Rank-5': result['Rank-5'],
            'mAP': result['mAP']
        }
    
    return results


# ============== 可视化 ==============
def save_ranking_results(query_images, gallery_images, query_pids, gallery_pids, distmat, save_path, top_k=10):
    """保存检索结果可视化
    
    保存前10个检索结果
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    num_queries = min(5, len(query_pids))
    
    fig, axes = plt.subplots(num_queries, top_k + 1, figsize=(20, num_queries * 2))
    
    for i in range(num_queries):
        # 查询图像
        axes[i, 0].imshow(query_images[i])
        axes[i, 0].set_title(f'Query\nPID:{query_pids[i]}', fontsize=8)
        axes[i, 0].axis('off')
        
        # 排序
        sorted_indices = np.argsort(distmat[i])[:top_k]
        
        for j, idx in enumerate(sorted_indices):
            match = gallery_pids[idx] == query_pids[i]
            color = 'green' if match else 'red'
            
            axes[i, j + 1].imshow(gallery_images[idx])
            axes[i, j + 1].set_title(
                f'Rank{j+1}\nPID:{gallery_pids[idx]}\n{"✓" if match else "✗"}',
                fontsize=7,
                color=color
            )
            axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"检索结果已保存到: {save_path}")


# ============== 主评估流程 ==============
def main():
    """无参数入口，从 sys.argv 解析参数（适配 main.py 调用）"""
    import sys
    parser = argparse.ArgumentParser(description='ReID模型评估')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet50', 'osnet', 'agw', 'resnet_ibn', 'pcb'],
                       help='骨干网络类型')
    parser.add_argument('--rerank', action='store_true', help='使用重排序')
    parser.add_argument('--output', type=str, default=None, help='输出结果JSON文件')

    args = parser.parse_args()
    
    # 检查数据集是否存在
    market_path = Path(args.data_dir) / 'Market-1501'
    if not market_path.exists():
        print(f"\n[提示] 未找到 Market-1501 数据集目录: {market_path}")
        print("[提示] 您有两种评估方式可选:")
        print()
        print("  方式1: 下载 Market-1501 数据集后评估")
        print(f"     python main.py train --download")
        print(f"     python main.py eval --model_path {args.model_path}")
        print()
        print("  方式2: 用您自定义的底库做交叉验证评估（推荐）")
        print("     python tools/evaluate_custom.py \\")
        print(f"       --model_path {args.model_path} \\")
        print(f"       --gallery rescue_gallery")
        print()
        return
    
    run_eval(args)


def run_eval(args):
    """实际评估逻辑"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from backbone import build_backbone

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据转换
    transform_test = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集 — 需显式指定 mode，否则默认找 bounding_box_train
    data_dir = Path(args.data_dir)

    query_dir = data_dir / 'Market-1501' / 'query'
    gallery_dir = data_dir / 'Market-1501' / 'bounding_box_test'

    from train import Market1501Dataset
    query_dataset = Market1501Dataset(str(query_dir), transform=transform_test, mode='query')
    gallery_dataset = Market1501Dataset(str(gallery_dir), transform=transform_test, mode='gallery')

    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    gallery_loader = DataLoader(gallery_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"查询集: {len(query_dataset)} 样本")
    print(f"底库: {len(gallery_dataset)} 样本")

    # 构建模型 — pretrained=False，权重从 checkpoint 加载
    model = build_backbone(args.backbone, num_classes=751, pretrained=False)

    # 加载权重
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"\n[错误] 无法加载模型文件: {args.model_path}")
        print(f"  原因: {e}")
        print(f"  该文件可能已损坏或未完整下载。")
        print(f"  建议: 删除该文件后重新训练/下载，或更换其他 checkpoint。")
        return
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()


    # 评估
    print("\n" + "=" * 50)
    print("开始评估...")
    print("=" * 50)
    
    results = evaluate_model(
        model, query_loader, gallery_loader,
        query_dataset.pids, gallery_dataset.pids,
        device, use_rerank=args.rerank
    )
    
    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果:")
    print("=" * 50)
    print(f"Rank-1:  {results['Rank-1']:.2f}%")
    print(f"Rank-5:  {results['Rank-5']:.2f}%")
    print(f"Rank-10: {results['Rank-10']:.2f}%")
    print(f"Rank-20: {results['Rank-20']:.2f}%")
    print(f"mAP:     {results['mAP']:.2f}%")
    if 'mINP' in results:
        print(f"mINP:    {results['mINP']:.2f}%")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_results = {
            'model': args.model_path,
            'backbone': args.backbone,
            'rerank': args.rerank,
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else list(v) 
                       for k, v in results.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        print(f"\n结果已保存到: {output_path}")
    
    return results


if __name__ == '__main__':
    main()
