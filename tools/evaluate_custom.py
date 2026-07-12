"""
针对自定义底库的标准化评估脚本
使用方法:
    python tools/evaluate_custom.py --model_path checkpoints/best_model.pth --gallery rescue_gallery
    
输出:
    1. 控制台: Rank-1, Rank-5, mAP, 混淆矩阵
    2. 图片: 相似度分布图 -> similarity_distribution.png
    3. 图片: 混淆矩阵 -> confusion_matrix.png
"""
import sys, os, argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from video_processor import load_reid_model, ReIDRetriever

def main():
    parser = argparse.ArgumentParser(description='自定义底库标准化评估')
    parser.add_argument('--model_path', default='checkpoints/best_model.pth')
    parser.add_argument('--gallery', default='rescue_gallery')
    parser.add_argument('--output', default='eval_results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)
    
    # 1. 加载模型
    print("="*60)
    print("加载模型...")
    model = load_reid_model(args.model_path, device)
    retriever = ReIDRetriever(model, device)
    
    # 2. 读取底库图片
    gallery_dir = Path(args.gallery)
    person_dirs = sorted([d for d in gallery_dir.iterdir() if d.is_dir() and not d.name.startswith('_')])
    print(f"\n底库人员: {[d.name for d in person_dirs]}")
    
    all_features = []  # [N, D]
    all_labels = []    # [N]
    label_names = []   # 人名列表
    label_to_id = {}
    
    for person_dir in person_dirs:
        pid = person_dir.name
        label_to_id[pid] = len(label_names)
        label_names.append(pid)
        imgs = sorted(person_dir.glob('*'))
        for p in imgs:
            if not p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']:
                continue
            img = cv2.imread(str(p))
            if img is None: continue
            hh, ww = img.shape[:2]
            feat = retriever.extract_feature(img, [0, 0, ww, hh])
            if feat is not None:
                all_features.append(feat)
                all_labels.append(label_to_id[pid])
    
    features = np.array(all_features)
    labels = np.array(all_labels)
    n_persons = len(label_names)
    print(f"总图片: {len(features)}, 人员数: {n_persons}")
    
    # 3. 留一法交叉验证 (Leave-One-Image-Out)
    print("\n" + "="*60)
    print("执行留一法交叉验证...")
    
    correct_top1 = 0
    correct_top5 = 0
    total = len(features)
    ap_per_query = []
    
    for i in range(total):
        query_feat = features[i]
        query_label = labels[i]
        
        # 其他为gallery
        gallery_feats = np.delete(features, i, axis=0)
        gallery_lbls = np.delete(labels, i, axis=0)
        
        # 计算相似度
        sims = np.dot(gallery_feats, query_feat)
        
        # 按相似度排序
        sorted_idx = np.argsort(sims)[::-1]
        sorted_labels = gallery_lbls[sorted_idx]
        sorted_sims = sims[sorted_idx]
        
        # 当前query的真实身份在gallery中的位置
        match_positions = np.where(sorted_labels == query_label)[0]
        
        # Rank-1
        if len(match_positions) > 0 and match_positions[0] < 1:
            correct_top1 += 1
        # Rank-5
        if len(match_positions) > 0 and match_positions[0] < 5:
            correct_top5 += 1
        
        # AP (Average Precision)
        if len(match_positions) > 0:
            relevant = sorted_labels == query_label
            precisions = []
            for k, is_rel in enumerate(relevant[:50], 1):
                if is_rel:
                    precisions.append(np.sum(relevant[:k]) / k)
            ap = np.mean(precisions) if precisions else 0
            ap_per_query.append(ap)
    
    rank1 = correct_top1 / total * 100
    rank5 = correct_top5 / total * 100
    mAP = np.mean(ap_per_query) * 100 if ap_per_query else 0
    
    print(f"\n{'='*60}")
    print(f"    评估结果 (留一法交叉验证)")
    print(f"{'='*60}")
    print(f"  总查询数: {total}")
    print(f"  人员数:    {n_persons}")
    print(f"  Rank-1:   {rank1:.2f}%")
    print(f"  Rank-5:   {rank5:.2f}%")
    print(f"  mAP:      {mAP:.2f}%")
    
    # 4. 混淆矩阵
    print("\n" + "="*60)
    print("混淆矩阵 (每张查询图的最匹配结果)")
    print("="*60)
    print(f"{'':>12}", end="")
    for n in label_names:
        print(f"{n:>12}", end="")
    print()
    
    confusion = np.zeros((n_persons, n_persons), dtype=int)
    for i in range(total):
        query_label = labels[i]
        query_feat = features[i]
        gallery_feats = np.delete(features, i, axis=0)
        gallery_lbls = np.delete(labels, i, axis=0)
        sims = np.dot(gallery_feats, query_feat)
        best_idx = np.argmax(sims)
        pred_label = gallery_lbls[best_idx]
        confusion[query_label, pred_label] += 1
    
    for i, name in enumerate(label_names):
        print(f"{name:>8}:", end="")
        for j in range(n_persons):
            print(f"{confusion[i,j]:>8}", end=" ")
        correct = confusion[i,i]
        total_i = np.sum(confusion[i,:])
        acc = correct/total_i*100 if total_i > 0 else 0
        print(f"  {acc:.0f}%")
    
    # 5. 相似度分布图
    print("\n生成相似度分布图...")
    same_sims = []
    diff_sims = []
    for i in range(total):
        for j in range(i+1, total):
            sim = np.dot(features[i], features[j])
            if labels[i] == labels[j]:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(same_sims, bins=30, alpha=0.7, color='green', label=f'Same Person\nn={len(same_sims)}')
    axes[0].hist(diff_sims, bins=30, alpha=0.7, color='red', label=f'Different\nn={len(diff_sims)}')
    axes[0].axvline(x=0.5, color='blue', linestyle='--', label='Threshold=0.5')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Similarity Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 混淆矩阵热图
    im = axes[1].imshow(confusion, cmap='Blues', aspect='auto')
    axes[1].set_xticks(range(n_persons))
    axes[1].set_yticks(range(n_persons))
    axes[1].set_xticklabels(label_names, rotation=45, ha='right')
    axes[1].set_yticklabels(label_names)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Confusion Matrix')
    for i in range(n_persons):
        for j in range(n_persons):
            axes[1].text(j, i, str(confusion[i,j]), ha='center', va='center',
                       color='white' if confusion[i,j] > confusion.max()/2 else 'black')
    
    plt.tight_layout()
    save_path = os.path.join(args.output, 'eval_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")
    
    # 6. 保存结果JSON
    results = {
        'model': os.path.basename(args.model_path),
        'gallery': args.gallery,
        'num_persons': n_persons,
        'num_queries': total,
        'rank1': float(round(rank1, 2)),
        'rank5': float(round(rank5, 2)),
        'mAP': float(round(mAP, 2)),
        'same_person_sim_mean': float(round(float(np.mean(same_sims)), 3)) if same_sims else 0,
        'diff_person_sim_mean': float(round(float(np.mean(diff_sims)), 3)) if diff_sims else 0,
    }
    json_path = os.path.join(args.output, 'eval_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {json_path}")
    
    print(f"\n{'='*60}")
    print(f"评估完成! 结果保存在 {args.output}/ 目录")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
