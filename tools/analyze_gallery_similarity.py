"""分析底库自相似度"""
import torch, numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from backbone import build_backbone

# 加载模型
ckpt = torch.load(str(Path(__file__).resolve().parent.parent / 'checkpoints' / 'best_model.pth'),
                  map_location='cpu', weights_only=False)
model = build_backbone(ckpt.get('backbone', 'nnlrnet'),
                       num_classes=ckpt.get('num_classes', 2), pretrained=False)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.CenterCrop((224, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 提取底库特征
gallery = Path(__file__).resolve().parent.parent / 'rescue_gallery'
features = {}
for person_dir in sorted(gallery.iterdir()):
    if not person_dir.is_dir() or person_dir.name.startswith('_'): continue
    feats = []
    for img_file in sorted(person_dir.glob('*.*'))[:30]:
        try:
            img = Image.open(img_file).convert('RGB')
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                out = model(tensor)
                if isinstance(out, tuple): out = out[1]
                feat = out.cpu().numpy().flatten()
                feat = feat / (np.linalg.norm(feat) + 1e-8)
                feats.append(feat)
        except Exception as e:
            print(f"  跳过 {img_file.name}: {e}")
    if feats:
        features[person_dir.name] = np.array(feats)

print('=' * 60)
print('  底库自相似度分析')
print('=' * 60)

for person, feats in features.items():
    n = len(feats)
    if n < 2:
        print(f'\n{person}: 只有 {n} 张，无法计算')
        continue
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(float(np.dot(feats[i], feats[j])))
    sims = np.array(sims)
    print(f'\n{person} ({n}张照片):')
    print(f'  同人相似度: 平均={sims.mean():.4f}  最高={sims.max():.4f}  最低={sims.min():.4f}')

if len(features) >= 2:
    print('\n--- 区分度判断 ---')
    names = list(features.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            cross_sims = []
            for f1 in features[names[i]]:
                for f2 in features[names[j]]:
                    cross_sims.append(float(np.dot(f1, f2)))
            cross_sims = np.array(cross_sims)

            # 同人相似度
            same_i = []
            fi = features[names[i]]
            for a in range(len(fi)):
                for b in range(a + 1, len(fi)):
                    same_i.append(float(np.dot(fi[a], fi[b])))
            same_j = []
            fj = features[names[j]]
            for a in range(len(fj)):
                for b in range(a + 1, len(fj)):
                    same_j.append(float(np.dot(fj[a], fj[b])))
            same_avg = (np.mean(same_i) + np.mean(same_j)) / 2 if same_i and same_j else 0
            cross_avg = cross_sims.mean()

            print(f'\n{names[i]} vs {names[j]}:')
            print(f'  异人相似度: 平均={cross_avg:.4f}  最高={cross_sims.max():.4f}')
            print(f'  同人平均: {same_avg:.4f}  |  异人平均: {cross_avg:.4f}')
            print(f'  区分度(同人-异人): {same_avg - cross_avg:.4f}')

            if same_avg - cross_avg < 0.05:
                print(f'  [严重] 混淆严重！两人特征几乎一样')
                print(f'     原因: 照片太少 + 两人可能穿相似衣服')
                print(f'     建议: 每人补拍不同衣服/不同角度的照片')
            elif same_avg - cross_avg < 0.1:
                print(f'  [警告] 区分度较低，容易混淆')
                print(f'     建议: 补充更多照片')
            else:
                print(f'  [OK] 区分度良好')
else:
    print('\n至少需要2个人才能做区分度分析')
