import torch
from pathlib import Path
c=torch.load('checkpoints/best_model.pth',map_location='cpu',weights_only=False)
print(f"模型: {c['backbone']}  | 类别={c['num_classes']}  | 轮数={c['epoch']}")
lm=c.get('label_mapping',{})
if lm:
    print(f"包含人员: {list(lm.keys())}")
print(f"训练指标: Rank-1={c.get('rank1','?'):.2f}%  mAP={c.get('mAP','?'):.2f}%")
# 查看底库
g=Path('rescue_gallery')
persons=[d for d in g.iterdir() if d.is_dir() and not d.name.startswith('_')]
print(f"\n底库: {len(persons)} 人")
for p in persons:
    imgs=list(p.glob('*.*'))
    ts=[i for i in imgs if i.suffix.lower() in ['.jpg','.jpeg','.png']]
    print(f"  {p.name}: {len(ts)} 张")
