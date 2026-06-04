import torch
from pathlib import Path
c=torch.load('checkpoints/best_model.pth',map_location='cpu',weights_only=False)
print(f"当前模型: classes={c['num_classes']}")
lm=c.get('label_mapping',{})
if lm:
    print(f"包含人员: {list(lm.keys())}")

# 看底库目录
g=Path('rescue_gallery')
persons=[d for d in g.iterdir() if d.is_dir() and not d.name.startswith('_')]
print(f"底库目录: {len(persons)} 个人")
for p in persons:
    imgs=list(p.glob('*.[jJ][pP][gG]')) + list(p.glob('*.[pP][nN][gG]'))
    print(f"  {p.name}: {len(imgs)} 张")
