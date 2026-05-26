"""分析 gt_bbox 中的人员ID分布"""
from pathlib import Path
from collections import Counter

data_dir = Path('data') / 'Market-1501'
src = data_dir / 'gt_bbox'

pids = []
for f in src.glob('*.jpg'):
    pid_str = f.stem.split('_')[0]
    pids.append(int(pid_str))

counter = Counter(pids)
print(f'总文件数: {len(pids)}')
print(f'唯一人员ID数: {len(counter)}')
print()

# 按ID范围统计
train_ids = [pid for pid in counter if 1 <= pid <= 751]
test_ids = [pid for pid in counter if 751 < pid <= 1501]
other_ids = [pid for pid in counter if pid == 0 or pid > 1501]

print(f'训练集ID (1-751): {len(train_ids)} 个, 共 {sum(counter[pid] for pid in train_ids)} 张')
print(f'测试集ID (752-1501): {len(test_ids)} 个, 共 {sum(counter[pid] for pid in test_ids)} 张')
print(f'其他ID: {len(other_ids)} 个, 共 {sum(counter[pid] for pid in other_ids)} 张')

print()
print('前20个ID的图片数:')
for pid, cnt in counter.most_common(20):
    print(f'  ID {pid:04d}: {cnt} 张')
