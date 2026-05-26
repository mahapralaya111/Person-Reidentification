"""从 gt_bbox 重建 bounding_box_train 和 bounding_box_test"""
import shutil
from pathlib import Path

data_dir = Path('data') / 'Market-1501'
src = data_dir / 'gt_bbox'
train_dir = data_dir / 'bounding_box_train'
test_dir = data_dir / 'bounding_box_test'

if not src.exists():
    print(f'[错误] gt_bbox 目录不存在')
    exit(1)

# 创建目录
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

train_count = 0
test_count = 0

for f in src.glob('*.jpg'):
    # 文件名格式: 0000_c1s1_000001_00.jpg
    pid_str = f.stem.split('_')[0]
    pid = int(pid_str)
    # Market-1501: ID 0001-0751 是训练集，0751-1501 是测试集
    if 1 <= pid <= 751:
        shutil.copy2(f, train_dir / f.name)
        train_count += 1
    elif 751 < pid <= 1501:
        shutil.copy2(f, test_dir / f.name)
        test_count += 1
    else:
        # pid=0000 是背景/干扰项，放入 test
        if pid == 0:
            shutil.copy2(f, test_dir / f.name)
            test_count += 1

print(f'bounding_box_train: {train_count} 张')
print(f'bounding_box_test: {test_count} 张')
print(f'总共: {train_count + test_count} 张')
print('重建完成!')
