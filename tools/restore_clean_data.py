"""恢复干净Market-1501数据 + 修复标签冲突"""
import shutil
from pathlib import Path

data_dir = Path('data') / 'Market-1501'
train_dir = data_dir / 'bounding_box_train'
backup_dir = data_dir / 'bounding_box_train_clean_backup'

# 如果之前有备份数据，直接恢复
if backup_dir.exists():
    print('找到备份数据，恢复中...')
    if train_dir.exists():
        shutil.rmtree(train_dir)
    backup_dir.rename(train_dir)
    print('恢复完成')
else:
    # 否则从gt_bbox重建
    print('从gt_bbox重建干净数据...')
    src = data_dir / 'gt_bbox'
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True)
    test_dir = data_dir / 'bounding_box_test'
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    
    train_count = 0
    test_count = 0
    for f in src.glob('*.jpg'):
        pid_str = f.stem.split('_')[0]
        pid = int(pid_str)
        if 1 <= pid <= 751:
            shutil.copy2(f, train_dir / f.name)
            train_count += 1
        else:
            shutil.copy2(f, test_dir / f.name)
            test_count += 1
    
    print(f'训练集: {len(list(train_dir.glob("*.jpg")))} 张')
    print(f'测试集: {len(list(test_dir.glob("*.jpg")))} 张')

# 恢复模型
shutil.copy2('checkpoints/best_model_safe_backup.pth', 'checkpoints/best_model_clean.pth')
print('干净模型已保存到 checkpoints/best_model_clean.pth')
print('数据已恢复，可以开始底库微调了')
