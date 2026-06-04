"""
一键整理底库照片为训练数据格式
把人物照片转为 Market-1501 格式用于训练
运行后即可用 main.py train 训练

使用方法:
    python tools/prepare_train_data.py                        # 默认使用 rescue_gallery
    python tools/prepare_train_data.py --source D:/my_photos  # 指定自定义底库目录

说明:
    将底库目录中每个人物文件夹的照片
    转为 data/Market-1501/bounding_box_train/ 下的标准格式
"""
import shutil, os, sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="整理底库照片为训练数据格式")
    parser.add_argument('--source', type=str, default='',
                       help='底库照片目录（每人一个子文件夹），默认使用 rescue_gallery')
    args, _ = parser.parse_known_args()

    project_root = Path(__file__).resolve().parent.parent
    
    if args.source:
        src = Path(args.source)
    else:
        src = project_root / 'rescue_gallery'
    
    dst = project_root / 'data' / 'rescue_train' / 'Market-1501'


    if not src.exists() or not src.is_dir():
        print(f"[错误] 底库目录不存在: {src}")
        print("请指定正确的人物照片目录路径")
        return

    # 创建标准 Market-1501 子目录
    train_dir = dst / 'bounding_box_train'
    test_dir = dst / 'bounding_box_test'
    query_dir = dst / 'query'
    for d in [train_dir, test_dir, query_dir]:
        d.mkdir(parents=True, exist_ok=True)


    persons = sorted([d for d in src.iterdir() if d.is_dir() and not d.name.startswith('_')])
    print(f"发现 {len(persons)} 个人员:")
    for p in persons:
        imgs = list(p.glob('*.*'))
        print(f"  {p.name}: {len(imgs)} 张照片")

    if not persons:
        print("请先在 rescue_gallery 下创建人物文件夹并放入照片")
        return

    # 复制到训练目录（每人80%做训练，20%做测试）
    total = 0
    for pid, person_dir in enumerate(persons, 1):
        imgs = sorted([f for f in person_dir.iterdir() if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')])
        n = len(imgs)
        n_test = max(1, n // 5)
        for idx, img in enumerate(imgs):
            # 文件名: {PID:04d}_c{cam_id}s1_{seq:06d}_{idx:02d}.jpg
            new_name = f'{pid:04d}_c1s1_{idx+1:06d}_{idx+1:02d}{img.suffix}'
            if idx < n_test:
                shutil.copy2(img, test_dir / new_name)
                if idx == 0:  # 第一张也放入query
                    shutil.copy2(img, query_dir / new_name)
            else:
                shutil.copy2(img, train_dir / new_name)
            total += 1

    print(f"\n数据整理完成!")
    print(f"  训练集: {len(list(train_dir.glob('*.*')))} 张 ({train_dir})")
    print(f"  测试集: {len(list(test_dir.glob('*.*')))} 张 ({test_dir})")
    print(f"  查询集: {len(list(query_dir.glob('*.*')))} 张 ({query_dir})")
    print(f"  人员数: {len(persons)}")
    print(f"  总图片: {total}")
    print(f"  输出目录: {dst}")
    print()
    print("="*55)
    print("现在可以开始训练了!")
    print("="*55)
    print()
    print("推荐训练命令（从底库微调）:")
    print(f"  python main.py train --data_dir ./data/rescue_train --backbone nnlrnet --epochs 30 --batch_size 16 --lr 0.0001 --use_cosine_annealing --finetune checkpoints/market1501_pretrained.pth")
    print()
    print("或从启动器选:")
    print("  一键快速训练 → 数据目录: ./data/rescue_train")


if __name__ == '__main__':
    main()
