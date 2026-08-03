"""后台启动完整训练流水线（独立窗口运行）"""
import subprocess, sys, os
from pathlib import Path

root = Path(__file__).resolve().parent.parent
py = sys.executable

print("=" * 55)
print("  启动完整训练流水线")
print("=" * 55)
print()
print("步骤1: 续训 Market-1501 (再30轮)")
print("步骤2: 底库微调 (100轮)")
print()
print(f"Python: {py}")
print(f"目录: {root}")
print()

# 步骤1：续训 Market-1501
print("[启动] 续训 Market-1501...")
proc1 = subprocess.Popen(
    [str(py), str(root / "main.py"), "train",
     "--backbone", "nnlrnet",
     "--epochs", "30",
     "--batch_size", "32",
     "--lr", "0.0003",
     "--use_circle_loss",
     "--use_cosine_annealing",
     "--resume", str(root / "checkpoints" / "best_model.pth")],
    cwd=str(root),
    creationflags=subprocess.CREATE_NEW_CONSOLE  # 新窗口
)
print(f"  PID: {proc1.pid} (独立窗口)")
print("  新窗口已打开，可以看到训练进度")
print()

# 备份预训练模型
import time
time.sleep(5)
import shutil
shutil.copy2(
    str(root / "checkpoints" / "best_model.pth"),
    str(root / "checkpoints" / "market1501_pretrained.pth")
)
print("[等待] 步骤1完成后，再启动步骤2")
print()
print("步骤1完成后，在这个窗口中输入 2 并按回车，启动底库微调：")
print("  或直接打开新终端运行:")
print(f"  {py} main.py train --backbone nnlrnet --epochs 100 --batch_size 32 --lr 0.0001 --use_circle_loss --use_cosine_annealing --finetune checkpoints/market1501_pretrained.pth")
print()

input("按回车键退出...")
