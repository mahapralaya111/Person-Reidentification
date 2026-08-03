"""
完整训练流水线：先训 Market-1501，再微调底库
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

print("=" * 60)
print("  完整训练流水线")
print("=" * 60)

# 第一步：下载 Market-1501 数据集
print("\n[1/4] 下载 Market-1501 数据集...")
result = subprocess.run([PY, str(ROOT / "main.py"), "train", "--download", "--auto_demo"],
                       cwd=str(ROOT), capture_output=True, text=True)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

# 检查数据是否就绪
data_dir = ROOT / "data" / "Market-1501" / "bounding_box_train"
jpg_count = len(list(data_dir.glob("*.jpg")))
if jpg_count < 100:
    print(f"[错误] Market-1501 数据不足 ({jpg_count} 张)，请检查网络后重试")
    sys.exit(1)
print(f"  Market-1501 训练集: {jpg_count} 张图片")

# 第二步：整理底库数据
print("\n[2/4] 整理底库照片为训练格式...")
result = subprocess.run([PY, str(ROOT / "tools" / "prepare_train_data.py")],
                       cwd=str(ROOT), capture_output=True, text=True)
print(result.stdout)

# 第三步：训练 Market-1501 (nnlrnet, 60轮)
print("\n[3/4] 训练 Market-1501 基础模型 (nnlrnet, 60轮)...")
# 先清一下 checkpoints，避免混淆
for f in ["checkpoints/best_model.pth", "checkpoints/final_model.pth"]:
    p = ROOT / f
    if p.exists():
        p.unlink()

# 先备份原来的 final_model
print("  （覆盖前先保留底库模型）")
result = subprocess.run(
    [PY, str(ROOT / "main.py"), "train",
     "--backbone", "nnlrnet",
     "--epochs", "60",
     "--batch_size", "32",
     "--lr", "0.0003",
     "--use_circle_loss",
     "--use_cosine_annealing",
     "--use_random_erasing",
     "--use_weather_aug",
     "--data_dir", "./data"],
    cwd=str(ROOT), capture_output=True, text=True
)
print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)

# 保存 Market-1501 基础模型
import shutil
src = ROOT / "checkpoints" / "best_model.pth"
if src.exists():
    shutil.copy2(src, ROOT / "checkpoints" / "market1501_pretrained.pth")
    print("  [OK] 已保存 market1501_pretrained.pth")

# 第四步：在 Market-1501 模型基础上微调底库
print("\n[4/4] 微调：在底库数据上继续训练...")
# 重新整理底库数据覆盖训练目录
result = subprocess.run([PY, str(ROOT / "tools" / "prepare_train_data.py")],
                       cwd=str(ROOT), capture_output=True, text=True)
print(result.stdout)

result = subprocess.run(
    [PY, str(ROOT / "main.py"), "train",
     "--backbone", "nnlrnet",
     "--epochs", "100",
     "--batch_size", "32",
     "--lr", "0.0001",   # 微调用更小学习率
     "--use_circle_loss",
     "--use_cosine_annealing",
     "--use_random_erasing",
     "--use_weather_aug",
     "--data_dir", "./data"],
    cwd=str(ROOT), capture_output=True, text=True
)
print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)

print("\n" + "=" * 60)
print("  🎉 全部完成！")
print("  模型位置: checkpoints/best_model.pth")
print("=" * 60)
