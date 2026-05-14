"""解压并整理 Market-1501 数据集"""
import zipfile, shutil, os
from pathlib import Path

d = Path('data')
zip_path = d / 'Market-1501' / 'Market-1501-v15.09.15.zip'
src = d / 'Market-1501-v15.09.15'
tgt = d / 'Market-1501'

# 先解压
if not src.exists():
    print('解压中...请等待（约2分钟）')
    with zipfile.ZipFile(str(zip_path)) as zf:
        zf.extractall(str(d))
    print('解压完成')
else:
    print('已解压，直接整理')

total_jpg = len(list(src.rglob('*.jpg')))
print(f'数据集共 {total_jpg} 张图片')

# 删除旧的 Market-1501 目录（整个删，避免残留）
print('清理旧数据...')
if tgt.exists():
    shutil.rmtree(tgt)
    print(f'  已删除 {tgt}')

# 重命名 Market-1501-v15.09.15 -> Market-1501
src.rename(tgt)
print(f'  已重命名为 {tgt}')

# 删除多余文件
for f in ['704c9-main.zip', 'Market-1501-v15.09.15.zip']:
    p = tgt / f
    if p.exists():
        p.unlink()
        print(f'  清理 {f}')

# 删除多余目录
for f in ['704c9-main']:
    p = tgt / f
    if p.exists():
        shutil.rmtree(p)
        print(f'  清理 {f}')

print('\n✅ 数据集整理完成！')
for sub in ['bounding_box_train', 'bounding_box_test', 'query']:
    n = len(list((tgt / sub).glob('*.jpg')))
    print(f'  {sub}: {n} 张图片')
