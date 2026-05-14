"""尝试多个源下载 Market-1501"""
import urllib.request, zipfile, os
from pathlib import Path

data_dir = Path('./data')
zip_path = data_dir / 'market1501.zip'
zip_path.parent.mkdir(parents=True, exist_ok=True)

urls = [
    'https://rdm.ding.pku.edu.cn/dataset/market1501/Market-1501-v15.09.15.zip',
    'https://github.com/charliememory/Market-1501/releases/download/v1.0/Market-1501-v15.09.15.zip',
]

success = False
for url in urls:
    print(f'尝试: {url}')
    try:
        urllib.request.urlretrieve(url, zip_path)
        size = zip_path.stat().st_size / 1024 / 1024
        print(f'下载成功! {size:.0f} MB')
        success = True
        break
    except Exception as e:
        print(f'失败: {e}')
        if zip_path.exists():
            zip_path.unlink()

if not success:
    print('\n所有在线源都不可用，请手动下载:')
    print('1. 打开 https://www.kaggle.com/datasets/pengcw/market1501')
    print('2. 下载 Market-1501-v15.09.15.zip')
    print('3. 解压到 data/Market-1501/ 目录')
    print()
    print('或者用百度网盘: 链接: https://pan.baidu.com/s/1ntIi2Op 密码: bd7x')
    exit(1)

print('正在解压...')
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(data_dir)
print('解压完成!')

extracted = data_dir / 'Market-1501-v15.09.15'
target = data_dir / 'Market-1501'
if extracted.exists():
    # 删除旧的被覆盖的目录
    import shutil
    if target.exists():
        shutil.rmtree(target)
    extracted.rename(target)

zip_path.unlink()
print('清理完成')

train_dir = target / 'bounding_box_train'
n = len(list(train_dir.glob('*.jpg')))
print(f'Market-1501 训练集: {n} 张图片')
print('下载完成!')
