"""检查 data 目录结构"""
from pathlib import Path
import os

data = Path('data')
print(f'data 目录: {data.resolve()}')
print()

print('子目录列表:')
for p in data.iterdir():
    if p.is_dir():
        total = len(list(p.rglob('*.jpg')))
        zips = len(list(p.rglob('*.zip')))
        print(f'  {p.name}/  ({total} jpg, {zips} zip)')
    else:
        size = p.stat().st_size / 1024 / 1024
        print(f'  {p.name} ({size:.1f} MB)')

print()
market = data / 'Market-1501'
if market.exists():
    print('Market-1501 子目录:')
    for p in market.iterdir():
        if p.is_dir():
            total = len(list(p.glob('*.jpg')))
            print(f'  {p.name}/  ({total} jpg)')
        else:
            size = p.stat().st_size / 1024
            print(f'  {p.name} ({size:.1f} KB)')

# 检查回收站或temp
temp_dirs = [
    Path(os.environ.get('TMP', 'C:/Temp')),
    Path(os.environ.get('TEMP', 'C:/Temp')),
    Path('C:/Users/MECHREVO/AppData/Local/Temp'),
]
print()
print('检查临时目录中是否有 Market-1501 相关文件...')
for td in temp_dirs:
    if td.exists():
        for f in td.glob('*market*'):
            print(f'  找到: {f}')
        for f in td.glob('*1501*'):
            print(f'  找到: {f}')
        for f in td.glob('*Market*'):
            print(f'  找到: {f}')
