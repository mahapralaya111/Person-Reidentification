"""
行人重识别系统 - 云服务器配置
所有云部署相关的配置集中在此，修改后重启服务生效
"""
import os
from pathlib import Path

# ============================================================
# 项目根目录（自动检测，无需修改）
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# 服务器基础配置
# ============================================================

# 监听地址（0.0.0.0 表示监听所有网络接口）
HOST = os.getenv('REID_HOST', '0.0.0.0')

# 端口（云服务器安全组需放行此端口）
PORT = int(os.getenv('REID_PORT', '5000'))

# 调试模式（生产环境必须为 False）
DEBUG = os.getenv('REID_DEBUG', 'false').lower() == 'true'

# 服务名称
SERVICE_NAME = '灾害应急搜救行人重识别系统'

# 服务版本
VERSION = '2.0.0'

# ============================================================
# 模型配置
# ============================================================

# 模型权重路径
MODEL_PATH = os.getenv('REID_MODEL', str(_PROJECT_ROOT / 'checkpoints' / 'best_model.pth'))

# 骨干网络类型 (resnet50, resnet34, osnet, agw, resnet_ibn, pcb)
BACKBONE = os.getenv('REID_BACKBONE', 'resnet50')

# 训练时的类别数
NUM_CLASSES = int(os.getenv('REID_NUM_CLASSES', '751'))

# ============================================================
# 图库配置
# ============================================================

# 图库目录（预加载后自动用于搜索）
GALLERY_DIR = os.getenv('REID_GALLERY', str(_PROJECT_ROOT / 'rescue_gallery_augmented'))

# 若增强图库不存在，回退到原始图库
_GALLERY_FALLBACK = str(_PROJECT_ROOT / 'rescue_gallery')
if not os.path.isdir(GALLERY_DIR):
    if os.path.isdir(_GALLERY_FALLBACK):
        GALLERY_DIR = _GALLERY_FALLBACK

# 搜索结果默认返回条数
DEFAULT_TOP_K = int(os.getenv('REID_TOP_K', '10'))

# 相似度阈值（高于此值视为同一人）
SIMILARITY_THRESHOLD = float(os.getenv('REID_THRESHOLD', '0.5'))

# ============================================================
# Web 界面配置
# ============================================================

# Streamlit 端口（与 Flask API 端口不同）
STREAMLIT_PORT = int(os.getenv('REID_STREAMLIT_PORT', '8501'))

# ============================================================
# 性能配置
# ============================================================

# Flask 工作进程数（仅在生产服务器模式下生效）
# 设置为 CPU 核心数 * 2 + 1 通常效果最佳
WORKERS = int(os.getenv('REID_WORKERS', '4'))

# ============================================================
# 日志配置
# ============================================================

LOG_DIR = os.getenv('REID_LOG_DIR', str(_PROJECT_ROOT / 'logs'))
os.makedirs(LOG_DIR, exist_ok=True)
