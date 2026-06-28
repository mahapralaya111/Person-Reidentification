# -*- coding: utf-8 -*-
"""
ReID模型 云服务器 API 服务
支持 REST API 调用、CORS、健康检查、远程图库管理
"""

import os
import sys
from pathlib import Path

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_src_dir = str(_PROJECT_ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# 加载云端配置
sys.path.insert(0, str(_PROJECT_ROOT))
from cloud_config import (
    SERVICE_NAME, VERSION, HOST, PORT, DEBUG,
    MODEL_PATH, BACKBONE, NUM_CLASSES,
    GALLERY_DIR, DEFAULT_TOP_K, SIMILARITY_THRESHOLD,
    WORKERS, LOG_DIR
)

from flask import Flask, request, jsonify, render_template_string
import torch
import numpy as np
from PIL import Image
import base64
import io
import logging
from datetime import datetime

# ============================================================
# Flask 应用初始化
# ============================================================
app = Flask(__name__)

# 添加 CORS 支持（云服务器必需：允许跨域请求）
try:
    from flask_cors import CORS
    CORS(app)
    print("[API] CORS 已启用")
except ImportError:
    print("[API] flask-cors 未安装，跨域请求可能受限")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, 'api.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger('reid-api')

# 全局推理器
infer = None
_startup_time = datetime.now()


def init_inferencer(model_path=None, backbone=None, num_classes=None):
    """初始化推理器"""
    global infer
    if infer is not None:
        return infer
    
    model_path = model_path or MODEL_PATH
    backbone = backbone or BACKBONE
    num_classes = num_classes or NUM_CLASSES
    
    logger.info(f"初始化推理器: model={model_path}, backbone={backbone}")
    from inference import ReidInferencer
    infer = ReidInferencer(
        model_path=model_path,
        backbone=backbone,
        num_classes=num_classes
    )
    logger.info("推理器就绪")
    return infer


# ============================================================
# API 路由
# ============================================================

@app.route('/', methods=['GET'])
def index():
    """服务状态页"""
    uptime = str(datetime.now() - _startup_time).split('.')[0]
    gallery_info = {}
    if infer and hasattr(infer, 'gallery_ids') and infer.gallery_ids:
        from collections import Counter
        person_counts = Counter(infer.gallery_ids)
        gallery_info = {
            'total_images': len(infer.gallery_ids),
            'total_persons': len(set(infer.gallery_ids)),
            'persons': {k: v for k, v in sorted(person_counts.items())}
        }
    
    return jsonify({
        'service': SERVICE_NAME,
        'version': VERSION,
        'status': 'running',
        'uptime': uptime,
        'model_loaded': infer is not None,
        'gallery': gallery_info,
        'endpoints': {
            '/': '此页面',
            '/health': '健康检查（GET）',
            '/feature': '提取图片特征（POST）',
            '/compare': '比较两张图片（POST）',
            '/search': '在底库中搜索（POST）',
            '/gallery': '查看/管理底库（GET/POST）',
            '/persons': '列出所有人员（GET）',
            '/reload': '重新加载模型或图库（POST）',
            '/upload': '上传图片到底库（POST）',
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """健康检查（用于云服务器负载均衡和监控）"""
    if infer is None:
        try:
            init_inferencer()
        except Exception:
            return jsonify({'status': 'initializing', 'service': SERVICE_NAME}), 503
    return jsonify({
        'status': 'ok',
        'service': SERVICE_NAME,
        'uptime': str(datetime.now() - _startup_time).split('.')[0],
        'device': str(infer.device),
        'gallery_size': len(infer.gallery_ids) if hasattr(infer, 'gallery_ids') and infer.gallery_ids else 0
    })


@app.route('/feature', methods=['POST'])
def extract_feature():
    """
    提取图片特征向量
    ---
    请求体（JSON）:
        {"image": "base64编码字符串"}
        {"image_url": "http://..."}
        {"image_path": "/path/to/file.jpg"}  (服务器本地文件)
    返回:
        {"feature": [512维向量], "norm": 1.0}
    """
    try:
        init_inferencer()
        data = request.json
        img = None

        if 'image' in data:
            img_data = base64.b64decode(data['image'])
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
        elif 'image_url' in data:
            import requests
            resp = requests.get(data['image_url'], timeout=30)
            img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        elif 'image_path' in data:
            img = Image.open(data['image_path']).convert('RGB')
        else:
            return jsonify({'error': '需要提供 image、image_url 或 image_path'}), 400

        transform = infer.transform
        img_tensor = transform(img).unsqueeze(0).to(infer.device)

        with torch.no_grad():
            outputs = infer.model(img_tensor)
            feature = outputs[1] if isinstance(outputs, tuple) else outputs

        feature = feature.cpu().numpy().flatten()

        return jsonify({
            'feature': feature.tolist(),
            'norm': float(np.linalg.norm(feature)),
            'feature_dim': len(feature)
        })

    except Exception as e:
        logger.error(f"特征提取失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/compare', methods=['POST'])
def compare():
    """
    比较两张图片的相似度
    ---
    请求体:
        {"image1": "base64...", "image2": "base64..."}
    返回:
        {"similarity": 0.95, "same_person": true}
    """
    try:
        init_inferencer()
        data = request.json
        transform = infer.transform

        def _decode(b64_str):
            return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert('RGB')

        img1 = _decode(data['image1'])
        img2 = _decode(data['image2'])

        t1 = transform(img1).unsqueeze(0).to(infer.device)
        t2 = transform(img2).unsqueeze(0).to(infer.device)

        with torch.no_grad():
            f1 = infer.model(t1)[1] if isinstance(infer.model(t1), tuple) else infer.model(t1)
            f2 = infer.model(t2)[1] if isinstance(infer.model(t2), tuple) else infer.model(t2)

        f1 = f1.cpu().numpy().flatten()
        f2 = f2.cpu().numpy().flatten()

        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        similarity = max(0.0, float(np.dot(f1, f2) / (norm1 * norm2 + 1e-8)))
        same_person = similarity > SIMILARITY_THRESHOLD

        return jsonify({
            'similarity': similarity,
            'same_person': same_person,
            'threshold': SIMILARITY_THRESHOLD
        })

    except Exception as e:
        logger.error(f"比较失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def search():
    """
    在底库中搜索相似行人
    ---
    请求体:
        {"image": "base64...", "top_k": 5, "threshold": 0.0}
    返回:
        {"results": [{"id": "张三", "similarity": 0.95}, ...], "query_norm": 1.0}
    """
    try:
        init_inferencer()
        data = request.json
        top_k = data.get('top_k', DEFAULT_TOP_K)
        min_sim = data.get('threshold', 0.0)

        # 检查底库
        if not hasattr(infer, 'gallery_features') or infer.gallery_features is None or len(infer.gallery_features) == 0:
            return jsonify({
                'error': '底库为空，请先通过 /gallery 或 /load_gallery 加载底库',
                'gallery_dir': GALLERY_DIR
            }), 400

        # 解码图片
        img_data = base64.b64decode(data['image'])
        img = Image.open(io.BytesIO(img_data)).convert('RGB')

        transform = infer.transform
        img_tensor = transform(img).unsqueeze(0).to(infer.device)

        with torch.no_grad():
            outputs = infer.model(img_tensor)
            query_feat = outputs[1] if isinstance(outputs, tuple) else outputs
        query_feat = query_feat.cpu().numpy().flatten()

        # 搜索
        raw_results = infer.search(query_feat, infer.gallery_features,
                                   infer.gallery_ids, top_k=top_k)

        # 过滤低于阈值的结果
        results = [r for r in raw_results if r[1] >= min_sim]

        return jsonify({
            'results': [
                {'id': str(r[0]), 'similarity': float(r[1])}
                for r in results
            ],
            'total_results': len(results),
            'query_norm': float(np.linalg.norm(query_feat))
        })

    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/gallery', methods=['GET', 'POST'])
def gallery():
    """
    查看/管理底库
    ---
    GET: 返回当前底库信息
    POST: 加载/重新加载底库
        请求体: {"gallery_dir": "/path/to/gallery"} (可选，默认使用配置)
    """
    if request.method == 'GET':
        if not hasattr(infer, 'gallery_ids') or not infer.gallery_ids:
            return jsonify({
                'loaded': False,
                'gallery_dir': GALLERY_DIR,
                'total_images': 0,
                'total_persons': 0,
                'persons': {}
            })

        from collections import Counter
        person_counts = Counter(infer.gallery_ids)
        return jsonify({
            'loaded': True,
            'gallery_dir': getattr(infer, '_gallery_dir', GALLERY_DIR),
            'total_images': len(infer.gallery_ids),
            'total_persons': len(set(infer.gallery_ids)),
            'persons': {k: v for k, v in sorted(person_counts.items())}
        })

    # POST: 加载底库
    try:
        init_inferencer()
        data = request.json or {}
        gallery_dir = data.get('gallery_dir', GALLERY_DIR)

        logger.info(f"加载底库: {gallery_dir}")
        features, ids, paths = infer.build_gallery(gallery_dir)

        infer.gallery_features = features
        infer.gallery_ids = ids
        infer.gallery_paths = paths
        infer._gallery_dir = gallery_dir

        from collections import Counter
        person_counts = Counter(ids)

        return jsonify({
            'status': 'success',
            'gallery_dir': gallery_dir,
            'total_images': len(ids),
            'total_persons': len(set(ids)),
            'feature_dim': features.shape[1] if len(features) > 0 else 0,
            'persons': {k: v for k, v in sorted(person_counts.items())}
        })

    except Exception as e:
        logger.error(f"加载底库失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/persons', methods=['GET'])
def list_persons():
    """
    列出底库中的所有人员
    ---
    返回:
        {"persons": ["张三", "李四", ...], "total": 2}
    """
    if not hasattr(infer, 'gallery_ids') or not infer.gallery_ids:
        return jsonify({'persons': [], 'total': 0})
    
    unique = sorted(set(infer.gallery_ids))
    return jsonify({'persons': unique, 'total': len(unique)})


@app.route('/reload', methods=['POST'])
def reload_service():
    """
    重新加载模型或图库（无需重启服务）
    ---
    请求体:
        {"type": "model", "model_path": "..."}
        {"type": "gallery", "gallery_dir": "..."}
        {"type": "all"}
    """
    global infer
    data = request.json or {}
    reload_type = data.get('type', 'all')

    try:
        if reload_type in ('model', 'all'):
            model_path = data.get('model_path', MODEL_PATH)
            logger.info(f"重新加载模型: {model_path}")
            infer = None
            init_inferencer(model_path=model_path)

        if reload_type in ('gallery', 'all'):
            gallery_dir = data.get('gallery_dir', GALLERY_DIR)
            logger.info(f"重新加载底库: {gallery_dir}")
            features, ids, paths = infer.build_gallery(gallery_dir)
            infer.gallery_features = features
            infer.gallery_ids = ids
            infer.gallery_paths = paths

        return jsonify({'status': 'success', 'message': f'已重新加载: {reload_type}'})

    except Exception as e:
        logger.error(f"重新加载失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_to_gallery():
    """
    上传图片到底库（远程更新底库）
    ---
    请求体 (multipart/form-data):
        image: 图片文件
        person_id: 人员ID
    或 JSON:
        {"image": "base64...", "person_id": "张三"}
    """
    try:
        init_inferencer()

        if request.files:
            # multipart 上传
            file = request.files['image']
            person_id = request.form.get('person_id', f'upload_{datetime.now().strftime("%H%M%S")}')
            img = Image.open(file).convert('RGB')
            
            # 保存到图库目录
            person_dir = Path(GALLERY_DIR) / person_id
            person_dir.mkdir(parents=True, exist_ok=True)
            save_path = person_dir / f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            img.save(save_path, quality=95)
            
            # 提取特征并加入内存底库
            if not hasattr(infer, 'gallery_features'):
                infer.gallery_features = []
                infer.gallery_ids = []
                infer.gallery_paths = []

            transform = infer.transform
            t = transform(img).unsqueeze(0).to(infer.device)
            with torch.no_grad():
                feat = infer.model(t)
                feat = feat[1] if isinstance(feat, tuple) else feat
            feat = feat.cpu().numpy().flatten()

            infer.gallery_features = np.vstack([infer.gallery_features, feat]) \
                if isinstance(infer.gallery_features, np.ndarray) and len(infer.gallery_features) > 0 \
                else np.array([feat])
            infer.gallery_ids.append(person_id)
            infer.gallery_paths.append(str(save_path))

            return jsonify({
                'status': 'success',
                'person_id': person_id,
                'saved_path': str(save_path),
                'gallery_total': len(infer.gallery_ids)
            })

        else:
            # JSON base64 上传
            data = request.json
            img_data = base64.b64decode(data['image'])
            person_id = data.get('person_id', f'upload_{datetime.now().strftime("%H%M%S")}')
            img = Image.open(io.BytesIO(img_data)).convert('RGB')

            person_dir = Path(GALLERY_DIR) / person_id
            person_dir.mkdir(parents=True, exist_ok=True)
            save_path = person_dir / f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            img.save(save_path, quality=95)

            # 提取特征并加入内存
            transform = infer.transform
            t = transform(img).unsqueeze(0).to(infer.device)
            with torch.no_grad():
                feat = infer.model(t)
                feat = feat[1] if isinstance(feat, tuple) else feat
            feat = feat.cpu().numpy().flatten()

            if not hasattr(infer, 'gallery_features') or infer.gallery_features is None or len(infer.gallery_features) == 0:
                infer.gallery_features = np.array([feat])
                infer.gallery_ids = [person_id]
                infer.gallery_paths = [str(save_path)]
            else:
                infer.gallery_features = np.vstack([infer.gallery_features, feat])
                infer.gallery_ids.append(person_id)
                infer.gallery_paths.append(str(save_path))

            return jsonify({
                'status': 'success',
                'person_id': person_id,
                'saved_path': str(save_path),
                'gallery_total': len(infer.gallery_ids)
            })

    except Exception as e:
        logger.error(f"上传失败: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# 启动入口
# ============================================================

def main():
    """启动API服务"""
    import argparse

    parser = argparse.ArgumentParser(description=f'{SERVICE_NAME} - 云服务器 API')
    parser.add_argument('--port', type=int, default=PORT, help=f'端口号 (默认: {PORT})')
    parser.add_argument('--host', type=str, default=HOST, help=f'监听地址 (默认: {HOST})')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='模型路径')
    parser.add_argument('--backbone', type=str, default=BACKBONE, help='骨干网络')
    parser.add_argument('--gallery', type=str, default=GALLERY_DIR, help='底库目录')
    parser.add_argument('--workers', type=int, default=WORKERS, help='工作进程数')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    # 初始化推理器并预加载底库
    global infer
    logger.info("=" * 50)
    logger.info(f"{SERVICE_NAME} v{VERSION}")
    logger.info(f"模型: {args.model}")
    logger.info(f"底库: {args.gallery}")
    logger.info(f"监听: {args.host}:{args.port}")
    logger.info("=" * 50)

    infer = init_inferencer(
        model_path=args.model,
        backbone=args.backbone
    )

    if args.gallery and os.path.isdir(args.gallery):
        logger.info(f"预加载底库: {args.gallery}")
        try:
            features, ids, paths = infer.build_gallery(args.gallery)
            infer.gallery_features = features
            infer.gallery_ids = ids
            infer.gallery_paths = paths
            infer._gallery_dir = args.gallery
            logger.info(f"底库加载完成: {len(set(ids))} 人, {len(ids)} 张图片")
        except Exception as e:
            logger.warning(f"底库预加载失败（可在启动后通过 /gallery 加载）: {e}")

    # 生产环境建议使用 gunicorn 替代 Flask 内置服务器
    if args.workers > 1 and not args.debug:
        try:
            from gunicorn.app.base import BaseApplication

            class GunicornApp(BaseApplication):
                def __init__(self, app, options):
                    self.application = app
                    self.options = options
                    super().__init__()

                def load_config(self):
                    for k, v in self.options.items():
                        self.cfg.set(k, v)

                def load(self):
                    return self.application

            options = {
                'bind': f'{args.host}:{args.port}',
                'workers': args.workers,
                'worker_class': 'sync',
                'timeout': 120,
                'accesslog': os.path.join(LOG_DIR, 'gunicorn_access.log'),
                'errorlog': os.path.join(LOG_DIR, 'gunicorn_error.log'),
                'loglevel': 'info',
                'preload_app': True,
            }
            logger.info(f"使用 Gunicorn 启动（{args.workers} 工作进程）")
            GunicornApp(app, options).run()
            return
        except ImportError:
            logger.info("gunicorn 未安装，使用 Flask 内置服务器（单进程）")

    # 使用 Flask 内置服务器
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
