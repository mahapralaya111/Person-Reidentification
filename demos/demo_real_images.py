"""
真实图片演示 - 抗恶劣环境行人重识别系统
使用深度学习ReID模型进行特征提取和检索
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from PIL import Image

# 设置中文字体
def setup_chinese_font():
    """配置中文字体，失败时回退到英文"""
    import matplotlib.font_manager as fm
    if sys.platform.startswith('win'):
        font_list = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'DengXian']
    else:
        font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']

    all_fonts = {f.name for f in fm.fontManager.ttflist}
    for font in font_list:
        if font in all_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return True
    # 检查本地是否有字体文件
    font_dirs = []
    if sys.platform.startswith('win'):
        font_dirs = [
            r'C:\Windows\Fonts',
            os.path.expanduser(r'~\AppData\Local\Microsoft\Windows\Fonts'),
        ]
    for fd in font_dirs:
        if os.path.exists(fd):
            fm.fontManager.addfont(fd)
            for font in font_list:
                if font in {f.name for f in fm.fontManager.ttflist}:
                    plt.rcParams['font.sans-serif'] = [font]
                    plt.rcParams['axes.unicode_minus'] = False
                    return True
    return False

HAS_CHINESE_FONT = setup_chinese_font()
# 强制使用英文标签（避免字体不全时出现方框）
USE_ENGLISH = not HAS_CHINESE_FONT

# 导入深度学习 ReID 模型
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from inference import ReidInferencer
    HAS_DEEP_MODEL = True
except ImportError as e:
    print(f"[警告] 无法导入深度学习模型: {e}")
    print("[信息] 将使用传统特征作为后备方案")
    HAS_DEEP_MODEL = False


class RealImageReIDDemo:
    """使用深度学习 ReID 模型的演示"""

    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.feature_db = []
        self.labels = []
        self.images = []

        # 初始化深度学习模型
        if HAS_DEEP_MODEL:
            model_path = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pth"
            print(f"[ReID] 检查模型路径: {model_path}")
            if model_path.exists():
                # 从checkpoint读取模型信息，避免硬编码
                try:
                    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
                    ckpt_backbone = ckpt.get('backbone', 'resnet50')
                    ckpt_num_classes = ckpt.get('num_classes', 751)
                    print(f"[ReID] 模型信息: backbone={ckpt_backbone}, classes={ckpt_num_classes}")
                except:
                    ckpt_backbone = 'resnet50'
                    ckpt_num_classes = 751
                self.inferencer = ReidInferencer(
                    model_path=str(model_path),
                    backbone=ckpt_backbone,
                    num_classes=ckpt_num_classes
                )
            else:
                print(f"[警告] 模型不存在({model_path})，使用随机初始化的特征")
                self.inferencer = ReidInferencer(
                    backbone='resnet50',
                    num_classes=751
                )

    def load_images_from_folder(self):
        """递归扫描文件夹及子文件夹中的所有图片（过滤非人员图片）"""
        print("=" * 70)
        print("步骤1: 从文件夹加载图片")
        print("=" * 70)

        supported_exts = ['.jpg', '.jpeg', '.png', '.bmp',
                         '.JPG', '.JPEG', '.PNG', '.BMP']
        image_files = []

        img_dir = Path(self.image_dir)
        if not img_dir.exists():
            print(f"[信息] 目录不存在，将创建: {img_dir}")
            img_dir.mkdir(parents=True, exist_ok=True)
        else:
            for f in sorted(img_dir.rglob('*')):  # 排序保证稳定顺序
                if f.is_file() and f.suffix in supported_exts:
                    # 过滤掉非人员图片（搜索结果图、临时文件等）
                    name_lower = f.name.lower()
                    if any(kw in name_lower for kw in ['search_results', '_query_temp', 'temp_', 'thumbnail']):
                        continue
                    image_files.append(str(f))

        print(f"在文件夹中找到 {len(image_files)} 张图片")
        for f in image_files:
            print(f"  - {os.path.basename(f)}")

        return image_files

    @staticmethod
    def _safe_imread(file_path):
        """支持中文路径的图片读取"""
        try:
            if not os.path.exists(file_path):
                print(f"  [错误] 文件不存在: {file_path}")
                return None
            pil_img = Image.open(file_path).convert('RGB')
            np_img = np.array(pil_img)
            return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"  [错误] 读取图片失败: {file_path} - {e}")
            return None

    def build_feature_database(self, image_files, labels=None):
        """使用深度学习模型构建特征数据库"""
        print("\n" + "=" * 70)
        print("步骤2: 使用深度学习模型提取特征并构建数据库")
        print("=" * 70)

        if labels is None:
            labels = [f"Person_{i+1:02d}" for i in range(len(image_files))]

        if HAS_DEEP_MODEL:
            print(f"[ReID] 使用 {self.inferencer.model.__class__.__name__} 模型提取特征...")
            valid_paths = []
            valid_labels = []
            valid_imgs = []
            for file_path, label in zip(image_files, labels):
                img = self._safe_imread(file_path)
                if img is not None:
                    valid_paths.append(file_path)
                    valid_labels.append(label)
                    valid_imgs.append(img)

            features = self.inferencer.extract_features_batch(valid_paths)

            for i, (feat, label, img) in enumerate(zip(features, valid_labels, valid_imgs)):
                print(f"  特征 {i+1}/{len(features)}: {label} - 维度: {len(feat)}, 范数: {np.linalg.norm(feat):.4f}")
                self.feature_db.append(feat)
                self.labels.append(label)
                disp = cv2.resize(img, (128, 256))
                self.images.append(disp)
        else:
            print("[信息] 使用传统特征（后备方案）")
            self._build_feature_db_fallback(image_files, labels)

        print(f"\n数据库构建完成！共 {len(self.feature_db)} 个样本")

    def _build_feature_db_fallback(self, image_files, labels):
        """后备方案：传统特征"""
        for file_path, label in zip(image_files, labels):
            print(f"\n处理图片: {os.path.basename(file_path)}")
            img = self._safe_imread(file_path)
            if img is None:
                continue
            features, img_display = self._extract_traditional_features(img)
            self.feature_db.append(features)
            self.labels.append(label)
            self.images.append(img_display)

    def _extract_traditional_features(self, image):
        """传统特征提取（后备方案）"""
        img = cv2.resize(image, (128, 256))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        features = []
        for channel, _ in [(img, 'BGR'), (hsv, 'HSV')]:
            for i in range(3):
                hist = cv2.calcHist([channel], [i], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                features.extend(hist)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, _ = cv2.cartToPolar(gx, gy)
        hist_grad, _ = np.histogram(mag.flatten(), bins=32, range=(0, mag.max() + 1e-5))
        features.extend(hist_grad / (hist_grad.sum() + 1e-5))
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        h = edges.shape[0]
        top_edges = np.sum(edges[:h//2, :] > 0)
        bottom_edges = np.sum(edges[h//2:, :] > 0)
        features.append(edge_density)
        features.append(top_edges / (bottom_edges + 1e-5))
        feat = np.array(features)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat, img

    def search_similar(self, query_image, top_k=3):
        """在数据库中搜索相似图片"""
        print("\n" + "=" * 70)
        print("步骤3: 执行相似度搜索")
        print("=" * 70)

        if len(self.feature_db) == 0:
            print("[错误] 数据库为空！")
            return [], None

        # 提取查询图片特征（使用系统临时目录，避免污染底库）
        if HAS_DEEP_MODEL:
            import tempfile
            temp_path = os.path.join(tempfile.gettempdir(), "_reid_query_temp.png")
            cv2.imwrite(temp_path, query_image)
            query_features = self.inferencer.extract_feature(temp_path)
            try:
                os.remove(temp_path)
            except OSError:
                pass
            print(f"[ReID] 查询特征维度: {len(query_features):d}")
        else:
            query_features, _ = self._extract_traditional_features(query_image)

        # backbone推理时已做L2归一化（范数≈1.0），无需再做特征中心化
        # 直接使用点积计算余弦相似度（对L2归一化的向量，点积=余弦相似度）
        db_feats = np.array(self.feature_db)
        q_norm = np.linalg.norm(query_features)
        
        similarities = []
        print("\n与数据库中每个样本的相似度:")
        print("-" * 50)

        for i, (feat, label) in enumerate(zip(self.feature_db, self.labels)):
            g_norm = np.linalg.norm(feat)
            if q_norm > 0 and g_norm > 0:
                sim = float(np.dot(query_features, feat) / (q_norm * g_norm))
            else:
                sim = 0.0
            similarities.append((i, label, sim))
            print(f"  {label}: {sim:.4f}")

        # 按相似度降序排列
        similarities.sort(key=lambda x: x[2], reverse=True)

        results = []
        print("\n" + "=" * 50)
        print(f"Top-{top_k} 搜索结果:")
        print("=" * 50)

        for rank, (idx, label, sim) in enumerate(similarities[:top_k], 1):
            if sim > 0.6:
                conf = "高置信度"
            elif sim > 0.4:
                conf = "中置信度"
            elif sim > 0.2:
                conf = "低置信度"
            else:
                conf = "极低置信度"
            print(f"  #{rank} {label:15s} 相似度: {sim:.4f}  [{conf}]")
            results.append({
                'rank': rank,
                'label': label,
                'similarity': sim,
                'image': self.images[idx]
            })

        return results, query_image

    def visualize_results(self, results, query_img):
        """可视化结果"""
        print("\n" + "=" * 70)
        print("步骤4: 生成可视化结果")
        print("=" * 70)

        n_results = len(results) + 1
        fig, axes = plt.subplots(1, n_results, figsize=(4 * n_results, 5))

        if n_results == 2:
            axes = [axes[0], axes[1]]

        axes[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        title = "Query"  # 统一使用英文避免字体问题
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].axis('off')

        for i, result in enumerate(results):
            axes[i + 1].imshow(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB))
            s = result['similarity']
            if USE_ENGLISH:
                conf_text = "High" if s > 0.6 else "Mid" if s > 0.4 else "Low" if s > 0.2 else "VeryLow"
                title = f"#{result['rank']} {result['label']}\nsim:{s:.4f} [{conf_text}]"
            else:
                conf_text = "\u9ad8" if s > 0.6 else "\u4e2d" if s > 0.4 else "\u4f4e" if s > 0.2 else "\u6781\u4f4e"
                title = f"#{result['rank']} {result['label']}\n{s:.4f}({conf_text})"
            axes[i + 1].set_title(title, fontsize=10)
            axes[i + 1].axis('off')

        plt.tight_layout()
        output_path = os.path.join(self.image_dir, 'search_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"结果图片已保存到: {output_path}")
        plt.close()

    def analyze_image(self, image_path):
        """分析单张图片"""
        print("\n" + "=" * 70)
        print(f"图片分析: {os.path.basename(image_path)}")
        print("=" * 70)

        img = self._safe_imread(image_path)
        if img is None:
            print("[错误] 无法读取图片")
            return

        print(f"尺寸: {img.shape[1]} x {img.shape[0]} 像素")
        print(f"通道数: {img.shape[2] if len(img.shape) > 2 else 1}")

        for i, name in enumerate(['Blue', 'Green', 'Red']):
            channel = img[:, :, i]
            print(f"  {name}通道: 均值={channel.mean():.1f}, 范围=[{channel.min()}, {channel.max()}]")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_count = np.sum(edges > 0)
        print(f"边缘数量: {edge_count} ({edge_count/(edges.size)*100:.1f}%)")

        return img

    def run_demo(self):
        """运行完整演示"""
        print("\n")
        print("*" * 70)
        print("*" + " " * 68 + "*")
        print("*" + "  面向灾害应急搜救的抗恶劣环境行人重识别系统  ".center(68) + "*")
        print("*" + "  Real Image Person Re-Identification Demo  ".center(68) + "*")
        print("*" + " " * 68 + "*")
        print("*" * 70)

        image_files = self.load_images_from_folder()

        if len(image_files) == 0:
            print("\n[警告] 文件夹中没有找到图片！")
            print(f"请将行人图片放入以下文件夹: {self.image_dir}")
            print("\n将创建示例图片进行演示...")
            self.create_sample_images()
            image_files = self.load_images_from_folder()

        print(f"\n加载了 {len(image_files)} 张图片")

        # 使用最后一张作为查询，其余构建底库
        query_file = image_files[-1]
        gallery_files = image_files[:-1]

        if not gallery_files:
            print("[错误] 底库至少需要一张图片！")
            return

        # 为底库生成标签：优先使用子文件夹名作为人员ID
        gallery_labels = []
        gallery_root = Path(self.image_dir)
        for f in gallery_files:
            rel = Path(f).parent
            if rel.name != gallery_root.name:
                gallery_labels.append(f"[{rel.name}]")
            else:
                gallery_labels.append(Path(f).stem[:12])

        self.build_feature_database(gallery_files, gallery_labels)

        query_img = self._safe_imread(query_file)
        if query_img is not None:
            results, query_processed = self.search_similar(query_img, top_k=min(3, len(gallery_files)))
            self.visualize_results(results, query_processed)
        else:
            print("[错误] 无法读取查询图片")

        print("\n" + "=" * 70)
        print("演示完成！")
        print("=" * 70)

    def create_sample_images(self):
        """创建示例图片"""
        print("\n创建示例行人图片...")

        sample_dir = self.image_dir
        os.makedirs(sample_dir, exist_ok=True)

        np.random.seed(42)

        sample_images = [
            ("person_01.jpg", [255, 100, 100], "Person 01 - Red Shirt"),
            ("person_02.jpg", [100, 255, 100], "Person 02 - Green Shirt"),
            ("person_03.jpg", [100, 100, 255], "Person 03 - Blue Shirt"),
            ("person_04.jpg", [255, 255, 100], "Person 04 - Yellow Shirt"),
            ("person_05.jpg", [100, 255, 255], "Person 05 - Cyan Shirt"),
        ]

        for filename, color, description in sample_images:
            filepath = os.path.join(sample_dir, filename)
            img = np.random.randint(50, 200, (256, 128, 3), dtype=np.uint8)
            cv2.rectangle(img, (30, 50), (98, 230), color, -1)
            cv2.circle(img, (64, 35), 25, (220, 180, 160), -1)
            cv2.putText(img, description.split(' - ')[0], (35, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(filepath, img)
            print(f"  [OK] 创建: {filename}")

        print("示例图片创建完成")


def main():
    """主函数"""
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description='行人重识别演示')
    parser.add_argument('--gallery', '-g', type=str, default=None,
                        help='底库目录路径 (默认: rescue_gallery)')
    parser.add_argument('--query', '-q', type=str, default=None,
                        help='指定查询图片路径 (可选)')
    args = parser.parse_args()
    
    start = time.time()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.gallery:
        image_dir = args.gallery
    else:
        image_dir = os.path.join(project_root, 'rescue_gallery')

    print(f"图片目录: {image_dir}")
    print(f"目录存在: {os.path.exists(image_dir)}")

    demo = RealImageReIDDemo(image_dir)

    try:
        demo.run_demo()
    except Exception as e:
        print(f"\n[错误] 演示运行异常: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start
    print(f"\n演示总耗时: {elapsed:.1f} 秒")
    print("\n按 Enter 键退出...")
    input()


if __name__ == '__main__':
    main()
