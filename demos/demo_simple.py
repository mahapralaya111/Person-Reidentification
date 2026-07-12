"""
抗恶劣环境行人重识别系统 - 简化演示版
可以直接运行，无需安装额外依赖（除了必要的库）
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# 修复中文显示问题
def setup_chinese_font():
    """配置matplotlib支持中文显示"""
    import matplotlib
    
    # Windows系统字体
    if sys.platform.startswith('win'):
        font_list = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    # Linux系统字体
    elif sys.platform.startswith('linux'):
        font_list = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
    # Mac系统字体
    elif sys.platform.startswith('darwin'):
        font_list = ['PingFang SC', 'Heiti SC', 'SimHei']
    else:
        font_list = ['DejaVu Sans']
    
    # 查找可用字体
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    
    for font in font_list:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            return True
    
    # 如果没有找到合适的中文字体，使用英文标签
    return False

# 设置中文字体
HAS_CHINESE_FONT = setup_chinese_font()

# 检查依赖
try:
    import torch
    import torch.nn as nn
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch未安装，将使用简化版演示")


class SimpleReIDDemo:
    """简化版ReID演示（不依赖深度学习框架）"""
    
    def __init__(self):
        print("=" * 60)
        print("面向灾害应急搜救的抗恶劣环境行人重识别系统")
        print("=" * 60)
        print()
        
        self.feature_db = []  # 特征库
        self.labels = []      # 标签
        self.images = []      # 原始图像
        
    def extract_simple_features(self, image):
        """
        简单的特征提取（演示用）
        实际系统中应使用深度学习模型
        """
        # 将图像调整为固定大小
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError("无法读取图像: {}".format(image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
        
        # 调整大小
        img_resized = cv2.resize(img, (128, 256))
        
        # 提取颜色直方图特征
        hist_features = []
        for channel in range(3):
            hist = cv2.calcHist([img_resized], [channel], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        
        # 提取HOG特征（简化版）
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        # 简单的梯度特征
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        hog_feature = mag.flatten()[:500]  # 取前500个梯度幅值
        
        # 合并特征
        features = np.array(hist_features + hog_feature.tolist())
        features = features / (np.linalg.norm(features) + 1e-5)  # 归一化
        
        return features, img_resized
    
    def add_to_database(self, image, label):
        """添加图像到特征库"""
        feat, img_resized = self.extract_simple_features(image)
        self.feature_db.append(feat)
        self.labels.append(label)
        self.images.append(img_resized)
        print("已添加: {}".format(label))
    
    def search(self, query_image, top_k=3):
        """搜索相似图像"""
        if len(self.feature_db) == 0:
            print("特征库为空，请先添加图像")
            return []
        
        query_feat, query_img = self.extract_simple_features(query_image)
        
        # 计算相似度
        similarities = []
        for feat in self.feature_db:
            sim = np.dot(query_feat, feat)
            similarities.append(sim)
        
        # 排序
        idxs = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in idxs:
            results.append({
                'label': self.labels[idx],
                'similarity': float(similarities[idx]),
                'image': self.images[idx]
            })
        
        return results, query_img
    
    def visualize_results(self, results, query_img):
        """可视化结果"""
        n_results = len(results) + 1
        fig, axes = plt.subplots(1, n_results, figsize=(4 * n_results, 4))
        
        if n_results == 2:
            axes = [axes[0], axes[1]]
        
        # 统一使用英文标签（避免字体问题导致方框）
        query_title = "Query Image"
        result_title = "Top {}: {}\nSim: {:.4f}"
        
        # 显示查询图像
        axes[0].imshow(query_img)
        axes[0].set_title(query_title, fontsize=12)
        axes[0].axis('off')
        
        # 显示结果
        for i, result in enumerate(results):
            axes[i + 1].imshow(result['image'])
            axes[i + 1].set_title(
                result_title.format(i+1, result['label'], result['similarity']),
                fontsize=10
            )
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('search_results.png', dpi=150, bbox_inches='tight')
        print("结果已保存到: search_results.png")
        plt.close()
    
    def run_demo(self):
        """运行演示"""
        print("")
        print("=" * 60)
        print("开始演示...")
        print("=" * 60)
        
        # 创建示例数据
        print("")
        print("1. 创建示例数据库...")
        
        # 生成示例图像（模拟不同人员）
        np.random.seed(42)
        for i in range(5):
            # 创建带有随机颜色的示例图像
            img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
            # 添加一些结构化特征（模拟行人）
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.rectangle(img, (30, 20), (98, 200), color, -1)
            self.add_to_database(img, 'person_{:03d}'.format(i+1))
        
        print("")
        print("数据库创建完成，共 {} 个样本".format(len(self.feature_db)))
        
        # 创建查询图像
        print("")
        print("2. 创建查询图像...")
        query_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        cv2.rectangle(query_img, (30, 20), (98, 200), (100, 150, 200), -1)
        
        # 搜索
        print("")
        print("3. 执行搜索...")
        results, query_processed = self.search(query_img, top_k=3)
        
        print("")
        print("搜索结果:")
        print("-" * 60)
        for i, result in enumerate(results):
            print("  Top {}: {} (相似度: {:.4f})".format(i+1, result['label'], result['similarity']))
        
        # 可视化
        print("")
        print("4. 生成可视化结果...")
        self.visualize_results(results, query_processed)
        
        print("")
        print("=" * 60)
        print("演示完成！")
        print("=" * 60)


def main():
    """主函数"""
    try:
        demo = SimpleReIDDemo()
        demo.run_demo()
        
        # 等待用户查看结果
        print("")
        print("=" * 60)
        print("按 Enter 键退出程序...")
        print("=" * 60)
        input()
        
    except Exception as e:
        print("")
        print("错误: {}".format(e))
        print("")
        print("请确保已安装必要的依赖:")
        print("  pip install numpy opencv-python matplotlib")
        
        # 等待用户查看错误
        print("")
        print("按 Enter 键退出...")
        input()


if __name__ == '__main__':
    main()
