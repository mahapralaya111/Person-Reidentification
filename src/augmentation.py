"""
抗恶劣环境行人重识别系统 - 环境增强模块
模拟各种恶劣环境：烟雾、雨雾、运动模糊、低光照、噪声等

使用方法：
    from augmentation import (
        WeatherAugmentation,
        MotionBlur,
        RainEffect,
        FogEffect,
        NoiseEffect,
        LowLightEffect
    )
    
    # 在训练时使用
    transform = transforms.Compose([
        WeatherAugmentation(p=0.5),  # 随机应用恶劣环境
        transforms.ToTensor(),
    ])
"""

import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance


# ============== 基础增强类 ==============
class BaseAugmentation:
    """基础增强类"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        return self.apply(img)
    
    def apply(self, img):
        raise NotImplementedError


# ============== 烟雾效果 ==============
class SmokeEffect(BaseAugmentation):
    """模拟烟雾效果
    
    烟雾特点：
    - 半透明遮罩
    - 边缘模糊
    - 颜色偏灰白
    """
    
    def __init__(self, p=0.3, intensity=0.3):
        super().__init__(p)
        self.intensity = intensity
    
    def apply(self, img):
        h, w = img.shape[:2]
        
        # 创建烟雾掩码
        smoke_mask = self._generate_smoke_mask(h, w)
        
        # 烟雾颜色（灰白色）
        smoke_color = np.array([180, 185, 190], dtype=np.uint8)
        
        # 混合烟雾和原图
        result = img.copy()
        for c in range(3):
            result[:, :, c] = np.clip(
                img[:, :, c] * (1 - smoke_mask * self.intensity) + 
                smoke_color[c] * smoke_mask * self.intensity,
                0, 255
            ).astype(np.uint8)
        
        # 轻微模糊
        result = cv2.GaussianBlur(result, (5, 5), 0)
        
        return result
    
    def _generate_smoke_mask(self, h, w):
        """生成烟雾掩码"""
        # 使用Perlin噪声生成烟雾
        mask = np.zeros((h, w), dtype=np.float32)
        
        # 多个烟雾团
        num_blobs = random.randint(3, 8)
        for _ in range(num_blobs):
            cx, cy = random.randint(0, w), random.randint(0, h)
            rx, ry = random.randint(w//4, w//2), random.randint(h//4, h//2)
            
            y, x = np.ogrid[:h, :w]
            ellipse = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
            blob = np.exp(-ellipse)
            mask = np.maximum(mask, blob)
        
        # 归一化
        mask = mask / (mask.max() + 1e-8)
        
        return mask


# ============== 雨雾效果 ==============
class FogEffect(BaseAugmentation):
    """模拟雾霾效果
    
    雾霾特点：
    - 大范围均匀遮罩
    - 对比度降低
    - 颜色偏灰蓝
    - 远处物体更模糊
    """
    
    def __init__(self, p=0.3, fog_density=0.5):
        super().__init__(p)
        self.fog_density = fog_density
    
    def apply(self, img):
        h, w = img.shape[:2]
        
        # 生成雾密度图（距离越远雾越浓）
        fog_mask = self._generate_fog_mask(h, w)
        
        # 雾颜色（灰蓝色）
        fog_color = np.array([180, 190, 200], dtype=np.uint8)
        
        # 混合
        result = img.copy()
        for c in range(3):
            result[:, :, c] = np.clip(
                img[:, :, c] * (1 - fog_mask * self.fog_density) + 
                fog_color[c] * fog_mask * self.fog_density,
                0, 255
            ).astype(np.uint8)
        
        # 添加轻微模糊模拟远景
        result = cv2.GaussianBlur(result, (3, 3), 1)
        
        return result
    
    def _generate_fog_mask(self, h, w):
        """生成雾密度图"""
        # 使用径向渐变模拟远近感
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        
        # 距离中心的距离
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        
        # 归一化（边缘更浓）
        fog_mask = dist / max_dist
        
        # 添加噪声
        noise = np.random.rand(h, w) * 0.1
        fog_mask = np.clip(fog_mask + noise, 0, 1)
        
        return fog_mask


# ============== 雨滴效果 ==============
class RainEffect(BaseAugmentation):
    """模拟雨天效果
    
    雨天特点：
    - 斜向雨滴条纹
    - 水珠附着
    - 光晕效果
    """
    
    def __init__(self, p=0.3, rain_intensity=0.3, angle=-30):
        super().__init__(p)
        self.rain_intensity = rain_intensity
        self.angle = angle  # 雨滴角度
    
    def apply(self, img):
        result = img.copy()
        h, w = img.shape[:2]
        
        # 生成雨滴
        num_drops = int(h * w * 0.01 * self.rain_intensity)
        
        for _ in range(num_drops):
            x = random.randint(0, w)
            y = random.randint(0, h)
            length = random.randint(5, 15)
            thickness = random.randint(1, 2)
            
            # 计算雨滴终点
            angle_rad = np.radians(self.angle)
            x2 = int(x + length * np.sin(angle_rad))
            y2 = int(y + length * np.cos(angle_rad))
            
            # 绘制雨滴
            cv2.line(result, (x, y), (x2, y2), (200, 210, 220), thickness)
        
        # 添加水珠
        num_droplets = int(num_drops * 0.3)
        for _ in range(num_droplets):
            x = random.randint(0, w - 5)
            y = random.randint(0, h - 5)
            radius = random.randint(1, 3)
            
            # 水珠
            cv2.circle(result, (x, y), radius, (220, 225, 230), -1)
            
            # 光晕
            cv2.circle(result, (x - 1, y - 1), radius // 2, (255, 255, 255), -1)
        
        # 整体稍微变暗变蓝
        result = result.astype(np.float32)
        result[:, :, 0] = np.clip(result[:, :, 0] * 1.05, 0, 255)  # 蓝色增强
        result[:, :, 2] = np.clip(result[:, :, 2] * 0.95, 0, 255)  # 红色减弱
        result = np.clip(result * 0.95, 0, 255).astype(np.uint8)
        
        return result


# ============== 运动模糊 ==============
class MotionBlur(BaseAugmentation):
    """模拟运动模糊
    
    运动模糊特点：
    - 沿运动方向的条纹
    - 边缘拖影
    """
    
    def __init__(self, p=0.3, blur_size=15, angle=None):
        super().__init__(p)
        self.blur_size = blur_size
        self.angle = angle
    
    def apply(self, img):
        h, w = img.shape[:2]
        
        # 随机角度或指定角度
        if self.angle is None:
            ksize = self.blur_size
            M = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), random.uniform(-45, 45), 1)
        else:
            ksize = self.blur_size
            M = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), self.angle, 1)
        
        # 创建运动模糊核
        kernel = np.zeros((ksize, ksize))
        cv2.line(kernel, (ksize // 2, 0), (ksize // 2, ksize - 1), 1, thickness=1)
        kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
        kernel = kernel / (kernel.sum() + 1e-8)
        
        # 应用模糊
        result = cv2.filter2D(img, -1, kernel)
        
        return result


# ============== 低光照效果 ==============
class LowLightEffect(BaseAugmentation):
    """模拟低光照/夜间环境
    
    低光照特点：
    - 整体变暗
    - 对比度降低
    - 可能有过曝/欠曝区域
    """
    
    def __init__(self, p=0.3, brightness_factor=0.3, add_noise=True):
        super().__init__(p)
        self.brightness_factor = brightness_factor
        self.add_noise = add_noise
    
    def apply(self, img):
        result = img.astype(np.float32)
        
        # 降低亮度
        result = result * self.brightness_factor
        
        # 降低对比度
        mean = result.mean()
        contrast_factor = random.uniform(0.5, 0.8)
        result = mean + (result - mean) * contrast_factor
        
        # 添加噪点模拟夜间噪声
        if self.add_noise:
            noise = np.random.normal(0, 10, result.shape)
            result = result + noise
        
        # 随机过曝区域（灯光）
        if random.random() > 0.5:
            num_overexposed = random.randint(1, 3)
            for _ in range(num_overexposed):
                x = random.randint(0, img.shape[1] - 30)
                y = random.randint(0, img.shape[0] - 30)
                cv2.circle(result, (x, y), random.randint(10, 30), (255, 255, 255), -1)
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result


# ============== 高斯噪声 ==============
class NoiseEffect(BaseAugmentation):
    """添加各种噪声
    
    噪声类型：
    - 高斯噪声
    - 椒盐噪声
    - 泊松噪声
    """
    
    def __init__(self, p=0.3, noise_type='gaussian', noise_level=0.02):
        super().__init__(p)
        self.noise_type = noise_type
        self.noise_level = noise_level
    
    def apply(self, img):
        result = img.astype(np.float32) / 255.0
        
        if self.noise_type == 'gaussian':
            noise = np.random.normal(0, self.noise_level, result.shape)
            result = result + noise
        
        elif self.noise_type == 'salt_pepper':
            # 椒盐噪声
            s_vs_p = 0.5
            amount = self.noise_level
            
            num_salt = np.ceil(amount * result.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in result.shape]
            result[tuple(coords)] = 1
            
            num_pepper = np.ceil(amount * result.size * (1 - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in result.shape]
            result[tuple(coords)] = 0
        
        elif self.noise_type == 'poisson':
            vals = len(np.unique(result))
            vals = 2 ** np.ceil(np.log2(vals))
            result = np.random.poisson(result * vals) / float(vals)
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result


# ============== 红外效果 ==============
class InfraredEffect(BaseAugmentation):
    """模拟红外热成像效果
    
    红外特点：
    - 灰度或伪彩色
    - 温度分布可视化
    - 低对比度
    """
    
    def __init__(self, p=0.2, colorize=False):
        super().__init__(p)
        self.colorize = colorize
    
    def apply(self, img):
        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 应用伪彩色映射
        if self.colorize:
            # 红外伪彩色
            result = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        else:
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        return result


# ============== 遮挡效果 ==============
class OcclusionEffect(BaseAugmentation):
    """模拟部分遮挡
    
    遮挡类型：
    - 随机矩形遮挡
    - 随机擦除
    """
    
    def __init__(self, p=0.3, max_area_ratio=0.2):
        super().__init__(p)
        self.max_area_ratio = max_area_ratio
    
    def apply(self, img):
        result = img.copy()
        h, w = img.shape[:2]
        
        num_occlusions = random.randint(1, 3)
        
        for _ in range(num_occlusions):
            # 随机大小的矩形
            oc_h = int(h * random.uniform(0.05, self.max_area_ratio))
            oc_w = int(w * random.uniform(0.05, self.max_area_ratio))
            
            x = random.randint(0, w - oc_w)
            y = random.randint(0, h - oc_h)
            
            # 用随机颜色填充
            fill_color = [random.randint(0, 50) for _ in range(3)]
            result[y:y+oc_h, x:x+oc_w] = fill_color
        
        return result


# ============== 姿态畸变效果 ==============
class PoseDistortionEffect(BaseAugmentation):
    """模拟搜救场景中的非常规姿态
    
    姿态类型：
    - 倒地姿势（俯卧/仰卧）
    - 匍匐姿势
    - 蜷缩姿势
    - 残缺肢体（部分身体被埋）
    """
    
    def __init__(self, p=0.3, distortion_type='all'):
        super().__init__(p)
        self.distortion_type = distortion_type
    
    def apply(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
            was_pil = True
        else:
            was_pil = False
        
        h, w = img.shape[:2]
        
        # 随机选择畸变类型
        if self.distortion_type == 'all':
            distortion_types = ['lying', 'crawling', 'crouching', 'partial']
            distortion = random.choice(distortion_types)
        else:
            distortion = self.distortion_type
        
        if distortion == 'lying':
            result = self._apply_lying(img, h, w)
        elif distortion == 'crawling':
            result = self._apply_crawling(img, h, w)
        elif distortion == 'crouching':
            result = self._apply_crouching(img, h, w)
        elif distortion == 'partial':
            result = self._apply_partial(img, h, w)
        else:
            result = img
        
        if was_pil:
            result = Image.fromarray(result)
        
        return result
    
    def _apply_lying(self, img, h, w):
        """倒地姿势：水平翻转+压扁"""
        # 随机选择俯卧或仰卧
        if random.random() > 0.5:
            img = cv2.flip(img, 0)  # 垂直翻转
        
        # 压扁模拟倒地
        scale_y = random.uniform(0.4, 0.7)
        new_h = int(h * scale_y)
        scaled = cv2.resize(img, (w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 添加到画布中央
        result = np.zeros_like(img)
        y_offset = (h - new_h) // 2
        result[y_offset:y_offset+new_h, :] = scaled
        
        # 添加阴影模拟地面
        shadow = np.ones((h, w, 3), dtype=np.float32) * 0.9
        shadow[y_offset:y_offset+new_h, :] = 1.0
        result = (result * shadow).astype(np.uint8)
        
        return result
    
    def _apply_crawling(self, img, h, w):
        """匍匐姿势：倾斜+不对称压扁"""
        # 随机角度倾斜（15-45度）
        angle = random.uniform(-40, 40)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))
        
        # 不对称压扁
        scale_x = random.uniform(0.6, 0.9)
        scale_y = random.uniform(0.5, 0.7)
        new_w, new_h = int(w * scale_x), int(h * scale_y)
        scaled = cv2.resize(rotated, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 放置在画布
        result = np.zeros_like(img)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled
        
        return result
    
    def _apply_crouching(self, img, h, w):
        """蜷缩姿势：中心缩放"""
        # 随机缩放（模拟蜷缩）
        scale = random.uniform(0.5, 0.75)
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 添加到画布
        result = np.zeros_like(img)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled
        
        # 添加模糊边缘模拟模糊衣物
        result = cv2.GaussianBlur(result, (5, 5), 0)
        
        return result
    
    def _apply_partial(self, img, h, w):
        """部分残缺：随机裁切"""
        # 随机裁切边缘
        crop_ratio = random.uniform(0.1, 0.3)
        
        crop_top = int(h * crop_ratio * random.random())
        crop_bottom = int(h * crop_ratio * random.random())
        crop_left = int(w * crop_ratio * random.random())
        crop_right = int(w * crop_ratio * random.random())
        
        # 裁切
        cropped = img[crop_top:h-crop_bottom, crop_left:w-crop_right]
        
        # 调整回原尺寸
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return result


# ============== 视角变换增强 ==============
class ViewpointAugmentation(BaseAugmentation):
    """模拟不同拍摄视角的增强
    
    视角类型：
    - 水平旋转（模拟左右视角变化）
    - 垂直旋转（模拟俯仰角度变化）
    - 仿射变换（模拟相机位置变化）
    - 透视变换（模拟远近和角度变化）
    """
    
    def __init__(self, p=0.3, viewpoint_type='all'):
        super().__init__(p)
        self.viewpoint_type = viewpoint_type
    
    def apply(self, img):
        try:
            if isinstance(img, Image.Image):
                img = np.array(img)
                was_pil = True
            else:
                was_pil = False
            
            # 确保图像有效
            if img is None or img.size == 0:
                return Image.fromarray(img) if was_pil else img
            
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                return Image.fromarray(img) if was_pil else img
            
            # 随机选择视角变换类型
            if self.viewpoint_type == 'all':
                viewpoint_types = ['horizontal', 'vertical', 'affine', 'perspective']
                viewpoint = random.choice(viewpoint_types)
            else:
                viewpoint = self.viewpoint_type
            
            if viewpoint == 'horizontal':
                result = self._apply_horizontal(img, h, w)
            elif viewpoint == 'vertical':
                result = self._apply_vertical(img, h, w)
            elif viewpoint == 'affine':
                result = self._apply_affine(img, h, w)
            elif viewpoint == 'perspective':
                result = self._apply_perspective(img, h, w)
            else:
                result = img
            
            # 确保结果有效
            if result is None or result.size == 0:
                result = img
            
            # 确保数据类型正确
            if not isinstance(result, np.ndarray):
                result = np.array(result)
            
            if result.dtype != np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)
            
            if was_pil:
                result = Image.fromarray(result)
            
            return result
        except Exception as e:
            # 发生任何异常时返回原图，确保训练不中断
            print(f"ViewpointAugmentation 出错: {e}，返回原图")
            return img
    
    def _apply_horizontal(self, img, h, w):
        """水平旋转（模拟左右视角变化）"""
        try:
            # 随机旋转角度（-15° 到 15°），模拟轻微左右转头
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            result = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            return result
        except Exception:
            return img
    
    def _apply_vertical(self, img, h, w):
        """垂直旋转（模拟俯仰角度变化）"""
        try:
            # 垂直方向的仿射变换，模拟抬头/低头
            # 轻微压缩顶部或底部
            scale_y = random.uniform(0.9, 1.1)  # 轻微垂直缩放
            shift_y = random.uniform(-0.05, 0.05) * h  # 轻微垂直平移
            
            src_points = np.float32([[0, 0], [w, 0], [0, h]])
            dst_points = np.float32([
                [0, shift_y],
                [w, shift_y],
                [0, h * scale_y + shift_y]
            ])
            
            # 检查变换是否有效（避免奇异矩阵）
            if self._is_affine_valid(src_points, dst_points):
                M = cv2.getAffineTransform(src_points, dst_points)
                result = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                return result
            else:
                return img
        except Exception:
            return img
    
    def _apply_affine(self, img, h, w):
        """仿射变换（模拟相机位置变化）"""
        try:
            # 随机仿射变换，模拟相机位置变化
            src_points = np.float32([[0, 0], [w, 0], [0, h]])
            dst_points = np.float32([
                [random.uniform(-0.1, 0.1) * w, random.uniform(-0.1, 0.1) * h],
                [w + random.uniform(-0.1, 0.1) * w, random.uniform(-0.1, 0.1) * h],
                [random.uniform(-0.1, 0.1) * w, h + random.uniform(-0.1, 0.1) * h]
            ])
            
            # 检查变换是否有效（避免奇异矩阵）
            if self._is_affine_valid(src_points, dst_points):
                M = cv2.getAffineTransform(src_points, dst_points)
                result = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                return result
            else:
                return img
        except Exception:
            return img
    
    def _apply_perspective(self, img, h, w):
        """透视变换（模拟远近和角度变化）"""
        try:
            # 随机透视变换，模拟远近和角度变化
            src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            
            # 控制变换强度，避免过度扭曲
            max_offset = min(w, h) * 0.15  # 最大偏移为尺寸的15%
            if max_offset < 1:  # 防止过小图像导致偏移为0
                max_offset = max(w, h) * 0.1
            
            dst_points = np.float32([
                [random.uniform(-max_offset, max_offset), random.uniform(-max_offset, max_offset)],
                [w + random.uniform(-max_offset, max_offset), random.uniform(-max_offset, max_offset)],
                [random.uniform(-max_offset, max_offset), h + random.uniform(-max_offset, max_offset)],
                [w + random.uniform(-max_offset, max_offset), h + random.uniform(-max_offset, max_offset)]
            ])
            
            # 检查透视变换是否有效（避免奇异矩阵）
            if self._is_perspective_valid(src_points, dst_points):
                M = cv2.getPerspectiveTransform(src_points, dst_points)
                result = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                return result
            else:
                return img
        except Exception:
            return img
    
    def _is_affine_valid(self, src_points, dst_points):
        """检查仿射变换是否有效（避免奇异矩阵）"""
        try:
            # 检查点是否共线
            src_array = np.array(src_points)
            dst_array = np.array(dst_points)
            
            # 计算行列式（2x2矩阵）
            # 使用前两个点的向量
            src_vec1 = src_array[1] - src_array[0]
            src_vec2 = src_array[2] - src_array[0]
            
            # 检查源点是否共线
            cross_product = np.cross(src_vec1, src_vec2)
            if abs(cross_product) < 1e-6:
                return False
            
            # 检查目标点是否共线
            dst_vec1 = dst_array[1] - dst_array[0]
            dst_vec2 = dst_array[2] - dst_array[0]
            cross_product = np.cross(dst_vec1, dst_vec2)
            if abs(cross_product) < 1e-6:
                return False
            
            return True
        except Exception:
            return False
    
    def _is_perspective_valid(self, src_points, dst_points):
        """检查透视变换是否有效（避免奇异矩阵）"""
        try:
            # 检查点是否共面（在2D中，四点不应有三点共线）
            src_array = np.array(src_points)
            dst_array = np.array(dst_points)
            
            # 检查源点
            for i in range(4):
                # 取其他三个点
                indices = [j for j in range(4) if j != i]
                vec1 = src_array[indices[1]] - src_array[indices[0]]
                vec2 = src_array[indices[2]] - src_array[indices[0]]
                cross_product = np.cross(vec1, vec2)
                if abs(cross_product) < 1e-6:
                    return False
            
            # 检查目标点
            for i in range(4):
                indices = [j for j in range(4) if j != i]
                vec1 = dst_array[indices[1]] - dst_array[indices[0]]
                vec2 = dst_array[indices[2]] - dst_array[indices[0]]
                cross_product = np.cross(vec1, vec2)
                if abs(cross_product) < 1e-6:
                    return False
            
            return True
        except Exception:
            return False


# ============== 组合天气增强 ==============
class WeatherAugmentation:
    """组合天气增强
    
    随机应用多种天气效果
    """
    
    def __init__(self, p=0.5, weather_types=None):
        """
        Args:
            p: 应用增强的概率
            weather_types: 要启用的天气类型列表，为None时启用所有
                         可选: ['smoke', 'fog', 'rain', 'motion_blur', 'low_light', 'noise', 'infrared', 'pose_distortion', 'viewpoint']
        """
        self.p = p
        
        if weather_types is None:
            weather_types = ['smoke', 'fog', 'rain', 'motion_blur', 'low_light', 'noise', 'infrared', 'pose_distortion', 'viewpoint']
        
        self.weather_types = weather_types
        self.augmentations = self._build_augmentations()
    
    def _build_augmentations(self):
        augs = []
        
        if 'smoke' in self.weather_types:
            augs.append(SmokeEffect(p=0.4))
        
        if 'fog' in self.weather_types:
            augs.append(FogEffect(p=0.4))
        
        if 'rain' in self.weather_types:
            augs.append(RainEffect(p=0.3))
        
        if 'motion_blur' in self.weather_types:
            augs.append(MotionBlur(p=0.3))
        
        if 'low_light' in self.weather_types:
            augs.append(LowLightEffect(p=0.3))
        
        if 'noise' in self.weather_types:
            augs.append(NoiseEffect(p=0.2))
        
        if 'infrared' in self.weather_types:
            augs.append(InfraredEffect(p=0.4))
        
        if 'pose_distortion' in self.weather_types:
            augs.append(PoseDistortionEffect(p=0.4))
        
        if 'viewpoint' in self.weather_types:
            augs.append(ViewpointAugmentation(p=0.4))
        
        return augs
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        # 随机选择一种天气效果
        aug = random.choice(self.augmentations)
        return aug(img)


# ============== torchvision 变换支持 ==============
class TorchWeatherAugmentation:
    """PyTorch版本的天气增强（用于GPU）"""
    
    def __init__(self, p=0.5, weather_types=None):
        self.p = p
        self.weather_types = weather_types or ['smoke', 'fog', 'rain', 'infrared', 'pose_distortion']
    
    def __call__(self, img):
        """img: Tensor [C, H, W]"""
        if random.random() > self.p:
            return img
        
        # 随机选择天气类型
        weather = random.choice(self.weather_types)
        
        if weather == 'smoke':
            return self._add_smoke(img)
        elif weather == 'fog':
            return self._add_fog(img)
        elif weather == 'rain':
            return self._add_rain(img)
        elif weather == 'infrared':
            return self._add_infrared(img)
        elif weather == 'pose_distortion':
            return self._add_pose_distortion(img)
        
        return img
    
    def _add_smoke(self, img):
        C, H, W = img.shape
        # 随机烟雾区域
        mask = torch.rand(H, W) * 0.3
        mask = mask.unsqueeze(0).expand(C, -1, -1)
        fog_color = torch.tensor([0.7, 0.72, 0.75]).view(3, 1, 1)
        return img * (1 - mask) + fog_color * mask
    
    def _add_fog(self, img):
        C, H, W = img.shape
        mask = torch.rand(H, W) * 0.4
        mask = mask.unsqueeze(0).expand(C, -1, -1)
        fog_color = torch.tensor([0.7, 0.74, 0.78]).view(3, 1, 1)
        return img * (1 - mask) + fog_color * mask
    
    def _add_rain(self, img):
        # 简单的雨线效果
        C, H, W = img.shape
        noise = torch.randn_like(img) * 0.05
        return img + noise
    
    def _add_infrared(self, img):
        """转换为红外风格（灰度化）"""
        # 将 RGB 转为灰度
        if img.shape[0] == 3:  # [C, H, W]
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            # 复制灰度到三个通道
            infrared = gray.unsqueeze(0).expand(3, -1, -1)
            return infrared
        return img
    
    def _add_pose_distortion(self, img):
        """简化的姿态畸变（用于GPU）"""
        C, H, W = img.shape
        # 随机裁切边缘模拟部分残缺
        crop_ratio = random.uniform(0.05, 0.15)
        h_crop = int(H * crop_ratio)
        w_crop = int(W * crop_ratio)
        
        # 随机裁切
        top = random.randint(0, h_crop)
        bottom = random.randint(0, h_crop)
        left = random.randint(0, w_crop)
        right = random.randint(0, w_crop)
        
        cropped = img[:, top:H-bottom, left:W-right]
        
        # 调整回原尺寸
        cropped = cropped.unsqueeze(0) if cropped.dim() == 2 else cropped
        result = torch.nn.functional.interpolate(
            cropped.unsqueeze(0), 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        return result[:C, :H, :W]


# ============== 极端天气增强 ==============
class ExtremeWeatherAugmentation:
    """极端天气组合增强（高难度训练）"""
    
    def __init__(self, p=0.3):
        self.p = p
    
    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        # 随机组合2-3种效果（包含红外和姿态）
        effects = [
            SmokeEffect(p=1.0, intensity=0.5),
            FogEffect(p=1.0, fog_density=0.6),
            RainEffect(p=1.0, rain_intensity=0.5),
            LowLightEffect(p=1.0, brightness_factor=0.2),
            NoiseEffect(p=1.0, noise_level=0.05),
            OcclusionEffect(p=1.0, max_area_ratio=0.25),
            InfraredEffect(p=1.0),
            PoseDistortionEffect(p=1.0),
        ]
        
        # 选择2-3个效果
        selected = random.sample(effects, min(3, len(effects)))
        
        for effect in selected:
            if isinstance(img, Image.Image):
                img = np.array(img)
            img = effect.apply(img)
        
        return img


# ============== 自适应增强（根据图像内容） ==============
class AdaptiveAugmentation:
    """自适应增强，根据图像内容自动调整增强参数"""
    
    def __init__(self, p=0.5):
        self.p = p
        self.weather_aug = WeatherAugmentation(p=p)
    
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img
        
        # 分析图像内容
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        brightness = gray.mean()
        contrast = gray.std()
        
        # 根据图像内容调整增强策略
        if brightness < 80:  # 暗光图像
            # 减少额外的暗光效果
            self.weather_aug.weather_types = ['smoke', 'fog', 'rain', 'motion_blur', 'noise']
        elif brightness > 180:  # 高亮图像
            # 减少雾效果
            self.weather_aug.weather_types = ['smoke', 'rain', 'motion_blur', 'low_light', 'noise']
        else:
            self.weather_aug.weather_types = ['smoke', 'fog', 'rain', 'motion_blur', 'low_light', 'noise']
        
        return self.weather_aug(img)


if __name__ == '__main__':
    # 测试各种增强效果
    import os
    
    # 创建测试图像
    test_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    
    print("测试各种天气增强效果...")
    
    effects = [
        ('原图', test_img),
        ('烟雾', SmokeEffect(p=1.0)(test_img)),
        ('雾霾', FogEffect(p=1.0)(test_img)),
        ('雨天', RainEffect(p=1.0)(test_img)),
        ('运动模糊', MotionBlur(p=1.0)(test_img)),
        ('低光照', LowLightEffect(p=1.0)(test_img)),
        ('噪声', NoiseEffect(p=1.0)(test_img)),
    ]
    
    print("所有增强效果测试完成!")
    print(f"支持的效果类型: smoke, fog, rain, motion_blur, low_light, noise, infrared, occlusion")
