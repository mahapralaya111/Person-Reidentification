#!/usr/bin/env python
"""
底库增强脚本
为每个底库图片生成多个增强版本，增加视角、姿态、环境多样性
"""

import os
import sys
import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# 导入增强模块
from augmentation import (
    WeatherAugmentation,
    ViewpointAugmentation,
    PoseDistortionEffect,
    SmokeEffect,
    FogEffect,
    RainEffect,
    MotionBlur,
    LowLightEffect,
    NoiseEffect,
    InfraredEffect,
    OcclusionEffect
)


class GalleryAugmenter:
    """底库增强器"""
    
    def __init__(self, output_dir, num_augmentations=5, augmentation_types=None):
        """
        初始化增强器
        
        Args:
            output_dir: 输出目录
            num_augmentations: 每张原图生成多少增强版本
            augmentation_types: 增强类型列表，可选:
                'viewpoint': 视角变换 (水平/垂直/仿射/透视)
                'pose': 姿态畸变 (倒地/匍匐/蜷缩)
                'weather': 天气增强 (烟雾/雾霾/雨天等)
                'all': 所有增强
        """
        self.output_dir = Path(output_dir)
        self.num_augmentations = num_augmentations
        
        if augmentation_types is None:
            augmentation_types = ['viewpoint', 'pose', 'weather']
        self.augmentation_types = augmentation_types
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化增强器
        self._init_augmenters()
    
    def _init_augmenters(self):
        """初始化各种增强器"""
        self.augmenters = []
        
        if 'all' in self.augmentation_types or 'viewpoint' in self.augmentation_types:
            # 视角增强 - 使用不同的视角类型
            self.viewpoint_types = ['horizontal', 'vertical', 'affine', 'perspective']
            self.augmenters.append(('viewpoint', ViewpointAugmentation(p=1.0, viewpoint_type='all')))
        
        if 'all' in self.augmentation_types or 'pose' in self.augmentation_types:
            # 姿态增强
            self.augmenters.append(('pose', PoseDistortionEffect(p=1.0)))
        
        if 'all' in self.augmentation_types or 'weather' in self.augmentation_types:
            # 天气增强 - 使用不同的天气类型
            weather_types = ['smoke', 'fog', 'rain', 'motion_blur', 'low_light', 'noise']
            self.augmenters.append(('weather', WeatherAugmentation(p=1.0, weather_types=weather_types)))
    
    def augment_image(self, image_path, person_name, image_index):
        """
        增强单张图片
        
        Args:
            image_path: 原始图片路径
            person_name: 人员姓名（用于组织目录）
            image_index: 图片索引（用于命名）
        
        Returns:
            list: 增强图片保存路径列表
        """
        try:
            # 加载图片（with 块退出前需加载数据，否则文件指针关闭后无法惰性读取）
            with Image.open(image_path) as img_file:
                img_file.load()
                img = img_file.copy()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_array = np.array(img)
            
            # 创建人员输出目录
            person_output_dir = self.output_dir / person_name
            person_output_dir.mkdir(exist_ok=True)
            
            saved_paths = []
            
            # 保存原始图片（作为参考）
            original_filename = f"{person_name}_original_{image_index:03d}.png"
            original_output_path = person_output_dir / original_filename
            img.save(original_output_path)
            saved_paths.append(str(original_output_path))
            
            # 生成增强版本
            for aug_idx in range(self.num_augmentations):
                aug_img = original_array.copy()
                aug_name_parts = [person_name]
                
                # 随机选择增强类型（可以组合多种增强）
                num_augs_to_apply = random.randint(1, 2)  # 应用1-2种增强
                selected_augmenters = random.sample(self.augmenters, min(num_augs_to_apply, len(self.augmenters)))
                
                for aug_type, augmenter in selected_augmenters:
                    if aug_type == 'viewpoint':
                        # 对于视角增强，随机选择具体类型
                        viewpoint_type = random.choice(self.viewpoint_types)
                        viewpoint_aug = ViewpointAugmentation(p=1.0, viewpoint_type=viewpoint_type)
                        aug_img = viewpoint_aug.apply(aug_img)
                        aug_name_parts.append(f"view_{viewpoint_type[:3]}")
                    else:
                        # 其他增强
                        if isinstance(aug_img, np.ndarray):
                            aug_img_pil = Image.fromarray(aug_img)
                        else:
                            aug_img_pil = aug_img
                        
                        aug_img_pil = augmenter(aug_img_pil)
                        
                        if isinstance(aug_img_pil, Image.Image):
                            aug_img = np.array(aug_img_pil)
                        else:
                            aug_img = aug_img_pil
                        
                        aug_name_parts.append(aug_type[:4])
                
                # 确保图片格式正确
                if isinstance(aug_img, np.ndarray):
                    if aug_img.dtype != np.uint8:
                        aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
                    aug_img_pil = Image.fromarray(aug_img)
                else:
                    aug_img_pil = aug_img
                
                # 生成文件名
                aug_name = "_".join(aug_name_parts) + f"_{aug_idx:03d}.png"
                aug_output_path = person_output_dir / aug_name
                
                # 保存增强图片
                aug_img_pil.save(aug_output_path)
                saved_paths.append(str(aug_output_path))
                
                print(f"  生成: {aug_output_path.name}")
            
            return saved_paths
            
        except Exception as e:
            import traceback
            print(f"  增强失败 {image_path}: {e}")
            traceback.print_exc()
            return []
    
    def augment_gallery(self, input_dir):
        """
        增强整个底库目录
        
        Args:
            input_dir: 输入目录，结构为:
                input_dir/
                    person1/
                        image1.jpg
                        image2.jpg
                    person2/
                        ...
        
        Returns:
            dict: 人员 -> 增强图片数量
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            print(f"错误: 输入目录不存在 {input_dir}")
            return {}
        
        statistics = {}
        
        # 遍历所有人员目录
        for person_dir in input_dir.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            print(f"\n处理人员: {person_name}")
            
            # 获取所有图片
            image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + \
                         list(person_dir.glob("*.jpeg")) + list(person_dir.glob("*.bmp"))
            
            if not image_files:
                print(f"  警告: 无图片文件在 {person_dir}")
                continue
            
            total_augmented = 0
            
            # 处理每张图片
            for img_idx, img_path in enumerate(image_files):
                print(f"  处理图片: {img_path.name}")
                saved_paths = self.augment_image(img_path, person_name, img_idx)
                total_augmented += len(saved_paths) - 1  # 减去原始图片
            
            statistics[person_name] = {
                'original': len(image_files),
                'augmented': total_augmented,
                'total': len(image_files) + total_augmented
            }
        
        return statistics


def main():
    parser = argparse.ArgumentParser(description="底库增强工具")
    parser.add_argument('--input', type=str, default='./rescue_gallery',
                       help='输入底库目录 (默认: ./rescue_gallery)')
    parser.add_argument('--output', type=str, default='./rescue_gallery_augmented',
                       help='输出目录 (默认: ./rescue_gallery_augmented)')
    parser.add_argument('--num_aug', type=int, default=5,
                       help='每张原图生成的增强图片数量 (默认: 5)')
    parser.add_argument('--aug_types', type=str, default='viewpoint,pose,weather',
                       help='增强类型，逗号分隔: viewpoint,pose,weather,all (默认: viewpoint,pose,weather)')
    
    args = parser.parse_args()
    
    # 解析增强类型
    augmentation_types = [t.strip() for t in args.aug_types.split(',')]
    
    print("=" * 60)
    print("底库增强工具")
    print("=" * 60)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"增强数量: {args.num_aug} 张/原图")
    print(f"增强类型: {', '.join(augmentation_types)}")
    print("=" * 60)
    
    # 创建增强器
    augmenter = GalleryAugmenter(
        output_dir=args.output,
        num_augmentations=args.num_aug,
        augmentation_types=augmentation_types
    )
    
    # 执行增强
    statistics = augmenter.augment_gallery(args.input)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("增强完成统计")
    print("=" * 60)
    
    total_original = 0
    total_augmented = 0
    
    for person, stats in statistics.items():
        print(f"{person}:")
        print(f"  原始图片: {stats['original']}")
        print(f"  增强图片: {stats['augmented']}")
        print(f"  总计: {stats['total']}")
        
        total_original += stats['original']
        total_augmented += stats['augmented']
    
    print("-" * 60)
    print(f"总计:")
    print(f"  原始图片: {total_original}")
    print(f"  增强图片: {total_augmented}")
    print(f"  总计: {total_original + total_augmented}")
    print(f"  扩充倍数: {(total_original + total_augmented) / max(total_original, 1):.1f}x")
    print("=" * 60)
    print(f"\n增强底库已保存到: {args.output}")
    print("可以使用增强后的底库进行更鲁棒的识别:")
    print(f"  python inference.py --query_dir drone_captures --gallery_dir {args.output}")


if __name__ == "__main__":
    main()