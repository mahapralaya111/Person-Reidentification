"""
基础模型类

定义所有ReID模型的基类，提供统一的接口
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np

from src.utils.logger import get_logger
from src.utils.types import Tensor, DeviceType, get_device, to_device

logger = get_logger(__name__)


class BaseReIDModel(nn.Module, ABC):
    """
    行人重识别模型基类
    
    所有ReID模型都应继承此类，并实现以下接口：
    1. forward: 前向传播
    2. extract_features: 特征提取
    3. get_config: 获取模型配置
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[DeviceType] = None
    ):
        """
        初始化模型
        
        Args:
            config: 模型配置字典
            device: 计算设备
        """
        super().__init__()
        
        # 设置设备
        self._device = get_device(device)
        logger.debug(f"模型初始化 - 设备: {self._device}")
        
        # 配置
        self.config = config or {}
        self._init_from_config()
        
        # 模型状态
        self._initialized = False
        self._trainable = True
    
    @abstractmethod
    def _init_from_config(self) -> None:
        """
        从配置初始化模型
        
        子类必须实现此方法，用于根据配置构建模型结构
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        x: Union[Tensor, Dict[str, Tensor]],
        return_feat: bool = True,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入数据，可以是张量或字典（多模态输入）
            return_feat: 是否返回特征向量
            **kwargs: 其他参数
            
        Returns:
            如果return_feat为True，返回特征向量
            否则返回(分类logits, 特征向量)
        """
        pass
    
    @abstractmethod
    def extract_features(
        self,
        x: Union[Tensor, Dict[str, Tensor]],
        normalize: bool = True
    ) -> Tensor:
        """
        提取特征向量
        
        Args:
            x: 输入数据
            normalize: 是否对特征进行L2归一化
            
        Returns:
            特征向量
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        获取模型配置
        
        Returns:
            模型配置字典
        """
        pass
    
    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
            metadata: 额外元数据
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.get_config(),
            'model_type': self.__class__.__name__,
            'version': '2.0.0'
        }
        
        if metadata:
            checkpoint.update(metadata)
        
        torch.save(checkpoint, path)
        logger.info(f"模型已保存: {path}")
    
    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[DeviceType] = None,
        strict: bool = True
    ) -> 'BaseReIDModel':
        """
        加载模型
        
        Args:
            path: 模型路径
            device: 计算设备
            strict: 是否严格加载权重
            
        Returns:
            加载的模型实例
        """
        logger.info(f"加载模型: {path}")
        
        # 加载检查点
        checkpoint = torch.load(path, map_location='cpu')
        
        # 验证检查点格式
        if 'model_state_dict' not in checkpoint:
            raise ValueError("检查点格式错误: 缺少'model_state_dict'")
        
        if 'config' not in checkpoint:
            raise ValueError("检查点格式错误: 缺少'config'")
        
        # 创建模型实例
        config = checkpoint['config']
        model = cls(config=config, device=device)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # 移动到指定设备
        model.to(get_device(device))
        
        # 设置为评估模式
        model.eval()
        
        logger.info(f"模型加载成功 - 类型: {checkpoint.get('model_type', 'unknown')}")
        
        return model
    
    def freeze(self) -> None:
        """冻结模型参数"""
        for param in self.parameters():
            param.requires_grad = False
        self._trainable = False
        logger.info("模型已冻结")
    
    def unfreeze(self) -> None:
        """解冻模型参数"""
        for param in self.parameters():
            param.requires_grad = True
        self._trainable = True
        logger.info("模型已解冻")
    
    def is_trainable(self) -> bool:
        """检查模型是否可训练"""
        return self._trainable
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        获取参数数量
        
        Args:
            trainable_only: 是否只统计可训练参数
            
        Returns:
            参数数量
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def get_device(self) -> torch.device:
        """获取模型所在设备"""
        return next(self.parameters()).device
    
    def summary(self) -> Dict[str, Any]:
        """
        获取模型摘要信息
        
        Returns:
            模型摘要字典
        """
        return {
            'model_type': self.__class__.__name__,
            'num_parameters': self.get_num_parameters(trainable_only=False),
            'num_trainable_parameters': self.get_num_parameters(trainable_only=True),
            'device': str(self.get_device()),
            'trainable': self.is_trainable(),
            'config': self.get_config()
        }
    
    def print_summary(self) -> None:
        """打印模型摘要"""
        summary = self.summary()
        
        logger.info("=" * 60)
        logger.info("模型摘要")
        logger.info("=" * 60)
        logger.info(f"模型类型: {summary['model_type']}")
        logger.info(f"总参数量: {summary['num_parameters']:,}")
        logger.info(f"可训练参数量: {summary['num_trainable_parameters']:,}")
        logger.info(f"设备: {summary['device']}")
        logger.info(f"可训练: {summary['trainable']}")
        
        # 打印配置
        logger.info("配置:")
        for key, value in summary['config'].items():
            if isinstance(value, (dict, list)):
                logger.info(f"  {key}: {type(value).__name__}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("=" * 60)


class MultiModalMixin:
    """多模态模型混入类"""
    
    def __init__(self):
        self._modalities = []
        self._fusion_method = 'concat'
        
    def register_modality(self, name: str, backbone: nn.Module) -> None:
        """
        注册模态
        
        Args:
            name: 模态名称
            backbone: 骨干网络
        """
        self._modalities.append(name)
        setattr(self, f"{name}_backbone", backbone)
        logger.debug(f"注册模态: {name}")
    
    def set_fusion_method(self, method: str) -> None:
        """
        设置特征融合方法
        
        Args:
            method: 融合方法，支持 'concat', 'add', 'attention'
        """
        valid_methods = ['concat', 'add', 'attention']
        if method not in valid_methods:
            raise ValueError(f"无效的融合方法: {method}，支持: {valid_methods}")
        
        self._fusion_method = method
        logger.info(f"设置特征融合方法: {method}")
    
    def fuse_features(self, features: Dict[str, Tensor]) -> Tensor:
        """
        融合多模态特征
        
        Args:
            features: 各模态特征字典
            
        Returns:
            融合后的特征
        """
        if not features:
            raise ValueError("特征字典不能为空")
        
        if len(features) == 1:
            # 单模态，直接返回
            return next(iter(features.values()))
        
        # 多模态融合
        if self._fusion_method == 'concat':
            # 拼接融合
            fused = torch.cat(list(features.values()), dim=1)
        elif self._fusion_method == 'add':
            # 相加融合
            fused = sum(features.values())
        elif self._fusion_method == 'attention':
            # 注意力融合
            fused = self._attention_fusion(features)
        else:
            raise ValueError(f"未知的融合方法: {self._fusion_method}")
        
        return fused
    
    def _attention_fusion(self, features: Dict[str, Tensor]) -> Tensor:
        """
        注意力融合
        
        Args:
            features: 各模态特征字典
            
        Returns:
            融合后的特征
        """
        # 这里实现简单的注意力融合
        # 实际项目中可能需要更复杂的注意力机制
        feature_list = list(features.values())
        
        # 计算注意力权重
        weights = []
        for feat in feature_list:
            # 使用全局平均池化后的值作为注意力权重
            weight = feat.mean(dim=[1, 2, 3] if feat.dim() == 4 else 1)
            weights.append(weight.unsqueeze(1))
        
        # 归一化权重
        weights = torch.softmax(torch.cat(weights, dim=1), dim=1)
        
        # 加权融合
        fused = sum(w * f for w, f in zip(
            weights.unbind(dim=1),
            feature_list
        ))
        
        return fused


__all__ = ['BaseReIDModel', 'MultiModalMixin']