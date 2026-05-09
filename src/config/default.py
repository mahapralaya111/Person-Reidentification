"""
默认配置定义

定义所有配置的数据类，提供类型安全和默认值
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum


class BackboneType(str, Enum):
    """骨干网络类型"""
    RESNET50 = "resnet50"
    RESNET34 = "resnet34"
    OSNET = "osnet"
    AGW = "agw"
    RESNET_IBN = "resnet_ibn"
    PCB = "pcb"


class LossType(str, Enum):
    """损失函数类型"""
    CROSS_ENTROPY = "cross_entropy"
    TRIPLET = "triplet"
    CIRCLE = "circle"
    COMBINED = "combined"


class ModalityType(str, Enum):
    """模态类型"""
    VISIBLE = "visible"
    INFRARED = "infrared"
    BOTH = "both"


@dataclass
class ModelConfig:
    """模型配置"""
    
    # 基础配置
    backbone: BackboneType = BackboneType.OSNET
    pretrained: bool = True
    feat_dim: int = 512
    dropout_rate: float = 0.5
    
    # 多模态配置
    modality: ModalityType = ModalityType.BOTH
    use_feature_fusion: bool = True
    fusion_method: str = "concat"  # concat, add, attention
    
    # 注意力机制
    use_attention: bool = True
    attention_type: str = "non_local"  # non_local, cbam, se
    
    # 部分特征
    use_part_features: bool = False
    num_parts: int = 6
    
    def __post_init__(self):
        """配置验证"""
        if self.feat_dim <= 0:
            raise ValueError(f"特征维度必须为正数: {self.feat_dim}")
        
        if self.dropout_rate < 0 or self.dropout_rate > 1:
            raise ValueError(f"Dropout率必须在0-1之间: {self.dropout_rate}")


@dataclass
class DataConfig:
    """数据配置"""
    
    # 数据集配置
    dataset_name: str = "Market1501"
    data_root: str = "./data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # 图像处理
    image_size: Tuple[int, int] = (256, 128)
    crop_size: Tuple[int, int] = (224, 112)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # 数据增强
    use_random_horizontal_flip: bool = True
    use_random_crop: bool = True
    use_color_jitter: bool = True
    use_random_erasing: bool = True
    random_erasing_prob: float = 0.5
    
    # 恶劣环境增强
    use_harsh_augmentation: bool = True
    harsh_augmentation_types: List[str] = field(default_factory=lambda: [
        "fog", "rain", "snow", "motion_blur", "low_light", "noise"
    ])
    harsh_augmentation_prob: float = 0.3
    
    # 批量配置
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    def __post_init__(self):
        """配置验证"""
        if not 0 <= self.train_split <= 1:
            raise ValueError(f"训练集比例必须在0-1之间: {self.train_split}")
        if not 0 <= self.val_split <= 1:
            raise ValueError(f"验证集比例必须在0-1之间: {self.val_split}")
        if not 0 <= self.test_split <= 1:
            raise ValueError(f"测试集比例必须在0-1之间: {self.test_split}")
        
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"数据集比例之和必须为1: {total}")
        
        if self.batch_size <= 0:
            raise ValueError(f"批次大小必须为正数: {self.batch_size}")


@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 基础配置
    epochs: int = 100
    learning_rate: float = 0.0001
    weight_decay: float = 5e-4
    momentum: float = 0.9
    
    # 损失函数
    loss_type: LossType = LossType.CIRCLE
    triplet_margin: float = 0.3
    circle_margin: float = 0.25
    circle_scale: float = 32
    
    # 学习率调度
    use_warmup: bool = True
    warmup_epochs: int = 5
    use_cosine_annealing: bool = True
    min_learning_rate: float = 1e-6
    
    # 优化器
    optimizer: str = "adam"  # adam, sgd, adamw
    use_gradient_clip: bool = True
    gradient_clip_norm: float = 1.0
    
    # 训练策略
    use_amp: bool = True  # 自动混合精度
    use_sync_bn: bool = False  # 同步批归一化，多GPU时使用
    save_best_only: bool = True
    save_every_n_epochs: int = 10
    
    # 早停
    use_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    
    # 日志和监控
    log_interval: int = 20
    eval_interval: int = 5
    tensorboard_log_dir: str = "./logs/tensorboard"
    
    def __post_init__(self):
        """配置验证"""
        if self.epochs <= 0:
            raise ValueError(f"训练轮数必须为正数: {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"学习率必须为正数: {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"权重衰减不能为负数: {self.weight_decay}")


@dataclass
class InferenceConfig:
    """推理配置"""
    
    # 模型加载
    model_path: str = "./checkpoints/best_model.pth"
    device: str = "auto"  # auto, cpu, cuda
    
    # 特征提取
    batch_size: int = 32
    normalize_features: bool = True
    use_fp16: bool = False  # 半精度推理
    
    # 搜索配置
    search_top_k: int = 10
    similarity_threshold: float = 0.2
    use_reranking: bool = True
    reranking_k1: int = 20
    reranking_k2: int = 6
    reranking_lambda: float = 0.3
    
    # 缓存配置
    use_feature_cache: bool = True
    cache_dir: str = "./cache/features"
    cache_expire_days: int = 30
    
    # 性能配置
    max_image_size: Tuple[int, int] = (1024, 1024)
    max_batch_memory_mb: int = 512
    
    def __post_init__(self):
        """配置验证"""
        if self.search_top_k <= 0:
            raise ValueError(f"搜索top-k必须为正数: {self.search_top_k}")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError(f"相似度阈值必须在0-1之间: {self.similarity_threshold}")
        if self.max_batch_memory_mb <= 0:
            raise ValueError(f"最大批次内存必须为正数: {self.max_batch_memory_mb}")


@dataclass
class WebUIConfig:
    """Web UI配置"""
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8501
    debug: bool = False
    
    # 界面配置
    theme: str = "light"
    page_title: str = "行人重识别系统"
    page_icon: str = "👤"
    
    # 功能配置
    enable_file_upload: bool = True
    enable_camera_input: bool = False
    enable_batch_processing: bool = True
    max_upload_size_mb: int = 100
    
    # 显示配置
    show_confidence_scores: bool = True
    show_feature_visualization: bool = True
    show_performance_stats: bool = True
    
    def __post_init__(self):
        """配置验证"""
        if not 1 <= self.port <= 65535:
            raise ValueError(f"端口号必须在1-65535之间: {self.port}")
        if self.max_upload_size_mb <= 0:
            raise ValueError(f"最大上传大小必须为正数: {self.max_upload_size_mb}")


@dataclass
class APIConfig:
    """API配置"""
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    
    # API配置
    api_prefix: str = "/api/v1"
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # 认证配置
    enable_auth: bool = False
    auth_token: Optional[str] = None
    rate_limit: int = 100  # 每分钟请求数
    
    # 文档配置
    enable_swagger: bool = True
    swagger_url: str = "/docs"
    
    def __post_init__(self):
        """配置验证"""
        if not 1 <= self.port <= 65535:
            raise ValueError(f"端口号必须在1-65535之间: {self.port}")
        if self.rate_limit <= 0:
            raise ValueError(f"速率限制必须为正数: {self.rate_limit}")