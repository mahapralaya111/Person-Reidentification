"""
配置管理模块

提供统一的配置管理功能，支持：
1. 配置文件加载和解析
2. 环境变量覆盖
3. 配置验证和默认值
4. 配置热更新

使用示例:
    from src.config import get_config
    config = get_config('training')
    print(config.learning_rate)
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from .default import TrainingConfig, ModelConfig, DataConfig, InferenceConfig


class ConfigError(Exception):
    """配置相关异常"""
    pass


@dataclass
class AppConfig:
    """应用配置"""
    name: str = "person-reid-system"
    version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"
    log_dir: str = "./logs"
    cache_dir: str = "./cache"
    checkpoint_dir: str = "./checkpoints"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """从字典创建"""
        return cls(**data)


class ConfigManager:
    """配置管理器"""
    
    _instance = None
    _configs: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._configs = {}
            self._load_default_configs()
            self._initialized = True
    
    def _load_default_configs(self):
        """加载默认配置"""
        # 应用配置
        self._configs['app'] = AppConfig()
        
        # 训练配置
        self._configs['training'] = TrainingConfig()
        
        # 模型配置
        self._configs['model'] = ModelConfig()
        
        # 数据配置
        self._configs['data'] = DataConfig()
        
        # 推理配置
        self._configs['inference'] = InferenceConfig()
    
    def load_config_file(self, config_type: str, file_path: Union[str, Path]) -> None:
        """
        从配置文件加载配置
        
        Args:
            config_type: 配置类型，如 'training', 'model' 等
            file_path: 配置文件路径
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ConfigError(f"配置文件不存在: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif suffix == '.json':
                    data = json.load(f)
                else:
                    raise ConfigError(f"不支持的配置文件格式: {suffix}")
            
            # 更新配置
            self.update_config(config_type, data)
            
        except Exception as e:
            raise ConfigError(f"加载配置文件失败: {e}")
    
    def update_config(self, config_type: str, updates: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            config_type: 配置类型
            updates: 更新数据
        """
        if config_type not in self._configs:
            raise ConfigError(f"未知的配置类型: {config_type}")
        
        config = self._configs[config_type]
        
        # 如果是dataclass，使用字段更新
        if hasattr(config, '__dataclass_fields__'):
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    print(f"[警告] 配置 {config_type} 没有字段: {key}")
        else:
            # 普通字典更新
            config.update(updates)
    
    def get_config(self, config_type: str) -> Any:
        """
        获取配置
        
        Args:
            config_type: 配置类型
            
        Returns:
            配置对象
        """
        if config_type not in self._configs:
            raise ConfigError(f"未知的配置类型: {config_type}")
        
        return self._configs[config_type]
    
    def save_config(self, config_type: str, file_path: Union[str, Path], format: str = 'yaml') -> None:
        """
        保存配置到文件
        
        Args:
            config_type: 配置类型
            file_path: 文件路径
            format: 文件格式，支持 'yaml', 'json'
        """
        config = self.get_config(config_type)
        
        if hasattr(config, 'to_dict'):
            data = config.to_dict()
        elif hasattr(config, '__dict__'):
            data = config.__dict__
        else:
            data = config
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if format == 'yaml':
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                elif format == 'json':
                    json.dump(data, f, ensure_ascii=False, indent=2)
                else:
                    raise ConfigError(f"不支持的格式: {format}")
        except Exception as e:
            raise ConfigError(f"保存配置失败: {e}")
    
    def load_env_variables(self) -> None:
        """加载环境变量覆盖配置"""
        # 应用配置
        app_config = self._configs['app']
        if os.getenv('REID_DEBUG'):
            app_config.debug = os.getenv('REID_DEBUG').lower() == 'true'
        if os.getenv('REID_LOG_LEVEL'):
            app_config.log_level = os.getenv('REID_LOG_LEVEL')
        
        # 训练配置
        training_config = self._configs['training']
        if os.getenv('REID_BATCH_SIZE'):
            training_config.batch_size = int(os.getenv('REID_BATCH_SIZE'))
        if os.getenv('REID_LEARNING_RATE'):
            training_config.learning_rate = float(os.getenv('REID_LEARNING_RATE'))
        
        # 模型配置
        model_config = self._configs['model']
        if os.getenv('REID_MODEL_BACKBONE'):
            model_config.backbone = os.getenv('REID_MODEL_BACKBONE')
        if os.getenv('REID_MODEL_FEAT_DIM'):
            model_config.feat_dim = int(os.getenv('REID_MODEL_FEAT_DIM'))


# 全局配置管理器实例
_config_manager = ConfigManager()


def get_config(config_type: str) -> Any:
    """
    获取配置的快捷函数
    
    Args:
        config_type: 配置类型
        
    Returns:
        配置对象
    """
    return _config_manager.get_config(config_type)


def load_config_file(config_type: str, file_path: Union[str, Path]) -> None:
    """
    加载配置文件的快捷函数
    
    Args:
        config_type: 配置类型
        file_path: 配置文件路径
    """
    _config_manager.load_config_file(config_type, file_path)


def update_config(config_type: str, updates: Dict[str, Any]) -> None:
    """
    更新配置的快捷函数
    
    Args:
        config_type: 配置类型
        updates: 更新数据
    """
    _config_manager.update_config(config_type, updates)


# 初始化时加载环境变量
_config_manager.load_env_variables()


# 导出常用配置
__all__ = [
    'ConfigError',
    'ConfigManager',
    'get_config',
    'load_config_file',
    'update_config',
    'AppConfig',
    'TrainingConfig',
    'ModelConfig',
    'DataConfig',
    'InferenceConfig'
]