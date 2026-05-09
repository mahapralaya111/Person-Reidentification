"""
类型定义模块

提供项目中的类型别名和类型定义，确保类型安全
"""

from typing import (
    TypeVar, Any, Dict, List, Tuple, Optional, Union, Callable,
    Sequence, Iterable, Iterator, Set, NamedTuple, NewType
)
from pathlib import Path
import numpy as np
import torch
from PIL import Image


# 基础类型别名
PathLike = Union[str, Path]
ImageType = Union[Image.Image, np.ndarray, torch.Tensor]
DeviceType = Union[torch.device, str]
DtypeType = Union[torch.dtype, np.dtype, type]

# 张量类型
Tensor = torch.Tensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
BoolTensor = torch.BoolTensor

# NumPy类型
NDArray = np.ndarray
FloatArray = np.ndarray  # 通常为float32或float64
IntArray = np.ndarray    # 通常为int32或int64

# 模型相关类型
ModelStateDict = Dict[str, Tensor]
OptimizerStateDict = Dict[str, Any]
CheckpointDict = Dict[str, Any]

# 配置相关类型
ConfigDict = Dict[str, Any]
HyperparamDict = Dict[str, Union[int, float, str, bool, List, Dict]]

# 数据相关类型
ImageShape = Tuple[int, int, int]  # (channels, height, width)
Bbox = Tuple[int, int, int, int]   # (x1, y1, x2, y2)
Keypoint = Tuple[float, float]     # (x, y)
Landmarks = List[Keypoint]

# 特征相关类型
FeatureVector = NDArray  # 一维特征向量
FeatureMatrix = NDArray  # 二维特征矩阵
SimilarityMatrix = NDArray  # 相似度矩阵

# 搜索结果类型
class SearchResult(NamedTuple):
    """搜索结果"""
    person_id: str
    similarity: float
    image_path: Optional[PathLike]
    confidence: float
    metadata: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'person_id': self.person_id,
            'similarity': float(self.similarity),
            'image_path': str(self.image_path) if self.image_path else None,
            'confidence': float(self.confidence),
            'metadata': self.metadata
        }


# 异常类型
class ReIDError(Exception):
    """基础异常类"""
    pass


class ModelError(ReIDError):
    """模型相关异常"""
    pass


class DataError(ReIDError):
    """数据相关异常"""
    pass


class ConfigError(ReIDError):
    """配置相关异常"""
    pass


class InferenceError(ReIDError):
    """推理相关异常"""
    pass


# 回调类型
ProgressCallback = Callable[[int, int, Dict[str, Any]], None]
ValidationCallback = Callable[[Dict[str, float]], bool]
CheckpointCallback = Callable[[Dict[str, Any], str], None]

# 泛型类型变量
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

# 设备工具函数
def get_device(device: Optional[DeviceType] = None) -> torch.device:
    """
    获取设备
    
    Args:
        device: 设备字符串或torch.device对象
        
    Returns:
        torch.device对象
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if isinstance(device, str):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise ModelError("CUDA不可用，请使用CPU")
        
        return torch.device(device)
    
    return device


def to_device(
    data: Union[Tensor, Dict[str, Tensor], List[Tensor], Any],
    device: DeviceType
) -> Any:
    """
    将数据移动到指定设备
    
    Args:
        data: 要移动的数据
        device: 目标设备
        
    Returns:
        移动到设备后的数据
    """
    target_device = get_device(device)
    
    if isinstance(data, Tensor):
        return data.to(target_device)
    elif isinstance(data, dict):
        return {k: to_device(v, target_device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, target_device) for item in data)
    else:
        return data


# 类型检查函数
def is_image_type(obj: Any) -> bool:
    """检查对象是否为图像类型"""
    return isinstance(obj, (Image.Image, np.ndarray, torch.Tensor))


def is_tensor(obj: Any) -> bool:
    """检查对象是否为张量"""
    return isinstance(obj, Tensor)


def is_ndarray(obj: Any) -> bool:
    """检查对象是否为NumPy数组"""
    return isinstance(obj, np.ndarray)


def is_path_like(obj: Any) -> bool:
    """检查对象是否为路径类型"""
    return isinstance(obj, (str, Path))


def ensure_path(path: PathLike) -> Path:
    """确保返回Path对象"""
    if isinstance(path, str):
        return Path(path)
    return path


def ensure_tensor(data: Any, device: Optional[DeviceType] = None) -> Tensor:
    """
    确保返回张量
    
    Args:
        data: 输入数据
        device: 目标设备
        
    Returns:
        张量
    """
    if is_tensor(data):
        tensor = data
    elif is_ndarray(data):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.tensor(data)
    
    if device is not None:
        tensor = tensor.to(get_device(device))
    
    return tensor


def ensure_ndarray(data: Any, dtype: Optional[DtypeType] = None) -> NDArray:
    """
    确保返回NumPy数组
    
    Args:
        data: 输入数据
        dtype: 目标数据类型
        
    Returns:
        NumPy数组
    """
    if is_ndarray(data):
        arr = data
    elif is_tensor(data):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.array(data)
    
    if dtype is not None:
        arr = arr.astype(dtype)
    
    return arr


# 类型转换装饰器
def typed_argument(func):
    """类型参数装饰器"""
    import inspect
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # 这里可以添加类型检查和转换逻辑
        # 暂时只返回原函数
        return func(*args, **kwargs)
    
    return wrapper


__all__ = [
    # 类型别名
    'PathLike', 'ImageType', 'DeviceType', 'DtypeType',
    'Tensor', 'FloatTensor', 'LongTensor', 'BoolTensor',
    'NDArray', 'FloatArray', 'IntArray',
    'ModelStateDict', 'OptimizerStateDict', 'CheckpointDict',
    'ConfigDict', 'HyperparamDict',
    'ImageShape', 'Bbox', 'Keypoint', 'Landmarks',
    'FeatureVector', 'FeatureMatrix', 'SimilarityMatrix',
    
    # 命名元组
    'SearchResult',
    
    # 异常类
    'ReIDError', 'ModelError', 'DataError', 'ConfigError', 'InferenceError',
    
    # 回调类型
    'ProgressCallback', 'ValidationCallback', 'CheckpointCallback',
    
    # 泛型变量
    'T', 'U', 'V',
    
    # 工具函数
    'get_device', 'to_device',
    'is_image_type', 'is_tensor', 'is_ndarray', 'is_path_like',
    'ensure_path', 'ensure_tensor', 'ensure_ndarray',
    
    # 装饰器
    'typed_argument'
]