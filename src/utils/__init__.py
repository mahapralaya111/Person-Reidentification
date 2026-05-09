"""
工具函数模块

提供日志系统与通用类型定义。
"""

from .logger import get_logger, setup_logging
from .types import *

__all__ = [
    'get_logger',
    'setup_logging',
]
