"""
核心模块

目前包含模型抽象基类与模型定义，数据/损失/引擎等子模块
随训练脚本（src/train.py）一并实现。
"""

from . import models

__all__ = ['models']
