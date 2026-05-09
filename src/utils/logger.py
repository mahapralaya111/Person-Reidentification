"""
日志系统模块

提供统一的日志记录功能，支持：
1. 多级别日志记录
2. 文件和控制台输出
3. 日志轮转
4. 结构化日志
5. 性能监控

使用示例:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("处理完成")
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import time


class CustomFormatter(logging.Formatter):
    """自定义日志格式化器"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[41m',  # 红底白字
        'RESET': '\033[0m'       # 重置
    }
    
    def __init__(self, use_color: bool = True, json_format: bool = False):
        """
        初始化格式化器
        
        Args:
            use_color: 是否使用颜色（控制台输出时）
            json_format: 是否输出JSON格式日志
        """
        super().__init__()
        self.use_color = use_color
        self.json_format = json_format
        
        # 标准格式
        self.standard_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # 详细格式（包含文件位置）
        self.detailed_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        
        self.date_format = '%Y-%m-%d %H:%M:%S'
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        
        if self.json_format:
            return self._format_json(record)
        
        # 根据日志级别选择格式
        if record.levelno >= logging.WARNING:
            fmt = self.detailed_format
        else:
            fmt = self.standard_format
        
        # 创建格式化器
        formatter = logging.Formatter(fmt, self.date_format)
        
        # 应用格式化
        result = formatter.format(record)
        
        # 添加颜色
        if self.use_color and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            result = f"{color}{result}{self.COLORS['RESET']}"
        
        return result
    
    def _format_json(self, record: logging.LogRecord) -> str:
        """格式化JSON日志"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'logger': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.threadName,
            'process': record.processName
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceFilter(logging.Filter):
    """性能监控过滤器"""
    
    def __init__(self, name: str = "", threshold_ms: float = 100.0):
        super().__init__(name)
        self.threshold_ms = threshold_ms
        self.timers: Dict[str, float] = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """过滤日志记录，添加性能信息"""
        if hasattr(record, 'duration_ms'):
            if record.duration_ms > self.threshold_ms:
                record.msg = f"[慢操作 {record.duration_ms:.1f}ms] {record.msg}"
        return True
    
    def start_timer(self, key: str) -> None:
        """开始计时"""
        self.timers[key] = time.time()
    
    def stop_timer(self, key: str) -> float:
        """结束计时，返回耗时（毫秒）"""
        if key in self.timers:
            duration_ms = (time.time() - self.timers[key]) * 1000
            del self.timers[key]
            return duration_ms
        return 0.0


def setup_logging(
    name: str = "person_reid",
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    max_file_size_mb: int = 100,
    backup_count: int = 5,
    json_format: bool = False
) -> logging.Logger:
    """
    设置日志系统
    
    Args:
        name: 日志器名称
        log_level: 日志级别
        log_dir: 日志目录
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        max_file_size_mb: 单个日志文件最大大小（MB）
        backup_count: 备份文件数量
        json_format: 是否使用JSON格式
        
    Returns:
        配置好的日志器
    """
    # 确保日志目录存在
    if log_dir is None:
        log_dir = "./logs"
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 创建日志器
    logger = logging.getLogger(name)
    
    # 避免重复设置
    if logger.handlers:
        return logger
    
    # 设置日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # 创建处理器
    handlers = []
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = CustomFormatter(use_color=True, json_format=json_format)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # 文件处理器
    if file_output:
        # 普通日志文件
        log_file = log_path / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_formatter = CustomFormatter(use_color=False, json_format=json_format)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        
        # 错误日志文件（只记录WARNING及以上）
        error_file = log_path / f"{name}.error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.WARNING)
        error_formatter = CustomFormatter(use_color=False, json_format=json_format)
        error_handler.setFormatter(error_formatter)
        handlers.append(error_handler)
    
    # 添加处理器到日志器
    for handler in handlers:
        logger.addHandler(handler)
    
    # 添加性能过滤器
    perf_filter = PerformanceFilter()
    logger.addFilter(perf_filter)
    
    # 避免日志传播到根日志器
    logger.propagate = False
    
    logger.info(f"日志系统初始化完成 - 级别: {log_level}, 目录: {log_path}")
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取日志器（快捷函数）
    
    Args:
        name: 日志器名称，如果为None则使用调用者模块名称
        
    Returns:
        日志器实例
    """
    if name is None:
        # 获取调用者模块名称
        import inspect
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
        else:
            name = 'unknown'
    
    return logging.getLogger(name)


class Timer:
    """计时器上下文管理器"""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
        """
        初始化计时器
        
        Args:
            name: 计时器名称
            logger: 日志器
            level: 日志级别
        """
        self.name = name
        self.logger = logger or get_logger(__name__)
        self.level = level
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, f"开始: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.logger.log(self.level, f"完成: {self.name} - 耗时: {duration:.3f}s")
    
    def get_elapsed(self) -> float:
        """获取已过时间（秒）"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


# 默认日志器
_default_logger = setup_logging()


def log_performance(func):
    """性能日志装饰器"""
    
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        timer = Timer(f"执行 {func.__name__}", logger)
        
        with timer:
            result = func(*args, **kwargs)
        
        return result
    
    return wrapper


__all__ = [
    'setup_logging',
    'get_logger',
    'Timer',
    'log_performance',
    'CustomFormatter',
    'PerformanceFilter'
]