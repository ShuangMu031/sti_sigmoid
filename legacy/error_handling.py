# -*- coding: utf-8 -*-
"""
错误处理和日志记录模块
提供自定义异常类和统一的日志配置
"""

import os
import sys
import logging
import traceback
import datetime
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any


# ============================
# 自定义异常类
# ============================

class ImageStitcherError(Exception):
    """
    图像拼接器基础异常类
    所有图像拼接相关的异常都应该继承此类
    """
    
    def __init__(self, message: str, error_code: int = 1, 
                 details: Optional[Dict[str, Any]] = None):
        """
        初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            details: 详细信息字典
        """
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.datetime.now().isoformat()
        self.stack_trace = traceback.format_exc()
        
        super().__init__(message)
    
    def __str__(self) -> str:
        """
        返回异常的字符串表示
        """
        base_msg = f"[{self.__class__.__name__}] 错误码 {self.error_code}: {super().__str__()}"
        if self.details:
            details_str = ", ".join([f"{k}={v}" for k, v in self.details.items()])
            base_msg += f" ({details_str})"
        return base_msg


class ImageLoadError(ImageStitcherError):
    """
    图像加载错误
    """
    
    def __init__(self, message: str, file_path: str = None):
        details = {'file_path': file_path} if file_path else {}
        super().__init__(message, error_code=101, details=details)


class ImageSaveError(ImageStitcherError):
    """
    图像保存错误
    """
    
    def __init__(self, message: str, file_path: str = None):
        details = {'file_path': file_path} if file_path else {}
        super().__init__(message, error_code=102, details=details)


class ImageProcessError(ImageStitcherError):
    """
    图像处理错误
    """
    
    def __init__(self, message: str, stage: str = None, details: dict = None):
        error_details = details or {}
        if stage:
            error_details['stage'] = stage
        super().__init__(message, error_code=103, details=error_details)


class FeatureDetectionError(ImageProcessError):
    """
    特征检测错误
    """
    
    def __init__(self, message: str, image_index: int = None):
        details = {'image_index': image_index} if image_index is not None else {}
        super().__init__(message, stage='feature_detection', details=details)


class RegistrationError(ImageProcessError):
    """
    图像配准错误
    """
    
    def __init__(self, message: str, image_pair: tuple = None):
        details = {'image_pair': image_pair} if image_pair else {}
        super().__init__(message, stage='registration', details=details)


class SeamFindingError(ImageProcessError):
    """
    接缝线寻找错误
    """
    
    def __init__(self, message: str, no_overlap: bool = False):
        details = {'no_overlap': no_overlap}
        super().__init__(message, stage='seam_finding', details=details)


class BlendingError(ImageProcessError):
    """
    图像融合错误
    """
    
    def __init__(self, message: str, method: str = None):
        details = {'method': method} if method else {}
        super().__init__(message, stage='blending', details=details)


class ConfigurationError(ImageStitcherError):
    """
    配置错误
    """
    
    def __init__(self, message: str, config_key: str = None):
        details = {'config_key': config_key} if config_key else {}
        super().__init__(message, error_code=201, details=details)


# ============================
# 日志配置功能
# ============================

class LogConfig:
    """
    日志配置类
    提供灵活的日志配置功能
    """
    
    # 默认日志格式
    DEFAULT_FORMAT = ('%(asctime)s - %(name)s - %(levelname)s - ' 
                    '[%(filename)s:%(lineno)d] - %(message)s')
    
    # 默认日期格式
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # 日志级别映射
    LEVEL_MAP = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    @classmethod
    def get_default_log_dir(cls) -> str:
        """
        获取默认日志目录
        
        Returns:
            str: 默认日志目录路径
        """
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        return log_dir
    
    @classmethod
    def setup_logger(cls, 
                    name: str = 'image_stitcher',
                    level: str = 'INFO',
                    log_file: Optional[str] = None,
                    log_to_console: bool = True,
                    max_bytes: int = 10*1024*1024,  # 10MB
                    backup_count: int = 5) -> logging.Logger:
        """
        设置日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别
            log_file: 日志文件路径，如果为None则使用默认路径
            log_to_console: 是否输出到控制台
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的备份文件数量
            
        Returns:
            logging.Logger: 配置好的日志记录器
        """
        # 获取日志级别
        numeric_level = cls.LEVEL_MAP.get(level.upper(), logging.INFO)
        
        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(numeric_level)
        
        # 清除已有的处理器
        if logger.handlers:
            logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            fmt=cls.DEFAULT_FORMAT,
            datefmt=cls.DEFAULT_DATE_FORMAT
        )
        
        # 添加文件处理器
        if log_file is None:
            # 使用默认日志目录
            log_dir = cls.get_default_log_dir()
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # 生成日志文件名
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        else:
            # 确保日志文件目录存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
        
        # 创建滚动文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 添加控制台处理器
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        logger.info(f"日志系统已初始化，级别: {level}, 日志文件: {log_file}")
        return logger


# ============================
# 错误处理工具函数
# ============================

def handle_exception(exception: Exception, 
                    logger: Optional[logging.Logger] = None,
                    re_raise: bool = False) -> None:
    """
    统一的异常处理函数
    
    Args:
        exception: 捕获的异常
        logger: 日志记录器，如果为None则使用默认日志记录器
        re_raise: 是否重新抛出异常
    """
    # 获取或创建日志记录器
    if logger is None:
        logger = logging.getLogger('image_stitcher')
    
    # 区分处理不同类型的异常
    if isinstance(exception, ImageStitcherError):
        # 处理自定义异常
        error_info = {
            'error_type': exception.__class__.__name__,
            'error_code': exception.error_code,
            'message': str(exception),
            'timestamp': exception.timestamp,
            'details': exception.details
        }
        
        # 根据错误级别记录日志
        if exception.error_code // 100 == 1:
            # 图像处理错误
            logger.error(f"图像处理错误: {str(exception)}")
            logger.debug(f"详细错误信息: {error_info}")
            logger.debug(f"堆栈跟踪:\n{exception.stack_trace}")
        elif exception.error_code // 100 == 2:
            # 配置错误
            logger.warning(f"配置错误: {str(exception)}")
            logger.debug(f"详细错误信息: {error_info}")
        else:
            # 其他自定义错误
            logger.error(f"自定义错误: {str(exception)}")
            logger.debug(f"详细错误信息: {error_info}")
            logger.debug(f"堆栈跟踪:\n{exception.stack_trace}")
    else:
        # 处理标准异常
        logger.error(f"未预期的错误: {str(exception)}")
        logger.debug(f"堆栈跟踪:\n{traceback.format_exc()}")
    
    # 重新抛出异常
    if re_raise:
        raise


def log_performance(func):
    """
    性能日志装饰器
    记录函数执行时间
    
    Args:
        func: 要装饰的函数
        
    Returns:
        callable: 装饰后的函数
    """
    def wrapper(*args, **kwargs):
        # 获取日志记录器
        logger = logging.getLogger(func.__module__)
        
        # 记录开始时间
        start_time = datetime.datetime.now()
        logger.debug(f"函数 {func.__name__} 开始执行")
        
        try:
            # 执行函数
            result = func(*args, **kwargs)
            
            # 记录结束时间
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 记录性能日志
            if duration > 1.0:  # 只记录执行时间超过1秒的函数
                logger.info(f"函数 {func.__name__} 执行完成，耗时: {duration:.2f}秒")
            else:
                logger.debug(f"函数 {func.__name__} 执行完成，耗时: {duration:.4f}秒")
            
            return result
            
        except Exception as e:
            # 记录异常
            logger.error(f"函数 {func.__name__} 执行出错: {str(e)}")
            logger.debug(f"堆栈跟踪:\n{traceback.format_exc()}")
            raise
    
    # 保留原函数的元数据
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    
    return wrapper


def validate_input(func):
    """
    输入验证装饰器
    用于验证函数输入参数
    
    Args:
        func: 要装饰的函数
        
    Returns:
        callable: 装饰后的函数
    """
    def wrapper(*args, **kwargs):
        # 获取日志记录器
        logger = logging.getLogger(func.__module__)
        logger.debug(f"验证函数 {func.__name__} 的输入参数")
        
        # 这里可以添加更复杂的参数验证逻辑
        # 目前仅作为一个框架
        
        return func(*args, **kwargs)
    
    # 保留原函数的元数据
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    
    return wrapper


# ============================
# 初始化默认日志记录器
# ============================

# 创建默认日志记录器
default_logger = LogConfig.setup_logger()

# 导出默认日志记录器
logger = default_logger


# ============================
# 便捷函数
# ============================

def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger: 日志记录器
    """
    return logging.getLogger(name)

def setup_basic_logging(level: str = 'INFO') -> None:
    """
    设置基本日志配置
    
    Args:
        level: 日志级别
    """
    LogConfig.setup_logger(name='image_stitcher', level=level)
