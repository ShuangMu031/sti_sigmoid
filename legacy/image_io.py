# -*- coding: utf-8 -*-
"""
图像输入输出处理模块
负责图像的加载、保存、格式转换等功能
"""

import os
import sys
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Union, Optional

# 设置日志
logger = logging.getLogger(__name__)


class ImageIOHandler:
    """
    图像输入输出处理器类
    封装图像的加载、保存和格式转换功能
    """
    
    # 支持的图像格式
    SUPPORTED_FORMATS = {
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
        '.webp', '.jp2', '.pbm', '.pgm', '.ppm'
    }
    
    def __init__(self, default_output_dir: str = 'output'):
        """
        初始化图像IO处理器
        
        Args:
            default_output_dir: 默认输出目录
        """
        self.logger = logger
        self.default_output_dir = default_output_dir
        
        # 确保默认输出目录存在
        self._ensure_directory(default_output_dir)
        
        self.logger.info(f"图像IO处理器已初始化，默认输出目录: {default_output_dir}")
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        加载单张图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            numpy.ndarray: 加载的图像数据，失败返回None
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                self.logger.error(f"图像文件不存在: {image_path}")
                return None
            
            # 检查文件格式
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.SUPPORTED_FORMATS:
                self.logger.warning(f"不推荐的图像格式: {ext}，尝试加载...")
            
            # 加载图像
            image = cv2.imread(image_path)
            
            if image is None:
                self.logger.error(f"无法加载图像: {image_path}")
                return None
            
            self.logger.info(f"成功加载图像: {image_path}，形状: {image.shape}")
            return image
            
        except Exception as e:
            self.logger.error(f"加载图像时出错: {str(e)}")
            return None
    
    def load_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        批量加载图像
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            List[numpy.ndarray]: 成功加载的图像列表
        """
        loaded_images = []
        failed_count = 0
        
        for path in image_paths:
            image = self.load_image(path)
            if image is not None:
                loaded_images.append(image)
            else:
                failed_count += 1
        
        self.logger.info(f"批量加载完成: 成功 {len(loaded_images)} 张，失败 {failed_count} 张")
        return loaded_images
    
    def load_images_from_directory(self, directory: str, 
                                 recursive: bool = False, 
                                 sort_by: str = 'name') -> List[Tuple[str, np.ndarray]]:
        """
        从目录加载图像
        
        Args:
            directory: 图像目录路径
            recursive: 是否递归子目录
            sort_by: 排序方式 ('name', 'date', 'size')
            
        Returns:
            List[Tuple[str, numpy.ndarray]]: (文件路径, 图像数据) 元组列表
        """
        try:
            # 检查目录是否存在
            if not os.path.isdir(directory):
                self.logger.error(f"目录不存在: {directory}")
                return []
            
            # 收集图像文件
            image_files = []
            
            if recursive:
                for root, _, files in os.walk(directory):
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in self.SUPPORTED_FORMATS:
                            full_path = os.path.join(root, file)
                            image_files.append(full_path)
            else:
                for file in os.listdir(directory):
                    full_path = os.path.join(directory, file)
                    if os.path.isfile(full_path):
                        ext = os.path.splitext(file)[1].lower()
                        if ext in self.SUPPORTED_FORMATS:
                            image_files.append(full_path)
            
            # 排序
            if sort_by == 'date':
                image_files.sort(key=lambda x: os.path.getmtime(x))
            elif sort_by == 'size':
                image_files.sort(key=lambda x: os.path.getsize(x))
            else:  # 'name'
                image_files.sort()
            
            # 加载图像
            result = []
            for file_path in image_files:
                image = self.load_image(file_path)
                if image is not None:
                    result.append((file_path, image))
            
            self.logger.info(f"从目录加载完成: {directory}，成功加载 {len(result)} 张图像")
            return result
            
        except Exception as e:
            self.logger.error(f"从目录加载图像时出错: {str(e)}")
            return []
    
    def save_image(self, image: np.ndarray, 
                  output_path: str, 
                  quality: int = 95, 
                  create_dir: bool = True) -> bool:
        """
        保存图像
        
        Args:
            image: 要保存的图像数据
            output_path: 输出文件路径
            quality: 图像质量 (0-100)，仅适用于JPEG等压缩格式
            create_dir: 是否自动创建输出目录
            
        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 验证图像数据
            if not isinstance(image, np.ndarray):
                self.logger.error("无效的图像数据类型")
                return False
            
            if image.ndim not in [2, 3]:
                self.logger.error(f"无效的图像维度: {image.ndim}")
                return False
            
            # 确保输出目录存在
            if create_dir:
                self._ensure_directory(os.path.dirname(output_path))
            
            # 获取文件扩展名
            ext = os.path.splitext(output_path)[1].lower()
            
            # 根据格式设置保存参数
            if ext in ['.jpg', '.jpeg']:
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif ext == '.png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, max(0, 10 - quality // 10))]
            elif ext == '.webp':
                params = [cv2.IMWRITE_WEBP_QUALITY, quality]
            else:
                params = []
            
            # 保存图像
            success = cv2.imwrite(output_path, image, params)
            
            if success:
                self.logger.info(f"图像已保存至: {output_path}，大小: {image.shape}")
            else:
                self.logger.error(f"保存图像失败: {output_path}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"保存图像时出错: {str(e)}")
            return False
    
    def save_image_batch(self, images: List[np.ndarray], 
                        base_path: str, 
                        prefix: str = 'image_', 
                        format: str = 'png', 
                        start_idx: int = 0) -> int:
        """
        批量保存图像
        
        Args:
            images: 图像列表
            base_path: 基础输出目录
            prefix: 文件名前缀
            format: 图像格式
            start_idx: 起始索引
            
        Returns:
            int: 成功保存的图像数量
        """
        # 确保基础目录存在
        self._ensure_directory(base_path)
        
        success_count = 0
        
        for i, image in enumerate(images):
            idx = start_idx + i
            filename = f"{prefix}{idx:04d}.{format}"
            output_path = os.path.join(base_path, filename)
            
            if self.save_image(image, output_path, create_dir=False):
                success_count += 1
        
        self.logger.info(f"批量保存完成: 成功 {success_count}/{len(images)} 张")
        return success_count
    
    def convert_to_standard_format(self, image: np.ndarray) -> np.ndarray:
        """
        将图像转换为标准格式 (BGR, 8-bit)
        
        Args:
            image: 输入图像
            
        Returns:
            numpy.ndarray: 标准化后的图像
        """
        try:
            # 确保图像是numpy数组
            if not isinstance(image, np.ndarray):
                raise TypeError("图像必须是numpy数组")
            
            # 灰度图转BGR
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # RGBA转BGR
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
            # RGB转BGR (如果需要)
            elif image.shape[2] == 3:
                # 假设OpenCV加载的已经是BGR，如果不是则转换
                # 这里可以根据实际情况调整
                return image
            
            # 其他情况，转换为8位
            if image.dtype != np.uint8:
                # 归一化到0-255
                min_val = image.min()
                max_val = image.max()
                if max_val > min_val:
                    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    return np.zeros_like(image, dtype=np.uint8)
            
            return image
            
        except Exception as e:
            self.logger.error(f"转换图像格式时出错: {str(e)}")
            return image  # 返回原始图像
    
    def resize_image(self, image: np.ndarray, 
                    target_width: int = None, 
                    target_height: int = None, 
                    keep_ratio: bool = True,
                    interp: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image: 输入图像
            target_width: 目标宽度
            target_height: 目标高度
            keep_ratio: 是否保持宽高比
            interp: 插值方法
            
        Returns:
            numpy.ndarray: 调整大小后的图像
        """
        try:
            height, width = image.shape[:2]
            
            if keep_ratio:
                # 计算新尺寸，保持宽高比
                if target_width is not None and target_height is not None:
                    ratio = min(target_width / width, target_height / height)
                elif target_width is not None:
                    ratio = target_width / width
                elif target_height is not None:
                    ratio = target_height / height
                else:
                    return image
                
                new_width = int(width * ratio)
                new_height = int(height * ratio)
            else:
                # 直接使用目标尺寸
                new_width = target_width if target_width is not None else width
                new_height = target_height if target_height is not None else height
            
            # 调整大小
            resized = cv2.resize(image, (new_width, new_height), interpolation=interp)
            
            self.logger.info(f"图像已调整大小: {width}x{height} → {new_width}x{new_height}")
            return resized
            
        except Exception as e:
            self.logger.error(f"调整图像大小时出错: {str(e)}")
            return image
    
    def get_image_info(self, image: np.ndarray) -> dict:
        """
        获取图像信息
        
        Args:
            image: 输入图像
            
        Returns:
            dict: 图像信息字典
        """
        try:
            info = {
                'shape': image.shape,
                'dtype': str(image.dtype),
                'width': image.shape[1] if image.ndim >= 2 else 0,
                'height': image.shape[0] if image.ndim >= 1 else 0,
                'channels': image.shape[2] if image.ndim >= 3 else 1,
                'size_bytes': image.nbytes,
                'min_value': float(image.min()),
                'max_value': float(image.max()),
                'mean_value': float(image.mean())
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"获取图像信息时出错: {str(e)}")
            return {}
    
    def _ensure_directory(self, directory: str) -> None:
        """
        确保目录存在
        
        Args:
            directory: 目录路径
        """
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                self.logger.info(f"创建目录: {directory}")
            except Exception as e:
                self.logger.error(f"创建目录失败: {str(e)}")


# 便捷函数
def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    便捷函数：加载图像
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        numpy.ndarray: 加载的图像，失败返回None
    """
    handler = ImageIOHandler()
    return handler.load_image(image_path)


def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> bool:
    """
    便捷函数：保存图像
    
    Args:
        image: 要保存的图像
        output_path: 输出文件路径
        quality: 图像质量
        
    Returns:
        bool: 成功返回True
    """
    handler = ImageIOHandler()
    return handler.save_image(image, output_path, quality)


def batch_process_images(input_dir: str, 
                         output_dir: str,
                         process_func: callable = None,
                         recursive: bool = False) -> List[str]:
    """
    批量处理图像
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        process_func: 处理函数，如果为None则仅复制图像
        recursive: 是否递归子目录
        
    Returns:
        List[str]: 成功处理的文件路径列表
    """
    handler = ImageIOHandler(output_dir)
    
    # 加载图像
    images_with_paths = handler.load_images_from_directory(input_dir, recursive)
    
    # 确保输出目录存在
    handler._ensure_directory(output_dir)
    
    success_paths = []
    
    for orig_path, image in images_with_paths:
        # 生成输出路径
        relative_path = os.path.relpath(orig_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        
        # 确保子目录存在
        handler._ensure_directory(os.path.dirname(output_path))
        
        # 处理图像
        if process_func is not None:
            try:
                processed_image = process_func(image)
                success = handler.save_image(processed_image, output_path)
            except Exception as e:
                handler.logger.error(f"处理图像失败: {orig_path}, 错误: {str(e)}")
                success = False
        else:
            # 直接保存原始图像
            success = handler.save_image(image, output_path)
        
        if success:
            success_paths.append(orig_path)
    
    handler.logger.info(f"批量处理完成: 成功 {len(success_paths)}/{len(images_with_paths)} 张")
    return success_paths
