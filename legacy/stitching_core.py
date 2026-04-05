# -*- coding: utf-8 -*-
"""
图像拼接核心模块
封装所有核心拼接功能，提供统一的接口给应用程序调用
"""

import os
import sys
import time
import logging
import numpy as np
import cv2
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入所需的模块
try:
    import config
    from saliency import saliency
    from edge_detect import canny_edge_detect
    from register_texture import texture_registration
    from homography_align import homography_warp
    from graph_cut import graph_cut_seam
    from poisson_refine import local_poisson_blend
    from pyramid_blend import laplacian_pyramid_blend, ownership_based_blending
    from overlap import find_overlap_region
    from seam_driven_ownership import seam_driven_ownership_allocation
    from utils import (ensure_directory, get_stitched_size, blend_gradient, 
                      draw_seam_line, load_images)
except ImportError as e:
    logging.error(f"导入模块时出错: {str(e)}")
    raise

# 设置日志
logger = logging.getLogger(__name__)


class ImageStitcher:
    """
    图像拼接器类
    封装完整的图像拼接流程
    """
    
    def __init__(self, config_params=None):
        """
        初始化图像拼接器
        
        Args:
            config_params: 配置参数字典，如果为None则使用默认配置
        """
        self.logger = logger
        self.logger.info("初始化图像拼接器...")
        
        # 加载配置
        self.config = vars(config) if config_params is None else config_params
        
        # 存储中间结果
        self.images = None
        self.warped_images = None
        self.masks = None
        self.seam_line = None
        self.ownership = None
        self.result = None
        
        # 记录处理时间
        self.timings = {}
    
    def load_and_preprocess_images(self, image_paths):
        """
        加载并预处理图像
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            self.logger.info(f"加载并预处理图像: {len(image_paths)}张")
            start_time = time.time()
            
            # 加载图像
            self.images = load_images(image_paths)
            
            # 确保图像格式正确
            for i in range(len(self.images)):
                if len(self.images[i].shape) == 2:
                    self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_GRAY2BGR)
                elif self.images[i].shape[2] == 4:
                    self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_RGBA2BGR)
            
            self.timings['load_images'] = time.time() - start_time
            self.logger.info(f"图像加载完成，耗时: {self.timings['load_images']:.2f}秒")
            return True
            
        except Exception as e:
            self.logger.error(f"加载图像时出错: {str(e)}")
            return False
    
    def detect_features_and_align(self):
        """
        检测特征并对齐图像
        
        Returns:
            bool: 成功返回True，失败返回False
        """
        if self.images is None:
            self.logger.error("未加载图像，无法进行特征检测和对齐")
            return False
        
        try:
            self.logger.info("开始特征检测和图像对齐...")
            start_time = time.time()
            
            # 确保有足够的图像
            if len(self.images) < 2:
                self.logger.error("至少需要2张图像进行拼接")
                return False
            
            # 使用第一张图像作为参考
            ref_image = self.images[0]
            warped_images = [ref_image]
            masks = [np.ones(ref_image.shape[:2], dtype=np.uint8) * 255]
            
            # 对每张后续图像进行配准和变换
            for i in range(1, len(self.images)):
                self.logger.info(f"配准图像 {i+1}/{len(self.images)}")
                
                # 纹理配准
                H, matches = texture_registration(ref_image, self.images[i])
                
                # 使用单应性变换对齐图像
                warped_img, mask = homography_warp(
                    self.images[i], 
                    H, 
                    ref_shape=ref_image.shape
                )
                
                # 合并结果
                warped_images.append(warped_img)
                masks.append(mask)
                
                # 更新参考图像为当前的拼接结果
                # 这里简化处理，实际应用中可能需要更复杂的合并策略
                ref_image = self._merge_images(warped_images[-2:], masks[-2:])
            
            # 计算最终的拼接尺寸
            self.logger.info("计算最终拼接尺寸...")
            final_shape = get_stitched_size(warped_images, masks)
            
            # 调整所有图像到最终尺寸
            self.logger.info(f"调整图像到最终尺寸: {final_shape[1]}x{final_shape[0]}")
            for i in range(len(warped_images)):
                warped_images[i], masks[i] = self._resize_to_final_shape(
                    warped_images[i], masks[i], final_shape
                )
            
            self.warped_images = warped_images
            self.masks = masks
            
            self.timings['alignment'] = time.time() - start_time
            self.logger.info(f"图像对齐完成，耗时: {self.timings['alignment']:.2f}秒")
            return True
            
        except Exception as e:
            self.logger.error(f"特征检测和对齐时出错: {str(e)}")
            return False
    
    def find_optimal_seam(self):
        """
        寻找最优接缝线
        
        Returns:
            bool: 成功返回True，失败返回False
        """
        if self.warped_images is None or self.masks is None:
            self.logger.error("未对齐图像，无法寻找接缝线")
            return False
        
        try:
            self.logger.info("开始寻找最优接缝线...")
            start_time = time.time()
            
            # 找到重叠区域
            overlap_mask = self._find_overlap_mask()
            
            if overlap_mask is None or not np.any(overlap_mask):
                self.logger.warning("未找到图像重叠区域，使用默认接缝")
                # 创建默认接缝线
                height, width = self.warped_images[0].shape[:2]
                self.seam_line = np.zeros((height, width), dtype=np.uint8)
                self.seam_line[:, width//2] = 255
                return True
            
            # 使用图割算法寻找最优接缝
            self.seam_line = graph_cut_seam(
                self.warped_images[0], 
                self.warped_images[1], 
                overlap_mask,
                self.config
            )
            
            self.timings['seam_finding'] = time.time() - start_time
            self.logger.info(f"接缝线寻找完成，耗时: {self.timings['seam_finding']:.2f}秒")
            return True
            
        except Exception as e:
            self.logger.error(f"寻找接缝线时出错: {str(e)}")
            return False
    
    def compute_ownership_map(self):
        """
        计算区域所有权分配图
        
        Returns:
            bool: 成功返回True，失败返回False
        """
        if self.seam_line is None:
            self.logger.error("未找到接缝线，无法计算所有权分配图")
            return False
        
        try:
            self.logger.info("开始计算区域所有权分配图...")
            start_time = time.time()
            
            # 计算所有权分配图
            self.ownership = seam_driven_ownership_allocation(
                self.warped_images[0], 
                self.warped_images[1], 
                self.seam_line
            )
            
            self.timings['ownership'] = time.time() - start_time
            self.logger.info(f"所有权分配完成，耗时: {self.timings['ownership']:.2f}秒")
            return True
            
        except Exception as e:
            self.logger.error(f"计算所有权分配图时出错: {str(e)}")
            return False
    
    def blend_images(self, use_ownership=True):
        """
        融合图像
        
        Args:
            use_ownership: 是否使用基于所有权的融合方法
            
        Returns:
            bool: 成功返回True，失败返回False
        """
        if self.warped_images is None:
            self.logger.error("未对齐图像，无法融合")
            return False
        
        try:
            self.logger.info("开始图像融合...")
            start_time = time.time()
            
            if use_ownership and self.ownership is not None:
                # 使用基于所有权的融合
                self.logger.info("使用基于所有权的融合方法")
                self.result = ownership_based_blending(
                    self.warped_images[0], 
                    self.warped_images[1], 
                    self.ownership,
                    pyramid_levels=self.config.get('PYRAMID_LEVELS', 6),
                    sigma_factor=self.config.get('BLEND_SIGMA', 1.0)
                )
            elif self.seam_line is not None:
                # 使用基于接缝线的融合
                self.logger.info("使用基于接缝线的融合方法")
                # 创建软掩码
                from pyramid_blend import label_to_soft_mask
                soft_mask = label_to_soft_mask(
                    self.seam_line, 
                    width=self.config.get('BLEND_WIDTH', 10)
                )
                
                # 使用拉普拉斯金字塔融合
                self.result = laplacian_pyramid_blend(
                    self.warped_images[0], 
                    self.warped_images[1], 
                    soft_mask,
                    pyramid_levels=self.config.get('PYRAMID_LEVELS', 6)
                )
            else:
                # 使用简单的加权融合
                self.logger.info("使用简单加权融合方法")
                self.result = self._merge_images(self.warped_images, self.masks)
            
            # 应用局部泊松融合进行细节优化
            if self.config.get('USE_POISSON_REFINEMENT', True):
                self.logger.info("应用局部泊松融合优化细节")
                # 创建掩码用于泊松融合
                if self.masks:
                    mask = self.masks[1] if len(self.masks) > 1 else None
                    self.result = local_poisson_blend(
                        self.warped_images[1], 
                        self.result, 
                        mask
                    )
            
            self.timings['blending'] = time.time() - start_time
            self.logger.info(f"图像融合完成，耗时: {self.timings['blending']:.2f}秒")
            return True
            
        except Exception as e:
            self.logger.error(f"融合图像时出错: {str(e)}")
            return False
    
    def run_complete_stitching(self, image_paths):
        """
        运行完整的图像拼接流程
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            tuple: (成功标志, 结果图像)
        """
        try:
            self.logger.info(f"开始完整拼接流程，处理 {len(image_paths)} 张图像")
            total_start_time = time.time()
            
            # 步骤1: 加载和预处理图像
            if not self.load_and_preprocess_images(image_paths):
                return False, None
            
            # 步骤2: 检测特征并对齐图像
            if not self.detect_features_and_align():
                return False, None
            
            # 步骤3: 寻找最优接缝线
            if not self.find_optimal_seam():
                return False, None
            
            # 步骤4: 计算区域所有权分配图
            self.compute_ownership_map()  # 即使失败也继续
            
            # 步骤5: 融合图像
            if not self.blend_images():
                return False, None
            
            total_time = time.time() - total_start_time
            self.timings['total'] = total_time
            
            # 打印时间统计
            self.logger.info("拼接完成！时间统计:")
            for key, value in self.timings.items():
                self.logger.info(f"  {key}: {value:.2f}秒")
            
            return True, self.result
            
        except Exception as e:
            self.logger.error(f"完整拼接流程出错: {str(e)}")
            return False, None
    
    def get_result(self):
        """
        获取拼接结果
        
        Returns:
            numpy.ndarray: 拼接后的图像
        """
        return self.result
    
    def get_intermediate_results(self):
        """
        获取中间处理结果
        
        Returns:
            dict: 包含中间结果的字典
        """
        return {
            'original_images': self.images,
            'warped_images': self.warped_images,
            'masks': self.masks,
            'seam_line': self.seam_line,
            'ownership': self.ownership,
            'timings': self.timings
        }
    
    def save_result(self, output_path):
        """
        保存拼接结果
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            bool: 成功返回True，失败返回False
        """
        if self.result is None:
            self.logger.error("没有拼接结果可保存")
            return False
        
        try:
            # 确保输出目录存在
            ensure_directory(os.path.dirname(output_path))
            
            # 保存图像
            cv2.imwrite(output_path, self.result)
            self.logger.info(f"结果已保存至: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存结果时出错: {str(e)}")
            return False
    
    # 辅助方法
    def _merge_images(self, images, masks):
        """
        合并多张图像
        
        Args:
            images: 图像列表
            masks: 对应图像的掩码列表
            
        Returns:
            numpy.ndarray: 合并后的图像
        """
        if len(images) == 0:
            return None
        
        # 初始化结果图像
        result = np.zeros_like(images[0], dtype=np.float32)
        weight_sum = np.zeros(images[0].shape[:2], dtype=np.float32)
        
        # 加权合并
        for img, mask in zip(images, masks):
            mask_float = mask.astype(np.float32) / 255.0
            mask_3channel = np.stack([mask_float, mask_float, mask_float], axis=2)
            
            result += img.astype(np.float32) * mask_3channel
            weight_sum += mask_float
        
        # 避免除以零
        weight_sum[weight_sum == 0] = 1
        weight_3channel = np.stack([weight_sum, weight_sum, weight_sum], axis=2)
        
        # 归一化
        result = result / weight_3channel
        
        return result.astype(np.uint8)
    
    def _resize_to_final_shape(self, image, mask, final_shape):
        """
        调整图像到最终尺寸
        
        Args:
            image: 输入图像
            mask: 对应的掩码
            final_shape: 最终尺寸 (height, width)
            
        Returns:
            tuple: (调整后的图像, 调整后的掩码)
        """
        # 计算图像在最终画布中的位置
        img_height, img_width = image.shape[:2]
        final_height, final_width = final_shape
        
        # 创建新的画布
        new_image = np.zeros((final_height, final_width, 3), dtype=np.uint8)
        new_mask = np.zeros((final_height, final_width), dtype=np.uint8)
        
        # 计算放置位置（居中）
        y_offset = (final_height - img_height) // 2
        x_offset = (final_width - img_width) // 2
        
        # 复制图像到新画布
        new_image[y_offset:y_offset+img_height, x_offset:x_offset+img_width] = image
        new_mask[y_offset:y_offset+img_height, x_offset:x_offset+img_width] = mask
        
        return new_image, new_mask
    
    def _find_overlap_mask(self):
        """
        找到图像重叠区域
        
        Returns:
            numpy.ndarray: 重叠区域掩码
        """
        if self.masks is None or len(self.masks) < 2:
            return None
        
        # 计算重叠区域
        overlap_mask = np.logical_and(self.masks[0] > 128, self.masks[1] > 128)
        return overlap_mask.astype(np.uint8) * 255


# 便捷函数
def stitch_images(image_paths, config_params=None):
    """
    便捷函数：直接拼接图像
    
    Args:
        image_paths: 图像文件路径列表
        config_params: 配置参数字典
        
    Returns:
        tuple: (成功标志, 结果图像)
    """
    stitcher = ImageStitcher(config_params)
    return stitcher.run_complete_stitching(image_paths)
