# -*- coding: utf-8 -*-
"""
图像拼接应用程序 - 主入口
将图像拼接功能封装为桌面应用程序
"""

import os
import sys
import logging
import cv2
import numpy as np
import time
from pathlib import Path

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stitching_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))


class StitchingApp:
    """
    图像拼接应用程序主类
    负责协调各个模块的工作，处理用户交互和图像拼接流程
    """
    
    def __init__(self):
        """
        初始化应用程序
        """
        self.logger = logger
        self.logger.info("初始化图像拼接应用程序...")
        
        # 存储用户选择的图像路径
        self.image_paths = []
        
        # 存储拼接参数配置
        self.config = self._load_default_config()
        
        # 存储拼接结果
        self.result_image = None
        
        # 进度回调函数
        self.progress_callback = None
        
        # 加载核心模块
        self._load_core_modules()
    
    def set_progress_callback(self, callback):
        """
        设置进度回调函数
        
        Args:
            callback: 回调函数，签名为 callback(step, total, message)
        """
        self.progress_callback = callback
    
    def _report_progress(self, step, total, message):
        """
        报告进度
        
        Args:
            step: 当前步骤
            total: 总步骤数
            message: 进度消息
        """
        if self.progress_callback:
            self.progress_callback(step, total, message)
    
    def _load_default_config(self):
        """
        加载默认配置参数
        """
        # 这里可以从config.py加载配置
        try:
            import config
            return vars(config)
        except ImportError:
            self.logger.warning("无法导入config模块，使用默认配置")
            return {
                'PYRAMID_LEVELS': 6,
                'CANNY_THRESH_1': 50,
                'CANNY_THRESH_2': 150,
                'BLEND_SIGMA': 1.0,
                'SEAM_BAND': 9,  # 融合带宽参数
                'FEATHER_RADIUS': 11  # 羽化半径
            }
    
    def _load_core_modules(self):
        """
        动态加载核心拼接模块
        """
        self.logger.info("加载核心拼接模块...")
        self.core_modules = {}
        
        # 尝试导入各个核心模块
        try:
            from edge_detect import canny_edge_detect
            from homography_align import homography_align
            from register_texture import registerTexture
            from saliency import mbs_saliency
            from overlap import compute_overlap_masks
            from graph_cut import graph_cut_seam
            from gradient_blend import gradient_blend_local
            
            self.core_modules['canny_edge_detect'] = canny_edge_detect
            self.core_modules['homography_align'] = homography_align
            self.core_modules['registerTexture'] = registerTexture
            self.core_modules['mbs_saliency'] = mbs_saliency
            self.core_modules['compute_overlap_masks'] = compute_overlap_masks
            self.core_modules['graph_cut_seam'] = graph_cut_seam
            self.core_modules['gradient_blend_local'] = gradient_blend_local
            
            self.logger.info("所有核心模块加载成功")
        except Exception as e:
            self.logger.error(f"加载模块时出错: {str(e)}")
            raise
    
    def load_images(self, image_paths):
        """
        加载用户选择的图像
        
        Args:
            image_paths: 图像文件路径列表
        """
        self.logger.info(f"加载图像: {len(image_paths)}张")
        self.image_paths = image_paths
        return True
    
    def update_config(self, new_config):
        """
        更新配置参数
        
        Args:
            new_config: 包含新参数的字典
        """
        self.config.update(new_config)
        self.logger.info(f"更新配置: {new_config}")
    
    def _extract_seam_line(self, label_map, overlap):
        """
        从标签图中提取接缝线位置
        
        参数:
            label_map: 图割生成的标签图(0表示属于图像2，1表示属于图像1)
            overlap: 重叠区域掩码
        
        返回:
            seam: 1像素宽的接缝线掩码
        """
        # 创建与标签图相同大小的零矩阵
        seam = np.zeros_like(label_map, np.uint8)
        # 检测水平方向的标签变化（左右相邻像素不同）
        seam[:, 1:] |= (label_map[:, 1:] != label_map[:, :-1])
        # 检测垂直方向的标签变化（上下相邻像素不同）
        seam[1:, :] |= (label_map[1:, :] != label_map[:-1, :])
        # 只保留重叠区域内的接缝
        seam &= overlap.astype(np.uint8)
        return seam
    
    def run_stitching(self):
        """
        执行图像拼接流程
        """
        try:
            total_steps = 8
            self.logger.info("开始图像拼接...")
            self._report_progress(0, total_steps, "开始图像拼接...")
            t0 = time.time()
            
            # 创建输出目录
            os.makedirs("./output", exist_ok=True)
            
            # 确保有足够的图像进行拼接
            if len(self.image_paths) < 2:
                self.logger.error("需要至少两张图像进行拼接")
                return False
            
            # 读取图像
            self._report_progress(1, total_steps, "正在加载图像...")
            img1 = cv2.imread(self.image_paths[0])
            img2 = cv2.imread(self.image_paths[1])
            
            if img1 is None or img2 is None:
                self.logger.error("无法读取图像文件")
                return False
            
            # 获取配置参数
            SEAM_BAND = self.config.get('SEAM_BAND', 9)
            FEATHER_RADIUS = self.config.get('FEATHER_RADIUS', 11)
            
            # 显著性和边缘检测
            self._report_progress(2, total_steps, "执行显著性和边缘检测...")
            sal1 = self.core_modules['mbs_saliency'](img1)
            sal2 = self.core_modules['mbs_saliency'](img2)
            edge1, _ = self.core_modules['canny_edge_detect'](img1)
            edge2, _ = self.core_modules['canny_edge_detect'](img2)
            
            # 计算同态变换矩阵
            self._report_progress(3, total_steps, "计算单应性变换矩阵...")
            H = self.core_modules['registerTexture'](img1, edge1, img2, edge2)
            self.logger.info(f"单应性变换矩阵: {H}")
            
            # 将图像变换到公共画布
            self._report_progress(4, total_steps, "图像对齐中...")
            img1_w, img2_w, sal1_w, sal2_w, edge1_w, edge2_w, valid1, valid2 = self.core_modules['homography_align'](
                img1, sal1, edge1,
                img2, sal2, edge2,
                H
            )
            
            # 计算重叠区域掩码
            _, _, overlap = self.core_modules['compute_overlap_masks'](img1_w, img2_w)
            
            # 图割接缝选择
            self._report_progress(5, total_steps, "执行图割接缝选择...")
            label_map = self.core_modules['graph_cut_seam'](
                img1_w, img2_w,  # 变换后的图像
                sal1_w, sal2_w,  # 变换后的显著性图
                edge1_w, edge2_w,  # 变换后的边缘图
                valid1, valid2,  # 有效区域掩码
                overlap  # 重叠区域掩码
            )
            
            # 硬合成
            self._report_progress(6, total_steps, "执行硬合成...")
            base = img2_w.copy()
            base[label_map == 1] = img1_w[label_map == 1]
            
            # 提取接缝线
            seam_line = self._extract_seam_line(label_map, overlap)
            
            # 局部泊松融合
            self._report_progress(6, total_steps, "执行局部梯度融合...")
            kernel = np.ones((SEAM_BAND, SEAM_BAND), np.uint8)
            seam_band = cv2.dilate(seam_line.astype(np.uint8), kernel)
            
            seam_band = cv2.erode(
                seam_band,
                np.ones((3, 3), np.uint8),
                iterations=2
            )
            
            poisson_mask = seam_band.astype(bool) & overlap
            
            result = self.core_modules['gradient_blend_local'](
                source=img1_w,
                target=base,
                mask=poisson_mask
            )
            
            # 接缝感知的羽化平滑
            self._report_progress(7, total_steps, "执行羽化平滑...")
            alpha = cv2.GaussianBlur(
                seam_band.astype(np.float32),
                (0, 0),
                sigmaX=FEATHER_RADIUS
            )
            
            alpha = alpha / (alpha.max() + 1e-6)
            alpha = np.clip(alpha, 0, 1)[..., None]
            
            result = (1 - alpha) * result + alpha * base
            self.result_image = result.astype(np.uint8)
            
            # 保存结果
            self._report_progress(8, total_steps, "正在保存结果...")
            default_output_path = "./output/result.png"
            cv2.imwrite(default_output_path, self.result_image)
            
            self.logger.info(f"拼接完成！总耗时: {time.time() - t0:.2f}s")
            self._report_progress(8, total_steps, "拼接完成!")
            return self.result_image
        except Exception as e:
            self.logger.error(f"拼接过程中出错: {str(e)}")
            return False
    
    def save_result(self, output_path):
        """
        保存拼接结果
        
        Args:
            output_path: 输出文件路径
        """
        if self.result_image is None:
            self.logger.error("没有可保存的拼接结果")
            return False
        
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # 保存图像
            cv2.imwrite(output_path, self.result_image)
            self.logger.info(f"保存结果到: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存结果时出错: {str(e)}")
            return False


def main():
    """
    应用程序主函数
    """
    app = StitchingApp()
    
    # 这里将初始化GUI并启动应用
    # 由于我们还没有实现GUI，这里只是打印一条消息
    print("图像拼接应用程序已初始化")
    print("请运行 'python gui_app.py' 来启动图形界面")


if __name__ == "__main__":
    main()