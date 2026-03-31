# utils.py
import os
import cv2
import numpy as np
from PIL import Image
import logging


def get_logger(name=None):
    """返回配置好的 logger（若已配置过则复用）。"""
    logger_name = name if name is not None else __name__
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def create_dir(dir_path):
    """创建文件夹（不存在则创建）"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def read_img(img_path):
    """读取图像，返回0-1的RGB浮点型数组（适配cv2/Pillow）"""
    # 用cv2读取（处理中文路径）
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{img_path}")
    # BGR转RGB + 归一化到0-1
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb / 255.0

def save_img(img, save_path):
    """保存0-1浮点型图像到指定路径（支持中文路径）"""
    # 归一化到0-255 uint8
    img_255 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    # RGB转BGR（cv2保存）
    img_bgr = cv2.cvtColor(img_255, cv2.COLOR_RGB2BGR)
    # 保存（处理中文路径）
    cv2.imencode(os.path.splitext(save_path)[1], img_bgr)[1].tofile(save_path)
    logger = get_logger('utils')
    logger.info(f"图像已保存：{save_path}")

def srgb_to_linear(img):
    img = img.astype(np.float32) / 255.0
    return np.power(img, 2.2)

def linear_to_srgb(img):
    img = np.clip(img, 0, 1)
    return np.power(img, 1/2.2) * 255.0

def matlab_cat3(mask_2d):
    """将2D掩码扩展为3通道，类似于MATLAB中的repmat操作
    
    Args:
        mask_2d: 2D的掩码数组
        
    Returns:
        3通道的掩码数组，形状为 (height, width, 3)
    """
    return np.repeat(mask_2d[:, :, np.newaxis], 3, axis=2)