import os
import cv2
import numpy as np


def ensure_directory(path):
    if path and not os.path.exists(path):
        os.makedirs(path)
    return path


def get_stitched_size(img1_w, img2_w):
    h = max(img1_w.shape[0], img2_w.shape[0])
    w = max(img1_w.shape[1], img2_w.shape[1])
    return h, w


def to_gray_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("输入图像不能为空")

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"不支持的图像形状: {img.shape}")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray
