import cv2
import numpy as np
from stitcher.config import CANNY_LOW_THRESH, CANNY_HIGH_THRESH, CANNY_SIGMA


def canny_edge_detect(img, low_thresh=None, high_thresh=None, sigma=None):
    low_thresh = CANNY_LOW_THRESH if low_thresh is None else low_thresh
    high_thresh = CANNY_HIGH_THRESH if high_thresh is None else high_thresh
    sigma = CANNY_SIGMA if sigma is None else sigma

    if low_thresh > 1:
        low_255 = int(np.clip(low_thresh, 0, 255))
    else:
        low_255 = int(np.clip(low_thresh * 255, 0, 255))
        
    if high_thresh > 1:
        high_255 = int(np.clip(high_thresh, 0, 255))
    else:
        high_255 = int(np.clip(high_thresh * 255, 0, 255))
    
    if low_255 >= high_255:
        raise ValueError(f"低阈值({low_255})不能大于等于高阈值({high_255})")
    if sigma <= 0:
        raise ValueError(f"高斯sigma需大于0，当前值：{sigma}")

    img_input = img.copy()
    if img_input.dtype == np.uint8:
        img_input = img_input / 255.0
    elif img_input.dtype not in [np.float32, np.float64]:
        img_input = img_input.astype(np.float64)
    img_input = np.clip(img_input, 0.0, 1.0)

    if len(img_input.shape) == 3 and img_input.shape[2] == 3:
        img_uint8 = (img_input * 255).astype(np.uint8)
        gray_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        gray_img = gray_uint8 / 255.0
    elif len(img_input.shape) == 2:
        gray_img = img_input.copy()
    else:
        raise ValueError(f"不支持的图像维度：{img_input.shape}")

    kernel_size = 2 * int(np.ceil(2 * sigma)) + 1
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    smoothed_img = cv2.GaussianBlur(
        gray_img,
        ksize=(kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE
    )

    canny_uint8 = cv2.Canny(
        (smoothed_img * 255).astype(np.uint8),
        threshold1=low_255,
        threshold2=high_255,
        L2gradient=True
    )
    edge_mask = canny_uint8 > 0

    edge_img = np.zeros_like(gray_img, dtype=np.float64)
    edge_img[edge_mask] = 1.0

    return edge_mask, edge_img
