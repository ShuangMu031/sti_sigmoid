import numpy as np
import cv2



def compute_overlap_masks(img1, img2, valid1=None, valid2=None):
    """
    兼容旧调用方式，同时优先使用上游传入的 valid mask。
    这样不会把真实黑色像素误判成无效区域。
    """
    if valid1 is not None and valid2 is not None:
        a = np.asarray(valid1, dtype=bool)
        b = np.asarray(valid2, dtype=bool)
        c = a & b
        return a, b, c

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    a = gray1 > 0
    b = gray2 > 0
    c = a & b

    return a, b, c
