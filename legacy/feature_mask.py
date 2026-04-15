import numpy as np
import cv2

def build_feature_valid_mask(
    saliency_map,
    edge_mask,
    object_mask=None,
    saliency_thresh=0.6,
    dilate_edge=True,
    edge_dilate_ksize=3
):
    """
    构建 SIFT 特征点的“合法区域掩码”

    合法区域 = 显著性区域 ∪ 边缘区域 ∪ 主体区域

    输出：
        feature_valid_mask : bool, shape (H, W)
    """

    H, W = saliency_map.shape

    # ---- 1. 显著性区域 ----
    saliency_mask = saliency_map > saliency_thresh

    # ---- 2. 边缘区域（可选膨胀，增强稳定性）----
    edge_mask = edge_mask.astype(bool)

    if dilate_edge:
        kernel = np.ones((edge_dilate_ksize, edge_dilate_ksize), np.uint8)
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel) > 0

    # ---- 3. 主体区域 ----
    if object_mask is not None:
        object_mask = object_mask.astype(bool)
    else:
        object_mask = np.zeros((H, W), dtype=bool)

    # ---- 4. 合并（并集，非常重要）----
    feature_valid_mask = saliency_mask | edge_mask | object_mask

    return feature_valid_mask
