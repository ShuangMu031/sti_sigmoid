import numpy as np
import cv2
import maxflow
from stitcher.algorithms.hist_otsu import histOstu
from stitcher.config import GC_SALIENCY_WEIGHT, GC_OBJECT_WEIGHT, GC_EDGE_PENALTY


def sigmoid(x, alpha, beta=20.0):
    return 1.0 / (1.0 + np.exp(-beta * (x - alpha)))


def compute_boundary(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape

    right = mask & ~np.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    left  = mask & ~np.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    down  = mask & ~np.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)
    up    = mask & ~np.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)

    return right | left | up | down


def build_graph_cut_cost(
    img1, img2,
    pMap1, pMap2,
    edge1, edge2,
    valid1, valid2
):
    eps = 1e-6
    INF = 1e9
    h, w = img1.shape[:2]

    # 颜色差异代价
    diff = np.linalg.norm(
        img1.astype(np.float32) - img2.astype(np.float32), axis=2
    )
    diff /= (diff.max() + eps)

    overlap = valid1 & valid2
    alpha = histOstu(diff[overlap]) if np.any(overlap) else 0.5
    color_cost = sigmoid(diff, alpha)

    # 显著性代价 - 优先选择显著性高的区域
    saliency_cost1 = pMap1 / (pMap1.max() + eps) if pMap1.max() > 0 else np.zeros_like(pMap1)
    saliency_cost2 = pMap2 / (pMap2.max() + eps) if pMap2.max() > 0 else np.zeros_like(pMap2)

    # 边缘代价 - 避免在边缘处切割
    edge_cost1 = edge1 / (edge1.max() + eps) if edge1.max() > 0 else np.zeros_like(edge1)
    edge_cost2 = edge2 / (edge2.max() + eps) if edge2.max() > 0 else np.zeros_like(edge2)

    # 物体代价 - 使用saliency和edge模拟物体区域，避免切割物体
    # GC_OBJECT_WEIGHT 用于惩罚 seam 穿过显著性高和边缘多的区域
    object_region1 = (pMap1 > 0.3) | (edge1 > 0.3)
    object_region2 = (pMap2 > 0.3) | (edge2 > 0.3)
    object_cost1 = object_region1.astype(np.float32)
    object_cost2 = object_region2.astype(np.float32)

    # 组合代价 - 集成GC_SALIENCY_WEIGHT, GC_OBJECT_WEIGHT, GC_EDGE_PENALTY
    data1 = color_cost + \
            GC_SALIENCY_WEIGHT * saliency_cost2 + \
            GC_OBJECT_WEIGHT * object_cost2 + \
            GC_EDGE_PENALTY * edge_cost2
    data2 = color_cost + \
            GC_SALIENCY_WEIGHT * saliency_cost1 + \
            GC_OBJECT_WEIGHT * object_cost1 + \
            GC_EDGE_PENALTY * edge_cost1

    # 确保有效区域
    data1[~valid1] = INF
    data2[~valid2] = INF

    # 平滑代价 - 基于梯度
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    grad = np.sqrt(gx**2 + gy**2)
    smooth = 1.0 / (grad + eps)

    smooth[~overlap] = 0.0
    return data1, data2, smooth, overlap


# 图割标签语义定义
LABEL_SELECT_IMG1 = 0  # 选择img1_w (source, 移动图像)
LABEL_SELECT_IMG2 = 1  # 选择img2_w (sink, 基础图像)


def graph_cut_seam(
    img1_w, img2_w,
    sal1_w, sal2_w,
    edge1_w, edge2_w,
    valid1, valid2,
    overlap
):
    """
    使用Graph Cut算法确定最佳接缝位置。
    使用ROI裁剪优化性能，只在重叠区域内计算图割。

    参数:
        img1_w: 对齐后的移动图像 (warped)
        img2_w: 对齐后的基础图像 (warped)
        sal1_w, sal2_w: 对齐后的显著性图
        edge1_w, edge2_w: 对齐后的边缘图
        valid1, valid2: 有效区域掩码
        overlap: 重叠区域掩码

    返回:
        label_map: uint8类型的标签图
                 0 (LABEL_SELECT_IMG1) = 选择img1_w的像素
                 1 (LABEL_SELECT_IMG2) = 选择img2_w的像素
    """
    h, w = overlap.shape

    # 如果没有重叠区域，直接返回全选择img2的标签图
    if not np.any(overlap):
        return np.full((h, w), LABEL_SELECT_IMG2, dtype=np.uint8)

    # 计算重叠区域的ROI (Region of Interest)
    ys, xs = np.where(overlap)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # 增加一些边距，确保边界条件正确处理
    margin = 10
    y0 = max(0, y0 - margin)
    y1 = min(h, y1 + margin + 1)
    x0 = max(0, x0 - margin)
    x1 = min(w, x1 + margin + 1)

    # 裁剪所有数据到ROI
    img1_roi = img1_w[y0:y1, x0:x1]
    img2_roi = img2_w[y0:y1, x0:x1]
    sal1_roi = sal1_w[y0:y1, x0:x1]
    sal2_roi = sal2_w[y0:y1, x0:x1]
    edge1_roi = edge1_w[y0:y1, x0:x1]
    edge2_roi = edge2_w[y0:y1, x0:x1]
    valid1_roi = valid1[y0:y1, x0:x1]
    valid2_roi = valid2[y0:y1, x0:x1]
    overlap_roi = overlap[y0:y1, x0:x1]

    # 在ROI内构建和求解图割
    h_roi, w_roi = overlap_roi.shape
    g = maxflow.Graph[float]()
    node_ids_roi = g.add_grid_nodes((h_roi, w_roi))

    INF = 1e6

    data1_roi, data2_roi, smooth_roi, _ = build_graph_cut_cost(
        img1_roi, img2_roi,
        sal1_roi, sal2_roi,
        edge1_roi, edge2_roi,
        valid1_roi, valid2_roi
    )

    D1_roi = data1_roi.copy()
    D2_roi = data2_roi.copy()

    D1_roi[~overlap_roi] = 0
    D2_roi[~overlap_roi] = 0

    g.add_grid_tedges(node_ids_roi, D1_roi, D2_roi)

    A_roi = valid1_roi
    B_roi = valid2_roi
    C_roi = overlap_roi

    boundary_B_roi = compute_boundary(B_roi)
    boundary_C_roi = compute_boundary(C_roi)

    seed_source_roi = boundary_B_roi & boundary_C_roi
    seed_sink_roi = boundary_C_roi & (~seed_source_roi)

    g.add_grid_tedges(
        node_ids_roi[seed_source_roi],
        0,
        INF
    )

    g.add_grid_tedges(
        node_ids_roi[seed_sink_roi],
        INF,
        0
    )

    g.add_grid_tedges(node_ids_roi[~valid1_roi], INF, 0)
    g.add_grid_tedges(node_ids_roi[~valid2_roi], 0, INF)

    structure = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=np.int32)

    g.add_grid_edges(
        node_ids_roi,
        smooth_roi,
        structure=structure,
        symmetric=True
    )

    g.maxflow()
    labels_roi = g.get_grid_segments(node_ids_roi)

    # 创建完整的label_map，默认选择img2
    label_map = np.full((h, w), LABEL_SELECT_IMG2, dtype=np.uint8)
    
    # 将ROI的结果复制到完整的label_map中
    label_map[y0:y1, x0:x1] = labels_roi.astype(np.uint8)

    return label_map
