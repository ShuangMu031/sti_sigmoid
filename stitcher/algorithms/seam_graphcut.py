import numpy as np
import cv2
import maxflow
from stitcher.algorithms.hist_otsu import histOstu
from stitcher.config import (
    GC_SALIENCY_WEIGHT,
    GC_OBJECT_WEIGHT,
    GC_EDGE_PENALTY,
    GC_ROI_MARGIN,
    GC_OBJECT_THRESH
)


def sigmoid(x, alpha, beta=20.0):
    return 1.0 / (1.0 + np.exp(-beta * (x - alpha)))


def compute_boundary(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape

    right = mask & ~np.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    left  = mask & ~np.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    down  = mask & ~np.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)
    up    = mask & ~np.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)

    return right | left | up | down


def _extract_overlap_roi(overlap, margin=GC_ROI_MARGIN):
    h, w = overlap.shape
    ys, xs = np.where(overlap)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    
    y0 = max(0, y0 - margin)
    y1 = min(h, y1 + margin + 1)
    x0 = max(0, x0 - margin)
    x1 = min(w, x1 + margin + 1)
    
    return y0, y1, x0, x1


def _build_unary_cost(
    img1, img2,
    pMap1, pMap2,
    edge1, edge2,
    valid1, valid2
):
    eps = 1e-6
    INF = 1e9
    h, w = img1.shape[:2]

    diff = np.linalg.norm(
        img1.astype(np.float32) - img2.astype(np.float32), axis=2
    )
    diff /= (diff.max() + eps)

    overlap = valid1 & valid2
    alpha = histOstu(diff[overlap]) if np.any(overlap) else 0.5
    color_cost = sigmoid(diff, alpha)

    saliency_cost1 = pMap1 / (pMap1.max() + eps) if pMap1.max() > 0 else np.zeros_like(pMap1)
    saliency_cost2 = pMap2 / (pMap2.max() + eps) if pMap2.max() > 0 else np.zeros_like(pMap2)

    edge_cost1 = edge1 / (edge1.max() + eps) if edge1.max() > 0 else np.zeros_like(edge1)
    edge_cost2 = edge2 / (edge2.max() + eps) if edge2.max() > 0 else np.zeros_like(edge2)

    object_region1 = (pMap1 > GC_OBJECT_THRESH) | (edge1 > GC_OBJECT_THRESH)
    object_region2 = (pMap2 > GC_OBJECT_THRESH) | (edge2 > GC_OBJECT_THRESH)
    object_cost1 = object_region1.astype(np.float32)
    object_cost2 = object_region2.astype(np.float32)

    data1 = color_cost + \
            GC_SALIENCY_WEIGHT * saliency_cost2 + \
            GC_OBJECT_WEIGHT * object_cost2 + \
            GC_EDGE_PENALTY * edge_cost2
    data2 = color_cost + \
            GC_SALIENCY_WEIGHT * saliency_cost1 + \
            GC_OBJECT_WEIGHT * object_cost1 + \
            GC_EDGE_PENALTY * edge_cost1

    data1[~valid1] = INF
    data2[~valid2] = INF

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    grad = np.sqrt(gx**2 + gy**2)
    smooth = 1.0 / (grad + eps)

    smooth[~overlap] = 0.0
    return data1, data2, smooth, overlap


def _build_seed_constraints(valid1_roi, valid2_roi, overlap_roi):
    INF = 1e6
    
    A_roi = valid1_roi
    B_roi = valid2_roi
    C_roi = overlap_roi

    boundary_B_roi = compute_boundary(B_roi)
    boundary_C_roi = compute_boundary(C_roi)

    seed_source_roi = boundary_B_roi & boundary_C_roi
    seed_sink_roi = boundary_C_roi & (~seed_source_roi)
    
    return seed_source_roi, seed_sink_roi, INF


def _solve_graphcut(node_ids_roi, unary1, unary2, smooth, seeds, valid1_roi, valid2_roi, overlap_roi):
    g = maxflow.Graph[float]()
    g.add_grid_nodes(node_ids_roi.shape)
    
    seed_source_roi, seed_sink_roi, INF = seeds
    
    D1_roi = unary1.copy()
    D2_roi = unary2.copy()
    
    D1_roi[~overlap_roi] = 0
    D2_roi[~overlap_roi] = 0
    
    g.add_grid_tedges(node_ids_roi, D1_roi, D2_roi)
    
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
        smooth,
        structure=structure,
        symmetric=True
    )
    
    g.maxflow()
    labels_roi = g.get_grid_segments(node_ids_roi)
    
    return labels_roi


def graph_cut_seam(
    img1_w, img2_w,
    sal1_w, sal2_w,
    edge1_w, edge2_w,
    valid1, valid2,
    overlap
):
    """
    使用Graph Cut算法确定最佳接缝位置。
    返回select_img1_mask: True表示选择img1_w的像素，False表示保留img2_w的像素。
    
    参数:
        img1_w: 对齐后的移动图像 (warped)
        img2_w: 对齐后的基础图像 (warped)
        sal1_w, sal2_w: 对齐后的显著性图
        edge1_w, edge2_w: 对齐后的边缘图
        valid1, valid2: 有效区域掩码
        overlap: 重叠区域掩码

    返回:
        select_img1_mask: bool类型的掩码
                        True = 选择img1_w的像素
                        False = 保留img2_w的像素
    """
    h, w = overlap.shape

    if not np.any(overlap):
        return np.full((h, w), False, dtype=bool)

    y0, y1, x0, x1 = _extract_overlap_roi(overlap)

    img1_roi = img1_w[y0:y1, x0:x1]
    img2_roi = img2_w[y0:y1, x0:x1]
    sal1_roi = sal1_w[y0:y1, x0:x1]
    sal2_roi = sal2_w[y0:y1, x0:x1]
    edge1_roi = edge1_w[y0:y1, x0:x1]
    edge2_roi = edge2_w[y0:y1, x0:x1]
    valid1_roi = valid1[y0:y1, x0:x1]
    valid2_roi = valid2[y0:y1, x0:x1]
    overlap_roi = overlap[y0:y1, x0:x1]

    h_roi, w_roi = overlap_roi.shape
    g = maxflow.Graph[float]()
    node_ids_roi = g.add_grid_nodes((h_roi, w_roi))

    data1_roi, data2_roi, smooth_roi, _ = _build_unary_cost(
        img1_roi, img2_roi,
        sal1_roi, sal2_roi,
        edge1_roi, edge2_roi,
        valid1_roi, valid2_roi
    )

    seeds = _build_seed_constraints(valid1_roi, valid2_roi, overlap_roi)
    labels_roi = _solve_graphcut(
        node_ids_roi, data1_roi, data2_roi, smooth_roi,
        seeds, valid1_roi, valid2_roi, overlap_roi
    )

    select_img1_mask = np.full((h, w), False, dtype=bool)
    select_img1_mask[y0:y1, x0:x1] = labels_roi.astype(bool)

    return select_img1_mask
