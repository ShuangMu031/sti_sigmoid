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

    # 组合代价
    data1 = color_cost + \
            GC_SALIENCY_WEIGHT * saliency_cost2 + \
            GC_EDGE_PENALTY * edge_cost2
    data2 = color_cost + \
            GC_SALIENCY_WEIGHT * saliency_cost1 + \
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


def graph_cut_seam(
    img1_w, img2_w,
    sal1_w, sal2_w,
    edge1_w, edge2_w,
    valid1, valid2,
    overlap
):
    h, w = overlap.shape
    g = maxflow.Graph[float]()
    node_ids = g.add_grid_nodes((h, w))

    INF = 1e6

    data1, data2, smooth, _ = build_graph_cut_cost(
        img1_w, img2_w,
        sal1_w, sal2_w,
        edge1_w, edge2_w,
        valid1, valid2
    )

    D1 = data1.copy()
    D2 = data2.copy()

    D1[~overlap] = 0
    D2[~overlap] = 0

    g.add_grid_tedges(node_ids, D1, D2)

    A = valid1
    B = valid2
    C = overlap

    boundary_B = compute_boundary(B)
    boundary_C = compute_boundary(C)

    seed_source = boundary_B & boundary_C
    seed_sink = boundary_C & (~seed_source)

    g.add_grid_tedges(
        node_ids[seed_source],
        0,
        INF
    )

    g.add_grid_tedges(
        node_ids[seed_sink],
        INF,
        0
    )

    g.add_grid_tedges(node_ids[~valid1], INF, 0)
    g.add_grid_tedges(node_ids[~valid2], 0, INF)

    structure = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=np.int32)

    g.add_grid_edges(
        node_ids,
        smooth,
        structure=structure,
        symmetric=True
    )

    g.maxflow()
    labels = g.get_grid_segments(node_ids)

    label_map = labels.astype(np.uint8)

    return label_map
