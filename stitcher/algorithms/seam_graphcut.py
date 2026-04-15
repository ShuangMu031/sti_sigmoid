import numpy as np
import cv2
import maxflow

from stitcher.algorithms.hist_otsu import histOstu
from stitcher.common.image_utils import to_gray_uint8
from stitcher.config import GC_SALIENCY_WEIGHT, GC_OBJECT_WEIGHT, GC_EDGE_PENALTY, GC_MODE


def sigmoid(x, alpha, beta=20.0):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-beta * (x - alpha)))


def normalize_map(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.size == 0:
        return x.astype(np.float32)

    mn = float(x.min())
    mx = float(x.max())

    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)

    return (x - mn) / (mx - mn + eps)


def compute_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)

    right = mask & ~np.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    left = mask & ~np.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    down = mask & ~np.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)
    up = mask & ~np.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)

    return right | left | up | down


def _build_smooth_normal(
    diff,
    img1,
    img2,
    pMap1,
    pMap2,
    edge1,
    edge2,
    overlap,
    saliency_weight,
    object_weight,
    edge_penalty,
):
    eps = 1e-6

    gray = to_gray_uint8(np.asarray(img1))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    smooth = 1.0 / (grad + eps)

    sal1_n = normalize_map(np.asarray(pMap1, dtype=np.float32))
    sal2_n = normalize_map(np.asarray(pMap2, dtype=np.float32))
    sal_union = np.maximum(sal1_n, sal2_n)

    sal_alpha = histOstu(sal_union[overlap]) if np.any(overlap) else 0.5
    sal_core = sigmoid(sal_union, sal_alpha, beta=12.0)

    edge1_n = normalize_map(np.asarray(edge1, dtype=np.float32))
    edge2_n = normalize_map(np.asarray(edge2, dtype=np.float32))
    edge_union = np.maximum(edge1_n, edge2_n)

    smooth = smooth * (1.0 + saliency_weight * sal_union + object_weight * sal_core)
    smooth = smooth / (1.0 + edge_penalty * edge_union + eps)

    return smooth


def _build_smooth_professional(
    diff,
    img1,
    img2,
    pMap1,
    pMap2,
    edge1,
    edge2,
    overlap,
    saliency_weight,
    object_weight,
    edge_penalty,
):
    eps = 1e-6

    img1_u8 = np.asarray(img1).astype(np.float32).astype(np.uint8)
    img2_u8 = np.asarray(img2).astype(np.float32).astype(np.uint8)

    gray1 = to_gray_uint8(img1_u8)
    gray2 = to_gray_uint8(img2_u8)

    gx1 = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, 3)
    gy1 = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, 3)
    grad1 = np.sqrt(gx1 ** 2 + gy1 ** 2)

    gx2 = cv2.Sobel(gray2, cv2.CV_32F, 1, 0, 3)
    gy2 = cv2.Sobel(gray2, cv2.CV_32F, 0, 1, 3)
    grad2 = np.sqrt(gx2 ** 2 + gy2 ** 2)

    grad_union = np.maximum(normalize_map(grad1), normalize_map(grad2))

    edge1_n = normalize_map(np.asarray(edge1, dtype=np.float32))
    edge2_n = normalize_map(np.asarray(edge2, dtype=np.float32))
    edge_union = np.maximum(edge1_n, edge2_n)

    sal1_n = normalize_map(np.asarray(pMap1, dtype=np.float32))
    sal2_n = normalize_map(np.asarray(pMap2, dtype=np.float32))
    sal_union = np.maximum(sal1_n, sal2_n)

    sal_alpha = histOstu(sal_union[overlap]) if np.any(overlap) else 0.5
    sal_core = sigmoid(sal_union, sal_alpha, beta=12.0)

    cut_friendly = (
        1.0
        + edge_penalty * edge_union
        + 0.75 * grad_union
    )

    cut_avoid = (
        1.0
        + saliency_weight * sal_union
        + object_weight * sal_core
        + 1.25 * diff
    )

    smooth = cut_avoid / (cut_friendly + eps)
    smooth = cv2.GaussianBlur(smooth.astype(np.float32), (0, 0), 0.8)

    return smooth


def build_graph_cut_cost(
    img1,
    img2,
    pMap1,
    pMap2,
    edge1,
    edge2,
    valid1,
    valid2,
    saliency_weight=None,
    object_weight=None,
    edge_penalty=None,
    mode=None,
):
    saliency_weight = GC_SALIENCY_WEIGHT if saliency_weight is None else saliency_weight
    object_weight = GC_OBJECT_WEIGHT if object_weight is None else object_weight
    edge_penalty = GC_EDGE_PENALTY if edge_penalty is None else edge_penalty
    mode = GC_MODE if mode is None else mode

    eps = 1e-6
    INF = 1e9

    valid1 = np.asarray(valid1, dtype=bool)
    valid2 = np.asarray(valid2, dtype=bool)
    overlap = valid1 & valid2

    img1_f = np.asarray(img1, dtype=np.float32)
    img2_f = np.asarray(img2, dtype=np.float32)

    diff = np.linalg.norm(img1_f - img2_f, axis=2)
    diff = diff / (diff.max() + eps)

    alpha = histOstu(diff[overlap]) if np.any(overlap) else 0.5
    data_base = sigmoid(diff, alpha)

    data1 = data_base.copy()
    data2 = data_base.copy()
    data1[~valid1] = INF
    data2[~valid2] = INF

    if mode == "professional":
        smooth = _build_smooth_professional(
            diff, img1, img2, pMap1, pMap2, edge1, edge2,
            overlap, saliency_weight, object_weight, edge_penalty,
        )
    else:
        smooth = _build_smooth_normal(
            diff, img1, img2, pMap1, pMap2, edge1, edge2,
            overlap, saliency_weight, object_weight, edge_penalty,
        )

    smooth[~overlap] = 0.0
    smooth = np.clip(smooth, 1e-4, 1e4).astype(np.float32)

    return data1.astype(np.float32), data2.astype(np.float32), smooth, overlap


def graph_cut_seam(
    img1_w,
    img2_w,
    sal1_w,
    sal2_w,
    edge1_w,
    edge2_w,
    valid1,
    valid2,
    overlap,
    saliency_weight=None,
    object_weight=None,
    edge_penalty=None,
    mode=None,
):
    saliency_weight = GC_SALIENCY_WEIGHT if saliency_weight is None else saliency_weight
    object_weight = GC_OBJECT_WEIGHT if object_weight is None else object_weight
    edge_penalty = GC_EDGE_PENALTY if edge_penalty is None else edge_penalty
    mode = GC_MODE if mode is None else mode

    overlap = np.asarray(overlap, dtype=bool)
    valid1 = np.asarray(valid1, dtype=bool)
    valid2 = np.asarray(valid2, dtype=bool)

    h, w = overlap.shape
    INF = 1e6

    g = maxflow.Graph[float]()
    node_ids = g.add_grid_nodes((h, w))

    data1, data2, smooth, _ = build_graph_cut_cost(
        img1_w,
        img2_w,
        sal1_w,
        sal2_w,
        edge1_w,
        edge2_w,
        valid1,
        valid2,
        saliency_weight=saliency_weight,
        object_weight=object_weight,
        edge_penalty=edge_penalty,
        mode=mode,
    )

    D1 = data1.copy()
    D2 = data2.copy()
    D1[~overlap] = 0.0
    D2[~overlap] = 0.0
    g.add_grid_tedges(node_ids, D1, D2)

    boundary_B = compute_boundary(valid2)
    boundary_C = compute_boundary(overlap)

    seed_source = boundary_B & boundary_C
    seed_sink = boundary_C & (~seed_source)

    if np.any(seed_source):
        g.add_grid_tedges(node_ids[seed_source], 0, INF)
    if np.any(seed_sink):
        g.add_grid_tedges(node_ids[seed_sink], INF, 0)

    if np.any(~valid1):
        g.add_grid_tedges(node_ids[~valid1], INF, 0)
    if np.any(~valid2):
        g.add_grid_tedges(node_ids[~valid2], 0, INF)

    structure = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.int32,
    )

    g.add_grid_edges(
        node_ids,
        smooth,
        structure=structure,
        symmetric=True,
    )

    g.maxflow()
    labels = g.get_grid_segments(node_ids)
    return labels.astype(np.uint8)
