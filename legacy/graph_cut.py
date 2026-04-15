# graph_cut.py
import numpy as np
import cv2
import maxflow
from hist_ostu import histOstu
import config

def sigmoid(x, alpha, beta=20.0):
    return 1.0 / (1.0 + np.exp(-beta * (x - alpha)))
def compute_boundary(mask: np.ndarray) -> np.ndarray:
    """
    等价 MATLAB: BR|BL|BU|BD
    mask: bool HxW
    return: bool HxW
    """
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

    diff = np.linalg.norm(
        img1.astype(np.float32) - img2.astype(np.float32), axis=2
    )
    diff /= (diff.max() + eps)

    overlap = valid1 & valid2
    alpha = histOstu(diff[overlap]) if np.any(overlap) else 0.5
    data_base = sigmoid(diff, alpha)

    data1 = data_base.copy()
    data2 = data_base.copy()
    data1[~valid1] = INF
    data2[~valid2] = INF

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    grad = np.sqrt(gx**2 + gy**2)
    smooth = 1.0 / (grad + eps)

    smooth[~overlap] = 0.0
    return data1, data2, smooth, overlap

# graph_cut.py


def graph_cut_seam(
    img1_w, img2_w,
    sal1_w, sal2_w,
    edge1_w, edge2_w,
    valid1, valid2,
    overlap
):
    """
    Graph Cut seam estimation.
    Return:
        label_map: HxW uint8, 0->img1, 1->img2
    """

    h, w = overlap.shape
    g = maxflow.Graph[float]()
    node_ids = g.add_grid_nodes((h, w))

    INF = 1e6

    # =========================================================
    # Unary term (data cost)
    # =========================================================
    # 颜色差（灰度即可，稳定）
    gray1 = cv2.cvtColor(img1_w, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2_w, cv2.COLOR_BGR2GRAY).astype(np.float32)
    color_diff = np.abs(gray1 - gray2)

    # 显著性差（论文核心）
    sal_diff = np.abs(sal1_w - sal2_w)

    # 边缘（鼓励 seam 走边）
    edge_cost = edge1_w + edge2_w

    # 组合 unary
    D1 = (
        0.4 * color_diff +
        0.4 * sal_diff +
        0.2 * edge_cost
    )
    D2 = D1.copy()   # 对称问题，单一数据项即可

    # 只在 overlap 内允许选择
    D1[~overlap] = 0
    D2[~overlap] = 0

    # 加入 terminal edge
    g.add_grid_tedges(node_ids, D1, D2)

    # =========================================================
    # Boundary seeds (MATLAB 等价)
    # =========================================================

    # A, B, C 定义（严格对齐 MATLAB）
    A = valid1
    B = valid2
    C = overlap

    boundary_B = compute_boundary(B)
    boundary_C = compute_boundary(C)

    # MATLAB:
    # imgseedR = (BR|BL|BU|BD) & (CR|CL|CU|CD)
    seed_source = boundary_B & boundary_C

    # imgseedB = boundary_C & ~imgseedR
    seed_sink = boundary_C & (~seed_source)

    # 强制 terminal
    # Source → img1 (label 0)
    g.add_grid_tedges(
        node_ids[seed_source],
        0,  # cost to source
        INF  # forbid sink
    )

    # Sink → img2 (label 1)
    g.add_grid_tedges(
        node_ids[seed_sink],
        INF,  # forbid source
        0
    )

    # =========================================================
    # Hard constraints（原有的，保留）
    # =========================================================
    # 不属于 img1 的区域，强制选 img2
    g.add_grid_tedges(node_ids[~valid1], INF, 0)

    # 不属于 img2 的区域，强制选 img1
    g.add_grid_tedges(node_ids[~valid2], 0, INF)

    # =========================================================
    # Pairwise term (smoothness)
    # =========================================================
    # 4-邻域
    structure = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=np.int32)

    # 平滑权重：在边缘处减弱平滑（防止切断物体）
    smooth_weight = 1.0 / (1.0 + edge_cost)

    g.add_grid_edges(
        node_ids,
        smooth_weight,
        structure=structure,
        symmetric=True
    )

    # =========================================================
    # Solve
    # =========================================================
    g.maxflow()
    labels = g.get_grid_segments(node_ids)

    # maxflow: False=source, True=sink
    # 我们统一：0->img1, 1->img2
    label_map = labels.astype(np.uint8)

    return label_map
