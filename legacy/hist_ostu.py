import numpy as np


def histOstu(edge_D, interval=0.02):
    """
    复刻 MATLAB histOstu.m：大津法计算 Sigmoid 最优阈值

    参数
    ----
    edge_D : 1D np.ndarray
        重叠区颜色差异值（建议已归一化到 [0,1]）
    interval : float, optional
        直方图区间步长（默认 0.02，与 graph_cut 接口对齐）

    返回
    ----
    alpha : float
        Otsu 最优阈值
    """

    # ---- 0. 输入清洗（工程健壮性）----
    edge_D = np.asarray(edge_D).astype(np.float64)
    edge_D = edge_D[np.isfinite(edge_D)]  # 去 NaN / inf

    if edge_D.size == 0:
        return 0.5

    # 限制在 [0,1]，防止颜色差溢出
    edge_D = np.clip(edge_D, 0.0, 1.0)

    # ---- 1. 构建直方图区间 ----
    bin_edges = np.arange(
        0.0, 1.0 + interval + 1e-8, interval
    )  # 长度 = num_x + 1

    num_x = len(bin_edges) - 1
    if num_x <= 1:
        return 0.5

    # 区间中点（与 counts 一一对应）
    xbins = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # ---- 2. 统计直方图 ----
    counts, _ = np.histogram(edge_D, bins=bin_edges)
    num_total = counts.sum()

    if num_total == 0:
        return 0.5

    # ---- 3. 概率与全局均值 ----
    pro_c = counts / num_total           # P(i)
    ut_c = pro_c * xbins                 # P(i) * μ(i)
    sum_ut = ut_c.sum()                  # 全局均值 μ_T

    # ---- 4. Otsu：最大类间方差 ----
    energy_max = -np.inf
    threshold_idx = 0

    for k in range(num_x):
        w_k = pro_c[:k + 1].sum()

        if w_k <= 1e-8 or w_k >= 1 - 1e-8:
            continue

        u_k = (pro_c[:k + 1] * xbins[:k + 1]).sum()

        sigma_b = (u_k - sum_ut * w_k) ** 2 / (w_k * (1.0 - w_k))

        if sigma_b > energy_max:
            energy_max = sigma_b
            threshold_idx = k

    # ---- 5. 最优阈值 ----
    alpha = xbins[threshold_idx] + interval / 2.0
    return float(alpha)
