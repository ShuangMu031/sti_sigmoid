import numpy as np


def histOstu(edge_D, interval=0.02):
    edge_D = np.asarray(edge_D).astype(np.float64)
    edge_D = edge_D[np.isfinite(edge_D)]

    if edge_D.size == 0:
        return 0.5

    edge_D = np.clip(edge_D, 0.0, 1.0)

    bin_edges = np.arange(0.0, 1.0 + interval + 1e-8, interval)
    num_x = len(bin_edges) - 1
    if num_x <= 1:
        return 0.5

    xbins = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    counts, _ = np.histogram(edge_D, bins=bin_edges)
    num_total = counts.sum()

    if num_total == 0:
        return 0.5

    pro_c = counts / num_total
    ut_c = pro_c * xbins
    sum_ut = ut_c.sum()

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

    alpha = xbins[threshold_idx] + interval / 2.0
    return float(alpha)
