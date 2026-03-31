# geometry/homography_core.py
import numpy as np


def normalize_points(pts: np.ndarray):
    """
    Normalized DLT 中的点归一化（Hartley normalization）

    Args:
        pts: (N, 2) ndarray

    Returns:
        pts_n:  (N, 3) normalized homogeneous points
        T:      (3, 3) normalization matrix
    """
    pts = np.asarray(pts, dtype=np.float64)

    mean = np.mean(pts, axis=0)
    std = np.std(pts)

    # 避免退化
    if std < 1e-8:
        std = 1.0

    T = np.array([
        [1.0 / std, 0.0, -mean[0] / std],
        [0.0, 1.0 / std, -mean[1] / std],
        [0.0, 0.0, 1.0]
    ])

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_n = (T @ pts_h.T).T

    return pts_n, T


def calc_homography_normalized(
    pts_src: np.ndarray,
    pts_dst: np.ndarray
):
    """
    使用 Normalized DLT 计算 Homography

    Args:
        pts_src: (N, 2) source points
        pts_dst: (N, 2) destination points

    Returns:
        H: (3, 3) homography matrix, normalized s.t. H[2,2] = 1
    """
    if pts_src.shape[0] < 4:
        raise ValueError("At least 4 point correspondences are required")

    pts1_n, T1 = normalize_points(pts_src)
    pts2_n, T2 = normalize_points(pts_dst)

    A = []
    for (x, y), (u, v) in zip(pts1_n[:, :2], pts2_n[:, :2]):
        A.append([-x, -y, -1,  0,  0,  0, u * x, u * y, u])
        A.append([ 0,  0,  0, -x, -y, -1, v * x, v * y, v])

    A = np.asarray(A, dtype=np.float64)

    # SVD
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)

    # Denormalize
    H = np.linalg.inv(T2) @ Hn @ T1

    # Normalize scale
    if abs(H[2, 2]) > 1e-8:
        H = H / H[2, 2]

    return H
