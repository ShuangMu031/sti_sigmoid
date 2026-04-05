import cv2
import numpy as np
from stitcher.common.logger import get_logger
from stitcher.config import SIFT_RATIO_THRESH, HIST_SHIFT_BIN_RATIO, RANSAC_REPROJ_THRESH

logger = get_logger(__name__)


def registerTexture(img1, edge1, img2, edge2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    gray1 = (gray1 * edge1).astype(np.uint8)
    gray2 = (gray2 * edge2).astype(np.uint8)

    sift = cv2.SIFT_create(
        contrastThreshold=0.0,
        edgeThreshold=500
    )

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        raise RuntimeError("SIFT failed")

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < SIFT_RATIO_THRESH * n.distance:
            good.append(m)

    if len(good) < 8:
        raise RuntimeError("Too few good matches")

    pts1 = np.array([kp1[m.queryIdx].pt for m in good])
    pts2 = np.array([kp2[m.trainIdx].pt for m in good])

    _, idx = np.unique(pts1, axis=0, return_index=True)
    pts1, pts2 = pts1[idx], pts2[idx]

    _, idx = np.unique(pts2, axis=0, return_index=True)
    pts1, pts2 = pts1[idx], pts2[idx]

    if len(pts1) < 8:
        raise RuntimeError("Too few matches after unique filter")

    pts1, pts2 = _histogram_filter(
        pts1, pts2,
        img1.shape,
        thr=HIST_SHIFT_BIN_RATIO
    )

    if len(pts1) < 8:
        raise RuntimeError("Too few matches after histogram filter")

    H0, mask = cv2.findHomography(
        pts2, pts1,
        cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESH
    )

    if H0 is None:
        raise RuntimeError("RANSAC failed")

    in1 = pts1[mask.ravel() == 1]
    in2 = pts2[mask.ravel() == 1]

    logger.info(f"Homography inliers: {len(in1)}")

    if len(in1) < 8:
        raise RuntimeError("Too few RANSAC inliers")

    H = _calc_homography_normalized(in2, in1)
    H = np.linalg.inv(H)
    return H


def _histogram_filter(pts1, pts2, shape, thr=0.1):
    h, w = shape[:2]

    dx = pts1[:, 0] - pts2[:, 0]
    dy = pts1[:, 1] - pts2[:, 1]

    xbins = np.arange(-w, w + 1e-6, w * thr)
    ybins = np.arange(-h, h + 1e-6, h * thr)

    hx, _ = np.histogram(dx, bins=xbins)
    hy, _ = np.histogram(dy, bins=ybins)

    ix = np.argmax(hx)
    iy = np.argmax(hy)

    x0 = max(ix - 2, 0)
    x1 = min(ix + 1, len(xbins) - 2)

    y0 = max(iy - 2, 0)
    y1 = min(iy + 1, len(ybins) - 2)

    mask = (
        (dx >= xbins[x0]) & (dx <= xbins[x1 + 1]) &
        (dy >= ybins[y0]) & (dy <= ybins[y1 + 1])
    )
    logger.info(f"Matches after hist filter: {len(pts1)}")

    return pts1[mask], pts2[mask]


def _normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)

    T = np.array([
        [1 / std, 0, -mean[0] / std],
        [0, 1 / std, -mean[1] / std],
        [0, 0, 1]
    ])

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_n = (T @ pts_h.T).T

    return pts_n, T


def _calc_homography_normalized(pts_src, pts_dst):
    pts1_n, T1 = _normalize_points(pts_src)
    pts2_n, T2 = _normalize_points(pts_dst)

    A = []
    for (x, y), (u, v) in zip(pts1_n[:, :2], pts2_n[:, :2]):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hn = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T2) @ Hn @ T1
    return H / H[2, 2]
