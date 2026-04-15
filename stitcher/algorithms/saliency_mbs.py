import cv2
import numpy as np
import time

from stitcher.common.logger import get_logger
from stitcher.common.image_utils import to_gray_uint8

logger = get_logger(__name__)

_SAL_WIN_COUNTER = 0
_SAL_WINDOWS = []


def getParam():
    param = {
        'verbose': False,
        'MAX_DIM': 400,
        'use_backgroundness': True,
        'use_geodesic': False,
        'remove_border': True,
        'colorSpace': 'LAB',
        'sigma_smooth': 1.0,
        'contrast_enhance': True,
        'theta': 0.1,
        'num_iter': 2,
        'erosion_size': 3,
        'dilation_size': 3
    }
    return param


def BG(I):
    I_gray = to_gray_uint8(I).astype(np.float64) / 255.0
    h, w = I_gray.shape

    border_h = int(0.1 * h)
    border_w = int(0.1 * w)
    top = I_gray[:border_h, :].mean()
    bottom = I_gray[-border_h:, :].mean()
    left = I_gray[:, :border_w].mean()
    right = I_gray[:, -border_w:].mean()

    bg = np.zeros_like(I_gray)
    for y in range(h):
        for x in range(w):
            d_top = y / h
            d_bottom = (h - y - 1) / h
            d_left = x / w
            d_right = (w - x - 1) / w
            bg[y, x] = (top * (1 - d_top) + bottom * (1 - d_bottom) + left * (1 - d_left) + right * (1 - d_right)) / 4.0

    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
    return bg


def fastMBS(I):
    h, w = I.shape
    dist = np.ones((h, w), dtype=np.float64) * 1e9

    dist[0, :] = 0
    dist[-1, :] = 0
    dist[:, 0] = 0
    dist[:, -1] = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            d1 = dist[y - 1, x] + abs(I[y, x] - I[y - 1, x])
            d2 = dist[y, x - 1] + abs(I[y, x] - I[y, x - 1])
            d3 = dist[y - 1, x - 1] + abs(I[y, x] - I[y - 1, x - 1])
            dist[y, x] = min(d1, d2, d3)

    for y in range(h - 2, 0, -1):
        for x in range(w - 2, 0, -1):
            d1 = dist[y + 1, x] + abs(I[y, x] - I[y + 1, x])
            d2 = dist[y, x + 1] + abs(I[y, x] - I[y, x + 1])
            d3 = dist[y + 1, x + 1] + abs(I[y, x] - I[y + 1, x + 1])
            dist[y, x] = min(dist[y, x], d1, d2, d3)

    dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
    dist = 1 - dist
    return dist


def fastGeodesic(I, theta=0.1):
    h, w = I.shape
    dist = np.ones((h, w), dtype=np.float64) * 1e9
    dist[0, :] = 0
    dist[-1, :] = 0
    dist[:, 0] = 0
    dist[:, -1] = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            d1 = dist[y - 1, x] + (abs(I[y, x] - I[y - 1, x]) > theta)
            d2 = dist[y, x - 1] + (abs(I[y, x] - I[y, x - 1]) > theta)
            d3 = dist[y - 1, x - 1] + (abs(I[y, x] - I[y - 1, x - 1]) > theta)
            dist[y, x] = min(d1, d2, d3)

    for y in range(h - 2, 0, -1):
        for x in range(w - 2, 0, -1):
            d1 = dist[y + 1, x] + (abs(I[y, x] - I[y + 1, x]) > theta)
            d2 = dist[y, x + 1] + (abs(I[y, x] - I[y, x + 1]) > theta)
            d3 = dist[y + 1, x + 1] + (abs(I[y, x] - I[y + 1, x + 1]) > theta)
            dist[y, x] = min(dist[y, x], d1, d2, d3)

    dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
    dist = 1 - dist
    return dist


def doMBS(I, param):
    if param['verbose']:
        logger.info("[doMBS] 开始MBS显著性检测...")
        start_total = time.time()

    h, w = I.shape[:2]
    scale = param['MAX_DIM'] / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    I_resize = cv2.resize(I, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    I_uint8 = (I_resize * 255).astype(np.uint8)
    I_lab = cv2.cvtColor(I_uint8, cv2.COLOR_BGR2LAB) / 255.0
    I_l = I_lab[:, :, 0]

    if param['verbose']:
        start_dist = time.time()
    if param['use_geodesic']:
        dist = fastGeodesic(I_l, param['theta'])
    else:
        dist = fastMBS(I_l)
        if param['verbose']:
            logger.info(f"[doMBS] 距离计算耗时：{time.time() - start_dist:.4f}s")

    if param['use_backgroundness']:
        if param['verbose']:
            start_bg = time.time()
        bg = BG(I_resize)
        dist = dist * (1 - bg)
        if param['verbose']:
            logger.info(f"[doMBS] 背景图融合耗时：{time.time() - start_bg:.4f}s")

    if param['verbose']:
        start_post = time.time()
    dist = cv2.resize(dist, (w, h), interpolation=cv2.INTER_LINEAR)

    if param['remove_border']:
        border = int(0.05 * min(h, w))
        dist[:border, :] = 0
        dist[-border:, :] = 0
        dist[:, :border] = 0
        dist[:, -border:] = 0

    dist = cv2.GaussianBlur(dist, (0, 0), sigmaX=param['sigma_smooth'])

    kernel_erode = np.ones((param['erosion_size'], param['erosion_size']), np.uint8)
    kernel_dilate = np.ones((param['dilation_size'], param['dilation_size']), np.uint8)
    dist_uint8 = (dist * 255).astype(np.uint8)
    dist_uint8 = cv2.morphologyEx(dist_uint8, cv2.MORPH_OPEN, kernel_erode)
    dist_uint8 = cv2.morphologyEx(dist_uint8, cv2.MORPH_CLOSE, kernel_dilate)
    dist = dist_uint8 / 255.0

    if param['contrast_enhance']:
        dist = cv2.equalizeHist((dist * 255).astype(np.uint8)) / 255.0

    pMap = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)

    if param['verbose']:
        logger.info(f"[doMBS] 后处理耗时：{time.time() - start_post:.4f}s")
        logger.info(f"[doMBS] 总耗时：{time.time() - start_total:.4f}s")

    return pMap


def mbs_saliency(img):
    paramMBplus = getParam()
    paramMBplus['verbose'] = True
    I = img.copy()
    pMap = doMBS(I, paramMBplus)
    return pMap
