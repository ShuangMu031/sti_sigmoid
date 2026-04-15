import cv2
import numpy as np


def warp_with_fallback(img, H, output_shape, order=1):
    if img is None or H is None:
        return None

    h, w = output_shape
    flags = cv2.INTER_NEAREST if order == 0 else cv2.INTER_LINEAR
    return cv2.warpPerspective(
        img, H, (w, h),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )



def homography_align(
    img1, pmap1, edge1,
    img2, pmap2, edge2,
    H
):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners = np.array([
        [0, 0, 1],
        [w1, 0, 1],
        [0, h1, 1],
        [w1, h1, 1]
    ]).T

    warped = H @ corners
    warped = warped[:2] / warped[2]

    min_x, min_y = np.floor(warped.min(axis=1)).astype(int)
    max_x, max_y = np.ceil(warped.max(axis=1)).astype(int)

    off_x = max(0, -min_x)
    off_y = max(0, -min_y)

    canvas_w = max(max_x + off_x, w2 + off_x)
    canvas_h = max(max_y + off_y, h2 + off_y)

    T = np.array([
        [1, 0, off_x],
        [0, 1, off_y],
        [0, 0, 1]
    ])

    H_shift = T @ H

    img1_w = warp_with_fallback(img1, H_shift, (canvas_h, canvas_w))
    pmap1_w = warp_with_fallback(pmap1, H_shift, (canvas_h, canvas_w))
    edge1_w = warp_with_fallback(edge1.astype(np.uint8), H_shift, (canvas_h, canvas_w), order=0) > 0

    # 显式 warp 全 1 mask，避免把真实黑色像素误判成无效区域
    mask1 = np.ones((h1, w1), dtype=np.uint8)
    valid1 = warp_with_fallback(mask1, H_shift, (canvas_h, canvas_w), order=0) > 0

    img2_c = np.zeros_like(img1_w)
    pmap2_c = np.zeros_like(pmap1_w)
    edge2_c = np.zeros_like(edge1_w)
    valid2 = np.zeros((canvas_h, canvas_w), dtype=bool)

    img2_c[off_y:off_y+h2, off_x:off_x+w2] = img2
    pmap2_c[off_y:off_y+h2, off_x:off_x+w2] = pmap2
    edge2_c[off_y:off_y+h2, off_x:off_x+w2] = edge2
    valid2[off_y:off_y+h2, off_x:off_x+w2] = True

    return (
        img1_w, img2_c,
        pmap1_w, pmap2_c,
        edge1_w, edge2_c,
        valid1, valid2
    )
