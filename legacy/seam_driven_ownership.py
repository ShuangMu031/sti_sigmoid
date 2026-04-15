import numpy as np
import cv2
from collections import deque

# =========================================================
# Seam-driven hard ownership assignment
# =========================================================

def extract_seam_barrier(label_map, overlap):
    """
    从 GraphCut 的 label_map 中提取 seam barrier。
    seam 定义为：overlap 内相邻像素 label 不同的位置。

    输出：
        barrier (H,W) bool，True 表示 seam 障碍
    """
    H, W = label_map.shape
    barrier = np.zeros((H, W), dtype=bool)

    for y in range(H):
        for x in range(W):
            if not overlap[y, x]:
                continue

            v = label_map[y, x]
            if x + 1 < W and overlap[y, x + 1] and label_map[y, x + 1] != v:
                barrier[y, x] = True
                barrier[y, x + 1] = True
            if y + 1 < H and overlap[y + 1, x] and label_map[y + 1, x] != v:
                barrier[y, x] = True
                barrier[y + 1, x] = True

    return barrier


# ---------------------------------------------------------
# Flood fill with seam as hard barrier
# ---------------------------------------------------------

def flood_fill_ownership(barrier, seed1, seed2, canvas_valid):
    H, W = barrier.shape
    ownership = np.full((H, W), 255, np.uint8)

    from collections import deque
    q = deque()

    # img1 seeds
    for y, x in zip(*np.where(seed1 & canvas_valid & (~barrier))):
        ownership[y, x] = 0
        q.append((y, x))

    # img2 seeds
    for y, x in zip(*np.where(seed2 & canvas_valid & (~barrier))):
        ownership[y, x] = 1
        q.append((y, x))

    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    while q:
        y, x = q.popleft()
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            if not canvas_valid[ny, nx]:
                continue
            if barrier[ny, nx]:
                continue
            if ownership[ny, nx] != 255:
                continue

            ownership[ny, nx] = ownership[y, x]
            q.append((ny, nx))

    return ownership

