import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg


def gradient_blend_local(source, target, mask):
    source = source.astype(np.float64)
    target = target.astype(np.float64)

    # 找到mask内的所有像素坐标
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return target.astype(np.uint8)

    # 创建像素到索引的映射
    pixel_to_idx = {}
    idx_to_pixel = []
    for i, (y, x) in enumerate(zip(ys, xs)):
        pixel_to_idx[(y, x)] = i
        idx_to_pixel.append((y, x))

    N = len(idx_to_pixel)
    rows, cols, data = [], [], []
    b = np.zeros((N, 3), np.float64)

    def add(i, j, v):
        rows.append(i)
        cols.append(j)
        data.append(v)

    # 对每个mask内的像素构建方程
    for i, (y, x) in enumerate(idx_to_pixel):
        add(i, i, 4)

        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx

            if (ny, nx) in pixel_to_idx:
                # 邻居也在mask内
                j = pixel_to_idx[(ny, nx)]
                add(i, j, -1)

                gs = source[y, x] - source[ny, nx]
                gt = target[y, x] - target[ny, nx]

                gs2 = np.dot(gs, gs)
                gt2 = np.dot(gt, gt)

                if gs2 > 2 * gt2:
                    grad = gs
                else:
                    grad = gt

                b[i] += grad
            else:
                # 邻居不在mask内，使用target的值
                b[i] += target[y, x]

    # 构建稀疏矩阵
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

    # 求解线性系统
    result = target.copy()
    for c in range(3):
        sol = splinalg.spsolve(A, b[:, c])
        for i, (y, x) in enumerate(idx_to_pixel):
            result[y, x, c] = np.clip(sol[i], 0, 255)

    return result.astype(np.uint8)
