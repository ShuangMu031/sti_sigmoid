import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg


def gradient_blend_local(source, target, mask):
    source = source.astype(np.float64)
    target = target.astype(np.float64)

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return target.astype(np.uint8)

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    src = source[y0:y1+1, x0:x1+1]
    tgt = target[y0:y1+1, x0:x1+1]
    msk = mask[y0:y1+1, x0:x1+1]

    # 只处理 mask 内的像素
    mask_ys, mask_xs = np.where(msk)
    N = len(mask_ys)
    
    if N == 0:
        return target.astype(np.uint8)
    
    # 创建像素到索引的映射
    pixel_to_idx = {}  # (y, x) -> idx
    for i, (y, x) in enumerate(zip(mask_ys, mask_xs)):
        pixel_to_idx[(y, x)] = i

    rows, cols, data = [], [], []
    b = np.zeros((N, 3), np.float64)

    def add(i, j, v):
        rows.append(i)
        cols.append(j)
        data.append(v)

    h, w = msk.shape
    
    for i, (y, x) in enumerate(zip(mask_ys, mask_xs)):
        p = i
        add(p, p, 4)

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx

            if 0 <= ny < h and 0 <= nx < w:
                if (ny, nx) in pixel_to_idx:
                    # 邻居在 mask 内，建立连接
                    q = pixel_to_idx[(ny, nx)]
                    add(p, q, -1)

                    gs = src[y, x] - src[ny, nx]
                    gt = tgt[y, x] - tgt[ny, nx]

                    gs2 = np.dot(gs, gs)
                    gt2 = np.dot(gt, gt)

                    if gs2 > 2 * gt2:
                        grad = gs
                    else:
                        grad = gt

                    b[p] += grad
                else:
                    # 邻居在 mask 外，使用目标图像值
                    b[p] += tgt[ny, nx]
            else:
                # 边界情况，使用目标图像值
                b[p] += tgt[y, x]

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

    # 求解
    out = tgt.copy()
    for c in range(3):
        sol = splinalg.spsolve(A, b[:, c])
        # 将解映射回原位置
        for i, (y, x) in enumerate(zip(mask_ys, mask_xs)):
            out[y, x, c] = sol[i]

    result = target.copy()
    result[y0:y1+1, x0:x1+1] = np.clip(out, 0, 255)

    return result.astype(np.uint8)
