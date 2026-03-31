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

    h, w = msk.shape
    N = h * w
    idx = np.arange(N).reshape(h, w)

    rows, cols, data = [], [], []
    b = np.zeros((N, 3), np.float64)

    def add(i, j, v):
        rows.append(i)
        cols.append(j)
        data.append(v)

    for y in range(h):
        for x in range(w):
            p = idx[y, x]

            if msk[y, x]:
                add(p, p, 4)

                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y + dy, x + dx

                    if 0 <= ny < h and 0 <= nx < w:
                        q = idx[ny, nx]
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
                        b[p] += tgt[y, x]
            else:
                add(p, p, 1)
                b[p] = tgt[y, x]

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

    out = np.zeros((h, w, 3), np.float64)
    for c in range(3):
        sol = splinalg.spsolve(A, b[:, c])
        out[..., c] = sol.reshape(h, w)

    result = target.copy()
    result[y0:y1+1, x0:x1+1] = np.clip(out, 0, 255)

    return result.astype(np.uint8)
