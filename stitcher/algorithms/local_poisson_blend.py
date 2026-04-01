import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from stitcher.config import POISSON_USE_FACTORIZED, POISSON_NEIGHBOR_MODE


def _build_mask_index(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None, None, 0

    N = len(ys)
    idx_map = -np.ones(mask.shape, dtype=np.int32)
    idx_map[mask] = np.arange(N)
    
    coords = np.column_stack((ys, xs))
    return idx_map, coords, N


def _assemble_poisson_system(source, target, mask, idx_map, coords, N):
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    
    rows, cols, data = [], [], []
    b = np.zeros((N, 3), np.float64)

    def add(i, j, v):
        rows.append(i)
        cols.append(j)
        data.append(v)

    for i, (y, x) in enumerate(coords):
        add(i, i, 4)

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx

            j = idx_map[ny, nx] if (0 <= ny < idx_map.shape[0] and 0 <= nx < idx_map.shape[1]) else -1
            
            if j != -1:
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
                if POISSON_NEIGHBOR_MODE == "target_neighbor":
                    if 0 <= ny < target.shape[0] and 0 <= nx < target.shape[1]:
                        b[i] += target[ny, nx]
                    else:
                        b[i] += target[y, x]
                else:
                    b[i] += target[y, x]

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return A, b


def _solve_poisson_channels(A, b, N):
    result_channels = []
    
    if POISSON_USE_FACTORIZED:
        solver = splinalg.factorized(A.tocsc())
        for c in range(3):
            sol = solver(b[:, c])
            result_channels.append(sol)
    else:
        for c in range(3):
            sol = splinalg.spsolve(A, b[:, c])
            result_channels.append(sol)
    
    return result_channels


def _write_back_solution(result, coords, sol_channels):
    for c in range(3):
        sol = sol_channels[c]
        for i, (y, x) in enumerate(coords):
            result[y, x, c] = np.clip(sol[i], 0, 255)
    return result


def gradient_blend_local(source, target, mask):
    idx_map, coords, N = _build_mask_index(mask)
    
    if N == 0:
        return target.astype(np.uint8)
    
    A, b = _assemble_poisson_system(source, target, mask, idx_map, coords, N)
    sol_channels = _solve_poisson_channels(A, b, N)
    
    result = target.copy()
    result = _write_back_solution(result, coords, sol_channels)
    
    return result.astype(np.uint8)
