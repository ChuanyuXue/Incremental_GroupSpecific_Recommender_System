import numpy as np
from core.lse_unit import als


def incre_lse(x: np.ndarray, y: np.ndarray
              , p: np.ndarray, q: np.ndarray, s: np.ndarray, t: np.ndarray, paras: dict):
    nstep = paras['nstep']
    lamb = paras['lambda']
    stop_all = paras['stop_all']

    maxs = x[:, 2].max() + 1
    maxt = x[:, 3].max() + 1

    diff_all = 999
    step = 0

    s_copy = s.copy()
    t_copy = t.copy()

    while diff_all > stop_all and step != nstep:
        old_s = s_copy.copy()
        old_t = t_copy.copy()
        ifactor = q[x[:, 1]] + t_copy[x[:, 3]]
        theta = y - np.sum(np.multiply(p[x[:, 0]], ifactor), axis=1)
        for v in range(maxs):
            ybag = theta[x[:, 2] == v]
            xbag = ifactor[x[:, 2] == v]
            s_copy[v] = als(ybag, xbag, lamb)

        ufactor = p[x[:, 0]] + s_copy[x[:, 2]]
        theta = y - np.sum(np.multiply(q[x[:, 1]], ufactor), axis=1)
        for j in range(maxt):
            ybag = theta[x[:, 3] == j]
            xbag = ufactor[x[:, 3] == j]
            t_copy[j] = als(ybag, xbag, lamb)
        diff_all = np.linalg.norm(old_s - s_copy) + np.linalg.norm(old_t - t_copy)
        step += 1

    # eli = 0.5
    # s = (1 - eli) * s + eli * s_copy
    # t = (1 - eli) * t + eli * t_copy

    return p, q, s, t, np.zeros(len(p)), np.zeros(len(q)), 0
