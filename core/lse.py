from core.lse_unit import als
from tools.tools import *


def lse(x: np.ndarray, y: np.ndarray, k: int, paras: dict, val=None)->tuple:
    maxp = max(x[:, 0].max(), val[0][:, 0].max()) + 1 if val else x[:, 0].max() + 1
    maxq = max(x[:, 1].max(), val[0][:, 1].max()) + 1 if val else x[:, 1].max() + 1
    maxs = max(x[:, 2].max(), val[0][:, 2].max()) + 1 if val else x[:, 2].max() + 1
    maxt = max(x[:, 3].max(), val[0][:, 3].max()) + 1 if val else x[:, 3].max() + 1

    p = np.array(np.random.normal(loc=0, scale=0.01, size=(maxp, k)), dtype='float64')
    q = np.array(np.random.normal(loc=0, scale=0.01, size=(maxq, k)), dtype='float64')
    s = np.array(np.random.normal(loc=0, scale=0.01, size=(maxs, k)), dtype='float64')
    t = np.array(np.random.normal(loc=0, scale=0.01, size=(maxt, k)), dtype='float64')

    user_bias = np.array(np.zeros(maxp), dtype='float64')
    item_bias = np.array(np.zeros(maxq), dtype='float64')
    global_bias = np.mean(y) if paras['bias'] else 0

    nstep = paras['nstep']
    lamb = paras['lambda']
    stop_all = paras['stop_all']
    stop_one = paras['stop_one']
    verbose = paras['verbose']
    have_bias = paras['bias']

    (p_old, q_old, s_old, t_old) = (p.copy(), q.copy(), s.copy(), t.copy())

    diff_all = 999
    step = 0
    old_loss = 999
    while diff_all > stop_all and step != nstep:
        if have_bias:
            for index, row in enumerate(x):
                err = y[index] - (np.sum((p[row[0]] + s[row[2]]) * (q[row[1]] + t[row[3]])) +
                         user_bias[row[0]] + item_bias[row[1]] + global_bias)
                user_bias[row[0]] += 0.1 * (err - lamb * user_bias[row[0]])
                item_bias[row[0]] += 0.1 * (err - lamb * item_bias[row[0]])
            bias = user_bias[x[:, 0]] + item_bias[x[:, 1]] + global_bias
        else:
            bias = 0

        diff_u = 999
        ifactor = q[x[:, 1]] + t[x[:, 3]]
        (p_new, s_new) = (p.copy(), s.copy())
        while diff_u > stop_one:
            theta = y - np.sum(np.multiply(s_new[x[:, 2]], ifactor), axis=1) - bias
            for u in range(maxp):
                ybag = theta[x[:, 0] == u]
                xbag = ifactor[x[:, 0] == u]
                p_new[u] = als(ybag, xbag, lamb)

            theta = y - np.sum(np.multiply(p_new[x[:, 0]], ifactor), axis=1) - bias
            for v in range(maxs):
                ybag = theta[x[:, 2] == v]
                xbag = ifactor[x[:, 2] == v]
                s_new[v] = als(ybag, xbag, lamb)

            diff_u = np.sum(np.sum(np.square(p_new - p))) / (k * maxp) + \
                     np.sum(np.sum(np.square(s_new - s))) / (k * maxs)
            (p, s) = (p_new.copy(), s_new.copy())
        if verbose:
            print('Users\' factors has iterated in ------ Step: ', step)

        diff_i = 999
        ufactor = p[x[:, 0]] + s[x[:, 2]]
        (q_new, t_new) = (q.copy(), t.copy())
        while diff_i > stop_one:
            theta = y - np.sum(np.multiply(t_new[x[:, 3]], ufactor), axis=1) - bias
            for i in range(maxq):
                ybag = theta[x[:, 1] == i]
                xbag = ufactor[x[:, 1] == i]
                q_new[i] = als(ybag, xbag, lamb)

            theta = y - np.sum(np.multiply(q_new[x[:, 1]], ufactor), axis=1) - bias
            for j in range(maxt):
                ybag = theta[x[:, 3] == j]
                xbag = ufactor[x[:, 3] == j]
                t_new[j] = als(ybag, xbag, lamb)

            diff_i = np.sum(np.sum(np.square(q_new - q))) / (k * maxq) + \
                     np.sum(np.sum(np.square(t_new - t))) / (k * maxt)
            (q, t) = (q_new.copy(), t_new.copy())
        if verbose:
            print('Items\' factors has iterated in ------ Step: ', step)

        diff_all = np.sum(np.sum(np.square(p[x[:, 0]] + s[x[:, 2]]
                                           - p_old[x[:, 0]] - s_old[x[:, 2]]))) / (k * len(x)) + \
            np.sum(np.sum(np.square(q[x[:, 1]] + t[x[:, 3]] - q_old[x[:, 1]]
                                    - t_old[x[:, 3]]))) / (k * len(x))
        (p_old, q_old, s_old, t_old) = (p, q, s, t)
        if verbose:
            print("Improvement is:", diff_all, '------', 'Step Now:', step)

        if val:
            (Xt, yt) = (val[0], val[1])
            val_hat = np.sum(np.multiply(p[Xt[:, 0]] + s[Xt[:, 2]], q[Xt[:, 1]] + t[Xt[:, 3]]), axis=1)
            loss = rmse_evaluate(yt, val_hat)
            if verbose:
                print("The loss on the valuation set is:", loss, '------', 'Step Now:', step)
            if loss < old_loss:
                old_loss = loss
            else:
                break

        if verbose:
            print('\n')
        step += 1
    return p, q, s, t, user_bias, item_bias, global_bias
