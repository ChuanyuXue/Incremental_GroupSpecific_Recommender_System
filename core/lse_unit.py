import numpy as np


def als(y: np.ndarray, x: np.ndarray, lam: float)-> np.ndarray:
    beta = np.zeros(x.shape[-1])
    if x.shape[0] == 0:
        return beta
    else:
        xp = np.mat(np.dot(np.transpose(x), x) + np.eye(np.shape(x)[1]) * lam)
        k = np.linalg.matrix_rank(xp)

        if k < np.shape(xp)[0]:
            beta = np.dot(np.linalg.pinv(xp), np.dot(np.transpose(x), y))
        else:
            beta = np.dot(xp.I, np.dot(np.transpose(x), y))
        return beta
