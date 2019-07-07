import numpy as np
import pandas as pd
from typing import *


def group_to_index(x: Iterable, groups: list)->int:
    index = [-1] * (max(max([a for b in groups for a in b])+1, max(x)+1))
    for i, group in enumerate(groups):
        for k in group:
            index[k] = i
    return np.array(index)[x]


def scale_by_half_acc(arr: np.ndarray)->np.ndarray:
    arr_ = arr.copy().astype(np.float64)
    key = np.unique(arr)
    key = key[key != 0]
    freq = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        freq[k] = v

    pr = np.array(list(freq.values())) / len(arr[arr != 0])

    pr_acc = pr.copy()
    for i in range(len(pr_acc) - 1):
        pr_acc[i + 1] = pr_acc[i] + pr_acc[i + 1]

    pr_half_acc = pr_acc - pr / 2

    pr_dict = dict(zip(key, pr_half_acc))
    for k in key:
        arr_[np.where(arr_ == k)] = pr_dict[k]
    return arr_


def rmse_evaluate(y: np.ndarray, yhat: Iterable)->float:
    return np.sqrt(np.mean(np.square(y-yhat)))


def matrix_to_tuplelist(mat: np.ndarray)->list:
    (x_i, y_i) = np.where(mat != 0)
    return [(str(x_i[i]), str(y_i[i]), mat[x_i[i]][y_i[i]]) for i in range(len(x_i)) if x_i[i] != y_i[i]]


def find_value_in_2dlist(value: int, listt: list)->int:
    for index, g in enumerate(listt):
        if value in g:
            return index
    return -1


def group_to_dict(group: list)->dict:
    result = {}
    for index, g in enumerate(group):
        for ins in g:
            result[ins] = index
    return result


def data_to_matrix(data: Union[np.ndarray, pd.DataFrame])->np.ndarray:
    umax, imax = max(data[0])+1, max(data[1])+1
    matrix = np.zeros(shape=(umax, imax))
    matrix[data[0], data[1]] = data[2]
    return matrix


def flatten(list2d: list)->list:
    return [a for b in list2d for a in b]

def extent_factor(factor: np.ndarray, lens: int)->np.ndarray:
    result = np.zeros(shape=(lens + 1, factor.shape[1]))
    result[0:factor.shape[0], 0:factor.shape[1]] = factor
    return result
