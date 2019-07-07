import numpy as np
import pandas as pd


def miscluster(x: pd.DataFrame, group_nums: int)->tuple:
    cu = x.groupby(0, as_index=False).count()
    cpu = np.percentile(cu[2],  np.linspace(0, group_nums, group_nums+1) * 100 / group_nums)
    ci = x.groupby(1, as_index=False).count()
    cpi = np.percentile(ci[2],  np.linspace(0, group_nums, group_nums+1) * 100 / group_nums)
    ug = [list(cu[(cu[2] < cpu[i+1]) & (cu[2] >= cpu[i])][0]) for i in range(group_nums)
          if list(cu[(cu[2] < cpu[i+1]) & (cu[2] >= cpu[i])][0]) != []]
    for i in list(cu[cu[2] == cpu[group_nums]][0]):
        ug[-1].append(i)
    ig = [list(ci[(ci[2] < cpi[i+1]) & (ci[2] >= cpi[i])][1]) for i in range(group_nums)
          if list(ci[(ci[2] < cpi[i+1]) & (ci[2] >= cpi[i])][1]) != []]
    for i in list(ci[ci[2] == cpi[group_nums]][1]):
        ig[-1].append(i)
    return ug, ig
