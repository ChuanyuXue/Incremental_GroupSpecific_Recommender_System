from cluster_methods.missingness_cluster import miscluster
from cluster_methods.projection_cluster import projcluster
from model.model import gsrs
from tools.tools import *
from sys import path
import pandas as pd
import numpy as np
import time

data = pd.read_csv(r'../data/movielens_data.csv', header=None)

data = data.sample(frac=1).reset_index(drop=True)
split_index = int(len(data) * 0.8)
train = data[:split_index].reset_index(drop=True)
test = data[split_index:].reset_index(drop=True)

ug,ig = miscluster(train, 10)
clusters = {}
clusters['uclusters'] = ug
clusters['iclusters'] = ig

paras = {'lambda':10, 'K':300, 'nstep':50, 'stopall':1e-3, 'stopone':1e-5, 'verbose':False, 'bias': False}
test1 = gsrs(paras)
test1.train(train[[0,1]], train[2], clusters)
result = np.array(test1.predict(test[[0,1]]))
print(str(ucold), str(icold), str(rmse_evaluate(test[2], result)), str(rmse_evaluate(test[2], result_1)))
