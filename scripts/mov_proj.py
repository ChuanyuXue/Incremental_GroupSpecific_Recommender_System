from cluster_methods.missingness_cluster import miscluster
from cluster_methods.projection_cluster import projcluster
from model.model import gsrs
from tools.tools import *
from sys import path
import pandas as pd
import numpy as np
import time


data = pd.read_csv(r'../data/movielens_data.csv', header=None)
file = open(r"../exp/mv_proj.txt",'w')
file.write('cv'+','+'i'+','+'j'+','+'rmse'+','+'ucold'+','+'icold'+','+'runtime'+','+'group_time'+'\n')
i, j = 0.9, 0.9
data = data.sample(frac=1).reset_index(drop=True)
split_index = int(len(data) * 0.8)
train = data[:split_index].reset_index(drop=True)
test = data[split_index:].reset_index(drop=True)
starttime = time.time()
ug,ig = projcluster(train, (i,j))
grouptime = str(time.time() - starttime)
clusters = {}
clusters['uclusters'] = ug
clusters['iclusters'] = ig
paras = {'lambda':10, 'K':5, 'nstep':50, 'stopall':1e-3, 'stopone':1e-5, 'verbose':False}
test1 = gsrs(paras)
test1.train(train[[0,1]], train[2], clusters)
result = test1.predict(test[[0,1]])
uset = set(train[0])
iset = set(train[1])
ucold = rmse_evaluate(result[~test[0].isin(uset)], test[~test[0].isin(uset)][2])
icold = rmse_evaluate(result[~test[1].isin(iset)], test[~test[1].isin(iset)][2])
print(str(rmse_evaluate(test[2], result)))
file.write(str(cv) + ',' + str(i) + ',' + str(j) + ',' + str(rmse_evaluate(test[2], result)) + ',' + str(
    ucold) + ',' + str(icold)+','+str(test1.runtime)+','+str(grouptime) + '\n')
file.close()