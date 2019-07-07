from cluster_methods.missingness_cluster import miscluster
from cluster_methods.projection_cluster import projcluster
from model.model import gsrs
from tools.tools import *
from sys import path
import pandas as pd
import numpy as np
import time

data = pd.read_csv(r'./data/movielens_data.csv', header=None)
file = open(r"./exp/movielens_orig.txt",'w')
file.write('cv'+','+'rmse'+','+'ucold'+','+'icold'+','+'runtime'+'\n')

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import SVD


# Creation of the dataframe. Column names are irrelevant.
df = data

reader = Reader(line_format='user item rating', sep=',')
# The columns must correspond to user id, item id and ratings (in that order).
data1 = Dataset.load_from_df(df[[0,1,2]], reader)

# We can now use this dataset as we please, e.g. calling cross_validate
print(cross_validate(SVD(biased=False), data1, cv=5))