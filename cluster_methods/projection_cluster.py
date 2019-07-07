import igraph
import numpy as np
import pandas as pd
import copy
from tools.tools import matrix_to_tuplelist
from tools.tools import scale_by_half_acc
from cluster_methods.update_cluster import cold_update


def proj_unit(list_edge: list)->list:
    if len(list_edge) == 0:
        return []
    if len(list_edge) == 1:
        return [[int(list_edge[0][0])]]
    g = igraph.Graph.TupleList(list_edge, directed=False, vertex_name_attr='name', edge_attrs=None, weights=True)
    clusters = g.clusters().giant().community_spinglass()

    cluster = []
    for i in clusters.subgraphs():
        cluster.append([int(x) for x in i.vs["name"]])

    tri_clusters = []
    for i in g.clusters().subgraphs():
        tri_clusters.append([int(x) for x in i.vs["name"]])
    del tri_clusters[[len(x) for x in tri_clusters].index(max([len(x) for x in tri_clusters]))]
    cluster.append([x for y in tri_clusters for x in y])
    return [x for x in cluster if len(x) != 0]


def get_similarity_matrix(x: pd.DataFrame, threshold: tuple)->tuple:
    ratings = np.zeros(shape=(x[0].max() + 1, x[1].max() + 1))
    ratings[x[0], x[1]] = x[2]

    ratings_copy = copy.deepcopy(ratings)
    for i in range(ratings.shape[0]):
        ratings_copy[i] = scale_by_half_acc(ratings[i])
    h1 = np.nan_to_num(np.corrcoef(ratings_copy))
    h1[(h1 >= -threshold[0]) & (h1 <= threshold[0])] = 0

    ratings_copy = copy.deepcopy(ratings)
    for i in range(ratings.shape[1]):
        ratings_copy[:, i] = scale_by_half_acc(ratings[:, i])
    h2 = np.nan_to_num(np.corrcoef(ratings_copy.T))
    h2[(h2 >= -threshold[1]) & (h2 <= threshold[1])] = 0

    return h1, h2


def projcluster(x: pd.DataFrame, threshold=(0.9, 0.9))->tuple:
    su, si = get_similarity_matrix(x, threshold=threshold)
    ug = proj_unit(matrix_to_tuplelist(su))
    ucold = list(x[~x[0].isin(set([a for b in ug for a in b]))][0])
    ig = proj_unit(matrix_to_tuplelist(si))
    icold = list(x[~x[1].isin(set([a for b in ig for a in b]))][1])
    return cold_update(x.groupby(0).count()[2], ug, ucold), \
           cold_update(x.groupby(1).count()[2], ig, icold)
