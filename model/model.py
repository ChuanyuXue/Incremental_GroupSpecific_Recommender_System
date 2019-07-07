import pandas as pd
from core.lse import *
from cluster_methods.projection_cluster import *
from cluster_methods.update_cluster import *
from core.incre_lse import *
import time


class gsrs():
    def __init__(self, paras=None, verbose=False):
        np.random.seed(1024)
        if not paras:
            paras = {'lambda': 10, 'K': 5, 'nstep': 50, 'stopall': 1e-3, 'stopone': 1e-5, 'bias': False}
        self.lamb = paras['lambda'] if 'lambda' in paras else 10
        self.K = paras['K'] if 'K' in paras else 5
        self.nstep = paras['nstep'] if 'nstep' in paras else 50
        self.stopall = paras['stopall'] if 'stopall' in paras else 1e-3
        self.stopone = paras['stopone'] if 'stopone' in paras else 1e-5
        self.bias = paras['bias'] if 'bias' in paras else False
        self.verbose = verbose

        self.p = None
        self.q = None
        self.s = None
        self.t = None

        self.global_bias = None
        self.user_bias = None
        self.item_bias = None

        self.ucluster = None
        self.icluster = None

        self.data = None
        self.user_set = None
        self.item_set = None

        self.runtime = None

    def train(self, train_set, label, clusters, monitor=None)->None:
        starttime = time.time()
        self.ucluster = uclusters = clusters['uclusters'] if 'uclusters' in clusters else\
            print("Fatal Err No User Clusters!")
        self.icluster = iclusters = clusters['iclusters'] if 'iclusters' in clusters else\
            print("Fatal Err No Item Clusters!")
        x = np.c_[np.array(train_set), group_to_index(train_set[0], uclusters),
                  group_to_index(train_set[1], iclusters)]

        self.data = pd.concat([train_set, label], axis=1)

        if monitor:
            xv = monitor[0]
            yv = monitor[1]
            self.data = pd.concat([self.data, pd.concat([xv, yv], axis=1)])
            (ug, ig) = self.cluster_for_colds([xv[xv[0] > max(x[:, 0])][0], xv[xv[1] > max(x[:, 1])][1]])
            val_x = np.c_[np.array(xv), group_to_index(xv[0], ug), group_to_index(xv[1], ig)]
            val_y = yv
            val = [val_x, val_y]
        else:
            val = None

        paras = dict()
        paras['nstep'] = self.nstep
        paras['lambda'] = self.lamb
        paras['stop_all'] = self.stopall
        paras['stop_one'] = self.stopone
        paras['verbose'] = self.verbose
        paras['bias'] = self.bias

        (self.p, self.q, self.s, self.t, self.user_bias, self.item_bias, self.global_bias)\
            = lse(x=x, y=np.array(label), k=self.K, paras=paras, val=val)

        if self.verbose:
            print("The training process has been successfully completed!")
        self.user_set = set(self.data[0])
        self.item_set = set(self.data[1])
        self.runtime = str(time.time() - starttime)


    def increase(self, train_set, label, clusters, monitor=None)->None:
        starttime = time.time()
        train_data = pd.concat([train_set, label], axis=1)
        self.data = pd.concat([self.data.dropna(), train_data]).reset_index(drop=True)

        x = np.c_[np.array(self.data[[0, 1]]), group_to_index(self.data[0], clusters['uclusters']),
                  group_to_index(self.data[1], clusters['iclusters'])]
        y = np.array(self.data[2])

        self.p = extent_factor(self.p, max(x[:, 0]))
        self.q = extent_factor(self.q, max(x[:, 1]))
        self.s = extent_factor(self.s, max(flatten(clusters['uclusters'])))
        self.t = extent_factor(self.t, max(flatten(clusters['iclusters'])))

        paras = dict()
        paras['nstep'] = self.nstep
        paras['lambda'] = self.lamb
        paras['stop_all'] = self.stopall
        paras['stop_one'] = self.stopone
        self.p, self.q, self.s, self.t, self.user_bias, self.item_bias, self.global_bias =\
        incre_lse(x, y,
                  self.p, self.q, self.s, self.t, paras
                  )

        self.runtime = str(time.time() - starttime)

    def cluster_for_colds(self, colds)->tuple:
        return (cold_update(self.data.groupby(0).count()[2], self.ucluster, colds[0]),
                cold_update(self.data.groupby(1).count()[2], self.icluster, colds[1]))

    def predict(self, test)->np.ndarray:
        self.data = pd.concat([self.data, test]).reset_index(drop=True)
        umax, imax = (len(self.p) - 1, len(self.q) - 1)
        ug, ig = self.cluster_for_colds([np.unique(test[~test[0].isin(flatten(self.ucluster))][0]),
                                         np.unique(test[~test[1].isin(flatten(self.icluster))][1])])

        x = np.c_[np.array(test), group_to_index(test[0], ug),
                  group_to_index(test[1], ig)]
        x[x[:, 2] >= len(self.s), 2] = 0
        x[x[:, 3] >= len(self.t), 3] = 0
        results = []
        for row in x:
            result = self.global_bias
            result += np.sum(np.multiply(self.s[row[2]], self.t[row[3]]))
            if row[0] <= umax:
                result += np.sum(np.multiply(self.p[row[0]], self.t[row[3]]))
                result += self.user_bias[row[0]]
            else:
                result += np.mean(self.user_bias)
            if row[1] <= imax:
                result += np.sum(np.multiply(self.s[row[2]], self.q[row[1]]))
                result += self.item_bias[row[1]]
            else:
                result += np.mean(self.item_bias)
            if row[0] <= umax and row[1] <= imax:
                result += np.sum(np.multiply(self.p[row[0]], self.q[row[1]]))
            results.append(result)
        return np.array(results)

    def score(self, data, label)->float:
        return rmse_evaluate(label.reshape(-1, 1), np.array(self.predict(data)).reshape((-1, 1)))
