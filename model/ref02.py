from tools.tools import *
import time


class Ref02:
    def __init__(self, k: int = 5):
        self.u_mat = None
        self.s_mat = None
        self.s_inv = None
        self.v_mat = None
        self.data = None
        self.k = k
        self.runtime = None

    def train(self, train_set: pd.DataFrame, label: pd.DataFrame)->None:
        starttime = time.time()
        self.data = pd.concat([train_set, label], axis=1)
        train_matrix = data_to_matrix(self.data)
        self.u_mat, self.s_mat, self.v_mat = np.linalg.svd(train_matrix, full_matrices=False)
        self.u_mat, self.v_mat = self.u_mat[:, :self.k], self.v_mat[:self.k, :].T
        self.s_mat = np.diag(self.s_mat[:self.k])
        self.s_inv = np.linalg.pinv(self.s_mat)
        self.runtime = str(time.time() - starttime)

    def increase(self, train_set: pd.DataFrame, label: pd.DataFrame)->None:
        starttime = time.time()
        train_data = pd.concat([train_set, label], axis=1)
        new_user_data, new_item_data = train_data[~train_data[0].isin(np.unique(self.data[0]))],\
            train_data[~train_data[1].isin(np.unique(self.data[1]))]
        try:
            self.u_mat = extent_factor(self.u_mat, max(new_user_data[0]))
            self.v_mat = extent_factor(self.v_mat, max(new_item_data[1]))
        except Exception:
            pass
        if None is self.data:
            self.data = pd.concat([self.data, pd.concat([self.data, train_data])])
        else:
            self.data = pd.concat([self.data, train_data])
        a_mat = data_to_matrix(self.data)
        a_mat_t = a_mat.T

        for index, row in new_user_data.iterrows():
            self.u_mat[int(row[0])] = np.dot(np.dot(a_mat[int(row[0])], self.v_mat),
                                        self.s_inv)
        for index, row in new_item_data.iterrows():
            self.v_mat[int(row[1])] = np.dot(np.dot(a_mat_t[int(row[1])], self.u_mat),
                                        self.s_inv)
        self.runtime = str(time.time() - starttime)

    def predict(self, test_set: pd.DataFrame)->Iterable:
        result_list = []
        u_set, i_set = set(self.data[0]), set(self.data[1])
        global_mean = np.mean(self.data[2])
        for index, row in test_set.iterrows():
            if row[0] in u_set and row[1] in i_set:
                result = np.matmul(np.matmul(self.u_mat[int(row[0])], self.s_mat), self.v_mat[int(row[1])])
            else:
                result = global_mean
            result_list.append(result)
        return result_list
