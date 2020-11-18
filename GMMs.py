import operator
import numpy as np
import time
import pickle
import mfcc_features as mf
from collections import defaultdict
from sklearn.cluster import KMeans
import warnings


class GMM:
    def __init__(self, max_iter=150, n_clusters=1, init_method='sample',
                 reg_covar = 1e-6, n_init = 1, tol = 1e-3, covariance_type='diag'):
        self.max_iter = max_iter
        self.converged = False
        self.init_method = init_method
        self.n_clusters = n_clusters
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.tol = tol
        self.covariance_type = covariance_type

    # def E_step(self, X):
    #     for _ in range(self.max_iter):

    # 平均分初始化
    def init_uniform(self, X):
        '''
        平均分初始化n_cluster个高斯分布
        返回各高斯分布的均值mean，diagonal方差variance，各部分比例prob, 每个样本的类别号label
        '''
        featnum_per_group = int(np.ceil(X.shape[0]/self.n_clusters))
        mean = np.zeros((self.n_clusters, X[0].shape[0]))
        variance = np.zeros((self.n_clusters, X[0].shape[0]))   # diagonal variances
        prob = np.zeros(self.n_clusters)
        label = np.zeros(X.shape[0])
        for i in range(self.n_clusters):
            mean[i] = np.sum(X[i*featnum_per_group: (i+1)*featnum_per_group], axis=0) / X[i*featnum_per_group: (i+1)*featnum_per_group].shape[0]
            variance[i] = np.sum((X[i*featnum_per_group: (i+1)*featnum_per_group] - mean[i]) * (X[i*featnum_per_group: (i+1)*featnum_per_group] - mean[i]), axis=0) / X[i*featnum_per_group: (i+1)*featnum_per_group].shape[0]
            prob[i] = X[i*featnum_per_group: (i+1)*featnum_per_group].shape[0] / X.shape[0]
            label[i*featnum_per_group: (i+1)*featnum_per_group] = i
        return mean, variance, prob, label

    def init_kmeans(self, X):
        '''
        输入样本X (n_samples, n_features)
        使用kmeans方法初始化均值、方差、权重
        '''
        print('Using Kmeans to Initialize...')
        label = KMeans(n_clusters=self.n_clusters, ).fit(X).labels_     # n_init=1,
        mean = np.zeros((self.n_clusters, X[0].shape[0]))
        variance = np.zeros((self.n_clusters, X[0].shape[0]))   # diagonal
        prob = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            # print(X[np.where(label==i)].shape)
            seg = X[np.where(label == i)]
            mean[i] = np.sum(seg, axis=0) / seg.shape[0]
            variance[i] = np.sum((seg - mean[i]) * (seg - mean[i]), axis=0) / seg.shape[0] + self.reg_covar
            prob[i] = seg.shape[0] / X.shape[0]
        # print(mean, variance, prob)
        return mean, variance, prob, label

    def _set_parameters(self, params):
        (self.pre_prob, self.mean, self.variance) = params

    def score(self, X):
        '''计算样本X属于该GMM模型的分值，'''
        if self.covariance_type == 'diag':
            mean2 = np.sum(self.mean ** 2 / self.variance, axis=1)[:, np.newaxis]
            X2 = np.dot(1 / self.variance, (X ** 2).T)
            X_mean = np.dot(self.mean / self.variance, X.T)
            var = np.prod(np.sqrt(self.variance), axis=1)[:, np.newaxis]
            gauss_prod_matrix = np.exp(-.5 * (mean2 + X2 - 2 * X_mean)) / (np.power((2 * np.pi), X[0].shape[0] / 2) * var)
            weighted_prob_matrix = self.pre_prob[:, np.newaxis] * gauss_prod_matrix  # p(h) * p(x|h)
            return np.mean(np.log(np.sum(weighted_prob_matrix, axis=0)))

    def fit(self, X):
        # loop BIC save and choose the best model
        # initialize
        # loop
        # e_step
        # m_step
        # initialize

        X = np.array(X)
        for init in range(self.n_init):
            # print('Initializing...')
            if self.init_method == 'sample':
                mean, variance, pre_prob, label = self.init_uniform(X)
            elif self.init_method == 'kmeans':
                mean, variance, pre_prob, label = self.init_kmeans(X)
            # print('Successfully initialized.')

            self.n_clusters = init + 1
            lower_bound = -np.infty
            max_lower_bound = -np.infty
            for n_iter in range(self.max_iter):
                prev_lower_bound = lower_bound

                # E step
                # 计算每个样本属于每个高斯分布的p(xi)
                if self.covariance_type == 'diag':
                    mean2 = np.sum(mean**2/variance, axis=1)[:, np.newaxis] # mean**2 / variance
                    X2 = np.dot(1/variance, (X**2).T)   # X**2 / variance
                    X_mean = np.dot(mean/variance, X.T)     # mean * X / variance
                    var = np.prod(np.sqrt(variance), axis=1)[:, np.newaxis]
                    gauss_prod_matrix = np.exp(-.5 * (mean2 + X2 - 2*X_mean)) / (np.power((2*np.pi), X[0].shape[0]/2) * var)

                weighted_prob_matrix = pre_prob[:, np.newaxis] * gauss_prod_matrix  # p(h) * p(x|h)
                weighted_prob_sum = np.sum(weighted_prob_matrix, axis=0)

                # 计算参数个数
                n_features = X[0].shape[0]
                if self.covariance_type == 'full':
                    cov_params = self.n_clusters * n_features * (n_features + 1) / 2
                elif self.covariance_type == 'diag':
                    cov_params = self.n_clusters * n_features
                mean_params = n_features * self.n_clusters
                n_params = cov_params + mean_params + self.n_clusters - 1

                lower_bound = 2 * np.mean(np.log(weighted_prob_sum)) * X.shape[0] - n_params * np.log(X.shape[0])

                # M step
                pos_prob_matrix = weighted_prob_matrix / weighted_prob_sum  # 每个样本属于每个类的后验概率p(h|x)
                pos_prob_sum = (np.sum(pos_prob_matrix, axis=1) + 10 * np.finfo(pos_prob_matrix.dtype).eps)[:, np.newaxis]

                mean = np.dot(pos_prob_matrix, X) / pos_prob_sum    # 均值
                if self.covariance_type == 'diag':
                    avg_X2 = np.dot(pos_prob_matrix, X * X) / pos_prob_sum
                    avg_means2 = mean ** 2
                    avg_X_means = mean * np.dot(pos_prob_matrix, X) / pos_prob_sum
                    variance = avg_X2 - 2 * avg_X_means + avg_means2 + self.reg_covar   # 方差
                pre_prob = np.sum(pos_prob_matrix, axis=1) / X.shape[0] # 更新先验概率

                change = lower_bound - prev_lower_bound
                if change < self.tol:
                    self.converged = True   # 收敛
                    break

            if lower_bound > max_lower_bound:   # 更新最优模型
                max_lower_bound = lower_bound
                best_params = (pre_prob, mean, variance)
                best_n_iter = n_iter
                best_n_clusters = self.n_clusters

        if not self.converged:
            print("Warning... Not converged...")

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound = max_lower_bound
        self.n_clusters = best_n_clusters

        # print("Best parameters: n_clusters = {}, lower_bound = {}".format(self.n_clusters, self.lower_bound))


    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        return (-2 * self.score(X) * X.shape[0] +
                self._n_parameters() * np.log(X.shape[0]))

# features = defaultdict(list)
# gmm = GMM()
# for k in range(10):
#     for i in range(1, 16):
#         path = 'records/digit_'+str(k) + '/' + str(i) + '_' + str(k) + '.wav'
#         feature = mf.lib_mfcc_feature(path)
#         features[k].extend(feature)
#     gmm.fit(features[k])
#     # feat = mf.lib_mfcc_feature('records/digit_'+str(k) + '/' + str(16) + '_' + str(k) + '.wav')
#     # print(gmm.score(feat))
#     break


class GMM_Set:
    def __init__(self, max_iter = 150):
        self.gmm_models = []    # 0-9 one model for one number
        self.label = [] # label
        self.max_iter = max_iter

    def add_new_model(self, features, label):
        '''添加GMM模型'''
        self.label.append(label)
        gmm = GMM()
        gmm.fit(features)
        self.gmm_models.append(gmm)

    def cal_gmm_score(self, gmm, input):
        return np.sum(gmm.score(input))

    def predict_num(self, input):
        '''对输入的音频特征input，预测其内容数字'''
        # print(len(input))
        scores = [self.cal_gmm_score(gmm, input) for gmm in self.gmm_models]  # log(p)
        result = [(self.label[index], value) for (index, value) in enumerate(scores)]
        p = max(result, key=operator.itemgetter(1))
        # print(p)
        return p[0]


class GMM_Model:
    def __init__(self, features):
        self.features = features
        self.gmms = GMM_Set()

    def train(self):
        start_time = time.time()
        for label, feature in self.features.items():
            try:
                self.gmms.add_new_model(feature, label)
            except Exception as e:
                print("%s failed" % (label))
        print("Model Training Time : {} seconds".format(time.time() - start_time))

    def dump(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, -1)

    # def predict(self, fs, signal):
    def predict(self, wav_path, method='Librosa'):
        try:
            if method == 'Speech_feature':
                feature = mf.get_mfcc_feature(wav_path)
            else:
                feature = mf.lib_mfcc_feature(wav_path)

        except Exception as e:
            print (e)
        return self.gmms.predict_num(feature)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            Model = pickle.load(f)
            return Model

