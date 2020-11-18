from sklearn.mixture import GaussianMixture
import operator
import numpy as np
import time
import pickle
import mfcc_features as mf

class GMM:

    def __init__(self, gmm_mixture_num = 1):
        self.gmm_models = []    # 0-9 one model for one number
        self.label = [] # label
        self.gmm_mixture_num = gmm_mixture_num

    def add_new_model(self, templates, label):
        self.label.append(label)
        gmm = GaussianMixture(self.gmm_mixture_num, covariance_type='diag')
        gmm.fit(templates)  # label自动从0开始标号
        self.gmm_models.append(gmm)

    def cal_gmm_score(self, gmm, input):
        # print(gmm.score(input))
        return np.sum(gmm.score(input))

    def predict_num(self, input):
        # print(len(input))
        scores = [self.cal_gmm_score(gmm, input) for gmm in self.gmm_models]  # log(p)
        # print(scores)
        # p = sorted(enumerate(scores), key=operator.itemgetter(1), reverse=True) # 按score降序
        # print(p)
        # p = [(str(self.label[i]), y, p[0][1] - y) for i, y in p]
        # print(p)
        result = [(self.label[index], value) for (index, value) in enumerate(scores)]
        # print(result)
        p = max(result, key=operator.itemgetter(1))
        # print(p)
        return p[0]


class GMM_Model_sklearn:

    def __init__(self, features):
        self.features = features
        self.gmms = GMM()

    def train(self):
        start_time = time.time()
        for label, feature in self.features.items():
            try:
                self.gmms.add_new_model(feature, label)
            except Exception as e:
                print("%s failed" % (label))
        print(time.time() - start_time, " seconds")

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