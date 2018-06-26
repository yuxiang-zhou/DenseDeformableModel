import sys, glob, argparse
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm
from sklearn.mixture import GaussianMixture
import menpo.io as mio


class FisherVector:
    def __init__(self, data, K=128):
        self.gmm = GaussianMixture(n_components=K, covariance_type='full')
        self.gmm.fit(data)
        self.K = K
        self._inv_covs = []

    #     for cov in self.gmm.covariances_:
    #         self._inv_covs.append(np.linalg.inv(cov))
    #
    # def _predict_proba(self, img):
    #     w = self.gmm.weights_
    #     mu = self.gmm.means_
    #     sig = np.array([np.diag(cov) for cov in self.gmm.covariances_])
    #
    #     probs = []
    #     for i in img:
    #         soft_w = []
    #         for k in range(self.K):
    #             soft_w.append(np.exp(0.5*((i-mu[k]).T.dot(self._inv_covs[k]).dot(i-mu[k])))/np.sum([0.5*((i-m).T.dot(ic).dot(i-m)) for m,ic in zip(mu, self._inv_covs)]))
    #         probs.append(soft_w)
    #
    #     return probs


    def generate(self,img, normalise=False):
        if not type(img) == list:
            img = [img]

        N = len(img)
        a = self.gmm.predict_proba(img)

        w = self.gmm.weights_
        mu = self.gmm.means_
        sig = np.array([np.diag(cov) for cov in self.gmm.covariances_])

        fv = []
        for k in range(self.K):
            fd = 1 / (N*np.sqrt(w[k])) * np.sum([a[p][k] * ((img[p] - mu[k]) / sig[k]) for p in range(N)], axis=0)
            sd = 1 / (N*np.sqrt(2*w[k])) * np.sum([a[p][k] * (np.power((img[p] - mu[k]) / sig[k], 2) - 1) for p in range(N)], axis=0)

            fv.append(fd)
            fv.append(sd)

        fv = np.concatenate(fv)
        if normalise:
            v = np.sqrt(abs(fv)) * np.sign(fv)
            fv = v / np.sqrt(np.dot(v, v))

        return fv
