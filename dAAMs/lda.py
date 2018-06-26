from numpy import linalg
from menpo.visualize import print_dynamic
from sklearn.utils.fixes import bincount
import itertools
import functools
import warnings
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np


def _class_means(X, y):
    """Compute class means.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target values.
    Returns
    -------
    means : array-like, shape (n_features,)
        Class means.
    """
    means = []
    classes = np.unique(y)
    for group in classes:
        Xg = X[y == group, :]
        means.append(Xg.mean(0))
    return np.asarray(means)


def lda(X, y, tol=0.00001, n_components=None):
    """SVD solver.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target values.
    """
    X,y = check_X_y(X,y)
    _, y_t = np.unique(y, return_inverse=True)
    priors_ = bincount(y_t) / float(len(y))
    classes_ = np.unique(y)
    n_samples, n_features = X.shape
    n_classes = len(classes_)


    print_dynamic('Calculate Class Mean')
    means_ = _class_means(X, y)

    Xc = []
    for idx, group in enumerate(classes_):
        Xg = X[y == group, :]
        Xc.append(Xg - means_[idx])

    xbar_ = np.dot(priors_, means_)

    Xc = np.concatenate(Xc, axis=0)

    print_dynamic('# 1) within (univariate) scaling by with classes std-dev')

    std = Xc.std(axis=0)
    # avoid division by zero in normalization
    std[std == 0] = 1.
    fac = 1. / (n_samples - n_classes)

    print_dynamic('# 2) Within variance scaling')
    X = np.sqrt(fac) * (Xc / std)
    # SVD of centered (within)scaled data
    U, S, V = linalg.svd(X, full_matrices=False)

    rank = np.sum(S > tol)
    if rank < n_features:
        warnings.warn("Variables are collinear.")
    # Scaling of within covariance is: V' 1/S
    scalings = (V[:rank] / std).T / S[:rank]

    print_dynamic('# 3) Between variance scaling')
    # Scale weighted centers
    X = np.dot(((np.sqrt((n_samples * priors_) * fac)) *
                    (means_ - xbar_).T).T, scalings)
    # Centers are living in a space with n_classes-1 dim (maximum)
    # Use SVD to find projection in the space spanned by the
    # (n_classes) centers
    _, S, V = linalg.svd(X, full_matrices=0)


    rank = np.sum(S > tol * S[0])
    scalings_ = np.dot(scalings, V.T[:, :rank])
    coef = np.dot(means_ - xbar_, scalings_)
    intercept_ = (-0.5 * np.sum(coef ** 2, axis=1)
                       + np.log(priors_))
    coef_ = np.dot(coef, scalings_.T)
    intercept_ -= np.dot(xbar_, coef_.T)

    return intercept_, coef_, classes_


def n_fold_generate(data, n_fold=4):
    it = itertools.groupby(data, lambda x: x[0])
    folded_data = [[] for i in range(n_fold)]

    for grp in it:
        for j,d in enumerate(chunk(list(grp[1]), n_fold)):
            folded_data[j].append(d)

    fdata = [functools.reduce(lambda x,y: x+y, f) for f in folded_data]
    return fdata


def chunk(seq, num):
  np.random.shuffle(seq)
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out


def decision_function(X, intercept_, coef_):
    X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
    scores = safe_sparse_dot(X, coef_.T, dense_output=True) + intercept_
    scores = scores.ravel() if scores.shape[1] == 1 else scores


    return scores


def predict(X, intercept_, coef_, classes_):
    X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
    scores = safe_sparse_dot(X, coef_.T, dense_output=True) + intercept_
    scores = scores.ravel() if scores.shape[1] == 1 else scores
    if len(scores.shape) == 1:
        indices = (scores > 0).astype(np.int)
    else:
        indices = scores.argmax(axis=1)
    return classes_[indices]
