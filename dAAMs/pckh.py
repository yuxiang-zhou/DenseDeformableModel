import numpy as np

scale = 4.5

def dist(pred, gt):
    return np.linalg.norm(pred - gt, axis=1) / scale

def accuracy(t, dists):
    return float(np.sum(dists <= t, axis=0)) / dists.shape[1]

def pckh(preds, gts):
    t_range = np.arange(0,0.51,0.01)
    dists = np.array([dists(p,g) for p,g in zip(preds, gts)])
    pckh = np.array([accuracy(t, dists) for t in t_range])
    return pckh, t
