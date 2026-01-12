from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler

class ADEvaluator:
    """
    Applicability Domain (AD) using standardized-space centroid distance.
    Default threshold: mean + 1*std  (same as D_th6 with K=1.0).
    """
    def __init__(self, Xtrain: np.ndarray, k: float = 1.0):
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(Xtrain)
        self.centroid = Xs.mean(axis=0)

        d_train = np.linalg.norm(Xs - self.centroid, axis=1)
        self.mean = float(d_train.mean())
        self.std  = float(d_train.std())
        self.k = float(k)
        self.thr = self.mean + self.k * self.std

    def evaluate(self, Xnew: np.ndarray):
        Xs = self.scaler.transform(Xnew)
        d = np.linalg.norm(Xs - self.centroid, axis=1)
        in_domain = d <= self.thr
        return d, in_domain, self.thr
