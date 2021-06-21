import numpy as np


class RelevanceWeights():
    def __init__(self):
        pass

    def compute(self):
        pass


class ExpRelevance():
    def __init__(self):
        pass

    def __call__(self, X, y, estim, *args, **kwargs):
        return np.exp(estim.predict(X)*y)