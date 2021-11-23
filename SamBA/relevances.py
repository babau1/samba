import numpy as np
from sklearn.tree import DecisionTreeClassifier

def positivate(a, axis=None):
    positive = np.where(a>-0.5)[0]
    a[positive]+=1
    return a


class ExpRelevance():
    def __init__(self, use_confid=False):
        self.use_confid = use_confid

    def __call__(self, X, y, estim, *args, **kwargs):
        if self.use_confid:
            if isinstance(estim, DecisionTreeClassifier) and estim.max_depth == 1:
                feat_used = estim.tree_.feature[0]
                thresh = estim.tree_.threshold[0]
                confid = np.abs(X[:, feat_used]-thresh)/np.max(np.abs(X[:, feat_used]-thresh))*estim.predict(X)
            elif hasattr(estim, "predict_proba") and callable(estim.predict_proba):
                prob_preds = estim.predict_proba(X)
                prob_preds[:,0] *= np.zeros(prob_preds.shape[0])-1
                confid = positivate(prob_preds[:,0])
            else:
                confid = estim.predict(X)
            return np.exp(confid*y)
        else:
            return np.exp(estim.predict(X)*y)


class MarginRelevance():

    def __call__(self, X, y, estim, *args, **kwargs):
        margin = estim.predict(X)*y
        # margin[margin<0]=0
        return margin
