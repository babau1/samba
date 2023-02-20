from sklearn.preprocessing import LabelBinarizer
from SamBA.neighborhood_classifiers import NHClassifier
import numpy as np


class TrainWeighting:
    def __init__(self):
        self.minus_binarizer = LabelBinarizer(neg_label=-1, pos_label=1)


class ExpTrainWeighting(TrainWeighting):

    def __init__(self, factor=2):
        TrainWeighting.__init__(self)
        self.factor = factor

    def __call__(self, estim, X, y, n_estimators=1, pred_train=False, *args, **kwargs):

        if isinstance(estim, NHClassifier):
            if pred_train:
                exps = np.exp(-estim._predict_on_train(X, n_estimators=n_estimators)*y*self.factor)
            else:
                exps = np.exp(-estim._predict_vote(X, n_estimators=n_estimators,
                                                   transform=False)*y*self.factor)
        else:
            exps = np.exp(-estim.predict(X) * y)
        return exps/np.sum(exps)


class SqExpTrainWeighting(ExpTrainWeighting):

    def __call__(self, estim, X, y, n_estimators=1, pred_train=False, *args, **kwargs):
        exps = ExpTrainWeighting.__call__(self, estim, X, y, n_estimators=n_estimators, pred_train=pred_train)
        return exps**2


class ZeroOneTrainWeighting(TrainWeighting):

    def __init__(self):
        TrainWeighting.__init__(self)

    def __call__(self, estim, X, y, n_estimators=1, pred_train=False, *args, **kwargs):
        if isinstance(estim, NHClassifier):
            if pred_train:
                failed_preds = -(np.sign(estim._predict_on_train(X, n_estimators=n_estimators))*y-1)
            else:
                failed_preds = -(np.sign(estim._predict_vote(X, n_estimators=n_estimators,
                                                      transform=False))*y-1)
        else:
            failed_preds = -(estim.predict(X)*y-1)/2
        return failed_preds/2