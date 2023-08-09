"""
    SamBA -- Sample Boosting Algorithm
    Copyright (C) 2023 Baptiste BAUVIN

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""


from sklearn.preprocessing import LabelBinarizer
from samba.neighborhood_classifiers import NeighborhoodClassifier
import numpy as np


class TrainWeighting:
    def __init__(self):
        self.minus_binarizer = LabelBinarizer(neg_label=-1, pos_label=1)


class ExpTrainWeighting(TrainWeighting):

    def __init__(self, factor=2):
        TrainWeighting.__init__(self)
        self.factor = factor

    def __call__(self, estim, X, y, n_estimators=1, pred_train=False, *args, **kwargs):

        if isinstance(estim, NeighborhoodClassifier):
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
        if isinstance(estim, NeighborhoodClassifier):
            if pred_train:
                failed_preds = -(np.sign(estim._predict_on_train(X, n_estimators=n_estimators))*y-1)
            else:
                failed_preds = -(np.sign(estim._predict_vote(X, n_estimators=n_estimators,
                                                      transform=False))*y-1)
        else:
            failed_preds = -(estim.predict(X)*y-1)/2
        return failed_preds/2