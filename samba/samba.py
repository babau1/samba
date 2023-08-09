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

from sklearn.base import clone
from sklearn.ensemble import BaseEnsemble
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from six import iteritems
from sklearn.utils import safe_mask
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import check_X_y, check_array
from sklearn.exceptions import NotFittedError

from samba.relevances import *
from samba.difficulties import *
from samba.vizualization import VizSamba
from samba.neighborhood_classifiers import NeighborhoodClassifier
from samba.utils import set_class_from_str


class SamBAClassifier(NeighborhoodClassifier, VizSamba):

    """
    An Neighborhood classifier generalizaing boosting with a similarity measure.
    It is an ensemble method that learns a set of features the with the same greedy method as Adaboost.
    However, it predicts using a similarity function on the subset of features selected during training.
    This class implements the algorithm known as SamBA [1].


    Parameters
    ----------
    base_estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is :class:`~sklearn.tree.DecisionTreeClassifier`
        initialized with `max_depth=1`.
        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.
    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`.
    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.
    relevance : object or string, default="ExpRelevance"
        The relevance function used to compute the weight of each estimator on
        each trinaing sample. All the available ones are provided in the
        relevances.py module.
    distance : object or string, default="EuclidianDist"
        The similarity measure used to compute the estimated weights in the
        prediction function. All the available ones are provided in the
        distences.py module.
    difficulty: object or string, default="ExpTrainWeighting"
        The difficutly function used to compute the difficulty of a sample,
        during the learning process. A difficult sample will be overweighted
        to try to classify it. All the available ones are provided in the
        difficulties.py module.
    keep_selected_features : bool, default=True
        If Ture, the similarity measure is computed only on the features
        selected during learning, if False it is computed on all the features.
        One of the main advantage of SamBA is to perform feature selection. Set
        to False uniquely if our are sure that the similarity on the entire
        set of features of X has sense.
    vote_compensate : bool, default=True
        If True, the algorithm will focus on each iteration on the samples that
        are the most difficult for the vote of all the preovous iterations.
        If False, on of the last iteration.
    b : float, default=1.0
        Hyper-parameter b controls the impact of the similarity on the weight
        estimation function, a bigger b means that only the closest samples
        have a meaningful vote, whereas a smaller b means that all the samples
        have a meaningful vote.
    a : float, default=0.1
        Hyper-parameter a controls the smoothness of the decision border, and
        the maximum weight of a sample in the weight estimation function.
    normalizer : object, default=None
        As SamBA relies on similarity functions, if the data needs to be
        normalized, one can set normalizer to RobustScaler, from the sklearn
        library.
    forced_diversity : bool, default=False
        If True, forced SamBA to choose different features at each iteration.
        Can be useful in the context of biomarker discovery, or in the case
        of an infinite loop.
    pred_train : bool, default=False
        If True, the difficulty of a sample is computed as a majority vote of
        experts, with the weights learned during the traning phase. If False,
        the weights of each traning sample a re-evaluated with the weight
        estimation fuunction for the difficulty computation.
        Intuitively, True leads to sparse votes that converge quickly but are
        prone to overfitting, and False leads to longer trnaing phases,
        but with possibly less overfitting.
    normalize_dists : bool, default=True
        During the prediction phase, if True the distances of all the samples
        are normalized, if not, they are fed raw to the weight estimation
        function. Has small impact on the prediction. Some approximation errors
         can happen when normalizing, but a normalized vector is easier to
         understand for interpretation.
    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_classes_ : int
        The number of classes.
    neig_weights_ : ndarray of floats, shape (n_estimators, n_samples)
        Weights for each estimator in the boosted ensemble,
        on each trnaing sample.
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances if supported by the
        ``estimator`` (when based on decision trees).
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=2,
                 estimator_params=tuple(),
                 relevance="ExpRelevance",
                 distance="EuclidianDist",
                 difficulty="ExpTrainWeighting",
                 keep_selected_features=True,
                 vote_compensate=True,
                 b=1.0,
                 a=0.1,
                 normalizer=None,
                 forced_diversity=False,
                 pred_train=False,
                 normalize_dists=True,
                 class_weight=None,
                 ):
        BaseEnsemble.__init__(self, base_estimator=base_estimator,
                              n_estimators=n_estimators,
                              estimator_params=estimator_params)
        self.b = b
        self.a = a
        self.pred_train = pred_train
        self.relevance = relevance
        self.distance = distance
        self.difficulty = difficulty
        self.normalize_dists = normalize_dists
        self.keep_selected_features = keep_selected_features
        self.vote_compensate = vote_compensate
        self.normalizer = set_class_from_str(normalizer)
        self.forced_diversity = forced_diversity
        self.class_weight = class_weight

    def fit(self, X, y, save_data=False, sample_weight=None, **fit_params):
        self._validate_estimator(default=DecisionTreeClassifier(max_depth=1))
        self._validate_functions()
        self.label_encoder_ = LabelEncoder()
        self.minus_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        self.feature_importances_ = None
        expanded_class_weight = None
        X, y = check_X_y(X, y, accept_sparse=False, )
        v = self._check_y(y)
        self.train_size_ = X.shape[0]

        if self.class_weight is not None:
            expanded_class_weight = compute_sample_weight(self.class_weight, y)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            if expanded_class_weight is not None:
                sample_weight *= expanded_class_weight
                sample_weight /= np.sum(sample_weight)
        else:
            if expanded_class_weight is not None:
                sample_weight = expanded_class_weight

        if self.normalizer is not None:
            X = self.normalizer.fit_transform(X)
        code_y = self.label_encoder_.fit_transform(y)
        sign_y = self.minus_binarizer_.fit_transform(code_y).reshape(y.shape)
        self.classes_ = self.label_encoder_.inverse_transform(self.minus_binarizer_.inverse_transform(np.unique(sign_y)))
        self._init_containers(X)

        self._init_greedy(X, sign_y, save_data, sample_weight=sample_weight)
        for iter_index in range(self.n_estimators-1):
            self._boost_loop(X, sign_y, iter_index+1, save_data,
                             sample_weight=sample_weight)
            if np.isnan(self.neig_weights_[0, iter_index + 1]):
                self.n_estimators = iter_index+2
                self.neig_weights_ = np.zeros((X.shape[0], self.n_estimators))
                self.neig_weights_[:, iter_index + 1] = 1
                break

        if np.sum(self.feature_importances_) != 0:
            self.feature_importances_/=np.sum(self.feature_importances_)

        self.support_feats_ = np.argsort(-self.feature_importances_)[:len(np.where(self.feature_importances_ != 0)[0])]
        self.support_ratio_ = len(self.support_feats_) / self.n_estimators
        return self

    def predict(self, X, save_data=None):
        X = check_array(X, accept_sparse=False)
        vote = self._predict_vote(X, save_data=save_data)
        return self.label_encoder_.inverse_transform(self.minus_binarizer_.inverse_transform(np.sign(vote))).reshape(X.shape[0])

    def predict_proba(self, X):
        X = check_array(X, accept_sparse=False)
        votes = self._predict_vote(X)
        probas = np.zeros((X.shape[0], self.n_classes_))
        for sample_ind, vote in enumerate(votes):
            if vote<0:
                probas[sample_ind, 0] = 0.5-vote/2
            else:
                probas[sample_ind, 1] = 0.5+vote/2
        for sample_ind, [proba_1, proba_2] in enumerate(probas):
            if proba_1 == 0:
                probas[sample_ind, 0] = 1 - proba_2
            if proba_2 == 0:
                probas[sample_ind, 1] = 1 - proba_1
        return probas

    def set_params(self, **kwargs):
        for parameter, value in iteritems(kwargs):
            setattr(self, parameter, value)
        return self

    def _validate_functions(self):
        self.relevance_ = set_class_from_str(self.relevance)
        self.distance_ = set_class_from_str(self.distance)
        self.difficulty_ = set_class_from_str(self.difficulty)
        if self.pred_train:
            self.difficulty_.pred_train = True

    def _get_tags(self):
        tags = BaseEnsemble._get_tags(self)
        tags["binary_only"] = True
        tags["poor_score"] = True
        return tags

    def _check_X(self, X):
        if not isinstance(X, np.ndarray):
            try:
                X = X.__array__()
            except Exception as e:
                raise e
        return X

    def _check_y(self, y):
        if isinstance(y, list):
            y = np.array(y)
        elif not isinstance(y, np.ndarray):
            try:
                y=y.__array__()
            except Exception as e:
                raise e
        if len(np.unique(y))>2:
            raise ValueError("Unknown label type: SamBA is only compatible with binary classification, for the moment ...")
        return y

    def _predict_on_train(self, X, n_estimators=None):
        if n_estimators is None:
            n_estimators = self.n_estimators
        pred = np.sum(np.array([estim.predict(X) * self.neig_weights_[:, estim_ind]
                       for estim_ind, estim in enumerate(self.estimators_[:n_estimators])]), axis=0)
        return pred

    def _step_predict_on_train(self, X):
        preds = np.zeros((X.shape[0], self.n_estimators))
        for n_estimators in range(self.n_estimators):
            preds[:, n_estimators] = self.minus_binarizer_.inverse_transform(np.sign(
                self._predict_on_train(X, n_estimators=n_estimators + 1))).reshape(
                X.shape[0])
        return preds

    def _predict_vote(self, X, base_decision=-1, n_estimators=None,
                      save_data=None, transform=True):
        # TODO : Predict only on the required features : nécessite une refaction
        #  de X, peut-être en sparse matrix, avec uniquement les valeurs des
        #  features utilisés non nuls, ou une réécriture du DT, ou un
        #  réapprentissage sur le dataset croppé à la fin du train en supposant
        #  que le processus est déterministe.
        X = self._check_X(X)
        if not hasattr(self, "estimators_"):
            raise NotFittedError
        if n_estimators is None:
            n_estimators = self.n_estimators
        if self.normalizer is not None and transform:
            X = self.normalizer.transform(X)
        if save_data is not None:
            self.saved_test = pd.DataFrame(columns=['Index', "Distance", "Relevance", "Weight", "Estim Index"])
            self.saved_ind_test = 0
        preds = np.zeros((X.shape[0], n_estimators))
        features_mask = np.zeros(X.shape[1], dtype=np.int64)
        for estim_index, estim in enumerate(self.estimators_[:n_estimators]):
            features_mask[np.where(estim.feature_importances_ != 0)[0]] = 1
            preds[:, estim_index] = estim.predict(X)
        votes = np.zeros(X.shape[0])
        for sample_index, sample in enumerate(X):
            dists = self.distance_(sample, self.train_samples_, features_mask)
            if self.normalize_dists:
                if np.sum(dists)==0:
                    dist_sum = 1
                else:
                    dist_sum = np.sum(dists)
                dists = dists/dist_sum
            weights = 1 / (self.a**self.b + dists ** self.b)
            if isinstance(self.relevance, MarginRelevance):
                weights /= np.sum(weights)
                weights = weights.reshape((self.train_size_, 1)) * np.sign(self.neig_weights_[:, :n_estimators])
            else:
                weights = weights.reshape((self.train_size_, 1)) * self.neig_weights_[:, :n_estimators]
                weights /= np.sum(weights)
            if save_data is not None and sample_index in save_data:
                for estim_index in range(self.n_estimators):
                    for train_sample_ind, (dist, relevance, weight) in enumerate(zip(dists, self.neig_weights_[:, estim_index], weights)):
                        self.saved_test.loc[self.saved_ind_test] = {"Index": train_sample_ind,
                                                                  'Distance': dist,
                                                                  "Relevance": relevance,
                                                                  "Weight": weight,
                                                                  "Estim Index": estim_index}
                        self.saved_ind_test += 1
            vote = np.sum(np.sum(weights, axis=0) * preds[sample_index])
            if vote == 0:
                vote = base_decision
            votes[sample_index] = vote
        return votes

    def single_sample_importances(self, X, transform=False, ):
        weights_mat = np.zeros((X.shape[0], X.shape[1]))
        if self.normalizer is not None and transform:
            X = self.normalizer.transform(X)
        self.features_mask = np.zeros(X.shape[1], dtype=np.int64)
        for estim_index, estim in enumerate(self.estimators_):
            self.features_mask[np.where(estim.feature_importances_ != 0)[0]] = 1
        for sample_index, sample in enumerate(X):
            dists = self.distance(sample, self.train_samples_,
                                  self.features_mask)
            if self.normalize_dists:
                if np.sum(dists) == 0:
                    dist_sum = 1
                else:
                    dist_sum = np.sum(dists)
                dists = dists / dist_sum
            weights = 1 / (self.a ** self.b + dists ** self.b)
            if isinstance(self.relevance, MarginRelevance):
                weights /= np.sum(weights)
                weights = weights.reshape((self.train_size_, 1)) * np.sign(
                    self.neig_weights_[:, :self.n_estimators])
            else:
                weights = weights.reshape(
                    (self.train_size_, 1)) * self.neig_weights_[:, :self.n_estimators]
                weights /= np.sum(weights)
            importances = np.array([estim.feature_importances_ for estim in self.estimators_])
            weights_mat[sample_index, :] = np.average(importances, weights=np.sum(weights, axis=0), axis=0)
        return weights_mat

    def step_predict(self, X):
        preds = np.zeros((X.shape[0], self.n_estimators))
        for n_estimators in range(self.n_estimators):
            preds[:, n_estimators] = self.zero_binarizer.fit_transform(np.sign(self._predict_vote(X, n_estimators=n_estimators+1))).reshape(X.shape[0])
        return preds

    def step_vote_predict(self, X):
        preds = np.zeros((X.shape[0], self.n_estimators))
        for n_estimators in range(self.n_estimators):
            preds[:, n_estimators] = self._predict_vote(X, n_estimators=n_estimators + 1)
        return preds

    def _boost_loop(self, X, y, iter_index=1, save_data=False, sample_weight=None):
        next_estimator = clone(self.base_estimator_)
        if self.forced_diversity:
            mask = np.zeros(X.shape, dtype=bool)
            chosen_features = np.where(self.estimators_[iter_index-1].feature_importances_ != 0)[0]
            mask[:, chosen_features] = True
            X_msk = np.ma.array(X, mask=safe_mask(X, mask))
            X_msk.fill_value=0
            X_msk = X_msk.filled()
        else:
            X_msk = X
        next_estimator.fit(X_msk, y,
                           sample_weight=self.train_weights_[:, iter_index])
        self.estimators_.append(next_estimator)
        if accuracy_score(next_estimator.predict(X_msk), y)==1.0:
            self.neig_weights_[:, iter_index] = np.nan
        else:
            self.neig_weights_[:, iter_index] = self.relevance_(X, y, next_estimator)
            if iter_index < self.n_estimators-1:
                if self.vote_compensate:
                    self.train_weights_[:,
                    iter_index + 1] = self.difficulty_(
                        self, X, y, n_estimators=iter_index + 1,
                        pred_train=self.pred_train)
                    if sample_weight is not None:
                        self.train_weights_[:,
                        iter_index + 1] *= sample_weight
                else:
                    self.train_weights_[:, iter_index + 1] *= self.difficulty(
                        next_estimator, X, y, pred_train=self.pred_train)
        # quit()
        self.feature_importances_ += next_estimator.feature_importances_
        if save_data:
            preds_train = np.sign(self._predict_on_train(X))
            preds = self.minus_binarizer_.fit_transform(self.predict(X)).reshape(
            y.shape)
            self._save_data(preds, preds_train, iter_index + 1, y)

    def _init_greedy(self, X, y, save_data, sample_weight=None):
        first_estimator = clone(self.base_estimator_)
        first_estimator.fit(X, y, sample_weight=sample_weight)
        self.estimators_.append(first_estimator)
        self.train_weights_[:, 1] = self.difficulty_(first_estimator, X, y,
                                                     pred_train=self.pred_train)
        if sample_weight is not None:
            self.train_weights_[:, 1]*=sample_weight
        self.neig_weights_[:, 0] = self.relevance_(X, y, first_estimator)
        self.feature_importances_ = first_estimator.feature_importances_
        if save_data:
            preds = self.minus_binarizer_.fit_transform(self.predict(X)).reshape(
                y.shape)
            self._save_data(preds, preds, 1, y)

    def _init_containers(self, X,):
        self.n_classes_ = 2
        self.train_samples_ = X
        self.n_features_in_ = X.shape[1]
        self.saved_data_ = pd.DataFrame(
            columns=["Iteration", "Pred", 'Pred Train"', "Margin", "Class", "X", "Y", "Weight"])
        self.saved_ind_ = 0
        self.train_weights_ = np.ones((X.shape[0], self.n_estimators + 1)) / X.shape[0]
        self.neig_weights_ = np.ones((X.shape[0], self.n_estimators))
        self.estimators_ = []
        self.feature_importances_ = np.zeros(X.shape[1])


if __name__ == "__main__":

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from samba.utils import gen_four_blobs
    rs = np.random.RandomState(7)

    X, y = gen_four_blobs(rs, n_samples=1000, unit=200)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True, random_state=rs)

    classifier = SamBAClassifier()

    classifier.fit(X_train, y_train)
    preds_train = classifier._step_predict_on_train(X_train)
    preds = classifier.predict(X_train)
    print("Train accuracy", accuracy_score(y_train, preds))
    print("Test accuracy", accuracy_score(y_test, classifier.predict(X_test)))

