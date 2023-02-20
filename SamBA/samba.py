from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, RobustScaler, LabelEncoder
from plotly import graph_objects as go
import plotly
from six import iteritems
from sklearn.utils import safe_mask
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import check_X_y, check_array
from sklearn.exceptions import NotFittedError

from SamBA.relevances import *
from SamBA.difficulties import *
from SamBA.vizualization import VizSamba
from SamBA.neighborhood_classifiers import NHClassifier
from SamBA.utils import set_class_from_str





class NeighborHoodClassifier(NHClassifier, VizSamba):

    def __init__(self,
                 base_estimator=None,
                 n_estimators=2,
                 estimator_params=tuple(),
                 relevance="ExpRelevance",
                 distance="EuclidianDist",
                 difficulty="ExpTrainWeighting",
                 keep_selected_features=True,
                 vote_compensate=True,
                 b=1,
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
        # Todo : label binarizer inside not outside
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
                # print("Broken because perfect {}/{}".format(iter_index+2, self.n_estimators))
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
        tags["binary_only"]=True
        return tags

    def _check_X(self, X):
        if not isinstance(X, np.ndarray):
            try:
                X = X.__array__()
            except e:
                raise e
        return X

    def _check_y(self, y):
        if isinstance(y, list):
            y = np.array(y)
        elif not isinstance(y, np.ndarray):
            try:
                y=y.__array__()
            except e:
                raise e
        if len(np.unique(y))>2:
            raise ValueError("Unknown label type: SamBA is only compatible with binary classification, for the moment ...")
        return y

    def _predict_on_train(self, X, n_estimators=None):
        if n_estimators is None:
            n_estimators = self.n_estimators
        pred = np.sum(np.array([estim.predict(X)* self.neig_weights_[:, estim_ind]
                       for estim_ind, estim in enumerate(self.estimators_[:n_estimators])]), axis=0)
        return pred

    def _step_predict_on_train(self, X):
        preds = np.zeros((X.shape[0], self.n_estimators))
        for n_estimators in range(self.n_estimators):
            preds[:, n_estimators] = self.zero_binarizer.fit_transform(np.sign(
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
               # /np.max(np.abs(self.votes))

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
            # X_msk = np.ma.array(X, mask=safe_mask(X, mask))
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
        # self.estim_weights = np.ones(self.n_estimators) / self.n_estimators
        self.estimators_ = []
        self.feature_importances_ = np.zeros(X.shape[1])





if __name__ == "__main__":

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import plotly.express as px
    from SamBA.utils import gen_four_blobs
    rs = np.random.RandomState(7)
    #
    X, y = gen_four_blobs(rs, n_samples=1000, unit=200)

    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=rs)
    #
    # classifier = NeighborHoodClassifier(n_estimators=5)
    # classifier.fit(X_train, y_train, save_data=True)
    #
    # from sklearn.ensemble import AdaBoostClassifier
    # from sklearn.tree import DecisionTreeClassifier
    # ada = AdaBoostClassifier(n_estimators=10, base_estimator=DecisionTreeClassifier(max_depth=1))
    # ada.fit(X_train, y_train)
    # print(accuracy_score(ada.predict(X_train), y_train))
    # print(accuracy_score(ada.predict(X_test), y_test))
    #
    #
    # print(accuracy_score(classifier.predict(X_train), y_train))
    # print(accuracy_score(classifier.predict(X_test), y_test))
    #
    from sklearn.datasets import make_moons
    # #
    # X, y = make_moons(n_samples=1000, shuffle=True, noise=0.1,
    #                   random_state=rs)
    # X = np.load("/home/baptiste/Documents/Datasets/cardiac_montreal/Cardiac_risk_LCMS/neg.npy")
    # y = np.load(
    #     "/home/baptiste/Documents/Datasets/cardiac_montreal/Cardiac_risk_LCMS/labels_neg.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True, random_state=rs)

    classifier = NeighborHoodClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1,
                                              splitter='best'),
        train_weighting=ExpTrainWeighting(),
        n_estimators=20,
        estimator_params=tuple(),
        keep_selected_features=True,
        normalizer=None)

    classifier.fit(X_train, y_train)
    preds_train = classifier._step_predict_on_train(X_train)
    preds = classifier.predict(X_train)
    np.set_printoptions(edgeitems=30, linewidth=100000,
                        formatter=dict(float=lambda x: "%.3g" % x))
    # print(preds[:10, :])
    # print(y_train[:10])
    # print(preds_train[:10, :])
    fig = go.Figure(data=plotly.graph_objs.Scatter(x=X_train[:10, 0], y=X_train[:10, 1], mode="markers", hovertext=[str(_) for _ in range(10)]))
    fig.show()

    # fig=px.scatter(classifier.saved_data, x="X", y="Y", animation_frame="Iteration",
    #            color="Pred", size="Weight", symbol="Class", color_continuous_scale='Bluered')
    # fig.show()
    # #
    # print([accuracy_score(y_train, pred) for pred in np.transpose(classifier.step_predict(X_train))])
    # print([accuracy_score(y_test, pred) for pred in np.transpose(classifier.step_predict(X_test))])
    # print(len(classifier.support_feats), X.shape[1],
    #       classifier.feature_importances_[classifier.support_feats])
