from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, RobustScaler
from plotly import graph_objects as go
import plotly
from six import iteritems

from SamBA.distances import *
from SamBA.relevances import *
from SamBA.vizualization import VizSamba
from SamBA.utils import set_class_from_str


class TrainWeighting:
    def __init__(self, pred_train=True):
        self.minus_binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        self.pred_train=pred_train


class ExpTrainWeighting(TrainWeighting):

    def __init__(self, factor=1, pred_train=True):
        TrainWeighting.__init__(self, pred_train=pred_train)
        self.factor = factor

    def __call__(self, estim, X, y, n_estimators=1, *args, **kwargs):

        if isinstance(estim, NeighborHoodClassifier):
            if self.pred_train:
                exps = np.exp(-estim._predict_on_train(X, n_estimators=n_estimators)*y*self.factor)
            else:
                exps = np.exp(-estim._predict_vote(X, n_estimators=n_estimators)*y*self.factor)
        else:
            exps = np.exp(-estim.predict(X) * y*self.factor)
        return exps/np.sum(exps)


class SqExpTrainWeighting(ExpTrainWeighting):

    def __call__(self, estim, X, y, n_estimators=1, *args, **kwargs):
        exps = ExpTrainWeighting.__call__(self, estim, X, y, n_estimators=n_estimators)
        return exps**2


class ZeroOneTrainWeighting(TrainWeighting):

    def __init__(self, pred_train=True):
        TrainWeighting.__init__(self, pred_train=pred_train)

    def __call__(self, estim, X, y, n_estimators=1, *args, **kwargs):
        if isinstance(estim, NeighborHoodClassifier):
            if self.pred_train:
                preds = -(np.sign(estim._predict_on_train(X, n_estimators=n_estimators))*y-1)
            else:
                preds = -(np.sign(estim._predict_vote(X, n_estimators=n_estimators))*y-1)
        else:
            preds = -(estim.predict(X)*y-1)
        return preds/2


class NeighborHoodClassifier(BaseEnsemble, ClassifierMixin, VizSamba):

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(max_depth=1,
                                                       splitter='best',
                                                       criterion='gini'),
                 n_estimators=2,
                 estimator_params=tuple(),
                 relevance=ExpRelevance(),
                 distance=EuclidianDist(),
                 train_weighting=ExpTrainWeighting(),
                 keep_selected_features=True,
                 vote_compensate=True,
                 b=1,
                 normalizer=None,
                 forced_diversity=False,
                 pred_train=True):
        BaseEnsemble.__init__(self, base_estimator=base_estimator,
                              n_estimators=n_estimators,
                              estimator_params=estimator_params)
        self.zero_binarizer = LabelBinarizer(neg_label=0, pos_label=1)
        self.minus_binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        self.b = b
        self.pred_train = pred_train
        self.relevance = set_class_from_str(relevance)
        self.distance = set_class_from_str(distance)
        self.train_weighting = set_class_from_str(train_weighting)
        self.distance.keep_selected_features = keep_selected_features
        self.keep_selected_features = keep_selected_features
        self.feature_importances_ = None
        self.vote_compensate = vote_compensate
        self.normalizer = set_class_from_str(normalizer)
        self.forced_diversity = forced_diversity


    def fit(self, X, y, save_data=False, **fit_params):
        self.train_size = X.shape[0]
        if isinstance(y, list):
            y = np.array(y)
        if self.normalizer is not None:
            X = self.normalizer.fit_transform(X)

        self.train_weighting.pred_train = self.pred_train
        sign_y = self.minus_binarizer.fit_transform(y).reshape(y.shape)
        self._init_containers(X)

        self._init_greedy(X, sign_y, save_data)
        for iter_index in range(self.n_estimators-1):
            self._boost_loop(X, sign_y, iter_index+1, save_data)
            if self.pred_train:
                preds = np.sign(self._predict_on_train(X))
            else:
                preds = np.sign(self._predict_vote(X))
            if accuracy_score(preds, sign_y)==1.0:
                print("Broken because perfect {}/{}".format(iter_index+2, self.n_estimators))
                self.n_estimators = iter_index+2
                break
        if np.sum(self.feature_importances_) != 0:
            self.feature_importances_/=np.sum(self.feature_importances_)
        self.support_feats = np.argsort(-self.feature_importances_)[:len(np.where(self.feature_importances_!=0)[0])]
        self.support_ratio = len(self.support_feats)/self.n_estimators

        return self

    def predict(self, X, save_data=None):
        vote = self._predict_vote(X, save_data=save_data)
        return self.zero_binarizer.fit_transform(np.sign(vote)).reshape(X.shape[0])

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], self.n_classes))
        self._predict_vote(X)
        for sample_ind, vote in enumerate(self.votes):
            if vote<0:
                probas[sample_ind, 0] = -vote
            else:
                probas[sample_ind, 1] = vote
        probas /= np.max(probas)
        for sample_ind, [proba_1, proba_2] in enumerate(probas):
            if proba_1 == 0:
                probas[sample_ind, 0] = 1 - proba_2
            if proba_2 == 0:
                probas[sample_ind, 1] = 1 - proba_1
        return probas

    def set_params(self, **kwargs):
        for parameter, value in iteritems(kwargs):
            setattr(self, parameter, value)
        if "pred_train" in kwargs:
            self.train_weighting.pred_train = kwargs["pred_train"]
        return self

    def _predict_on_train(self, X, n_estimators=None):
        if n_estimators is None:
            n_estimators = self.n_estimators
        pred = np.sum(np.array([estim.predict(X)*self.neig_weights[:, estim_ind]
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
                      save_data=None):
        # TODO : Predict only on the required features : nécessite une refaction
        #  de X, peut-être en sparse matrix, avec uniquement les valeurs des
        #  features utilisés non nuls, ou une réécriture du DT.
        if n_estimators is None:
            n_estimators = self.n_estimators
        if self.normalizer is not None:
            X = self.normalizer.fit_transform(X)
        if save_data is not None:
            self.saved_test = pd.DataFrame(columns=['Index', "Distance", "Relevance", "Weight", "Estim Index"])
        preds = np.zeros((X.shape[0], n_estimators))
        self.features_mask = np.zeros(X.shape[1], dtype=np.int64)
        for estim_index, estim in enumerate(self.estimators_[:n_estimators]):
            self.features_mask[np.where(estim.feature_importances_ != 0)[0]] = 1
            preds[:, estim_index] = estim.predict(X)
        self.votes = np.zeros(X.shape[0])
        for sample_index, sample in enumerate(X):
            dists = self.distance(sample, self.train_samples, self.features_mask)
            weights = 1 / (dists) ** self.b
            if isinstance(self.relevance, MarginRelevance):
                weights /= np.sum(weights)
                weights = weights.reshape((self.train_size, 1))*np.sign(self.neig_weights[:, :n_estimators])
            else:
                weights = weights.reshape((self.train_size, 1))*self.neig_weights[:, :n_estimators]
                weights /= np.sum(weights)
            if sample_index==save_data:
                for estim_index in range(self.n_estimators):
                    for train_sample_ind, (dist, relevance, weight) in enumerate(zip(dists, self.neig_weights[:, estim_index], weights)):
                        self.saved_test = self.saved_test.append({"Index": train_sample_ind,
                                                                  'Distance': dist,
                                                                  "Relevance": relevance,
                                                                  "Weight": weight,
                                                                  "Estim Index": estim_index},ignore_index=True)
            vote = np.sum(
                self.estim_weights[:n_estimators] * np.sum(weights, axis=0) *
                preds[sample_index])
            if vote == 0:
                vote = base_decision
            self.votes[sample_index] = vote
        return self.votes/np.max(np.abs(self.votes))

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

    def _boost_loop(self, X, y, iter_index=1, save_data=False):
        next_estimator = clone(self.base_estimator)
        if self.forced_diversity and iter_index == 1:
            modded_X = X.copy()
            chosen_features = np.where(self.estimators_[0].feature_importances_ !=0)
            modded_X[:, chosen_features] = 0
            next_estimator.fit(modded_X, y,
                               sample_weight=self.train_weights[:, iter_index])
            del modded_X
        else:
            next_estimator.fit(X, y,
                               sample_weight=self.train_weights[:, iter_index])
        self.estimators_.append(next_estimator)
        self.neig_weights[:, iter_index] = self.relevance(X, y, next_estimator)
        if iter_index<self.n_estimators-1:
            if self.vote_compensate:
                self.train_weights[:, iter_index + 1] = self.train_weighting(
                    self, X, y, n_estimators=iter_index+1)
            else:
                self.train_weights[:, iter_index + 1] *= self.train_weighting(
                    next_estimator, X, y)
        self.feature_importances_ += next_estimator.feature_importances_
        if save_data:
            preds_train = np.sign(self._predict_on_train(X))
            preds = self.minus_binarizer.fit_transform(self.predict(X)).reshape(
            y.shape)
            self._save_data(preds, preds_train, iter_index + 1, y)

    def _init_greedy(self, X, y, save_data):
        first_estimator = clone(self.base_estimator)
        first_estimator.fit(X, y)
        self.estimators_.append(first_estimator)
        self.train_weights[:, 1] = self.train_weighting(first_estimator, X, y)
        self.neig_weights[:, 0] = self.relevance(X, y, first_estimator)
        self.feature_importances_ = first_estimator.feature_importances_
        if save_data:
            preds = self.minus_binarizer.fit_transform(self.predict(X)).reshape(
                y.shape)
            self._save_data(preds, preds, 1, y)

    def _init_containers(self, X,):
        self.n_classes = 2
        self.train_samples = X
        self.saved_data = pd.DataFrame(
            columns=["Iteration", "Pred", 'Pred Train"', "Margin", "Class", "X", "Y", "Weight"])
        self.train_weights = np.ones((X.shape[0], self.n_estimators+1))/X.shape[0]
        self.neig_weights = np.ones((X.shape[0], self.n_estimators))
        self.estim_weights = np.ones(self.n_estimators) / self.n_estimators
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
        train_weighting=ExpTrainWeighting(pred_train=True),
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
