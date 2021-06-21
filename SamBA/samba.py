from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from plotly import graph_objects as go
import plotly
import os

from SamBA.distances import *
from SamBA.relevances import *


class ExpTrainWeighing():

    def __init__(self):
        self.minus_binarizer = LabelBinarizer(neg_label=-1, pos_label=1)

    def __call__(self, estim, X, y, *args, **kwargs):
        if isinstance(estim, NeighborHoodClassifier):
            exps = np.exp(-estim._predict_vote(X)*y)
        else:
            exps = np.exp(-estim.predict(X) * y)
        return exps/np.sum(exps)


class NeighborHoodClassifier(BaseEnsemble, ClassifierMixin):

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(max_depth=1,
                                                       splitter='best',
                                                       criterion='gini'),
                 n_estimators=2,
                 estimator_params=tuple(),
                 relevance=ExpRelevance,
                 distance=EuclidianDist,
                 train_weighting=ExpTrainWeighing,
                 keep_selected_features=True):
        BaseEnsemble.__init__(self, base_estimator=base_estimator,
                              n_estimators=n_estimators,
                              estimator_params=estimator_params)
        self.zero_binarizer = LabelBinarizer(neg_label=0, pos_label=1)
        self.minus_binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        self.relevance = relevance()
        self.distance = distance(keep_selected_features=keep_selected_features)
        self.train_weighting = train_weighting()
        self.keep_selected_features = keep_selected_features
        self.feature_importances_ = None

    def fit(self, X, y, save_data=False, **fit_params):
        if isinstance(y, list):
            y = np.array(y)
        sign_y = self.minus_binarizer.fit_transform(y).reshape(y.shape)
        self._init_containers(X)
        self._init_greedy(X, sign_y, save_data)
        for iter_index in range(self.n_estimators-1):
            self._boost_loop(X, sign_y, iter_index, save_data)

        self.feature_importances_/=np.sum(self.feature_importances_)
        self.support_feats = np.where(self.feature_importances_!=0)[0]
        self.support_ratio = len(self.support_feats)/self.n_estimators
        return self

    def predict(self, X):
        self._predict_vote(X)
        return self.zero_binarizer.fit_transform(np.sign(self.votes)).reshape(X.shape[0])

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], self.n_classes))
        self._predict_vote(X)
        for sample_ind, vote in enumerate(self.votes):
            if vote<1:
                probas[sample_ind, 0] = -vote
            else:
                probas[sample_ind, 1] = vote
        return probas/np.max(probas)

    def plot_projection(self, X, y, save=True, path=".", rs=42, contour=False):
        if self.support_feats.shape[0] == 1:
            sec_dim = np.random.RandomState(rs).uniform(0,1, size=X.shape[0])
            fig = self._plot_2d(X, y, sec_dim=sec_dim, contour=contour)
        elif self.support_feats.shape[0]==2:
            fig = self._plot_2d(X, y, contour=contour)
        else:
            best_feats = np.argsort(-self.feature_importances_)[:3]
            fig = go.Figure()
            labels = np.unique(y)
            for label in labels:
                data = X[np.where(y == label)[0], best_feats]
                fig.add_trace(go.Scatter3d(x=data[:, 0],
                                           y=data[:, 1],
                                           z=data[:, 2],
                                           name="Class {}".format(label + 1),
                                           mode="markers",
                                           marker=dict(
                                               size=1, )))
        if save:
            plotly.offline.plot(fig, filename=os.path.join(path, "projection_fig.html"), auto_open=False)
        else:
            fig.show()

    def get_feature_importance(self, x_test, sample_name="NoName",
                               feature_names=None, limit=10):
        if feature_names is None:
            feature_names = ["Feat {}".format(ind) for ind in range(x_test.shape[0])]
        feat_imp = self._predict_one_sample(x_test)
        n_important_feat = len(np.where(feat_imp!=0)[0])
        if n_important_feat>limit:
            add_limit = "The list was shortened, because, there were more than " \
                        "{} important features, but you can increase this limit " \
                        "by setting limit=<your value> in " \
                        "`get_feature_importance` arguments.".format(limit)
        else:
            add_limit = ""
        best_feats_ind = np.argsort(-feat_imp)
        out_str = "For sample {}, the feature importances are : \n".format(sample_name)
        for i in range(min(n_important_feat, limit)):
            out_str += "\t - {} has an importance of {}% \n".format(feature_names[best_feats_ind[i]], round(feat_imp[best_feats_ind[i]], 2),)
        out_str+= add_limit
        return out_str

    def _plot_2d(self, X, y, sec_dim=None, contour=False):
        if sec_dim is None:
            sec_dim = X[:, self.support_feats[1]]
        fig = go.Figure()
        labels = np.unique(y)
        for label in labels:
            indices = np.where(y == label)[0]
            fig.add_trace(go.Scatter(x=X[indices, self.support_feats[0]],
                                     y=sec_dim[indices],
                                     name="Class {}".format(label + 1),
                                     mode="markers",
                                     marker=dict(
                                         size=3, )))
        if contour:
            fig.add_trace(go.Contour(
                z=-self._predict_vote(X),
                x=X[:, self.support_feats[0]],
                y=sec_dim,
                line_smoothing=0.85,
                contours_coloring='heatmap',
                colorscale='RdBu',
            ))
        return fig

    def _predict_one_sample(self, sample):
        feat_importances = np.array([estim.feature_importances_ for estim in self.estimators_])
        dists = self.distance(sample, self.train_samples, self.features_mask)
        weights = self.estim_weights*np.sum(np.transpose(self.neig_weights)*dists, axis=1)
        importances = np.sum(feat_importances*weights.reshape((10,1)), axis=0)
        importances /= np.sum(importances)
        return importances

    def _predict_vote(self, X, base_decision=-1):
        # TODO : Predict only on the required features : nécessite une refaction
        #  de X, peut-être en sparse matrix, avec uniquement les valeurs des
        #  features utilisés non nuls, ou une réécriture du DT.
        preds = np.zeros((X.shape[0], self.n_estimators))
        self.features_mask = np.zeros(X.shape[1], dtype=np.int64)
        for estim_index, estim in enumerate(self.estimators_):
            self.features_mask[np.where(estim.feature_importances_ != 0)[0]] = 1
            preds[:, estim_index] = estim.predict(X)
        self.votes = np.zeros(X.shape[0])
        for sample_index, sample in enumerate(X):
            dists = self.distance(sample, self.train_samples, self.features_mask)
            vote = np.sum(self.estim_weights*np.sum(np.transpose(self.neig_weights)*dists, axis=1)*preds[sample_index])
            if vote==0:
                vote = -1
            self.votes[sample_index] = vote
        return self.votes/np.max(np.abs(self.votes))

    def _save_data(self, preds, iter, y):
        for sample_index, (sample_class, sample, prediction, vote, weight) in enumerate(zip(y, self.train_samples, preds, self.votes, self.train_weights[:, iter-1])):
            self.saved_data = self.saved_data.append({"Iteration":int(iter),
                                                      "Pred": np.sign(prediction),
                                                      "Margin": vote,
                                                      "Class": sample_class,
                                                      "X": sample[0],
                                                      "Y": sample[1],
                                                      "Weight":weight},
                                                     ignore_index=True)

    def _boost_loop(self, X, y, iter_index=0, save_data=False):
        next_estimator = clone(self.base_estimator)
        next_estimator.fit(X, y,
                           sample_weight=self.train_weights[:, iter_index])
        self.estimators_.append(next_estimator)
        self.train_weights[:, iter_index + 1] = self.train_weighting(
            self, X, y)
        self.feature_importances_ += next_estimator.feature_importances_
        self.neig_weights[:, iter_index + 1] = self.relevance(X, y,
                                                              next_estimator) / \
                                               X.shape[0]
        if save_data:
            preds = self.minus_binarizer.fit_transform(self.predict(X)).reshape(
                y.shape)
            self._save_data(preds, iter_index + 2, y)

    def _init_greedy(self, X, y, save_data):
        first_estimator = clone(self.base_estimator)
        first_estimator.fit(X, y)
        self.estimators_.append(first_estimator)
        self.train_weights[:, 0] = self.train_weighting(first_estimator, X, y)
        self.neig_weights[:, 0] = self.relevance(X, y, first_estimator) / \
                                  X.shape[0]
        if save_data:
            preds = self.minus_binarizer.fit_transform(self.predict(X)).reshape(
                y.shape)
            self._save_data(preds, 1, y)

    def _init_containers(self, X,):
        self.n_classes = 2
        self.train_samples = X
        self.saved_data = pd.DataFrame(
            columns=["Iteration", "Pred", "Margin", "Class", "X", "Y", "Weight"])
        self.train_weights = np.ones((X.shape[0], self.n_estimators))
        self.neig_weights = np.ones((X.shape[0], self.n_estimators))
        self.estim_weights = np.ones(self.n_estimators) / self.n_estimators
        self.estimators_ = []
        self.feature_importances_ = np.zeros(X.shape[1])


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    rs = np.random.RandomState(7)
    #
    # n_samples = 2000
    # n_pos = int(n_samples/2)
    # n_neg = n_samples-n_pos
    # unit = 450
    # n_features = 2
    # scale = 0.5
    # X = np.zeros((n_samples, n_features))
    # centers = [np.array([1,1]),
    #           np.array([-1,-1]),
    #           np.array([1,-1]),
    #           np.array([-1,1]), ]
    # y = np.ones(n_samples)
    # y[:n_neg] = 0
    #
    # X[:unit, 0] = rs.normal(centers[0][0], scale=scale, size=unit)
    # X[:unit, 1] = rs.normal(centers[0][1], scale=scale, size=unit)
    #
    # X[unit:n_pos, 0] = rs.normal(centers[1][0], scale=scale, size=n_pos-unit)
    # X[unit:n_pos, 1] = rs.normal(centers[1][1], scale=scale, size=n_pos-unit)
    #
    # X[n_pos:n_pos+unit, 0] = rs.normal(centers[2][0], scale=scale, size=unit)
    # X[n_pos:n_pos+unit, 1] = rs.normal(centers[2][1], scale=scale, size=unit)
    #
    # X[n_pos+unit:, 0] = rs.normal(centers[3][0], scale=scale, size=n_neg-unit)
    # X[n_pos+unit:, 1] = rs.normal(centers[3][1], scale=scale, size=n_neg-unit)
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
    ratios = [0.01, 0.015, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    # ratios = [0.10]
    for ratio in ratios:
        X, y = make_moons(n_samples=1000, shuffle=True, noise=0.1,
                          random_state=rs)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, shuffle=True, random_state=rs)

        classifier = NeighborHoodClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=2,
                                                  splitter='best'),
            n_estimators=10,
            estimator_params=tuple(),
            keep_selected_features=True)

        classifier.fit(X_train, y_train, save_data=True)
        preds = classifier.predict(X_test)
        #
        #
        print(ratio, accuracy_score(y_test, preds), classifier.support_ratio)
        # classifier.plot_projection(X_train, y_train, save=False, rs=rs, contour=True)
        # print(classifier.get_feature_importance(X[0,:], sample_name="Test sample", feature_names=["Abs", "Ord"]))
        # # fig = plot_2d(X_train, y_train)
