from sklearn.datasets import make_moons,make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import h5py
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import zero_one_loss, balanced_accuracy_score
from SamBA.samba import NeighborHoodClassifier, ExpTrainWeighting
from SamBA.relevances import MarginRelevance, ExpRelevance
from SamBA.distances import EuclidianDist
from sklearn.preprocessing import RobustScaler
import plotly.express as px
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


import plotly.io as pio
pio.kaleido.scope.mathjax = None


def plot_step(X, y, n_estim=20, rs=np.random.RandomState(42), b=2, a=0.001, forced_diversity=False, normalizer=RobustScaler(), title="", train_sizes=[0.8]):
    preds_df = pd.DataFrame(columns=["x0", "x1", "y", "pred", "iterations",
                                     "Balanced Accuracy"])
    for train_size in train_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=rs,
                                                            shuffle=True, train_size=train_size)
        # noisy_X_train = add_noise(X_train, beta=0.1)

        classifier = NeighborHoodClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1,
                                              splitter='best'),
        distance=EuclidianDist(),
        relevance=ExpRelevance(),
        n_estimators=n_estim,
        normalizer=normalizer,
        forced_diversity=forced_diversity,
        b=b, a=a, pred_train=False, normalize_dists=True)

        ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1,
                                                                       splitter='best'),
                                 n_estimators=n_estim, algorithm="SAMME")
        dt = DecisionTreeClassifier(max_depth=1, splitter="best")
        dt.fit(X_train, y_train)
        print(zero_one_loss(dt.predict(X_train), y_train))
        print("Samba")
        classifier.fit(X_train, y_train)
        for estim in classifier.estimators_:
            pred = estim.predict(classifier.normalizer.transform(X_train))
            pred[pred==-1] = 0
            print(zero_one_loss(pred, y_train))
        print("Adaboost")
        ada.fit(X_train, y_train)
        for estim in ada.estimators_:
            print(zero_one_loss(estim.predict(X_train),
                                          y_train))

        train_step_preds_samba = classifier.step_predict(X_train)
        train_step_preds_ada = np.transpose(
            np.array([preds for preds in ada.staged_predict(X_train)]))

        test_step_preds_samba = classifier.step_predict(X_test)
        test_step_preds_ada = np.transpose(
            np.array([preds for preds in ada.staged_predict(X_test)]))

        preds_df = pd.DataFrame(columns=["x0", "x1", "y", "pred", "iterations",
                                         "Balanced accuracy"])
        train_accuracies_samba = [balanced_accuracy_score(y_train, pred) for
                                  pred in np.transpose(train_step_preds_samba)]
        train_accuracies_ada = [balanced_accuracy_score(y_train, pred) for pred
                                in np.transpose(train_step_preds_ada)]
        test_accuracies_samba = [balanced_accuracy_score(y_test, pred) for pred
                                 in np.transpose(test_step_preds_samba)]
        test_accuracies_ada = [balanced_accuracy_score(y_test, pred) for pred in
                               np.transpose(test_step_preds_ada)]

        train_accuracies_samba += [train_accuracies_samba[-1] for i in range(
            n_estim - len(np.transpose(train_step_preds_samba)))]
        train_accuracies_ada += [train_accuracies_ada[-1] for i in range(
            n_estim - len(np.transpose(train_step_preds_ada)))]
        test_accuracies_samba += [test_accuracies_samba[-1] for i in range(
            n_estim - len(np.transpose(test_step_preds_samba)))]
        test_accuracies_ada += [test_accuracies_ada[-1] for i in range(
            n_estim - len(np.transpose(test_step_preds_ada)))]

        for estim_ind, acc in enumerate(train_accuracies_samba):
            preds_df = preds_df.append(
                {"iterations": estim_ind + 1, "Balanced accuracy": acc,
                 "Classifier": "SamBA", "Set": "Train"}, ignore_index=True)
        for estim_ind, acc in enumerate(train_accuracies_ada):
            preds_df = preds_df.append(
                {"iterations": estim_ind + 1, "Balanced accuracy": acc,
                 "Classifier": "Adaboost", "Set": "Train"}, ignore_index=True)
        for estim_ind, acc in enumerate(test_accuracies_samba):
            preds_df = preds_df.append(
                {"iterations": estim_ind + 1, "Balanced accuracy": acc,
                 "Classifier": "SamBA", "Set": "Test"}, ignore_index=True)
        for estim_ind, acc in enumerate(test_accuracies_ada):
            preds_df = preds_df.append(
                {"iterations": estim_ind + 1, "Balanced accuracy": acc,
                 "Classifier": "Adaboost", "Set": "Test"}, ignore_index=True)
        fig = px.line(preds_df, x="iterations", y="Balanced accuracy",
                      color="Classifier", line_dash="Set")
        fig.update_layout(
            # title=title,
            font=dict(
                family="Computer Modern",
                size=18, ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray',
                         showline=True, linewidth=2, linecolor='black',
                         mirror=True)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray',
                         showline=True, linewidth=2, linecolor='black',
                         mirror=True)
        return fig
def add_noise(X, beta=0.05, rs=np.random.RandomState(42)):
    n_samples=X.shape[0]
    sample_inds = np.arange(n_samples)
    print((0.5-beta)*n_samples)
    n_noisy = int((0.5-beta)*n_samples)
    noisy_X = X.copy()
    max_X = np.max(X)
    for col_index in range(X.shape[1]):
        noised_samples = rs.choice(sample_inds,
                                   size=n_noisy,
                                   replace=False)
        noisy_X[noised_samples, col_index] = rs.uniform(0, max_X, size=n_noisy)
    return noisy_X

def aggregate_two_X(X, y, rs=np.random.RandomState(42)):
    # new_X = rs.uniform(np.min(X), np.max(X), size=(X.shape[0]*2, 2*X.shape[1]))
    new_X = -np.ones((X.shape[0] * 2, 2 * X.shape[1]))
    new_X[:X.shape[0], :X.shape[1]] = X
    new_X[-X.shape[0]:, -X.shape[1]:] = X
    new_y = np.zeros(y.shape[0]*2)
    new_y[:y.shape[0]] = y
    new_y[-y.shape[0]:] = -y+1
    return X, y

# X = np.load("/home/baptiste/Documents/Datasets/cardiac_montreal/Cardiac_risk_LCMS/pos.npy")
# y = np.load("/home/baptiste/Documents/Datasets/cardiac_montreal/Cardiac_risk_LCMS/labels_pos.npy")
# print(X.shape)
# # X, y = aggregate_two_X(X[:, :1000], y)
# fig = plot_step(X, y, normalizer=RobustScaler(), title="Cardiac LCMS")
#
# fig.write_image("cardiac_converge_norm.pdf")
# "/home/baptiste/Documents/Datasets/bleuets/bleuets.hdf5"
dset_file = h5py.File("/home/baptiste/Documents/Datasets/BioBanqCovid/meatbolomics_BioBanQ.hdf5", 'r')
for view_ind in range(dset_file["Metadata"].attrs["nbView"]):
    X = dset_file["View{}".format(view_ind)][...]
    y = dset_file["Labels"][...]
    print(X.shape, y.shape)
    view_name = dset_file['View{}'.format(view_ind)].attrs["name"]

    print(X.shape)
    fig = plot_step(X, y, normalizer=RobustScaler(), title=view_name)

    fig.write_image("figures/{}_converge_norm.pdf".format(view_name), width=1000, height=500)