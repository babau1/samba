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
from SamBA.distances import EuclidianDist, MixedDistance, Jaccard
from sklearn.preprocessing import RobustScaler
import plotly.express as px
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Balance : Jaccard
# Abalone : [Euclidian(), Jaccard()], [[0,1,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,0,0,]]
# Australian : [Euclidian(), Jaccard()], [[0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0]]
# BUPA : Euclidean()
# Cylinder : [Euclidean(), Jaccard()], [[0,0,0,0,0,
#                                        0,0,0,0,0,
#                                        0,0,0,0,0,
#                                        0,0,0,0,0,
#                                        1,1,1,1,1,
#                                        1,1,1,1,1,
#                                        1,1,1,1,1,
#                                        1,1,1,1,0],
#                                       [1,1,1,1,1,
#                                        1,1,1,1,1,
#                                        1,1,1,1,1,
#                                        1,1,1,1,1,
#                                        0,0,0,0,0,
#                                        0,0,0,0,0,
#                                        0,0,0,0,0,
#                                        0,0,0,0,1]]
# hepatitis : [Euclidean(), Jaccard()], [[1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0],[0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1]]
# Ionosphere : Euclidean()
# Yeast : Euclidean()

def plot_step(X, y, n_estim=50, rs=np.random.RandomState(42), b=6, a=0.0001, forced_diversity=False, normalizer=None, n_iter=10, title=""):
    test_accuracies_ada_full = np.zeros(shape=(n_iter, n_estim))
    test_accuracies_samba_full = np.zeros(shape=(n_iter, n_estim))
    train_accuracies_ada_full = np.zeros(shape=(n_iter, n_estim))
    train_accuracies_samba_full = np.zeros(shape=(n_iter, n_estim))
    for i in range(n_iter):
        print(i, n_iter)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs, shuffle=True, train_size=0.7)

        classifier = NeighborHoodClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1,
                                                  splitter='best'),
            distance=MixedDistance(distances=[EuclidianDist(), Jaccard()],
                                   feature_map=[[0, 1, 1, 0, 0, 0, 1, 0, 0, 1,
                                                 0, 0, 1, 1],
                                                [1, 0, 0, 1, 1, 1, 0, 1, 1, 0,
                                                 1, 1, 0, 0]]),
            relevance=ExpRelevance(),
            n_estimators=n_estim,
            normalizer=normalizer,
            forced_diversity=forced_diversity,
            b=b, a=a, pred_train=False, normalize_dists=True)

        param_distributions = {"a": uniform(), "b":uniform(loc=0, scale=8), "n_estimators":randint(low=1, high=30), "pred_train":[False]}
        search1 = RandomizedSearchCV(classifier, param_distributions, n_iter=50,
                                     scoring="balanced_accuracy", n_jobs=5,
                                     random_state=42)

        ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1,
                                                                       splitter='best'),
                                 n_estimators=n_estim)

        classifier = classifier.fit(X_train, y_train)
        print("Samba", len(np.where(classifier.feature_importances_!=0)[0]))
        ada = ada.fit(X_train, y_train)
        print("Ada", len(np.where(ada.feature_importances_ != 0)[0]))

        classifier.fit(X_train, y_train)
        ada.fit(X_train, y_train)

        train_step_preds_samba = classifier.step_predict(X_train)
        train_step_preds_ada = np.transpose(
            np.array([preds for preds in ada.staged_predict(X_train)]))

        test_step_preds_samba = classifier.step_predict(X_test)
        test_step_preds_ada = np.transpose(
            np.array([preds for preds in ada.staged_predict(X_test)]))

        preds_df = pd.DataFrame(columns=["x0", "x1", "y", "pred", "iterations", "Balanced accuracy"])
        train_accuracies_samba = [balanced_accuracy_score(y_train, pred) for pred in np.transpose(train_step_preds_samba)]
        train_accuracies_ada = [balanced_accuracy_score(y_train, pred) for pred in np.transpose(train_step_preds_ada)]
        test_accuracies_samba = [balanced_accuracy_score(y_test, pred) for pred in np.transpose(test_step_preds_samba)]
        test_accuracies_ada = [balanced_accuracy_score(y_test, pred) for pred in np.transpose(test_step_preds_ada)]

        train_accuracies_samba += [train_accuracies_samba[-1] for i in range(n_estim - len(np.transpose(train_step_preds_samba)))]
        train_accuracies_ada += [train_accuracies_ada[-1] for i in range(n_estim - len(np.transpose(train_step_preds_ada)))]
        test_accuracies_samba +=  [test_accuracies_samba[-1] for i in range(n_estim - len(np.transpose(test_step_preds_samba)))]
        test_accuracies_ada += [test_accuracies_ada[-1] for i in range(n_estim - len(np.transpose(test_step_preds_ada)))]

        print("Ada ", train_accuracies_ada[-1], test_accuracies_ada[-1])
        print("Samba ", train_accuracies_samba[-1], test_accuracies_samba[-1])

        test_accuracies_ada_full[i] = test_accuracies_ada
        test_accuracies_samba_full[i] = test_accuracies_samba
        train_accuracies_ada_full[i] = train_accuracies_ada
        train_accuracies_samba_full[i] = train_accuracies_samba

    train_accuracies_ada = np.mean(train_accuracies_ada_full, axis=0)
    test_accuracies_ada = np.mean(test_accuracies_ada_full, axis=0)
    train_accuracies_samba = np.mean(train_accuracies_samba_full, axis=0)
    test_accuracies_samba = np.mean(test_accuracies_samba_full, axis=0)

    train_accuracies_ada_std = np.std(train_accuracies_ada_full, axis=0)
    test_accuracies_ada_std = np.std(test_accuracies_ada_full, axis=0)
    train_accuracies_samba_std = np.std(train_accuracies_samba_full, axis=0)
    test_accuracies_samba_std = np.std(test_accuracies_samba_full, axis=0)

    for estim_ind, acc in enumerate(train_accuracies_samba):
        preds_df = preds_df.append({"iterations":estim_ind+1, "Balanced accuracy":acc, "Classifier":"SamBA", "Set":"Train", "STD":train_accuracies_samba_std[estim_ind]}, ignore_index=True)
    for estim_ind, acc in enumerate(train_accuracies_ada):
        preds_df = preds_df.append({"iterations":estim_ind+1, "Balanced accuracy":acc, "Classifier":"Adaboost", "Set":"Train", "STD":train_accuracies_ada_std[estim_ind]}, ignore_index=True)
    for estim_ind, acc in enumerate(test_accuracies_samba):
        preds_df = preds_df.append({"iterations":estim_ind+1, "Balanced accuracy":acc, "Classifier":"SamBA", "Set":"Test", "STD":test_accuracies_samba_std[estim_ind]}, ignore_index=True)
    for estim_ind, acc in enumerate(test_accuracies_ada):
        preds_df = preds_df.append({"iterations": estim_ind+1, "Balanced accuracy": acc, "Classifier": "Adaboost", "Set":"Test", "STD":test_accuracies_ada_std[estim_ind]}, ignore_index=True)
    print("Ada ", train_accuracies_ada[-1], test_accuracies_ada[-1])
    print("Samba ", train_accuracies_samba[-1], test_accuracies_samba[-1])
    fig = px.line(preds_df, x="iterations", y="Balanced accuracy", error_y="STD", color="Classifier", line_dash="Set")
    fig.update_layout(
        title=title+", a={}, b={}".format(a, b),
        font=dict(
            family="Computer Modern",
            size=18,),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray',
                     showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray',
                     showline=True, linewidth=2, linecolor='black', mirror=True)
    return fig

# X = np.load("/home/baptiste/Documents/Datasets/cardiac_montreal/Cardiac_risk_LCMS/pos.npy")
# y = np.load("/home/baptiste/Documents/Datasets/cardiac_montreal/Cardiac_risk_LCMS/labels_pos.npy")
#
# fig = plot_step(X, y, normalizer=RobustScaler(), title="Cardiac LCMS")
#
# fig.write_image("cardiac_converge_norm.png", width=1000, height=500)


for file_name in ["/home/baptiste/Documents/Datasets/UCI/both/australian.hdf5",
                  "/home/baptiste/Documents/Datasets/study_med/study_med_soyless.hdf5",
                  "/home/baptiste/Documents/Datasets/recover/hdf5/recover.hdf5",
                  "/home/baptiste/Documents/Datasets/QIN/QIN.hdf5",
                  "/home/baptiste/Documents/Datasets/Metagenomes/metagenomes-relab/metagenome.hdf5"]:
    dset_file = h5py.File(file_name, 'r')
    for view_ind in range(dset_file["Metadata"].attrs["nbView"]):
        X = dset_file["View{}".format(view_ind)][...]
        y = dset_file["Labels"][...]

        view_name = dset_file['View{}'.format(view_ind)].attrs["name"]
        print(X.shape)
        fig = plot_step(X, y, title=view_name)

        fig.write_image("{}_converge_norm.png".format(view_name), width=1000, height=500)
    quit()