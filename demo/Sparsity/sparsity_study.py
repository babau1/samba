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


def plot_step(X, y, n_estim=20, rs=np.random.RandomState(42), b=2.0, a=0.1, forced_diversity=False, normalizer=RobustScaler(), title=""):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs, shuffle=True)

    classifier = NeighborHoodClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1,
                                              splitter='best'),
        distance=EuclidianDist(),
        relevance=ExpRelevance(),
        n_estimators=n_estim,
        normalizer=normalizer,
        forced_diversity=forced_diversity,
        b=b, a=a, pred_train=True, normalize_dists=True)

    param_distributions = {"a": uniform(), "b":uniform(loc=0, scale=8), "n_estimators":randint(low=1, high=30), "pred_train":[False]}
    search1 = RandomizedSearchCV(classifier, param_distributions, n_iter=50,
                                 scoring="balanced_accuracy", n_jobs=5,
                                 random_state=42)

    ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1,
                                                                   splitter='best'),
                             n_estimators=n_estim)
    param_distributions = {"n_estimators": randint(low=1, high=30)}
    search2 = RandomizedSearchCV(ada, param_distributions, n_iter=50,
                                 scoring="balanced_accuracy", n_jobs=5,
                                 random_state=42)
    classifier = search1.fit(X_train, y_train).best_estimator_
    cl_estim = search1.best_params_['n_estimators']
    print(search1.best_params_)
    ada = search2.fit(X_train, y_train).best_estimator_
    print(search2.best_params_)
    ada_estim = search1.best_params_['n_estimators']
    classifier.fit(X_train, y_train)
    ada.fit(X_train, y_train)

    n_estim = max(ada_estim, cl_estim)

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

    for estim_ind, acc in enumerate(train_accuracies_samba):
        preds_df = preds_df.append({"iterations":estim_ind+1, "Balanced accuracy":acc, "Classifier":"SamBA", "Set":"Train"}, ignore_index=True)
    for estim_ind, acc in enumerate(train_accuracies_ada):
        preds_df = preds_df.append({"iterations":estim_ind+1, "Balanced accuracy":acc, "Classifier":"Adaboost", "Set":"Train"}, ignore_index=True)
    for estim_ind, acc in enumerate(test_accuracies_samba):
        preds_df = preds_df.append({"iterations":estim_ind+1, "Balanced accuracy":acc, "Classifier":"SamBA", "Set":"Test"}, ignore_index=True)
    for estim_ind, acc in enumerate(test_accuracies_ada):
        preds_df = preds_df.append({"iterations": estim_ind+1, "Balanced accuracy": acc, "Classifier": "Adaboost", "Set":"Test"}, ignore_index=True)
    fig = px.line(preds_df, x="iterations", y="Balanced accuracy", color="Classifier", line_dash="Set")
    fig.update_layout(
        # title=title,
        font=dict(
            family="Computer Modern",
            size=18,),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray',showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray',showline=True, linewidth=2, linecolor='black', mirror=True)
    return fig

# X = np.load("/home/baptiste/Documents/Datasets/cardiac_montreal/Cardiac_risk_LCMS/pos.npy")
# y = np.load("/home/baptiste/Documents/Datasets/cardiac_montreal/Cardiac_risk_LCMS/labels_pos.npy")
#
# fig = plot_step(X, y, b=2.0, normalizer=RobustScaler(), title="Cardiac LCMS")
#
# fig.write_image("cardiac_converge_norm.png", width=1000, height=500)


for file_name in ["/home/baptiste/Documents/Datasets/UCI/both/ionosphere.hdf5",
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
        fig = plot_step(X, y, b=2.0, normalizer=RobustScaler(), title=view_name)

        fig.write_image("figures/{}_converge_norm.png".format(view_name), width=1000, height=500)
    quit()