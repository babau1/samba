from SamBA.samba import NeighborHoodClassifier
from SamBA.distances import *
from SamBA.relevances import *
from sklearn.datasets import make_moons, make_gaussian_quantiles
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from plotly import graph_objs as go
import os
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from randomscm.randomscm import RandomScmClassifier
from cb_boost.cb_boost import CBBoostClassifier
from pyscm.scm import SetCoveringMachineClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline


def add_label_noise(y_train, noise_percentage=0.10, rs=None):
    ones = np.where(y_train == 1)[0]
    zeros = np.where(y_train == 0)[0]

    noisy_ones = rs.choice(ones, size=int(y_train.shape[0] * noise_percentage / 2),
                           replace=False)
    noisy_zeros = rs.choice(zeros, size=int(y_train.shape[0] * noise_percentage / 2),
                            replace=False)
    y_train[noisy_ones] = 0
    y_train[noisy_zeros] = 1

    return y_train


def generate_mesh_space(X, n_steps=10, ):
    mesh_space_x1, mesh_space_x2 = np.meshgrid(
        np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), n_steps),
        np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), n_steps))
    mesh_space = np.concatenate(((mesh_space_x1.flatten()).reshape(
        (n_steps ** 2, 1)), (mesh_space_x2.flatten()).reshape(
        (n_steps ** 2, 1))),
        axis=1)
    return mesh_space


def plot_contour_adaboost(X, y, clf=AdaBoostClassifier(), contour=True,
                          n_estimator=None, test_preds=None, test_samples=None,
                          test_labels=None, size=5, n_steps=10,
                          symbols=["x", "circle"]):
    fig = go.Figure()
    labels = np.unique(y)
    preds = clf.predict(X)

    support_feats = [0,1]
    colors = np.array(["Blue" if label == 1 else "Red" for label in preds])
    if test_preds is not None:
        colors_test = np.array(
            ["Blue" if label == 1 else "Red" for label in test_preds])
    for label in labels:
        if contour:
            opacity = 0.9
        else:
            opacity = 0.7
        indices = np.where(y == label)[0]
        fig.add_trace(go.Scatter(x=X[indices, support_feats[0]],
                                 y=X[indices, support_feats[1]],
                                 opacity=opacity,
                                 name="Class {}".format(label + 1),
                                 mode="markers",
                                 marker=dict(symbol=symbols[int(label)],
                                     size=size, color=colors[indices])))
        if test_samples is not None:
            indices = np.where(np.sign(test_labels) == label)[0]
            fig.add_trace(go.Scatter(x=test_samples[indices, support_feats[0]],
                                     y=test_samples[indices, support_feats[1]],
                                     opacity=opacity,
                                     name="Class {}".format(label + 1),
                                     mode="markers",
                                     marker=dict(symbol=symbols[int(label)],
                                                 size=2 * size,
                                                 color=colors_test[indices])))

    if contour:
        mesh_space = generate_mesh_space(X, n_steps=n_steps)
        fig.add_trace(go.Contour(
            z=clf.predict_proba(mesh_space)[:, 1],
            x=mesh_space[:, support_feats[0]],
            y=mesh_space[:, support_feats[1]],
            line_smoothing=0.85,
            contours_coloring='heatmap',
            colorscale='RdBu',
            showscale=False,
            # contours=dict(
            #     start=0,
            #     end=1,
            #     size=1e-3,
            # )
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          showlegend=False,
                          margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(visible=False, showgrid=False)
        fig.update_yaxes(visible=False, showgrid=False)
    return fig


def plot_contour_gif_adaboost(X, y, ada=AdaBoostClassifier(),
                         temp_folder="temp/", save_path=None, title="",
                         template="plotly"):
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)
        for estim_index in range(ada.n_estimators):
            fig = plot_contour_adaboost(X, y, clf=ada, contour=True,
                                        n_estimator=estim_index+1,)
            fig.write_image(os.path.join(temp_folder, "gif_{}.png").format(estim_index), scale=2.0)

        img, *imgs = [Image.open(os.path.join(temp_folder, "gif_{}.png").format(i)) for i in range(ada.n_estimators)]
        img.save(fp=save_path, format='GIF', append_images=imgs,
                 save_all=True, duration=200, loop=0)

        for fname in os.listdir(temp_folder):
            os.remove(os.path.join(temp_folder, fname))
        os.rmdir(temp_folder)

def plot_contour_img(classifiers, X_train, y_train, X_test, y_test, n_steps, dataset,
                 show=False ):
    for classifier in classifiers:
        fig = plot_contour_adaboost(X_train, y_train, clf=classifier,
                                            test_samples=X_test, test_labels=y_test,
                                            test_preds=classifier.predict(X_test),
                                            n_steps=n_steps)

        fig.update_layout(
            {"xaxis_showticklabels": False, "yaxis_showticklabels": False})


        fig.write_image(os.path.join("figures", "{}_{}.pdf".format(dataset, classifier.__class__.__name__), ))
        fig.write_image(os.path.join("figures", "{}_{}.png".format(dataset,
                                                                   classifier.__class__.__name__), ))
        fig.update_layout(title=classifier.__class__.__name__,
                          xaxis_title="classifier.__class__.__name__",
                          showlegend=True
                          )
        if show:
            fig.show()


def make_spirals(n_samples, rs):
    theta_mul = 3
    N = int(n_samples / 2)
    theta = np.sqrt(
        rs.rand(N)) * 4 * np.pi  # np.linspace(0,2*pi,100)

    r_a = theta_mul * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + rs.randn(N, 2)

    r_b = -theta_mul * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + rs.randn(N, 2)

    res_a = np.append(x_a, np.zeros((N, 1)), axis=1)
    res_b = np.append(x_b, np.ones((N, 1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    rs.shuffle(res)

    X = res[:, :2]
    y = res[:, -1]
    return X, y

def plot_contour(datasets=["spiral", "moons", "circles", ], noise=None,
                 n_samples=400, rs=np.random.RandomState(42), b=2, a=0.1,
                 pred_train=True, n_estimators=2, n_splits=10, n_steps=200,
                 classifiers = [AdaBoostClassifier()]):
    cols = [("Dataset", "")]+ [(clf.__class__.__name__, set) for clf in classifiers for set in ["Train", "Test"] ]
    res_df = pd.DataFrame(columns=cols)
    res_df.columns = pd.MultiIndex.from_tuples(res_df.columns, names=['Algorithm', 'Set'])

    for dataset in datasets:
        print(dataset)
        if dataset=="Moons":
            X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.1,
                              random_state=rs)
        if dataset=="Moons Noisy":
            X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.1,
                              random_state=rs)
        elif dataset=="circles":
            X, y = make_gaussian_quantiles(n_samples=n_samples, n_features=2,
                                           n_classes=2,
                                           shuffle=True, random_state=rs)
        elif dataset=="Spirals":
            X, y = make_spirals(n_samples, rs)
        elif dataset == "Spirals Noisy":
            X, y = make_spirals(n_samples, rs)
        elif dataset == "spiral_unif":
            thetas = rs.uniform(0, 2*np.pi, size=n_samples)
            rays = rs.uniform(size=n_samples)
            y = np.zeros(n_samples)
            for ind, (r, theta) in enumerate(zip(rays, thetas)):
                if 0.1<r / theta <0.2 :
                    y[ind]=0
                else:
                    y[ind]=1
            X = np.array([rays*np.cos(thetas), rays*np.sin(thetas)]).transpose()
        if "Noisy" in dataset:
            noisy_dims = rs.uniform(low=np.min(X), high=np.max(X),
                                    size=(X.shape[0], 50), )
            X = np.concatenate((X, noisy_dims), axis=1)

        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=rs)

        accs = np.zeros((len(classifiers), 2, n_splits,))

        for split_ind, (train_index, test_index) in enumerate(sss.split(X, y)):
            print("")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if noise is not None:
                noisy_y = add_label_noise(y_train, noise_percentage=noise, rs=rs)
            else:
                noisy_y = y_train.copy()
                noise = 0
            for clf_index, clf in enumerate(classifiers):
                clf.fit(X_train, noisy_y)
                accs[clf_index, 0, split_ind] = accuracy_score(y_train,clf.predict(X_train))
                accs[clf_index, 1, split_ind] = accuracy_score(y_test, clf.predict(X_test))
                print("\t", clf.__class__.__name__, accuracy_score(y_test, clf.predict(X_test)))
        mean_accs = accs.mean(axis=2).round(2)
        std_accs = accs.std(axis=2).round(2)
        con_dict = dict(((clf.__class__.__name__, set),
                                     "${}$ \scriptsize$\pm {}$".format(mean_accs[clf_ind,set_ind],
                                                                       std_accs[clf_ind,set_ind]))
                                    for clf_ind, clf in enumerate(classifiers)
                                    for set_ind, set in enumerate(['Train', 'Test']))
        con_dict[('Dataset', "")] = dataset

        res_df = res_df.append(con_dict,
                               ignore_index=True)
        print(res_df.to_latex(escape=False, index=False,
                              column_format="|l|"+"".join(["c|" for _ in range(len(classifiers)*2)])))

        if "Noisy" not in dataset:
            plot_contour_img(classifiers, X_train, y_train, X_test,
                             y_test, n_steps, dataset)


if __name__=="__main__":
    log_reg_regu = LogisticRegression(C=10)
    log_reg_n_regu = LogisticRegression(C=10001.0)
    spline = SplineTransformer(n_knots=4, knots="quantile", degree=4)
    classifiers = [
        # LogisticRegression(Cs=10,),
        LogisticRegression(C=10001.0, ),
        Pipeline([("spline", spline), ("ridge", log_reg_n_regu)]),
        AdaBoostClassifier(),
                   NeighborHoodClassifier(n_estimators=4,
                                         normalizer=None,
                                         relevance=ExpRelevance(),
                                         distance=EuclidianDist(),
                                         b=20,
                                         a=1e-15,
                                         pred_train=False,
                                         # forced_diversity=False,
                                          forced_diversity=True,
                                          ),
                   SVC(probability=True, C=0.1, gamma=1.1),
                   KNeighborsClassifier(),
                   RandomForestClassifier(),
                   DecisionTreeClassifier(),
                   RandomScmClassifier(),
                   CBBoostClassifier(n_stumps=200), SetCoveringMachineClassifier()
    ]
    plot_contour(n_samples=1000, noise=None,
                 # datasets=['Moons', 'Spirals', "Moons Noisy", "Spirals Noisy",],# "moons", "circles", "spiral"],
                 datasets=["Spirals", "Moons"],
                 rs=np.random.RandomState(42), classifiers=classifiers,
                 # n_splits=10,
                 n_splits=1,
                 n_steps=500)