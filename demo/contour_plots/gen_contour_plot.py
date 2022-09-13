from SamBA.samba import NeighborHoodClassifier, ZeroOneTrainWeighting
from SamBA.distances import *
from SamBA.relevances import *
import plotly
from sklearn.datasets import make_moons, make_gaussian_quantiles, make_blobs, load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from plotly import graph_objs as go
import os
import math
from PIL import Image
from sklearn.preprocessing import RobustScaler


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


def plot_contour_adaboost(X, y, clf=AdaBoostClassifier(), contour=True, n_estimator=None):
    fig = go.Figure()
    labels = np.unique(y)
    preds = [pred for pred in clf.staged_decision_function(X)]
    if n_estimator is not None:
        preds = preds[n_estimator-1]
    else:
        preds = preds[-1]
    signs = np.sign(preds)
    support_feats = np.argsort(-clf.feature_importances_)[:2]
    symbols = np.array(["x" if label == y[0] else "circle" for label in y])
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
                                 marker=dict(
                                     size=5, symbol=symbols[indices])))
    scaled_preds = preds.copy()
    # print(np.unique(scaled_preds))
    if len(np.unique(scaled_preds))!=2:
        neg = np.where(scaled_preds<0)
        pos = np.where(scaled_preds>0)
        scaled_preds[neg] /=-np.max(scaled_preds[neg])
        scaled_preds[pos] /= np.min(scaled_preds[pos])
        scaled_preds = np.log(np.abs(scaled_preds))
        scaled_preds[neg] /= np.max(scaled_preds[neg])
        scaled_preds[pos] /= np.max(scaled_preds[pos])
        scaled_preds *= signs
    # print(scaled_preds)
    if contour:
        fig.add_trace(go.Contour(
            z=-scaled_preds,
            x=X[:, support_feats[0]],
            y=X[:, support_feats[1]],
            line_smoothing=0.85,
            contours_coloring='heatmap',
            colorscale='RdBu',
            showscale=False
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          showlegend=False,
                          margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
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


def plot_contour(datasets=["spiral", "moons", "circles", ], noise=None,
                 n_samples=400, rs=np.random.RandomState(42), b=2, a=0.1,
                 pred_train=True, n_estimators=2):
    for dataset in datasets:
        if dataset=="moons":
            X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.1, random_state=rs)
        elif dataset=="circles":
            X, y = make_gaussian_quantiles(n_samples=n_samples, n_features=2,
                                           n_classes=2,
                                           shuffle=True, random_state=rs)
        elif dataset=="spiral":
            # b=1
            theta_mul = 3
            N = int(n_samples/2)
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

        elif dataset == "spiral_unif":
            mul=3
            thetas = rs.uniform(0, 2*np.pi, size=n_samples)
            rays = rs.uniform(size=n_samples)
            y = np.zeros(n_samples)
            print(np.mean(rays/thetas))
            for ind, (r, theta) in enumerate(zip(rays, thetas)):
                if 0.1<r / theta <0.2 :
                    y[ind]=0
                else:
                    y[ind]=1
            X = np.array([rays*np.cos(thetas), rays*np.sin(thetas)]).transpose()

        # print(X)
        # print(y)
        # zers = np.where(y == 0)[0]
        # ones = np.where(y==1)[0]
        # print(len(zers), len(ones), len(zers)/len(ones))
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=X[zers, 0], y=X[zers, 1],mode='markers'))
        # fig.add_trace(go.Scatter(x=X[ones, 0], y=X[ones, 1],mode='markers'))
        # fig.show()

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)

        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        print(X.shape, y.shape)
        if noise is not None:
            noisy_y = add_label_noise(y_train, noise_percentage=noise, rs=rs)
        else:
            noisy_y = y_train.copy()
            noise = 0
        clf = NeighborHoodClassifier(n_estimators=n_estimators,
                                     normalizer=None,
                                     relevance=ExpRelevance(),
                                     distance=EuclidianDist(),
                                     b=b,
                                     a=a,
                                     pred_train=pred_train,
                                     forced_diversity=True)

        clf.fit(X_train, noisy_y)
        ada = AdaBoostClassifier(n_estimators=20, base_estimator=DecisionTreeClassifier(max_depth=1))
        ada.fit(X_train, noisy_y)
        # print(ada.decision_function(X))
        # quit()
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        from sklearn.metrics import accuracy_score
        print("ada : ", accuracy_score(ada.predict(X_train), y_train), accuracy_score(ada.predict(X_test), y_test))
        print("Train score : {}".format(accuracy_score(y_train, y_pred_train)))
        print("Test score : {}".format(accuracy_score(y_test, y_pred_test)))
        # y[train_index] = noisy_y+0.5
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((noisy_y, y_test), axis=0)
        print("Plotting {}".format(dataset))
        import os
        fig = clf._plot_2d(X_train, y_train, sec_dim=None, contour=True,
                            random_state=42,
                            n_estimators=None,
                            title="title", template="plotly")
        fig.write_image("{}_{}.pdf".format(dataset, clf.n_estimators),)
        fig = plot_contour_adaboost(X_train, y_train, clf=ada)
        fig.write_image("{}_{}_ada.pdf".format(dataset, ada.n_estimators),)
        clf.plot_contour_gif(X_train, y_train,
                             save_path="contour_{}_{}.gif".format(dataset, int(
                                 noise * 100)),
                             title="Train noise : {}".format(
                                 noise * 100))
        plot_contour_gif_adaboost(X_train, y_train, ada=ada,
                                  save_path="contour_ada_{}_{}.gif".format(
                                      dataset,
                                      int(
                                          noise * 100)),
                                  title="Train noise : {}".format(
                                      noise * 100))



# print("Breast Cancer")
#
# X, y = load_breast_cancer(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
#                                                     shuffle=True,
#                                                     random_state=rs)
# clf = NeighborHoodClassifier(n_estimators=15,
#                              normalizer=RobustScaler())
# clf.fit(X_train, y_train)
# clf.plot_contour_gif(X, y, save_path="contour_bc.gif")
# plotly.offline.plot(fig, filename="contour_bc.html",
#                             auto_open=False)

# print("Four Blobs")
# from SamBA.utils import gen_four_blobs
#
# X, y = gen_four_blobs(rs=rs, n_samples=2*n_samples, unit=int(n_samples/4)-30,
#                       n_pos=int(n_samples/2))
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
#                                                     shuffle=True,
#                                                     random_state=rs)
# noisy_y = add_label_noise(y_train)
# clf = NeighborHoodClassifier(n_estimators=15,
#                              normalizer=None,
#                              relevance=MarginRelevance(),
#                              distance=EuclidianDist())
# clf.fit(X_train, noisy_y)
# clf.plot_contour_gif(X, y, save_path="contour_four_blobs_noisy.gif")

if __name__=="__main__":
    plot_contour(n_samples=1000, noise=None, datasets=["spiral_unif",], n_estimators=5, b=6, a=0.00002, pred_train=False, rs=np.random.RandomState(42))
    # plot_contour(noise=.10, n_samples=600, datasets=["moons"], b=1, rs=np.random.RandomState(43))