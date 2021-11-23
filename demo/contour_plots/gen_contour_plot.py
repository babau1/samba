from SamBA.samba import NeighborHoodClassifier, ZeroOneTrainWeighting
from SamBA.distances import *
from SamBA.relevances import *
import plotly
from sklearn.datasets import make_moons, make_gaussian_quantiles, make_blobs, load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
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


def plot_contour(datasets=["spiral", "moons", "circles", ], noise=None, n_samples=400, rs=np.random.RandomState(42), b=2):
    for dataset in datasets:
        if dataset=="moons":
            X, y = make_moons(n_samples=n_samples, shuffle=True, noise=0.1, random_state=rs)
        elif dataset=="circles":
            X, y = make_gaussian_quantiles(n_samples=n_samples, n_features=2,
                                           n_classes=2,
                                           shuffle=True, random_state=rs)
        elif dataset=="spiral":
            b=1
            theta_mul = 1.5
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
        clf = NeighborHoodClassifier(n_estimators=15,
                                     normalizer=None,
                                     relevance=ExpRelevance(),
                                     distance=EuclidianDist(),
                                     b=b,
                                     pred_train=False,
                                     forced_diversity=True)

        clf.fit(X_train, noisy_y)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        from sklearn.metrics import accuracy_score

        print("Train score : {}".format(accuracy_score(y_train, y_pred_train)))
        print("Test score : {}".format(accuracy_score(y_test, y_pred_test)))
        y[train_index] = noisy_y+0.5
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((noisy_y, y_test), axis=0)
        print("Plotting {}".format(dataset))
        import os
        fig = clf._plot_2d(X, y, sec_dim=None, contour=True,
                            random_state=42,
                            n_estimators=None,
                            title="title", template="plotly")
        fig.write_image("{}_{}.pdf".format(dataset, clf.n_estimators),)
        # clf.plot_contour_gif(X, y, save_path="contour_{}_{}.gif".format(dataset, int(noise*100)),
        #                      title="Train noise : {}".format(
        #                          noise * 100))

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
    plot_contour(n_samples=200, datasets=["spiral", "moons", "circles", ], b=2)
    # plot_contour(noise=.10, n_samples=600, datasets=["moons"], b=1, rs=np.random.RandomState(43))