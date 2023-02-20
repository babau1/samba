import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score


def add_label_noise(y_train, noise_percentage=0.05, rs=np.random.RandomState(42)):
    ones = np.where(y_train == 1)[0]
    zeros = np.where(y_train == 0)[0]

    noisy_ones = rs.choice(ones, size=int(y_train.shape[0] * noise_percentage / 2),
                           replace=False)
    noisy_zeros = rs.choice(zeros, size=int(y_train.shape[0] * noise_percentage / 2),
                            replace=False)
    y_train[noisy_ones] = 0
    y_train[noisy_zeros] = 1

    return y_train


def test_noise(X, y, models, noise_levels=[0.05, 0.1,
                                           0.15, 0.2, 0.25, 0.30, 0.35, 0.40,
                                           0.45, 0.46, 0.47, 0.48, 0.49],
               n_splits=5, rs=np.random.RandomState(42), train_size=0.5,
               param_grid={"n_estimators":[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
               name="", n_jobs=4):
    res_df = pd.DataFrame(columns=["Model Name", "Noise level",
                                   "Accuracy", "STD", "Set"])
    for noise_level in noise_levels:
        print("Noise level ", noise_level)
        for b in [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4]:
            train_accs = np.zeros((n_splits, len(models)))
            test_accs = np.zeros((n_splits, len(models)))
            for split_ind in range(n_splits):
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    shuffle=True,
                                                                    random_state=rs,
                                                                    train_size=train_size)
                y_train = add_label_noise(y_train, noise_percentage=noise_level)
                for model_ind, model in enumerate(models):
                    if hasattr(model, 'b'):
                        model.b = b
                    clf = GridSearchCV(model,
                                       param_grid=param_grid,
                                       cv=StratifiedKFold(n_splits=n_splits,
                                                          shuffle=True,
                                                          random_state=rs),
                                       n_jobs=n_jobs)
                    search = clf.fit(X_train, y_train)
                    train_preds = search.predict(X_train)
                    train_accs[split_ind, model_ind] = accuracy_score(y_train, train_preds)
                    test_preds = search.predict(X_test)
                    test_accs[split_ind, model_ind] = accuracy_score(y_test, test_preds)

            for model_ind, model in enumerate(models):
                train_acc = np.mean(train_accs[:, model_ind])
                train_std = np.std(train_accs[:, model_ind])
                res_df = res_df.append({"Model Name": model.__class__.__name__,
                                        "Noise level": noise_level,
                                        "Accuracy": train_acc,
                                        "STD": train_std,
                                        "Set": "Train",
                                        "b": b},
                                       ignore_index=True)
                test_acc = np.mean(test_accs[:, model_ind])
                test_std = np.std(test_accs[:, model_ind])
                res_df = res_df.append({"Model Name": model.__class__.__name__,
                                        "Noise level": noise_level,
                                        "Accuracy": test_acc,
                                        "STD": test_std,
                                        "Set": "Test",
                                        "b":b},
                                       ignore_index=True)

    res_df.to_csv("figures/noise_robustness_df_exp_{}.csv".format(name))


if __name__=='__main__':
    from sklearn.datasets import make_moons, make_classification
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from SamBA.samba import NeighborHoodClassifier
    from SamBA.distances import *
    from SamBA.relevances import *
    from SamBA.samba import ExpTrainWeighting
    from SamBA.utils import gen_four_blobs

    rs = np.random.RandomState(43)
    for dataset in ["moons", "mk_clf", "four_blobs"]:
        if dataset == "four_blobs":
            X, y = gen_four_blobs(rs=rs, n_samples=1000)
        elif dataset == "moons":
            X, y = make_moons(n_samples=1000, shuffle=True, random_state=rs, noise=0.1)
        elif dataset == "mk_clf":
            X, y = make_classification(n_samples=1000, n_features=20, n_informative=3,
                                        n_redundant=2, n_repeated=0, n_classes=2,
                                        n_clusters_per_class=3, weights=None, flip_y=0,
                                        class_sep=1.0, hypercube=True, shift=0.0,
                                        scale=1.0, shuffle=True, random_state=rs)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                            shuffle=True,
                                                            random_state=rs)
        samba = NeighborHoodClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1,
                                                           splitter='best',
                                                           criterion='gini'),
                    n_estimators=2,
                    train_weighting=ExpTrainWeighting(),
                    relevance=ExpRelevance(),
                    distance=EuclidianDist(),
                    pred_train=False)
        adabooost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,
                                                           splitter='best',
                                                           criterion='gini'))
        test_noise(X, y, models=[samba, adabooost], rs=rs, name=dataset,
                   n_jobs=4)