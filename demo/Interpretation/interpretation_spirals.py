import numpy as np
import plotly.express as px
from sklearn.model_selection import StratifiedShuffleSplit
from samba.samba import SamBAClassifier
from samba.distances import EuclidianDist
from sklearn.metrics import balanced_accuracy_score


def make_spirals(n_samples, rs):
    theta_mul = 3
    N = int(n_samples / 2)
    theta = np.sqrt(
        rs.rand(N)) * 4 * np.pi

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

clf = SamBAClassifier(n_estimators=17, forced_diversity=True, distance=EuclidianDist(), b=4, a=1e-10)
n_splits = 1
rs = np.random.RandomState(42)
X, y = make_spirals(n_samples=1000, rs=rs)

sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=rs)
for split_ind, (train_index, test_index) in enumerate(sss.split(X, y)):
    print("")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    print("\t", clf.__class__.__name__,
              balanced_accuracy_score(y_test, clf.predict(X_test)))
    print(X.shape)
    mat = clf.single_sample_importances(X_test)
    sum_mat = np.sum(mat, axis=0)
    non_zer = np.where(sum_mat!=0)[0]
    most_var = []
    for ind in non_zer:
        if np.argmax(mat[:, ind]) not in most_var:
            most_var.append(np.argmax(mat[:, ind]))
        if np.argmin(mat[:, ind]) not in most_var:
            most_var.append(np.argmin(mat[:, ind]))
    good_feats = mat[:, non_zer]
    fig = px.imshow(np.transpose(np.round(good_feats[[14,105,158,52,71, 69]], 2)), aspect="auto",
                    color_continuous_scale=["white", "black"], text_auto=True)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      showlegend=False,
                      coloraxis_showscale=False,
                      xaxis_title="Test Samples",
                      yaxis_title = "Support Features",
                      font_size=25)
    fig.update_xaxes(visible=True, )
    fig.update_yaxes(visible=True, showgrid=False, tickmode = 'linear',
                     tick0 = 1, dtick = 1)
    fig.write_image("figures/feature_importances_samples.pdf", width=1000, height=500 )
    # fig.show()