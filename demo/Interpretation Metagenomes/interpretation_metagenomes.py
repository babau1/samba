import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import StratifiedShuffleSplit
from SamBA.samba import NeighborHoodClassifier
from SamBA.distances import ExpEuclidianDist, EuclidianDist
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def generate_mesh_space(X, n_steps=10, support_feats=[0,1], rs=np.random.RandomState(42)):
    space = rs.uniform(0,1, size=(n_steps**len(support_feats), X.shape[1]))
    print(space.shape)
    mesh_space_xs = np.meshgrid(*[np.linspace(np.min(X[:, support_feats[i]]),
                                              np.max(X[:, support_feats[i]]),
                                              n_steps)
                                  for i in range(len(support_feats))])
    # mesh_space = np.concatenate(((mesh_space_x.flatten()).reshape(
    #     (n_steps ** 2, 1)) for mesh_space_x in mesh_space_xs),
    #     axis=1)
    for i in range(len(support_feats)):
        space[:, support_feats[i]] = mesh_space_xs[i].flatten()
    return space

def plot_contour_adaboost(X, y, clf=NeighborHoodClassifier(), contour=True,
                          n_estimator=None, test_preds=None, test_samples=None,
                          test_labels=None, size=5, n_steps=10,
                          symbols=["x", "circle"]):
    fig = go.Figure()
    labels = np.unique(y)
    preds = clf.predict(X)
    imps = clf.feature_importances_.copy()
    support_feats = np.zeros(clf.n_estimators, dtype=int)
    for i in range(clf.n_estimators):
        support_feats[i] = np.argmax(imps)
        imps[np.argmax(imps)] = 0
    print(support_feats)


    colors = np.array(["Blue" if label == 1 else "Red" for label in preds])
    if test_preds is not None:
        colors_test = np.array(
            ["Blue" if label == 1 else "Red" for label in test_preds])
    if contour:
        mesh_space = generate_mesh_space(X, n_steps=n_steps,
                                         support_feats=support_feats)
        # probs = clf.predict_proba(X)[:, 1]
        # preds = clf.predict(X)
        # for i in range(X.shape[0]):
        #     print(probs[i], preds[i], y[i])

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
            # ),
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          showlegend=False,
                          margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(visible=False, showgrid=False)
        fig.update_yaxes(visible=False, showgrid=False)
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


    return fig

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

# hdf5_file = h5py.File('/home/baptiste/Documents/Datasets/study_med/study_med.hdf5', 'r')
# # feature_ids = ["GO:0002183", "GO:0000150"]
# #
# # hdf5_file = h5py.File("/home/baptiste/Documents/Datasets/Metagenomes/metagenomes-relab/metagenome.hdf5", "r")
#
# labels = hdf5_file["Labels"][...]
#
# go_dataset = hdf5_file["View0"][...]
# # go_feature_ids = [id_.decode() for id_ in hdf5_file["Metadata"]["feature_ids-View2"][...]]
# # indices = [ind for ind in range(len(go_feature_ids))
# #            if go_feature_ids[ind] in feature_ids]
# # dims = go_dataset[:, indices]
clf = NeighborHoodClassifier(n_estimators=17, forced_diversity=True, distance=EuclidianDist(), b=4, a=1e-10)
# clf = AdaBoostClassifier(n_estimators=2, base_estimator=DecisionTreeClassifier(max_depth=1))
n_splits = 1
rs = np.random.RandomState(42)
X, y = make_spirals(n_samples=1000, rs=rs)
# y = labels

sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=rs)
for split_ind, (train_index, test_index) in enumerate(sss.split(X, y)):
    print("")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    # accs[clf_index, 0, split_ind] = balanced_accuracy_score(y_train,
    #                                                clf.predict(X_train))
    # accs[clf_index, 1, split_ind] = balanced_accuracy_score(y_test,
    #                                                clf.predict(X_test))
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
    fig.show()
# mean_accs = accs.mean(axis=2).round(2)
# std_accs = accs.std(axis=2).round(2)
# con_dict = dict(((clf.__class__.__name__, set),
#                  "${}$ \scriptsize$\pm {}$".format(mean_accs[clf_ind, set_ind],
#                                                    std_accs[clf_ind, set_ind]))
#                 for clf_ind, clf in enumerate(classifiers)
#                 for set_ind, set in enumerate(['Train', 'Test']))
# con_dict[('Dataset', "")] = dataset

# res_df = res_df.append(con_dict,
#                        ignore_index=True)
# print(res_df.to_latex(escape=False, index=False,
#                       column_format="|l|" + "".join(
#                           ["c|" for _ in range(len(classifiers) * 2)])))

# if "Noisy" not in dataset:
#     fig = plot_contour_adaboost(X, y, clf, n_steps=2,)




# fig.show()
# print(dims.shape)
