import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import pandas
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from SamBA.samba import NeighborHoodClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lineartree.lineartree import LinearTreeClassifier
import warnings

class LWLR():

    def __init__(self, tau=0.05, reg=0.0001, threshold=1e-6, random_state=42):
        self.reg = reg
        self.threshold = threshold
        self.tau = tau
        self.random_state = random_state


    def weights(self, x_train, x):
        sq_diff = (x_train - x) ** 2
        norm_sq = sq_diff.sum(axis=1)
        return np.ravel(np.exp(- norm_sq / (2 * self.tau ** 2)))

    def logistic(self, x_train):
        return np.ravel(1 / (1 + np.exp(-x_train.dot(self.theta))))

    def fit(self, X, y, **fit_params):
        self.X = X
        self.y = y

    def train(self, x):
        self.w = self.weights(self.X, x)
        self.theta = np.zeros(self.X.shape[1])
        gradient = np.ones(self.X.shape[1]) * np.inf
        while np.linalg.norm(gradient) > self.threshold:
            # compute gradient
            h = self.logistic(self.X)
            gradient = self.X.T.dot(
                self.w * (np.ravel(self.y) - h)) - self.reg * self.theta
            # Compute Hessian
            D = np.diag(-(self.w * h * (1 - h)))
            H = self.X.T.dot(D).dot(self.X) - self.reg * np.identity(
                self.X.shape[1])
            # weight update
            self.theta = self.theta - np.linalg.inv(H).dot(gradient)

    def predict(self, X):
        preds = []
        for x in X:
            self.train(x)
            preds.append(np.array(self.logistic(X) > 0.5).astype(int)[0])
        return np.array(preds)

def gen_graphs(title="nest", algs=[]):
    """
    Function that generates a random dataset of increasingly high dimension,
    and evaluates the training and predicting durations of all the algorithms
    provided in algs
    """
    rs = np.random.RandomState(42)
    algs_names = ["SamBA", "Adaboost", "KNN", "SVM-RBF", "Random Forest",
                  "Decision Tree", "Lasso", "XGBoost", "Gradient Boosting", "Linear Tree"]
    n_samples_list = [500, 2000, ]
    n_features_list = [10, 100, 1000, 2000, 5000,
                       50000]
    df_train = pandas.DataFrame()
    df_test = pandas.DataFrame()
    for n_features in n_features_list:
        print(n_features)
        for n_samples in n_samples_list:
            print("\t", n_samples)
            for clf, alg_name in zip(algs, algs_names):
                if n_features > 1000 and alg_name=="Linear Tree":
                    df_test = df_test.append(
                        {"Algorithm": alg_name, "# Estimators": n_estimators,
                         "# Samples": n_samples, "# Features": n_features,
                         "Duration (s)": None, "Phase": "Test"},
                        ignore_index=True)
                    df_train = df_train.append(
                        {"Algorithm": alg_name, "# Estimators": n_estimators,
                         "# Samples": n_samples, "# Features": n_features,
                         "Duration (s)": None, "Phase": "Train"},
                        ignore_index=True)
                else:
                    print("\t\t", alg_name)
                    X = rs.uniform(0, 1, size=(n_samples, n_features))
                    y = rs.randint(0, 2, size=n_samples)
                    beg = time.time()
                    clf.fit(X, y)
                    end = time.time()
                    train_duration = end-beg
                    print("\t\t\t Train: ", train_duration)
                    df_train = df_train.append(
                        {"Algorithm": alg_name, "# Estimators": n_estimators,
                         "# Samples": n_samples, "# Features": n_features,
                         "Duration (s)": train_duration, "Phase": "Train"}, ignore_index=True)
                    beg = time.time()
                    _ = clf.predict(X)
                    end = time.time()
                    test_duration = end-beg
                    print("\t\t\t Test: ", test_duration)
                    df_test = df_test.append(
                        {"Algorithm": alg_name, "# Estimators": n_estimators,
                         "# Samples": n_samples, "# Features": n_features,
                         "Duration (s)": test_duration, "Phase": "Test"}, ignore_index=True)

    # Plot train durations

    fig = px.line(df_train, x="# Features", y="Duration (s)", color="Algorithm",
                  line_dash="# Samples", log_y=True,)
    fig.update_traces(showlegend = False, )
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='black', dash='dash'),
                             name='2000 samples', showlegend=True), )
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='black', ),
                             name='500 samples', showlegend=True))
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='#636EFA', ),
                             name='SamBA', showlegend=True))
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='#EF553B', ),
                             name='Adaboost', showlegend=True))
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='#00CC96', ),
                             name='KNN', showlegend=True))
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='#AB63FA', ),
                             name='SVM-RBF', showlegend=True))
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='#FFA15A', ),
                             name='Random Forest', showlegend=True))
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='#19D3F3', ),
                             name='Decision Tree', showlegend=True))
    fig.update_traces(line=dict(width=2))
    fig.update_layout(legend_title_text='', font_family="Computer Modern", paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)', font_size=20)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', tickfont_size=10 ,  showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.write_image("figures/train_duration_pred_{}.pdf".format(title))


    # Plot test duration
    fig = px.line(df_test, x="# Features", y="Duration (s)", color="Algorithm",
                  line_dash="# Samples", log_y=True)
    fig.update_traces(showlegend=False, )
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='black', dash='dash'),
                             name='2000 samples', showlegend=True), )
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='black', ),
                             name='500 samples', showlegend=True))
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='#636EFA', ),
                             name='SamBA', showlegend=True))
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='#EF553B', ),
                             name='Adaboost', showlegend=True))
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='#00CC96', ),
                             name='KNN', showlegend=True))
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='#AB63FA', ),
                             name='SVM-RBF', showlegend=True))
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='#FFA15A', ),
                             name='Random Forest', showlegend=True))
    fig.add_trace(go.Scatter(x=[0], y=[0],
                             mode='lines',
                             line=dict(color='#19D3F3', ),
                             name='Decision Tree', showlegend=True))
    fig.update_traces(line=dict(width=2))
    fig.update_layout(legend_title_text='', font_family="Computer Modern", paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)', font_size=20)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black',)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', tickfont_size=10 ,  showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_traces(line=dict(width=2))
    fig.show()
    fig.write_image("figures/test_duration_pred_{}.pdf".format(title))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Generate the figure with same number of base estimators
    n_estimators=10
    algs = [NeighborHoodClassifier(n_estimators=n_estimators,
                          base_estimator=DecisionTreeClassifier(max_depth=1)),
            AdaBoostClassifier(n_estimators=n_estimators,
                          base_estimator=DecisionTreeClassifier(max_depth=1)),
            KNeighborsClassifier(), SVC(),
            RandomForestClassifier(n_estimators=n_estimators),
            DecisionTreeClassifier(max_depth=n_estimators), Lasso(),
            XGBClassifier(n_estimators=n_estimators), GradientBoostingClassifier(n_estimators=n_estimators),
            LinearTreeClassifier(base_estimator=LogisticRegression(), max_depth=n_estimators),
            # LWLR()
            ]
    gen_graphs("nest", algs)


    # Generate the figure with the number of base estimators returned in the
    # Performance experimentation
    import math
    algs = [NeighborHoodClassifier(n_estimators=7,
                          base_estimator=DecisionTreeClassifier(max_depth=1)),
            AdaBoostClassifier(n_estimators=146,
                          base_estimator=DecisionTreeClassifier(max_depth=1)),
            KNeighborsClassifier(), SVC(),
            RandomForestClassifier(n_estimators=127),
            DecisionTreeClassifier(max_depth=int(math.log2(17))),
            Lasso(),
            XGBClassifier(n_estimators=172),
            GradientBoostingClassifier(n_estimators=1135),
            LinearTreeClassifier(base_estimator=LogisticRegression(), max_depth=7),
            # LWLR()
            ]
    gen_graphs("var", algs)