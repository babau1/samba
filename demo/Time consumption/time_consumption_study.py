import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import time
import pandas
from sklearn.ensemble import AdaBoostClassifier
from SamBA.samba import NeighborHoodClassifier
from sklearn.tree import DecisionTreeClassifier

import plotly.io as pio
pio.kaleido.scope.mathjax = None

rs = np.random.RandomState(42)

algs = [NeighborHoodClassifier, AdaBoostClassifier]
algs_names = ["SamBA", "Adaboost"]
n_samples_list = [500, 2000]
n_features_list = [10, 100, 1000, 2000, 5000, 50000, 100000, ]#500000] #
n_estimators=10


df_train = pandas.DataFrame()
df_test = pandas.DataFrame()
for n_features in n_features_list:
    print(n_features)
    for n_samples in n_samples_list:
        print("\t", n_samples)
        for alg, alg_name in zip(algs, algs_names):
            X = rs.uniform(0, 1, size=(n_samples, n_features))
            y = rs.randint(0, 2, size=n_samples)
            clf = alg(n_estimators=n_estimators,
                      base_estimator=DecisionTreeClassifier(max_depth=1))
            beg = time.time()
            clf.fit(X, y)
            end = time.time()
            train_duration = end-beg
            df_train = df_train.append(
                {"Algorithm": alg_name, "# Estimators": n_estimators,
                 "# Samples": n_samples, "# Features": n_features,
                 "Duration (s)": train_duration, "Phase": "Train"}, ignore_index=True)
            beg = time.time()
            y_pred = clf.predict(X)
            end = time.time()
            test_duration = end-beg
            df_test = df_test.append(
                {"Algorithm": alg_name, "# Estimators": n_estimators,
                 "# Samples": n_samples, "# Features": n_features,
                 "Duration (s)": test_duration, "Phase": "Test"}, ignore_index=True)

fig = px.line(df_train, x="# Features", y="Duration (s)", color="# Samples",
              line_dash="Algorithm", log_y=True,)
fig.update_traces(showlegend = False, )
fig.add_trace(go.Scatter(x=[0], y=[0],
                mode='lines', line=dict(color='black', dash='dash'),
                name='Adaboost',showlegend=True), )
fig.add_trace(go.Scatter(x=[0], y=[0],
                         mode='lines',
                         line=dict(color='black', ),
                         name='SamBA', showlegend=True))
fig.add_trace(go.Scatter(x=[0], y=[0],
                         mode='lines',
                         line=dict(color='#636EFA', ),
                         name='500 samples', showlegend=True))
fig.add_trace(go.Scatter(x=[0], y=[0],
                         mode='lines',
                         line=dict(color='#EF553B', ),
                         name='2000 samples', showlegend=True))
fig.update_traces(line=dict(width=2))
fig.update_layout(legend_title_text='', font_family="Computer Modern", paper_bgcolor='rgba(0,0,0,0)',
plot_bgcolor='rgba(0,0,0,0)', font_size=20)
fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', tickfont_size=10 ,  showgrid=True, gridwidth=1, gridcolor='LightGray')
fig.write_image("train_duration_pred.pdf")

fig = px.line(df_test, x="# Features", y="Duration (s)", color="# Samples",
              line_dash="Algorithm", log_y=True)
fig.update_traces(showlegend=False, )
fig.add_trace(go.Scatter(x=[0], y=[0],
                         mode='lines',
                         line=dict(color='black', dash='dash'),
                         name='Adaboost', showlegend=True), )
fig.add_trace(go.Scatter(x=[0], y=[0],
                         mode='lines',
                         line=dict(color='black', ),
                         name='SamBA', showlegend=True))
fig.add_trace(go.Scatter(x=[0], y=[0],
                         mode='lines',
                         line=dict(color='#636EFA', ),
                         name='500 samples', showlegend=True))
fig.add_trace(go.Scatter(x=[0], y=[0],
                         mode='lines',
                         line=dict(color='#EF553B', ),
                         name='2000 samples', showlegend=True))
fig.update_traces(line=dict(width=2))
fig.update_layout(legend_title_text='', font_family="Computer Modern", paper_bgcolor='rgba(0,0,0,0)',
plot_bgcolor='rgba(0,0,0,0)', font_size=20)
fig.update_xaxes(showline=True, linewidth=1, linecolor='black',)
fig.update_yaxes(showline=True, linewidth=1, linecolor='black', tickfont_size=10 ,  showgrid=True, gridwidth=1, gridcolor='LightGray')
fig.update_traces(line=dict(width=2))
fig.write_image("test_duration_pred.pdf")