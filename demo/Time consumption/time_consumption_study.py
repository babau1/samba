import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import time
import pandas
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from SamBA.samba import NeighborHoodClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import math
# import plotly.io as pio
# pio.kaleido.scope.mathjax = None

rs = np.random.RandomState(42)

n_estimators=10
algs = [NeighborHoodClassifier(n_estimators=n_estimators,
                      base_estimator=DecisionTreeClassifier(max_depth=1)),
        AdaBoostClassifier(n_estimators=n_estimators,
                      base_estimator=DecisionTreeClassifier(max_depth=1)),
        KNeighborsClassifier(), SVC(),
        RandomForestClassifier(n_estimators=n_estimators),
        DecisionTreeClassifier(max_depth=n_estimators)]
algs_names = ["SamBA", "Adaboost", "KNN", "SVM-RBF", "Random Forest", "Decision Tree"]
n_samples_list = [500, 2000, ]
n_features_list = [10, 100, 1000, 2000, 5000, 50000]#, 100000, ]#500000] #



df_train = pandas.DataFrame()
df_test = pandas.DataFrame()
for n_features in n_features_list:
    print(n_features)
    for n_samples in n_samples_list:
        print("\t", n_samples)
        for clf, alg_name in zip(algs, algs_names):
            X = rs.uniform(0, 1, size=(n_samples, n_features))
            y = rs.randint(0, 2, size=n_samples)
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
fig.write_image("figures/train_duration_pred_nest.pdf")

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
fig.write_image("figures/test_duration_pred_nest.pdf")