import sklearn
import plotly
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

precision = 2000
b = 8
a = 0.1

x1 = np.array([0.0,0.02,0.06,0.1, 0.2, 0.22, 0.23, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
xs = np.linspace(min(x1)-0.05, max(x1)+0.05, precision)

w = np.array([0.4,0.4,0.1,0.4, 0.4, 0.4, 0.4, 0.4 ,0.4, 0.4, 0.4, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
w = w/np.sum(w)

w_x = np.zeros(precision)

for ind_x, x in enumerate(xs):
    dists = np.array([1/(1+a**b)-1/(a**b+np.linalg.norm(x - x_1_i)**b) for x_1_i in x1])
    w_x[ind_x] = np.sum([w_i*dist/np.sum(dists) for w_i, dist in zip(w, dists)])
mean = np.zeros(precision)+np.mean(w_x)
# ada = np.zeros(precision)+np.log(1/(1-np.mean(w>np.mean(w)))-1)/len(w)
fig = go.Figure()

fig.add_trace(go.Scatter(x=x1, y=w, mode='markers', name="Empirical weights"))
for sample,weight in zip(x1,w):
    fig.add_shape(type="line", xref="x", yref="y", x0=sample, y0=0, x1=sample, y1=weight,
                  line=dict(
                      color="#A9A9A9",
                      width=3,
                  ),)

# model = make_pipeline(PolynomialFeatures(5),
#                       Ridge(alpha=1e-3))
# model.fit(x1.reshape(-1, 1), w)
# preds = model.predict(xs.reshape(-1, 1))
fig.add_trace(go.Scatter(x=xs, y=mean, mode='lines', name="Mean Samba weight"))
# fig.add_trace(go.Scatter(x=xs, y=preds, mode='lines', name="Spline weight"))
fig.add_trace(go.Scatter(x=xs, y=w_x, mode='lines', name="Estimated weight"))
# print(mean)
# print(ada)

fig.update_layout(title="Samba weighting in 1d, with b = {} ; a = {}".format(b, a), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',  xaxis_title="x",
    yaxis_title="weight", font_color="lightgray")
fig.update_xaxes(showline=True, linewidth=2, linecolor='lightgray',showgrid=False)
fig.update_yaxes(showline=True, linewidth=2, linecolor='lightgray',showgrid=False)
# fig.show()
fig.write_image("weight_viz_1d.png", width=1000, height=500)



#
# x1 = np.array([5, 5, 1, 3, 4, 5])
# x2 = np.array([0.5, 5, 2, 3, 5, 4])
#
# xs, ys = np.linspace(0, max(x1), precision), np.linspace(0, max(x2), precision)
#
# w = np.array([0.1, 0.1, 0.4, 0.1, 0.4, 0.4])
# w = w/np.sum(w)
#
# w_x = np.zeros((precision, precision))
#
# for ind_x, x in enumerate(xs):
#     for ind_y, y in enumerate(ys):
#         dists = np.array([1/np.linalg.norm(np.array([x, y])-np.array([x_1_i, x_2_i]))**b for x_1_i, x_2_i in zip(x1, x2)])
#         w_x[ind_x, ind_y] = np.sum([w_i * dist/np.sum(dists) for w_i, dist in zip(w, dists)])
#
# ada = np.zeros((precision, precision))+np.mean(w)
#
# fig = go.Figure(data=go.Surface(z=w_x, x=xs, y=ys, opacity=0.8))
# fig.add_trace(go.Surface(z=ada, x=xs, y=ys, opacity=0.5))
#
# fig.update_layout(title="Samba weighting in 2d, with b = {}".format(b))
# fig.show()
# fig.write_image("weight_viz_2d.png", width=1000, height=500)

