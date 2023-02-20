import sklearn
import plotly
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

colors = px.colors.qualitative.Plotly

def gen_data(precision, random_state, locs=[0.25,0.75], scale=0.1,
             blob_size=100, lin_space_lag=0.05, weights=[2.7,0.36],
             noisy_sm=[0, 1,2], noisy_bg=[0, 1 , 2]):
    first_blob = random_state.uniform(low=0.05, high=0.45, size=blob_size)
    second_blob = random_state.uniform(low=0.55, high=0.95, size=blob_size)
    X = np.concatenate((first_blob, second_blob))
    Xs = np.linspace(min(X)-lin_space_lag, max(X)+lin_space_lag, precision)

    w_first_blob = np.zeros(shape=first_blob.shape) + weights[0]
    w_second_blob = np.zeros(shape=second_blob.shape) + weights[1]
    w = np.concatenate((w_first_blob, w_second_blob))
    w[noisy_sm] = weights[1]
    w[[blob_size + _ for _ in noisy_bg]] = weights[0]
    w = w / np.sum(w)

    w_x = np.zeros(precision)

    return X, Xs, w, w_x


def plot_figure(X, xs, w, w_x, fig, a_s=[0, 0.02], bs=[1.5,4], row=1, b=2, ind_b=0):
    if row!=1 or ind_b!=1:
        legend=False
    else:
        legend=True
    for ind_x, x in enumerate(xs):
        dists = np.array(
            [1 / (a_s[0] + (np.linalg.norm(x - x_1_i)) ** b) for x_1_i in X])
        w_x[ind_x] = np.sum(
            [w_i * dist / np.sum(dists) for w_i, dist in zip(w, dists)])
    mean = np.zeros(precision) + np.mean(w_x)

    fig.add_trace(
        go.Scatter(x=X, y=w, mode='markers',
                   name=r"$\text{Empirical: }\Omega_{t,i}$",
                   marker=dict(color="black"),
                   showlegend=legend
                   ),
                row=row, col=ind_b+1)
    for sample, weight in zip(X, w):
        fig.add_shape(type="line", xref="x", yref="y", x0=sample, y0=0,
                      x1=sample,
                      y1=weight, line=dict(color="#A9A9A9", width=2, ),
                      row=row, col=ind_b+1)

    fig.add_trace(go.Scatter(x=xs, y=mean, mode='lines',
                             name=r"$\text{Mean: }\frac{1}{m}\sum_{i=1}^m \Omega_{t,i}$",
                             line={"color":colors[2]},
                   showlegend=legend),
                  row=row, col=ind_b+1)
    fig.add_trace(go.Scatter(x=xs, y=w_x, mode='lines',
                            name=r"$\text{Estimated: }\omega^{0,"+"{}".format(b)+r"}_t(x)$",
                             line={"color": colors[0]},
                   showlegend=legend),
                  row=row, col=ind_b+1)
    for ind_x, x in enumerate(xs):
        dists = np.array(
            [1 / (a_s[1] + (np.linalg.norm(x - x_1_i)) ** b) for x_1_i in X])
        w_x[ind_x] = np.sum(
            [w_i * dist / np.sum(dists) for w_i, dist in zip(w, dists)])
    fig.add_trace(
        go.Scatter(x=xs, y=w_x, mode='lines',
                   name=r"$\text{Estimated: }\omega^{0.02,"+"{}".format(b)+r"}_t(x)$",
                   line={"color":colors[1]},
                   showlegend=legend),
    row=row, col=ind_b+1)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="X",
        yaxis_title="Weight",
        font_size=25)
    fig.update_xaxes(showline=True, linecolor='black', showgrid=False)
    fig.update_yaxes(showline=True, linecolor='black', showgrid=False)
    return fig


precision = 2000
blob_size = 20




random_state = np.random.RandomState(42)


# X, xs, w, w_x = gen_data(precision, random_state, locs=[0.25,0.75], scale=0.1,
#              blob_size=blob_size, lin_space_lag=0.05, weights=[0.4,0.05],
#                          noisy_sm=[], noisy_bg=[])
#
# fig = plot_figure(X, xs, w, w_x,)
# fig.show()
# fig.write_image("weight_viz_1d.pdf", height=500, width=1000)
#
#
# X, xs, w, w_x = gen_data(precision, random_state, locs=[0.25,0.75], scale=0.1,
#              blob_size=blob_size, lin_space_lag=0.05, weights=[0.4,0.05])
#
# fig = plot_figure(X, xs, w, w_x)
# fig.show()
# fig.write_image("weight_viz_1d_noisy.pdf", height=500, width=1000)
bs = [1.5,4]
fig = make_subplots(rows=2, cols=len(bs),
                    subplot_titles=("Pure, b = 1.5", "Pure, b = 4", "Noisy, b = 1.5", "Noisy, b = 4"))
for ind_b, b in enumerate(bs):


    X, xs, w, w_x = gen_data(precision, random_state, locs=[0.25,0.75], scale=0.1,
                 blob_size=blob_size, lin_space_lag=0.05, weights=[0.4,0.05],
                             noisy_sm=[], noisy_bg=[])

    fig = plot_figure(X, xs, w, w_x, fig, row=1, b=b, ind_b=ind_b)

    X, xs, w, w_x = gen_data(precision, random_state, locs=[0.25,0.75], scale=0.1,
                 blob_size=blob_size, lin_space_lag=0.05, weights=[0.4,0.05])

    fig = plot_figure(X, xs, w, w_x, fig, row=2, b=b, ind_b=ind_b)
fig.show()
fig.write_image("figures/weight_viz_1d_noisy_b1.pdf", height=1000, width=2000)