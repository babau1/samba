import numpy as np
import plotly
import plotly.graph_objs as go

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from SamBA.samba import NeighborHoodClassifier

def generate_heterogenous(n_samples, n_features, n_env=4, n_classes=2, rs=np.random.RandomState(42)):
    n_sample_per_env_per_class = int(n_samples/n_classes/n_env)
    print(n_sample_per_env_per_class)
    n_sample_per_env = n_sample_per_env_per_class * n_classes
    n_samples = n_sample_per_env*n_env
    X = rs.normal(0, 2, size=(n_samples, n_features))
    print(X.shape)
    y = np.zeros(n_samples)
    for env_ind in range(n_env):
        X_env_1 = rs.normal(1, 0.5, size=(n_sample_per_env_per_class))
        X[env_ind*n_sample_per_env:env_ind*n_sample_per_env+n_sample_per_env_per_class, env_ind] = X_env_1
        y[env_ind*n_sample_per_env:env_ind*n_sample_per_env+n_sample_per_env_per_class] = n_env%2
        X_env_2 = rs.normal(-1, 0.5, size=(n_sample_per_env_per_class))
        print("plif")
        print((env_ind+1)*n_sample_per_env_per_class)
        print((env_ind+2)*n_sample_per_env_per_class)
        X[env_ind*n_sample_per_env+n_sample_per_env_per_class:env_ind*n_sample_per_env+2*n_sample_per_env_per_class, env_ind] = X_env_2
        y[env_ind*n_sample_per_env+n_sample_per_env_per_class:env_ind*n_sample_per_env+2*n_sample_per_env_per_class] = (n_env+1)%2
        print(y)
    return X, y

X, y = generate_heterogenous(n_samples=800, n_features=4, n_env=4)
pos_ind = np.where(y==1)[0]
print(len(pos_ind))
neg_ind = np.where(y==0)[0]
print(len(neg_ind))
fig = go.Figure(data=go.Scatter(x=X[pos_ind, 0], y=X[pos_ind, 1], mode="markers"))
fig.add_trace(go.Scatter(x=X[neg_ind, 0], y=X[neg_ind, 1], mode="markers"))
fig.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=np.random.RandomState(42), shuffle=True, train_size=0.7)
ada = AdaBoostClassifier(n_estimators=10)
ada.fit(X_train,y_train)
print(ada.score(X_test, y_test))

clf = NeighborHoodClassifier(n_estimators=10, a=0.0000001, b=.5)
clf.fit(X, y)
clf.plot_projection(X_train, y_train, contour=True, force_2d=True, save=False)
print(clf.score(X_test, y_test))
