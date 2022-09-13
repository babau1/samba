import numpy as np
from SamBA.samba import NeighborHoodClassifier
from SamBA.distances import MultiEnvDist, EuclidianDist
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

n_samples = 500
rs = np.random.RandomState(43)
X = rs.uniform(-2, 2, size=(n_samples,20))
X1 = rs.normal(1, 1, size=(int(n_samples*0.75/2),5))
X2 = rs.normal(-1, 1, size=(int(n_samples*0.75/2),5))
X[:int(n_samples*0.75/2), :5] = X1
X[int(n_samples*0.75/2):2*int(n_samples*0.75/2), :5] = X2
X1 = rs.normal(1, 1, size=(int(n_samples*0.25/2),5))
X2 = rs.normal(-1, 1, size=(int(n_samples*0.25/2),5))
X[n_samples-int(n_samples*0.25/2)*2:n_samples-int(n_samples*0.25/2), 5:10] = X1
X[n_samples-int(n_samples*0.25/2):, 5:10] = X2
X[:2*int(n_samples*0.75/2), -1] = 1
X[2*int(n_samples*0.75/2):, -1] = 0
X[:2*int(n_samples*0.75/2), -2] = 1
X[2*int(n_samples*0.25/2):, -2] = 0

y = np.zeros(n_samples)
y[int(n_samples*0.75/2):2*int(n_samples*0.75/2)] = 1
y[n_samples-int(n_samples*0.25/2):] = 1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)

sb = NeighborHoodClassifier(a=.001, b=4, distance=MultiEnvDist(base_val=10, env_features=[-1,-2]),
                            n_estimators=50)

sb = sb.fit(X_train, y_train)

y_pred = sb.predict(X_test)
print(sb.feature_importances_)
fig = sb.plot_projection(X, y, contour=True, force_2d=True )
print(y_pred)
print(balanced_accuracy_score(y_train, sb.predict(X_train)))
print(balanced_accuracy_score(y_test, y_pred))
