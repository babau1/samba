"""
    SamBA -- Sample Boosting Algorithm
    Copyright (C) 2023 Baptiste BAUVIN

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
from samba.difficulties import ExpTrainWeighting, SqExpTrainWeighting, ZeroOneTrainWeighting
from samba.relevances import ExpRelevance, MarginRelevance
from samba.distances import *
from sklearn.preprocessing import RobustScaler


def set_class_from_str(string, classes=None):
    if isinstance(string, str):
        all_classes = [ExpTrainWeighting, RobustScaler, SqExpTrainWeighting,
                       ZeroOneTrainWeighting, ExpRelevance, MarginRelevance,
                       ExpEuclidianDist, EuclidianDist, ExpPolarDist, PolarDist,
                       RobustScaler]
        if classes is not None:
            all_classes.append(classes)
        for avail_class in all_classes:
            if string == avail_class.__name__:
                return avail_class()
        raise ValueError(
            "Argument {} not valid as a class name; available class names are {}".format(
                string, ", ".join([cl.__name__ for cl in classes])))
    else:
        return string

def gen_four_blobs(rs=np.random.RandomState(42), n_samples=1000,
                   unit=int(1000/4), n_pos=int(1000/2), n_features=2,
                   scale=0.5):
    n_neg = n_samples-n_pos
    X = np.zeros((n_samples, n_features))
    centers = [np.array([1,1]),
              np.array([-1,-1]),
              np.array([1,-1]),
              np.array([-1,1]), ]
    y = np.ones(n_samples)
    y[:n_neg] = 0

    X[:unit, 0] = rs.normal(centers[0][0], scale=scale, size=unit)
    X[:unit, 1] = rs.normal(centers[0][1], scale=scale, size=unit)

    X[unit:n_pos, 0] = rs.normal(centers[1][0], scale=scale, size=n_pos-unit)
    X[unit:n_pos, 1] = rs.normal(centers[1][1], scale=scale, size=n_pos-unit)

    X[n_pos:n_pos+unit, 0] = rs.normal(centers[2][0], scale=scale, size=unit)
    X[n_pos:n_pos+unit, 1] = rs.normal(centers[2][1], scale=scale, size=unit)

    X[n_pos+unit:, 0] = rs.normal(centers[3][0], scale=scale, size=n_neg-unit)
    X[n_pos+unit:, 1] = rs.normal(centers[3][1], scale=scale, size=n_neg-unit)
    return X, y