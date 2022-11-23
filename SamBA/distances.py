import numpy as np
from sklearn.gaussian_process.kernels import RBF

class Distances:

    def __init__(self, keep_selected_features=True):
        self.keep_selected_features = keep_selected_features

    def keep_chosen_features(self, set, feature_mask):
        if self.keep_selected_features:
            indices = np.where(feature_mask!=0)[0]
        else:
            return np.arange(set.shape[1])
        return indices

    def get_norms(self, sample, set, feature_mask):
        indices = self.keep_chosen_features(set, feature_mask)
        norms = np.linalg.norm(sample[indices] - set[:, indices], axis=1)
        return norms

    def __call__(self, sample, set, feature_mask, base_val=0, *args, **kwargs):
        if np.sum(feature_mask)==0:
            return np.zeros(set.shape[0])+base_val
        else:
            return self.get_dist(sample, set, feature_mask, *args, **kwargs)

class Jaccard(Distances):

    def get_dist(self, sample, set, feature_mask):
        indices = self.keep_chosen_features(set, feature_mask)
        jacc = np.mean(sample[indices]==set[:, indices], axis=1)
        return jacc


class EuclidianDist(Distances):

    def get_dist(self, sample, set, feature_mask):
        norms = self.get_norms(sample, set, feature_mask)
        return norms/np.sum(norms)

class RBFKernel(Distances, RBF):

    def __init__(self, keep_selected_features = True, length_scale=1.0):
        RBF.__init__(self, length_scale=length_scale)
        Distances.__init__(self, keep_selected_features=keep_selected_features)

    def __call__(self, sample, set, feature_mask, base_val=0, *args, **kwargs):
        indices = self.keep_chosen_features(set, feature_mask)
        return RBF.__call__(self, sample[:, indices], set[:, indices])


class MultiEnvDist(Distances):

    def __init__(self, keep_selected_features=True, base_val=1, env_features=[-1,-2,-3]):
        self.base_val = base_val
        self.env_features = env_features
        Distances.__init__(self, keep_selected_features=keep_selected_features)

    def get_dist(self, sample, set, feature_mask, ):
        norms = np.zeros(set.shape[0])
        if self.base_val != 0:
            if self.env_features is None:
                raise AttributeError("Must specify env_features for multi-env distance")
            for row_ind, row in enumerate(set):
                norms[row_ind] = (1-np.equal(row[self.env_features],sample[self.env_features]).all())*self.base_val
        norms += self.get_norms(sample, set, feature_mask)
        return norms / np.sum(norms)


class ExpEuclidianDist(Distances):

    def get_dist(self, sample, set, feature_mask, *args, **kwargs):
        norms = self.get_norms(sample, set, feature_mask)
        norms = norms/np.sum(norms)
        return np.exp(norms)


class PolarDist(Distances):

    def cart2pol(self, x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    def get_dist(self,  sample, set, feature_mask):
        indices = self.keep_chosen_features(set, feature_mask)
        vals = (np.linalg.norm(sample[indices])-np.linalg.norm(set[:, indices], axis=1))**2
        return vals


class ExpPolarDist(PolarDist):

    def get_dist(self,  sample, set, feature_mask, *args, **kwargs):
        return np.exp(1/PolarDist.get_dist(self, sample, set, feature_mask,))


class MixedDistance(Distances):

    def __init__(self, distances = [EuclidianDist(), Jaccard()],
                 feature_map=[[0, 1], [1, 0]], keep_selected_features=True):
        super(MixedDistance, self).__init__(keep_selected_features=keep_selected_features)
        if len(distances)!=len(feature_map):
            raise ValueError("distances and features must be two lists of same "
                             "length, here, they are of lengths {} and {} "
                             "respactively".format(len(distances),
                                                   len(feature_map)))
        self.distances = distances
        self.feature_map = feature_map

    def get_dist(self, sample, set, feature_mask, base_val=np.nan, *args, **kwargs):
        dists = np.zeros((len(self.distances), set.shape[0]))
        for ind, (distance, feature_map) in enumerate(zip(self.distances,
                                                          self.feature_map)):
            dist_feature_mask = feature_mask*feature_map
            dists[ind, :] = distance.__call__(sample, set, dist_feature_mask,
                                           base_val=base_val, *args, **kwargs)
        dists = np.delete(dists, np.where(np.isnan(np.sum(dists, axis=1)))[0], 0)
        return np.mean(dists, axis=0)
