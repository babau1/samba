import numpy as np



class Distances:

    def __init__(self, keep_selected_features=True):
        self.keep_selected_features = keep_selected_features

    def keep_chosen_features(self, sample, set, feature_mask):
        if self.keep_selected_features:
            indices = np.where(feature_mask!=0)[0]
        else:
            return np.arange(set.shape[1])
        return indices

    def get_norms(self, sample, set, feature_mask, base_val=1/5):
        indices = self.keep_chosen_features(sample, set, feature_mask)
        norms = np.linalg.norm(sample[indices] - set[:, indices], axis=1)
        norms[norms == 0] = base_val
        return norms


class EuclidianDist(Distances):

    def __call__(self, sample, set, feature_mask, base_val=1/5,*args, **kwargs):
        norms = self.get_norms(sample, set, feature_mask, base_val)
        return 1/norms


class ExpEuclidianDist(Distances):

    def __call__(self, sample, set, feature_mask, base_val=1/5,*args, **kwargs):
        norms = self.get_norms(sample, set, feature_mask, base_val)
        return 1/np.exp(norms)


class PolarDist(Distances):

    def cart2pol(self, x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    def __call__(self,  sample, set, feature_mask, base_val=1/5, *args, **kwargs):
        indices = self.keep_chosen_features(sample, set, feature_mask)
        vals = (np.linalg.norm(sample[indices])-np.linalg.norm(set[:, indices], axis=1))**2
        vals[vals==0] = base_val
        return 1/vals