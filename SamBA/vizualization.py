import plotly
import numpy as np
import plotly.graph_objects as go
import os
from PIL import Image
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier


class VizSamba():

    def plot_projection(self, X, y, save=True, path=".", rs=42, contour=False,
                        template="plotly", feature_ids=None, force_2d=False,):
        if self.normalizer is not None:
            new_X = self.normalizer.fit_transform(X)
        else:
            new_X = X.copy()
        if self.support_feats_.shape[0] == 1:
            sec_dim = np.random.RandomState(rs).uniform(0,1, size=new_X.shape[0])
            fig = self._plot_2d(X, y, sec_dim=sec_dim, contour=contour, feature_ids=feature_ids)
        elif self.support_feats_.shape[0]==2 or force_2d:
            if force_2d:
                best_feats = np.argsort(-self.feature_importances_)[:2]
                supp = self.support_feats_
                self.support_feats_ = best_feats
            fig = self._plot_2d(X, y, contour=contour, feature_ids=feature_ids)
            self.support_feats_ = supp
        else:
            best_feats = np.argsort(-self.feature_importances_)[:3]
            fig = go.Figure()
            labels = np.unique(y)
            data = new_X[:, best_feats]
            for label in labels:
                fig.add_trace(go.Scatter3d(x=data[np.where(y == label)[0], 0],
                                           y=data[np.where(y == label)[0], 1],
                                           z=data[np.where(y == label)[0], 2],
                                           name="Class {}".format(label + 1),
                                           mode="markers",
                                           marker=dict(
                                               size=1, )))
            fig.update_layout(template=template)
            if feature_ids is not None:
                fig.update_layout(scene = dict(xaxis_title=feature_ids[best_feats[0]].decode(),
                                  yaxis_title=feature_ids[best_feats[1]].decode(),
                                  zaxis_title=feature_ids[best_feats[2]].decode(),))
        if save:
            plotly.offline.plot(fig, filename=os.path.join(path, "projection_fig.html"), auto_open=False)
        else:
            fig.show()

    def get_sample_feature_importance(self, x_test, sample_name="NoName",
                               feature_names=None, limit=10):
        if feature_names is None:
            feature_names = ["Feat {}".format(ind) for ind in range(x_test.shape[0])]
        if self.normalizer is not None:
            new_X = self.normalizer.transform(x_test)
        else:
            new_X = x_test.copy()
        feat_imp = self._predict_one_sample(new_X)
        n_important_feat = len(np.where(feat_imp!=0)[0])
        if n_important_feat>limit:
            add_limit = "The list was shortened, because, there were more than " \
                        "{} important features, but you can increase this limit " \
                        "by setting limit=<your value> in " \
                        "`get_feature_importance` arguments.".format(limit)
        else:
            add_limit = ""
        best_feats_ind = np.argsort(-feat_imp)
        out_str = "For sample {}, the feature importances are : \n".format(sample_name)
        for i in range(min(n_important_feat, limit)):
            out_str += "\t - {} has an importance of {}% \n".format(feature_names[best_feats_ind[i]], round(feat_imp[best_feats_ind[i]], 2),)
        out_str+= add_limit
        return out_str

    def _plot_2d(self, X, y, sec_dim=None, contour=False,
                 random_state=np.random.RandomState(42),
                 n_estimators=None, title="", template="plotly",
                 feature_ids=None, test_samples=None, test_labels=None,
                 size=5, test_preds=None, symbols = ["x", "circle"],
                 n_steps=10 ):
        if sec_dim is None:
            if X.shape[1] < 2 :
                sec_index = 0
                # sec_dim = np.random.uniform(low=X.min(), high=X.max(),
                #                             size=X.shape[0],
                #                             random_state=random_state)
                # if test_samples is not None:
                    # sec_dim_test = np.random.uniform(low=test_samples.min(), high=test_samples.max(),
                    #                             size=test_samples.shape[0],
                    #                             random_state=random_state)
            # elif len(self.support_feats)==2:
            #     feat_1 = self.support_feats[0]
            #     sec_index = self.support_feats[1]
                # sec_dim = X[:, [_ for _ in [0, 1] if _ != feat_1][0]]
                # if test_samples is not None:
                #     sec_dim_test = test_samples[:, [_ for _ in [0, 1] if _ != feat_1][0]]
            elif len(self.support_feats_)<2:
                sec_index = np.random.choice(X.shape[1], size=1, random_state=random_state)
                # sec_dim = X[:, feat]
                # if test_samples is not None:
                #     sec_dim_test = test_samples[:, feat]
            else:
                sec_index = self.support_feats_[1]
                # sec_dim = X[:, self.support_feats[1]]
                # if test_samples is not None:
                #     sec_dim_test = test_samples[:, self.support_feats[1]]
        fig = go.Figure()
        labels = np.unique(y)
        preds = self.predict(X)

        # preds = self.zero_binarizer.fit_transform(np.sign(preds))
        # print(self._predict_vote(X, n_estimators=n_estimators))
        colors = np.array(["Blue" if label == 1 else "Red" for label in preds])

        if test_preds is not None:
            colors_test = np.array(["Blue" if label == 1 else "Red" for label in test_preds])
        for label in labels:
            if contour:
                opacity = 0.9
            else:
                opacity = 0.7
            indices = np.where(y == label)[0]
            fig.add_trace(go.Scatter(x=X[indices, self.support_feats_[0]],
                                     y=X[indices, sec_index],
                                     opacity=opacity,
                                     name="Class {}".format(label + 1),
                                     mode="markers",
                                     marker=dict(symbol = symbols[int(label)],
                                     size=size, color=colors[indices])))
            if test_samples is not None:
                indices = np.where(np.sign(test_labels) == label)[0]
                fig.add_trace(go.Scatter(x=test_samples[indices, self.support_feats_[0]],
                                         y=test_samples[indices, sec_index],
                                         opacity=opacity,
                                         name="Class {}".format(label + 1),
                                         mode="markers",
                                         marker=dict(symbol=symbols[int(label)],
                                         size=2*size, color=colors_test[indices])))
        if contour:
            mesh_space = generate_mesh_space(X, n_steps=n_steps)
            fig.add_trace(go.Contour(
                z=self.predict_proba(mesh_space,)[:, 1],
                x=mesh_space[:, self.support_feats_[0]],
                y=mesh_space[:,sec_index],
                line_smoothing=0.0,
                contours_coloring='heatmap',
                colorscale='RdBu',
                showscale=False
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
            fig.update_xaxes(visible=False, showgrid=False)
            fig.update_yaxes(visible=False, showgrid=False)
            if n_estimators is not None and isinstance(self.base_estimator, DecisionTreeClassifier) and self.base_estimator.max_depth == 1:
                fig.update_layout(title=title+"<br> Feat: {}, threshold: {}".format(self.estimators_[n_estimators-1].tree_.feature[0],
                                                                         self.estimators_[n_estimators-1].tree_.threshold[0] ),
                                  template=template)
        if feature_ids is not None:
            fig.update_layout(xaxis_title=feature_ids[self.support_feats_[0]].decode(),
                              yaxis_title=feature_ids[self.support_feats_[1]].decode())
        return fig

    def plot_contour_gif(self, X, y, sec_dim=None,
                         random_state=np.random.RandomState(42),
                         temp_folder="temp/", save_path=None, title="",
                         template="plotly"):
        if self.normalizer is not None:
            X = self.normalizer.transform(X)
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)
        try:
            for estim_index in range(self.n_estimators):
                fig = self._plot_2d(X, y, sec_dim=sec_dim, contour=True,
                                    random_state=random_state,
                                    n_estimators=estim_index+1,
                                    title=title, template=template)
                fig.write_image(os.path.join(temp_folder, "gif_{}.png").format(estim_index), scale=2.0)

            img, *imgs = [Image.open(os.path.join(temp_folder, "gif_{}.png").format(i)) for i in range(self.n_estimators)]
            img.save(fp=save_path, format='GIF', append_images=imgs,
                     save_all=True, duration=200, loop=0)
        except Exception as exc:
            print(exc)
            pass
        for fname in os.listdir(temp_folder):
            os.remove(os.path.join(temp_folder, fname))
        os.rmdir(temp_folder)

    def _predict_one_sample(self, sample):
        feat_importances = np.array([estim.feature_importances_ for estim in self.estimators_])
        dists = self.distance(sample, self.train_samples, self.features_mask)
        weights = self.estim_weights*np.sum(np.transpose(self.neig_weights)*dists, axis=1)
        importances = np.sum(feat_importances*weights.reshape((10,1)), axis=0)
        importances /= np.sum(importances)
        return importances

    def _save_data(self, preds, preds_train, iter, y):
        for sample_index, (sample_class, sample, prediction, pred_train, vote, weight) in enumerate(zip(y, self.train_samples, preds, preds_train, self.votes, self.train_weights[:, iter])):
            self.saved_data.loc[self.saved_ind] = {"Iteration":int(iter),
                                                      "Index": sample_index,
                                                      "Pred": prediction,
                                                      "Pred Train":pred_train,
                                                      "Margin": vote*sample_class,
                                                      "Class": sample_class,
                                                      "X": sample[0],
                                                      "Y": sample[1],
                                                      "Weight":weight}
            self.saved_ind+=1

    def plot_saved_data(self, template="plotly"):
        fig = px.scatter(self.saved_data, x="X", y="Y", animation_frame="Iteration",
           color="Pred", symbol="Class", color_continuous_scale='Bluered', template=template)
        return fig

def generate_mesh_space(X, n_steps=10, ):
    mesh_space_x1, mesh_space_x2 = np.meshgrid(
        np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), n_steps),
        np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), n_steps))
    mesh_space = np.concatenate(((mesh_space_x1.flatten()).reshape(
        (n_steps ** 2, 1)), (mesh_space_x2.flatten()).reshape(
        (n_steps ** 2, 1))),
        axis=1)
    return mesh_space