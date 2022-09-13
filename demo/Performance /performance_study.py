import os
import pandas as pd

# result_dir =
# dataset_name =

def get_n_feature(file_name, view_name=None):
    feat_imp_df = pd.read_csv(file_name, delimiter=',', header=0, index_col=0)
    if view_name is not None:
        good_ind = [ind for ind in feat_imp_df.index if ind.startswith(view_name)]
        feat_imp_df = feat_imp_df.loc[good_ind]
    n_feat = dict((col_name, sum(feat_imp_df[col_name]!=0.0)) for col_name in feat_imp_df.columns)
    return n_feat

def get_scores(file_name):
    scores_df = pd.read_csv(file_name, delimiter=',', header=0, index_col=0)
    return scores_df

def get_iter_res(path, dataset_name, view_name):
    n_feat_mean = None
    mean_scores = None
    n_iter = 0
    for name in os.listdir(path):
        if name.startswith('iter'):
            n_iter+=1
            new_path = os.path.join(path, name)
            if os.path.isfile(os.path.join(new_path, "feature_importances", "{}-{}-feature_importances_dataframe.csv".format(dataset_name, view_name))):
                n_feat_dict = get_n_feature(os.path.join(new_path, "feature_importances", "{}-{}-feature_importances_dataframe.csv".format(dataset_name, view_name)))
            else:
                n_feat_dict = get_n_feature(os.path.join(new_path, "feature_importances","{}_dataframe.csv".format(dataset_name, view_name)), view_name=view_name)
            if n_feat_mean is not None:
                for key, val in n_feat_dict.items():
                    n_feat_mean[key]+=val
            else:
                n_feat_mean = n_feat_dict
            scores_df = get_scores(os.path.join(new_path, "{}-balanced_accuracy_p.csv".format(dataset_name)))
            if mean_scores is None:
                mean_scores = scores_df
            else:
                mean_scores+=scores_df
    # print(n_feat_mean)
    n_feat_mean = dict((k, v/n_iter) for k, v in n_feat_mean.items() )
    mean_scores /= n_iter
    return n_feat_mean, mean_scores

def get_path(dataset_name, base_path="results"):
    path = os.path.join(base_path, dataset_name)
    for exp_name in os.listdir(path):
        if dataset_name=="tnbc_mazid" and exp_name.endswith('09__'):
            return os.path.join(path, exp_name)
        elif dataset_name=="bleuets" and exp_name.endswith('05__'):
            return os.path.join(path, exp_name)
        elif exp_name.split("-")[0].endswith("16") or exp_name.split("-")[0].endswith("17"):
            return os.path.join(path, exp_name)


if __name__ == "__main__":

    dataset_names = [
        "abalone",
        "australian",
        # "balance",
        "bupa",
        "cylinder",
        "hepatitis",
        "ionosphere",
        "pima",
        "yeast",
        "tnbc_mazid",
        "bleuets",
    ]
    view_names = [
        "Abalone_data",
        "australian",
        # "balance_data",
        "bupa_data",
        "cylinder_data",
        "hepatitis_data",
        "ionosphere_data",
        "pima",
        "yeast_data",
        ["clinic", "methyl", "mirna", "rna", "rna_iso"],
        "metabolomics_bleuets"
    ]
    alg_names = ["Decision Tree", "Adaboost", "Samba", "Random Scm", "Scm",
                 "Random Forest"]
    train_res_df = pd.DataFrame(columns=["Dataset"] + alg_names)
    test_res_df = pd.DataFrame(columns=["Dataset"] + alg_names)
    for dataset_name, view_name in zip(dataset_names, view_names):
        path = get_path(dataset_name)
        if isinstance(view_name, list):
            for v_name in view_name:
                n_feat, scores = get_iter_res(path, dataset_name, v_name)
                good_cols = [col for col in scores.columns if col.endswith(v_name)]
                scores = scores[good_cols]
                test_res_df = test_res_df.append(dict(dict((" ".join(col.split("-")[0].split("_")).title(), "{} ({})".format(round(scores[col]['Test'], 2), n_feat[col.split("-")[0]])) for col in scores.columns), **{"Dataset":dataset_name+"-"+v_name}), ignore_index=True)
        else:
            n_feat, scores = get_iter_res(path, dataset_name, view_name)
            test_res_df = test_res_df.append(dict(dict((" ".join(
                col.split("-")[0].split("_")).title(), "{} ({})".format(
                round(scores[col]['Test'], 2), n_feat[col.split("-")[0]])) for
                                                       col in scores.columns),
                                                  **{"Dataset": dataset_name}),
                                             ignore_index=True)
    test_res_df.to_csv('tnbc_test_res.csv')
    # get_n_feature("/home/baptiste/Documents/Gitwork/summit/results/bupa/debug_started_2022_03_16-14_08_02__/iter_1/feature_importances/bupa-bupa_data-feature_importances_dataframe.csv")
    # get_iter_res("/home/baptiste/Documents/Gitwork/summit/results/bupa/debug_started_2022_03_16-14_28_57__", "bupa", "bupa_data")