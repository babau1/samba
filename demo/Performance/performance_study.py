import os
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings
import numpy as np
import re


def get_n_feature(file_name, view_name=None,):
    feat_imp_df = pd.read_csv(file_name, delimiter=',', header=0, index_col=0)
    if view_name is not None:
        good_ind = [ind for ind in feat_imp_df.index if ind.startswith(view_name)]
        feat_imp_df = feat_imp_df.loc[good_ind]
    n_feat = dict((col_name, sum(feat_imp_df[col_name]!=0.0)) if not sum(feat_imp_df[col_name])==0
                  else (col_name, "all") for col_name in feat_imp_df.columns )
    return n_feat


# def get_scores(file_name):
#     scores_df = pd.read_csv(file_name, delimiter=',', header=0, index_col=0)
#     return scores_df


def get_iter_res(path, dataset_name, view_name):
    n_feat_mean = None
    mean_scores = None
    n_feats = None
    scores = None
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
                    if not isinstance(val, str):
                        n_feat_mean[key]+=val
                        n_feats[key].append(val)
            else:
                n_feat_mean = n_feat_dict
                n_feats = dict((key, [val]) for key, val in n_feat_dict.items())
            # scores_df = get_scores(os.path.join(new_path, "{}-balanced_accuracy_p.csv".format(dataset_name)))
            # if mean_scores is None:
            #     mean_scores = scores_df
            # else:
            #     mean_scores+=scores_df
    n_feats_mean = dict((k, np.mean(np.array(v))) if isinstance(v[0], int) else (k, 'all') for k, v in n_feats.items())
    n_feats_std = dict((k, np.std(v))  if isinstance(v[0], int) else (k, 0) for k, v in n_feats.items())
    # n_feat_mean = dict((k, v/n_iter) if isinstance(v, int) else (k,"all") for k, v in n_feat_mean.items() )
    # mean_scores /= n_iter
    return n_feats_mean, n_feats_std


def get_path(dataset_name, base_path="results"):
    path = os.path.join(base_path, dataset_name)
    for exp_name in os.listdir(path):
        if dataset_name=="tnbc_mazid" and exp_name.endswith('09__'):
            return os.path.join(path, exp_name)
        elif dataset_name=="bleuets" and exp_name.endswith('05__'):
            return os.path.join(path, exp_name)
        elif dataset_name=="study_med" and exp_name.endswith('30__'):
            return os.path.join(path, exp_name)
        elif exp_name.split("-")[0].endswith("16") or exp_name.split("-")[0].endswith("25"):
            return os.path.join(path, exp_name)


def format_latex(latex):
    latex = latex.replace("\\toprule\n", "")
    latex = latex.replace("\\midrule\n", "")
    latex = latex.replace("\\bottomrule\n", "")
    latex = latex.replace("\n", " \\hline\n")
    latex = latex.replace("_", ".")
    latex = latex.replace("Samba", r"\algo")
    latex = latex.replace("Knn", "KNN")
    latex = latex.replace("Svm Rbf", "SVM-RBF")
    latex = latex.replace("Random Forest", "Rand. Forest")
    latex = latex.replace("Decision Tree", "Dec. Tree")
    latex = latex.replace("Gradient Boosting", "Grad. Boost.")
    latex = latex.replace("Xgboost", "XGBoost")
    latex = re.sub("& +0.", "& .", latex)
    latex = latex.replace("& .7 ", "& .70 ")
    latex = latex.replace("& .8 ", "& .80 ")
    return latex

# def pm_string(string, res):
#     return "{} $\pm$ {}".format(round(res[string].loc["Test"], 2), round(res[string].loc["Test STD"], 2))

def get_name(string, alg_dict):
    return alg_dict[string.split("-")[0]]

def get_scores(df, v_name):
    cols_to_keep = []
    new_names = {}
    for col_name in df.columns:
        if col_name.endswith(v_name):
            cols_to_keep.append(col_name)
            new_names[col_name] = col_name.split('-')[0]
    scores = df[cols_to_keep]
    scores = scores.rename(new_names, axis=1)
    return scores.loc["Test"], scores.loc["Test STD"]


def plot_acc_and_feats(n_feat, scores, std_scores, v_name):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    alg_names = {"samba": "SamBA",
                 "adaboost": "Adaboost",
                 "xgboost": "XGBoost",
                 "gradient_boosting": "GB",
                 "random_forest": "RF",
                 "decision_tree": "DT",
                 "lasso": "Lasso"
                 }
    n_feat = dict((alg_name, n_feat[alg_name]) for alg_name in alg_names)
    n_feat = dict(sorted(n_feat.items(), key=lambda item: item[1]))
    fig.add_trace(
        go.Bar(x=[alg_names[key] for key in n_feat.keys() if key in alg_names],
               y=[n_feat[key] for key in n_feat.keys() if key in alg_names],
               # error_y=dict(type='data', array=[std_feats[key] for key in std_feats.keys() if key in alg_names], visible=True),
               name="# Features", offsetgroup=0, marker_color='#DC3912'),
        secondary_y=False,
    )
    # cols = [_ for _ in scores.columns if _.endswith(v_name) and _.split("-")[0] in alg_names]
    fig.add_trace(
        go.Bar(x=[alg_names[key] for key in scores.keys() if key in alg_names],
               y=[scores[key] for key in scores.keys() if key in alg_names],
               error_y=dict(type='data',
                            array=[std_scores[key] for key in std_scores.keys()
                                   if key in alg_names], visible=True),
               name="Balanced Accuracy (Test)", offsetgroup=1,
               marker_color='#3366CC'),
        secondary_y=True,
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_size=25,
        legend=dict(
            yanchor="top",
            y=1.25,
            xanchor="left",
            x=0.01),
        margin=dict(l=0, r=0, t=0, b=0),
        yaxis=dict(showline=True, linecolor='#DC3912',
                   showgrid=False),
        yaxis2=dict(showline=True, linecolor='#3366CC',
                    showgrid=False),
    )
    fig.write_image("figures/perf_and_feats_{}.pdf".format(v_name), height=420,
                    width=1000)
    fig.update_layout(font_color="white")
    fig.write_image("figures/perf_and_feats_{}_w.pdf".format(v_name),
                    height=420, width=1000)




if __name__ == "__main__":
    # path_res_dir = "/home/baptiste/Documents/Gitwork/neighborhood_classifier/demo/Performance/results/samba_res_uai/projects/def-corbeilj/bbauvin/results/metagenome/debug_started_2023_05_25-10_16_18_uai2023"
    path_res_dir = os.path.join("results","metagenome","debug_started_2023_05_24-16_19_21_uai2023")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset_names = ["metagenome"]
        view_names = [["cog", "ec", "go", "kegg_module", "kegg_pathway",
                       "taxa.family", "taxa.phylum", "taxa.genus"]]
        view_sizes = {"ec": 2736, "taxa.family": 101, "go": 11946, "cog": 24,
                      "taxa.genus": 72, "taxa.phylum": 37, "kegg_pathway": 414,
                      "kegg_module": 682}
        alg_names = ["Decision Tree", "Adaboost", "Samba", "Random Scm", "Scm",
                     "Random Forest", "Xgboost", "Gradient Boosting", "Lasso",
                     "SamBA+DT"]
        alg_dict = {"decision_tree": "Decision Tree", "adaboost": "Adaboost",
                    "samba": "SamBA", "random_forest": "Random Forest", "knn": 'KNN',
                    "svm_rbf": "SVM-RBF", "xgboost": "XGBoost",
                    "gradient_boosting": "Gradient Boosting",
                    "lasso": "Lasso", "samba_dt": "SamBA+DT", "_lasso": "None",
                    "+decision_tree": "None"}
        train_res_df = pd.DataFrame(columns=["Dataset"] + alg_names)
        test_res_df = pd.DataFrame(columns=["Dataset"] + alg_names)
        test_accs_df = pd.DataFrame(columns=["Dataset"] + alg_names)
        feat_res_df = pd.DataFrame(columns=["Dataset"] + alg_names)
        # test_res_df_1 = pd.DataFrame(columns=["Dataset"] + alg_names)
        for dataset_name, view_name in zip(dataset_names, view_names):
            # path = get_path(dataset_name)
            # if isinstance(view_name, list):
            res = pd.read_csv(os.path.join(path_res_dir, "metagenome-mean_on_10_iter-balanced_accuracy_p.csv"), index_col=0, header=0)

            # res = res.rename(dict((col_name, "_"+col_name) for col_name in res.columns if col_name.startswith("lasso") or col_name.startswith('svm')), axis="columns")
            # res2 = pd.read_csv("/home/baptiste/Documents/Gitwork/summit/results/metagenome/debug_started_2023_04_14-07_50_02_th/metagenome-mean_on_10_iter-balanced_accuracy_p.csv", index_col=0, header=0)
            # res3 = pd.read_csv("/home/baptiste/Documents/Gitwork/summit/results/metagenome/debug_started_2023_04_21-06_21_34_th/metagenome-mean_on_10_iter-balanced_accuracy_p.csv", index_col=0, header=0)
            # res3 = res3.rename(dict((col_name, "_" + col_name) for col_name in res3.columns if col_name.startswith("decision")), axis="columns")
            # res = pd.concat((res, res2, res3), axis=1)

            # full_res = pd.DataFrame(columns=["Dataset", "Decision Tree",
            #                                  "Adaboost", "SamBA",
            #                                  "Random Forest", 'KNN', "SVM-RBF",
            #                                  "Gradient Boosting", "XGBoost",
            #                                  "Lasso", "SamBA+DT"])
            for v_name in view_name:
                print(v_name)
                scores, std_scores = get_scores(res, v_name)
                # dict_to_append = dict((get_name(_, alg_dict), pm_string(_, res))
                #                       for _ in res.columns if _.endswith(v_name)
                #                       and _.split("-")[0] in alg_dict)
                # dict_to_append.update({"Dataset": v_name + "\\hfill ({})".format(view_sizes[v_name])})
                # full_res = full_res.append(dict_to_append, ignore_index=True)
                n_feat, std_feats = get_iter_res(path_res_dir, dataset_name, v_name)
                plot_acc_and_feats(n_feat, scores, std_scores, v_name)
                # for key in keys:
                #     if key.startswith('lasso'):
                #         n_feat["+"+key] = n_feat.pop(key)
                # keys = n_feat.keys()
                # n_feat = dict((key, val) if not key.startswith("svm")
                #               else ("+"+key, val) for key, val in n_feat.items() )
                # scores = scores.rename(dict((col_name, "+"+col_name)
                #                             for col_name in scores.columns
                #                             if col_name.startswith("lasso")
                #                             or col_name.startswith("svm")),
                #                        axis="columns")
                # n_feat_1, scores_1 = get_iter_res("/home/baptiste/Documents/Gitwork/summit/results/metagenome/debug_started_2023_04_14-07_50_02_th/",
                #                                   dataset_name, v_name,
                #                                   view_sizes)
                # n_feat.update(n_feat_1)
                # n_feat_2, scores_2 = get_iter_res( "/home/baptiste/Documents/Gitwork/summit/results/metagenome/debug_started_2023_04_21-06_21_34_th/",
                #                                    dataset_name, v_name,
                #                                    view_sizes)
                # n_feat_2 = dict((key, val) if not key.startswith("decision") else ("+" + key, val) for key, val in n_feat_2.items())
                # scores_2 = scores_2.rename(dict((col_name, "+" + col_name)
                #                                 for col_name in scores_2.columns
                #                                 if col_name.startswith("decision")),
                #                            axis="columns")
                # n_feat.update(n_feat_2)
                # scores = pd.concat((scores, scores_1, scores_2), axis=1)

                # good_cols = [col for col in scores.keys() if
                #              col.endswith(v_name)]
                # scores = scores[good_cols]
                test_res_df = test_res_df.append(dict(dict((" ".join(key.split("_")).title(),
                                                            "{} \\scriptsize ({}) \\normalsize ".format(round(val, 2), n_feat[key]))
                                                           for key, val in scores.items()),
                                                      **{"Dataset":v_name + " \\hfill ({})".format(view_sizes[v_name])}), ignore_index=True)
                test_accs_df = test_accs_df.append(dict(dict((" ".join(key.split("_")).title(),
                               "{} $\\pm$ {} ".format(
                                   round(val, 2), round(std_scores[key], 2)))
                              for key, val in scores.items()),
                         **{"Dataset": v_name + " \\hfill ({})".format(
                             view_sizes[v_name])}), ignore_index=True)
                feat_res_df = feat_res_df.append(dict(dict((" ".join(key.split("_")).title(),
                               "{} $\\pm$ {} ".format(
                                   round(val, 2), round(std_feats[key], 2)))
                              if not isinstance(val, str) else (" ".join(key.split("_")).title(), "all") for key, val in n_feat.items()),
                         **{"Dataset": v_name + " \\hfill ({})".format(
                             view_sizes[v_name])}), ignore_index=True)
            col_list = ["Dataset", "Samba", "Adaboost", 'Xgboost',
                             'Gradient Boosting', 'Svm Rbf', "Knn",
                             "Random Forest", "Decision Tree", "Lasso"]
            print(format_latex(feat_res_df[col_list].to_latex(index=False,
                                                              escape=False,
                                                              column_format="|l|c|c|c|c|c|c|c|c|c|",)))
            print(format_latex(test_accs_df[col_list].to_latex(index=False,
                                                              escape=False,
                                                              column_format="|l|c|c|c|c|c|c|c|c|c|", )))

                    # test_res_df_1 = test_res_df_1.append(dict(dict((" ".join(col.split("-")[0].split("_")).title(), "{} ({})".format( round(scores[col].loc['Test'],2),n_feat[col.split("-")[0]])) for col in scores.columns), **{"Dataset": v_name + " ({})".format(view_sizes[v_name])}), ignore_index=True)
            # else:
            #     n_feat, scores = get_iter_res(path_res_dir, dataset_name, view_name, view_sizes)
            #     test_res_df = test_res_df.append(dict(dict((" ".join(
            #         col.split("-")[0].split("_")).title(), "{} ({})".format(
            #         round(scores[col]['Test'], 2), n_feat[col.split("-")[0]])) for
            #                                                col in scores.columns),
            #                                           **{"Dataset": dataset_name}),
            #                                      ignore_index=True)

        test_res_df.to_csv('figures/{}.csv'.format(dataset_name))
        print(test_res_df.columns)
        latex = test_res_df[col_list].to_latex(index=False, escape=False,
                                               column_format="|l|c|c|c|c|c|c|c|c|c|",)

        print(format_latex(latex))