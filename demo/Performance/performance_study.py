import os
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# result_dir =
# dataset_name =

def get_n_feature(file_name, view_name=None, view_sizes=None):
    feat_imp_df = pd.read_csv(file_name, delimiter=',', header=0, index_col=0)
    if view_name is not None:
        good_ind = [ind for ind in feat_imp_df.index if ind.startswith(view_name)]
        feat_imp_df = feat_imp_df.loc[good_ind]
    n_feat = dict((col_name, sum(feat_imp_df[col_name]!=0.0)) if not feat_imp_df[col_name].isnull().values.any()
                  else (col_name, "all") for col_name in feat_imp_df.columns )
    return n_feat

def get_scores(file_name):
    scores_df = pd.read_csv(file_name, delimiter=',', header=0, index_col=0)
    return scores_df

def get_iter_res(path, dataset_name, view_name, view_sizes):
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
                n_feat_dict = get_n_feature(os.path.join(new_path, "feature_importances","{}_dataframe.csv".format(dataset_name, view_name)), view_name=view_name, view_sizes=view_sizes)
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
    n_feat_mean = dict((k, v/n_iter) if isinstance(v, int) else (k,"all") for k, v in n_feat_mean.items() )
    mean_scores /= n_iter
    return n_feat_mean, mean_scores

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
    return latex

if __name__ == "__main__":

    dataset_names = [
        # "study_med"]
      "metagenome"]
    #     "abalone",
    #     "australian",
    #     # "balance",
    #     "bupa",
    #     "cylinder",
    #     "hepatitis",
    #     "ionosphere",
    #     "pima",
    #     "yeast",
    #     "tnbc_mazid",
    #     "bleuets",
    # ]
    view_names = [
        # "Study_med_data",]
        ["cog", "ec", "go", "kegg_module", "kegg_pathway", "taxa.family", "taxa.phylum", "taxa.genus"]]
    #     "Abalone_data",
    #     "australian",
    #     "balance_data",
    #     "bupa_data",
    #     "cylinder_data",
    #     "hepatitis_data",
    #     "ionosphere_data",
    #     "pima",
    #     "yeast_data",
    #     ["clinic", "methyl", "mirna", "rna", "rna_iso"],
    #     "metabolomics_bleuets"
    # ]
    view_sizes = {"ec":2736, "taxa.family": 101, "go":11946, "cog":24,
                  "taxa.genus":72, "taxa.phylum":37, "kegg_pathway":414,
                  "kegg_module":682}
    alg_names = ["Decision Tree", "Adaboost", "Samba", "Random Scm", "Scm",
                 "Random Forest", "XGBoost", "Gradient Boosting", "Lasso", "SamBA+DT"]
    alg_dict = {"decision_tree":"Decision Tree", "adaboost":"Adaboost",
                "samba":"SamBA", "random_forest":"Random Forest", "knn":'KNN',
                "svm_rbf":"SVM-RBF", "xgboost":"XGBoost",
                "gradient_boosting":"Gradient Boosting",
                "lasso":"Lasso", "samba_dt":"SamBA+DT", "_lasso":"plif", "+decision_tree":"plif"}
    train_res_df = pd.DataFrame(columns=["Dataset"] + alg_names)
    test_res_df = pd.DataFrame(columns=["Dataset"] + alg_names)
    test_res_df_1 = pd.DataFrame(columns=["Dataset"] + alg_names)
    for dataset_name, view_name in zip(dataset_names, view_names):
        path = get_path(dataset_name)
        if isinstance(view_name, list):
            res = pd.read_csv(
                "/demo/Performance/results/metagenome/debug_started_2022_11_25-11_47_00_th/metagenome-mean_on_10_iter-balanced_accuracy_p.csv",
                index_col=0, header=0)
            res = res.rename(dict((col_name, "_"+col_name) for col_name in res.columns if col_name.startswith("lasso") or col_name.startswith('svm')), axis="columns")
            res2 = pd.read_csv(
                "/home/baptiste/Documents/Gitwork/summit/results/metagenome/debug_started_2023_04_14-07_50_02_th/metagenome-mean_on_10_iter-balanced_accuracy_p.csv",
                index_col=0, header=0)
            res3 = pd.read_csv(
                "/home/baptiste/Documents/Gitwork/summit/results/metagenome/debug_started_2023_04_21-06_21_34_th/metagenome-mean_on_10_iter-balanced_accuracy_p.csv",
                index_col=0, header=0)
            res3 = res3.rename(dict(
                (col_name, "_" + col_name) for col_name in res3.columns if
                col_name.startswith("decision")), axis="columns")
            res = pd.concat((res, res2, res3), axis=1)
            print(res)
            full_res = pd.DataFrame(columns=["Dataset", "Decision Tree",
                                             "Adaboost", "SamBA",
                                             "Random Forest", 'KNN', "SVM-RBF",
                                             "Gradient Boosting", "XGBoost",
                                             "Lasso", "SamBA+DT"])
            for v_name in view_name:
                full_res = full_res.append(dict(dict((alg_dict[plif.split("-")[0]],
                                                      "{} $\pm$ {}".format(round(res[plif].loc["Test"], 2),
                                                                         round(res[plif].loc["Test STD"], 2)))
                                                     for plif in res.columns
                                                     if plif.endswith(v_name)
                                                     and plif.split("-")[0] in alg_dict),
                                                **{"Dataset":v_name+"\\hfill ({})".format(view_sizes[v_name])}),
                                           ignore_index=True)
                n_feat, scores = get_iter_res(path, dataset_name, v_name, view_sizes)
                keys = n_feat.keys()
                for key in keys:
                    if key.startswith('lasso'):
                        n_feat["+"+key] = n_feat.pop(key)
                keys = n_feat.keys()
                n_feat = dict((key, val) if not key.startswith("svm") else ("+"+key, val) for key, val in n_feat.items() )
                # for key in keys:
                #     if key.startswith('svm_rbf'):
                #         n_feat["+"+key] = n_feat.pop(key)
                scores = scores.rename(dict((col_name, "+"+col_name) for col_name in scores.columns if col_name.startswith("lasso") or col_name.startswith("svm")), axis="columns")
                n_feat_1, scores_1 = get_iter_res("/home/baptiste/Documents/Gitwork/summit/results/metagenome/debug_started_2023_04_14-07_50_02_th/", dataset_name, v_name,
                                              view_sizes)
                n_feat.update(n_feat_1)
                n_feat_2, scores_2 = get_iter_res(
                    "/home/baptiste/Documents/Gitwork/summit/results/metagenome/debug_started_2023_04_21-06_21_34_th/",
                    dataset_name, v_name,
                    view_sizes)
                n_feat_2 = dict((key, val) if not key.startswith("decision") else (
                "+" + key, val) for key, val in n_feat_2.items())
                scores_2 = scores_2.rename(dict((col_name, "+" + col_name) for col_name in scores_2.columns if col_name.startswith("decision")),axis="columns")
                n_feat.update(n_feat_2)
                scores = pd.concat((scores, scores_1, scores_2), axis=1)

                good_cols = [col for col in scores.columns if
                             col.endswith(v_name)]
                scores = scores[good_cols]
                print(n_feat)
                print(scores)
                fig = make_subplots(specs=[[{"secondary_y":True}]])
                alg_names = {"adaboost": "Adaboost",
                             "decision_tree": "DT",
                             "random_forest": "RF",
                             "samba": "SamBA",
                             }
                fig.add_trace(
                    go.Bar(x=[alg_names[key] for key in n_feat.keys() if key in alg_names], y=[n_feat[key] for key in n_feat.keys() if key in alg_names],
                               name="# Features", offsetgroup=0, marker_color='#DC3912'),
                    secondary_y=False,
                )
                cols = [plif for plif in scores.columns if plif.endswith(v_name) and plif.split("-")[0] in alg_names]
                print(scores[cols])
                print([alg_names[plif.split("-")[0]] for plif in cols if plif.split("-")[0] in alg_names ])
                print(scores[cols].to_numpy())
                fig.add_trace(
                    go.Bar(x=[alg_names[plif.split("-")[0]] for plif in cols if plif.split("-")[0] in alg_names ], y=scores[cols].to_numpy()[1,:],
                               name="Balanced Accuracy (Test)",offsetgroup=1,  marker_color='#3366CC'),
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
                # fig.update_xaxes(showline=True, linecolor='black',
                #                  showgrid=False)
                # fig.update_yaxes(showline=True, linecolor='black',
                #                  showgrid=False)
                # fig.show()
                fig.write_image("figures/perf_and_feats_{}.pdf".format(v_name), height=420, width=1000)
                fig.update_layout(font_color="white")
                fig.write_image("figures/perf_and_feats_{}_w.pdf".format(v_name), height=420, width=1000)
                test_res_df = test_res_df.append(dict(dict((" ".join(col.split("-")[0].split("_")).title(), "{} \\scriptsize ({}) \\normalsize ".format(round(scores[col].loc['Test'], 2), n_feat[col.split("-")[0]])) for col in scores.columns), **{"Dataset":v_name + " \\hfill ({})".format(view_sizes[v_name])}), ignore_index=True)
                test_res_df_1 = test_res_df_1.append(dict(dict((" ".join(col.split("-")[0].split("_")).title(), "{} ({})".format( round(scores[col].loc['Test'],2),n_feat[col.split("-")[0]])) for col in scores.columns), **{"Dataset": v_name + " ({})".format(view_sizes[v_name])}), ignore_index=True)
        else:
            n_feat, scores = get_iter_res(path, dataset_name, view_name, view_sizes)
            test_res_df = test_res_df.append(dict(dict((" ".join(
                col.split("-")[0].split("_")).title(), "{} ({})".format(
                round(scores[col]['Test'], 2), n_feat[col.split("-")[0]])) for
                                                       col in scores.columns),
                                                  **{"Dataset": dataset_name}),
                                             ignore_index=True)
    print(format_latex(full_res.to_latex(index=False, escape=False, column_format="|l|c|c|c|c|c|c|")))
    test_res_df.to_csv('figures/study_med_test_res.csv')
    print(test_res_df.columns)
    latex = test_res_df[["Dataset", "Adaboost", "Decision Tree", 'Samba', 'Random Forest', 'Svm Rbf', "Knn", "Gradient Boosting", "Xgboost", "Samba Dt", "Lasso"]].to_latex(index=False, escape=False, column_format="|l|c|c|c|c|c|",)
    latex = latex.replace("\\toprule\n", "")
    latex = latex.replace("\\midrule\n", "")
    latex = latex.replace("\\bottomrule\n", "")
    latex = latex.replace("\n", " \\hline\n")
    latex = latex.replace("_", ".")
    latex = latex.replace("Samba", r"\algo")
    latex = latex.replace("Knn", "KNN")
    latex = latex.replace("Svm Rbf", "SVM-RBF")
    print(format_latex(latex))
    print(test_res_df_1[["Dataset", "Adaboost", "Decision Tree", 'Samba', 'Random Forest', 'Svm Rbf', "Knn", "Gradient Boosting", "Xgboost", "Samba Dt", "Lasso"]].to_markdown())
    # get_n_feature("/home/baptiste/Documents/Gitwork/summit/results/bupa/debug_started_2022_03_16-14_08_02__/iter_1/feature_importances/bupa-bupa_data-feature_importances_dataframe.csv")
    # get_iter_res("/home/baptiste/Documents/Gitwork/summit/results/bupa/debug_started_2022_03_16-14_28_57__", "bupa", "bupa_data")