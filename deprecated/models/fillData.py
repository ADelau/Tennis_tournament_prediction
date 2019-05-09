import pandas as pd
import numpy as np
from fancyimpute import KNN, IterativeImputer


def simple_fill(dataset):
    statColumns = ("ace", "df", "svpt", "first_in", "first_won", "second_won", "sv_gms",
                   "bp_saved", "bp_faced")

    for player in ("_1", "_2"):
        for stat in statColumns:
            print(stat)
            for i, sample in dataset.iterrows():
                if(np.isnan(sample[stat + "_avg" + player])):
                    continue

                if(np.isnan(sample[stat + "_common" + player])):
                    dataset.loc[i, stat + "_common" + player] = dataset[stat + "_avg" + player][i]

                if(np.isnan(sample[stat + "_prev" + player])):
                    dataset.loc[i, stat + "_prev" + player] = dataset[stat + "_common" + player][i]

    numericalColumns = ("age_1", "atp_ranking_1", "atp_points_1", "height_1", "ratio_avg_1", "ratio_prev_1", "ratio_common_1",
                        "ace_avg_1", "df_avg_1", "svpt_avg_1", "first_in_avg_1", "first_won_avg_1", "second_won_avg_1", "sv_gms_avg_1",
                        "bp_saved_avg_1", "bp_faced_avg_1", "ace_prev_1", "df_prev_1", "svpt_prev_1", "first_in_prev_1", "first_won_prev_1",
                        "sv_gms_prev_1", "bp_saved_prev_1", "bp_faced_prev_1", "ace_common_1", "df_common_1", "svpt_common_1",
                        "first_in_common_1", "first_won_common_1", "second_won_common_1", "sv_gms_common_1", "bp_saved_common_1",
                        "bp_faced_common_1", "age_2", "atp_ranking_2", "atp_points_2", "height_2", "ratio_avg_2", "ratio_prev_2", "ratio_common_2",
                        "ace_avg_2", "df_avg_2", "svpt_avg_2", "first_in_avg_2", "first_won_avg_2", "second_won_avg_2", "sv_gms_avg_2",
                        "bp_saved_avg_2", "bp_faced_avg_2", "ace_prev_2", "df_prev_2", "svpt_prev_2", "first_in_prev_2", "first_won_prev_2",
                        "sv_gms_prev_2", "bp_saved_prev_2", "bp_faced_prev_2", "ace_common_2", "df_common_2", "svpt_common_2",
                        "first_in_common_2", "first_won_common_2", "second_won_common_2", "sv_gms_common_2", "bp_saved_common_2",
                        "bp_faced_common_2")

    categoricalColumns = (("hand_1", "R"), ("hand_2", "R"))

    for column in numericalColumns:
        dataset[column] = dataset[column].fillna(dataset[column].mean())

    for column, value in categoricalColumns:
        dataset[column] = dataset[column].fillna(value)

    dataset.fillna(method="ffill", inplace=True)
    dataset.fillna(method="bfill", inplace=True)

    return dataset


def knn_fill(dataset):
    return KNN(k=3).fit_transform(dataset)


def iterative_fill(dataset):
    return IterativeImputer().fit_transform(dataset)
