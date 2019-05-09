#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from fillData import knn_fill as fill

YEAR_BEGIN = 2001
YEAR_END = 2009


to_drop = ['winner_hand', 'winner_name', 'loser_hand', 'loser_name',
           'winner_h2h_ratio', 'winner_h2h_minutes', 'winner_h2h_ace', 'winner_h2h_df', 'winner_h2h_1st_in', 'winner_h2h_1st_win', 'winner_h2h_2nd_win', 'winner_h2h_break_saved', 'winner_h2h_break_lost',
           'loser_h2h_ratio', 'loser_h2h_minutes', 'loser_h2h_ace', 'loser_h2h_df', 'loser_h2h_1st_in', 'loser_h2h_1st_win', 'loser_h2h_2nd_win', 'loser_h2h_break_saved', 'loser_h2h_break_lost']


def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter, encoding="latin_1")


def swap_column(dataset, column1, column2):
    colList = list(dataset)

    a = colList.index(column1)
    b = colList.index(column2)

    colList[a], colList[b] = colList[b], colList[a]

    return dataset


def symmetric(dataset):
    symmetricDataset = dataset.copy(deep=True)

    toSwap = ['ht', 'age', 'rank', 'rank_points',
              'common_ratio', 'common_minutes', 'common_ace', 'common_df', 'common_1st_in',
              'common_1st_win', 'common_2nd_win', 'common_break_saved', 'common_break_lost',
              'avg_ratio', 'avg_minutes', 'avg_ace', 'avg_df', 'avg_1st_in', 'avg_1st_win',
              'avg_2nd_win', 'avg_break_saved', 'avg_break_lost']

    for swap in toSwap:
        symmetricDataset = swap_column(symmetricDataset, "winner_" + swap, "loser_" + swap)

    return symmetricDataset


def prepareData():
    dataset = load_from_csv("data_07.csv")
    dataset = dataset.drop(columns=to_drop, axis=1)

    dates = dataset["date"]
    cols = list(dataset)

    data = fill(dataset.loc[:, dataset.columns != 'date'])
    dataset = pd.DataFrame(data, columns=cols[1:])
    dataset["date"] = dates

    nbSamples = len(dataset["date"])

    symmetricDataset = symmetric(dataset)

    winning = pd.Series(data=np.full(nbSamples, 1), name="outcome")
    losing = pd.Series(data=np.full(nbSamples, 0), name="outcome")

    dataset = pd.concat([dataset, winning], axis=1, sort=False)
    symmetricDataset = pd.concat([symmetricDataset, losing], axis=1, sort=False)

    dataset = pd.concat([dataset, symmetricDataset], axis=0, sort=False, ignore_index=True)

    dataset.to_csv("preparedDataset.csv", header=True, index=False)


if __name__ == "__main__":
    prepareData()
