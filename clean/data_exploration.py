import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from fancyimpute import KNN


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


if __name__ == '__main__':

    df = load_from_csv("dataset_07.csv")
    df.dropna().hist(bins=100)
    plt.show()

    #df_filled = KNN(k=3).fit_transform(df.loc[:, df.columns != 'date'])
