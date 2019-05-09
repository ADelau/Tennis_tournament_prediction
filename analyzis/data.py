import pandas as pd
from datetime import date
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SEED = 11

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
    return pd.read_csv(path, delimiter=delimiter, encoding = "latin_1")

def encode(dataset):
    return pd.get_dummies(dataset)

def myToordinal(date):
    return date.toordinal()

def load_dataset(path):
    dataset = load_from_csv(path)

    dataset["date"] = dataset["date"].astype("datetime64")
    dataset["date"] = dataset["date"].map(myToordinal, na_action = "ignore")
    dataset["date"] = dataset["date"].astype("int64")

    """
    dataset["round"] = dataset["round"].astype("category")
    dataset["surface"] = dataset["surface"].astype("category")
    dataset["court"] = dataset["court"].astype("category")
    dataset["level"] = dataset["level"].astype("category")
    dataset["hand_1"] = dataset["hand_1"].astype("category")
    dataset["hand_2"] = dataset["hand_2"].astype("category")
    """

    dataset = dataset.select_dtypes(["number"])

    return dataset

def compute_difference():
    dataset =  load_dataset("data/preparedDataset.csv")

    toSwap = ['ht', 'age', 'rank', 'rank_points',
              'common_ratio', 'common_minutes', 'common_ace', 'common_df', 'common_1st_in',
              'common_1st_win', 'common_2nd_win', 'common_break_saved', 'common_break_lost',
              'avg_ratio', 'avg_minutes', 'avg_ace', 'avg_df', 'avg_1st_in', 'avg_1st_win',
              'avg_2nd_win', 'avg_break_saved', 'avg_break_lost']

    newDataset = dataset["date"]
    newDataset = pd.concat((newDataset, dataset["outcome"]), axis = 1)

    for column in toSwap:
        winnerSerie = dataset["winner_" + column]
        looserSerie = dataset["loser_" + column]
        diffSerie = winnerSerie.subtract(looserSerie)
        diffSerie.rename(column)
        newDataset = pd.concat((newDataset, diffSerie), axis = 1)

    newDataset.to_csv("diffDataset.csv", header = True, index = False)

def load_train_set():
    return load_dataset("data/preparedDataset.csv")

def build_dataset():
    dataset = load_train_set()
    date = dataset["date"].copy(deep = True)
    date.rename("date_", inplace = True)
    dataset = pd.concat((dataset, date), axis = 1)
    #dataset = encode(dataset)

    return dataset

# Test de preprocess avec pca et lda, voir le nombre de composants à garder avec un screeplot.
def build_dataset_pca():
    dataset = load_train_set()
    featureSelectionSet, trainSet = train_test_split(dataset, train_size = 0.3, random_state = SEED)
    trainSet.reset_index(drop = True, inplace = True)
    date = trainSet["date"]
    outcome = trainSet["outcome"]
    trainSet = trainSet.drop("outcome", axis = 1)
    featureSelectionSet = featureSelectionSet.drop("outcome", axis = 1)
    pca = PCA(n_components = 3, whiten = True)
    pca.fit(featureSelectionSet)
    pcaSet = pca.transform(trainSet)
    pcaSet = pd.DataFrame(pcaSet)
    pcaSet = pd.concat((pcaSet, date, outcome), axis = 1)
    print("nans: {}".format(pcaSet.isnull().sum().sum()))
    pcaSet.to_csv("pca.csv", header = True, index = False)
    return pcaSet

def build_dataset_lda():
    dataset = load_train_set()
    featureSelectionSet, trainSet = train_test_split(dataset, train_size = 0.3, random_state = SEED)
    trainSet.reset_index(drop = True, inplace = True)

    date = trainSet["date"]
    outcome = trainSet["outcome"]
    trainSet = trainSet.drop("outcome", axis = 1)
    outcomeSelection = featureSelectionSet["outcome"]
    featureSelectionSet = featureSelectionSet.drop("outcome", axis = 1)

    lda = LinearDiscriminantAnalysis(n_components = 1)
    lda.fit(featureSelectionSet, outcomeSelection)
    ldaSet = lda.transform(trainSet)

    ldaSet = pd.DataFrame(ldaSet)
    ldaSet = pd.concat((ldaSet, date, outcome), axis = 1)

    print("nans: {}".format(ldaSet.isnull().sum().sum()))
    ldaSet.to_csv("lda.csv", header = True, index = False)

    return ldaSet

def scree_plot_pca():
    dataset = load_train_set()
    dataset = dataset.drop("outcome", axis = 1)
    featureSelectionSet, trainSet = train_test_split(dataset, train_size = 0.3, random_state = SEED)
    pca = PCA(whiten = True)
    pca.fit(featureSelectionSet)

    np.savetxt("pca_components.csv", pca.components_, delimiter=",")

    varExplained = pca.explained_variance_ratio_.cumsum()
    components = [x+1 for x in range(len(varExplained))]

    plt.plot(components, varExplained, 'ro-', linewidth = 2)
    plt.title("scree plot pca")
    plt.xlabel("nb components")
    plt.ylabel("cumulative variance explained")
    plt.savefig("plots/pca.jpg")
    plt.close()

def scree_plot_lda():
    dataset = load_train_set()
    featureSelectionSet, trainSet = train_test_split(dataset, train_size = 0.3, random_state = SEED)
    outcome = featureSelectionSet["outcome"]
    featureSelectionSet = featureSelectionSet.drop("outcome", axis = 1)
    lda = LinearDiscriminantAnalysis()
    lda.fit(featureSelectionSet, outcome)

    print("coef:")
    print(lda.coef_)

    varExplained = lda.explained_variance_ratio_.cumsum()
    components = [x+1 for x in range(len(varExplained))]

    plt.plot(components, varExplained, 'ro-', linewidth = 2)
    plt.title("scree plot lda")
    plt.xlabel("nb components")
    plt.ylabel("cumulative variance explained")
    plt.savefig("plots/lda.jpg")
    plt.close()

if __name__ == "__main__":
    compute_difference()



    