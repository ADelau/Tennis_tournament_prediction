#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from datetime import date
from data import build_dataset, build_dataset_pca, build_dataset_lda

def random_search_param():
    estimator = RandomForestClassifier()

    dataset = build_dataset()

    n_estimators = [100] #Pas besoin de l'optimiser, un n_estimator élevé produire toujours de meilleurs résultats
    max_features = ["sqrt", "auto"]
    max_depth = [int(x) for x in np.linspace(start = 10, stop = 200, num = 20)]
    max_depth.append(None)
    min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 200, num = 100)]
    bootstrap = ["True", "False"]
   
    param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'bootstrap': bootstrap}

    search = cv.RandomizedSearchCV(estimator, param_grid, n_iter = 50, cv = 3)
    search.fit(dataset, "RandomForestRandomSearch.txt")

def grid_search_param():
	
    estimator = RandomForestClassifier()

    dataset = build_dataset()

    # /!\ modifier en fonction de ce que retournera RandomSearch sinon la on est parti pour 3 ans
    n_estimators = [100]
    max_features = ["sqrt", "auto"]
    max_depth = [int(x) for x in np.linspace(start = 10, stop = 50, num = 5)]
    min_samples_split = [int(x) for x in np.linspace(start = 120, stop = 170, num = 6)]
    bootstrap = ["True", "False"]

    param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'bootstrap': bootstrap}

    search = cv.GridSearchCV(estimator, param_grid, cv = 5)
    search.fit(dataset, "RandomForestGridSearch.txt")

def feature_importance():
    estimator = RandomForestClassifier(max_depth = 10, min_samples_split = 170, bootstrap = True)

    dataset = build_dataset()

    testDate = date(2018, 1, 1)

    trainSet = dataset[dataset["date"] < testDate.toordinal()]
    testSet = dataset[dataset["date"] >= testDate.toordinal()]

    trainX = trainSet.drop("outcome", axis = 1)
    trainY = trainSet["outcome"]

    testX = testSet.drop("outcome", axis = 1)
    testY = testSet["outcome"]

    estimator.fit(trainX, trainY)

    scores = [x for x in estimator.feature_importances_]
    featuresNames = trainX.columns

    tmp = []
    for i in range(len(scores)):
        tmp.append((scores[i], featuresNames[i]))

    tmp = sorted(tmp)

    file = open("features_importance", "w")

    for i in range(len(scores)):
        file.write("{} = {}\n".format(tmp[i][1], tmp[i][0]))

    file.close()

def compute_score():
    estimator = RandomForestClassifier(max_depth = 10, min_samples_split = 170, bootstrap = True)

    dataset = build_dataset()

    print(cv.cross_val_proba_score(dataset, estimator, 3))

if __name__ == "__main__":
    compute_score()