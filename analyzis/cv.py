#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import date
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import random
import math
from itertools import product

LAST_DATA_YEAR = 2009

def cross_val_score(dataset, estimator, cv, scoreFunction = mean_squared_error):
	
	score = 0

	for year in range(LAST_DATA_YEAR - cv + 1, LAST_DATA_YEAR + 1):

		print("fold {}".format(year - LAST_DATA_YEAR + cv))
		
		testDate = date(year, 1, 1)

		trainSet = dataset[dataset["date"] < testDate.toordinal()]
		testSet = dataset[dataset["date"] >= testDate.toordinal()]

		trainSet.drop("date", axis = 1)
		testSet.drop("date", axis = 1)

		trainX = trainSet.drop("outcome", axis = 1)
		trainY = trainSet["outcome"]

		testX = testSet.drop("outcome", axis = 1)
		testY = testSet["outcome"]

		estimator.fit(trainX, trainY)
		predictY = estimator.predict(testX)

		score += scoreFunction(testY, predictY)

	score /= cv

	return score

def cross_val_accuracy(dataset, estimator, cv):
	return cross_val_score(dataset, estimator, cv, accuracy_score)

def cross_val_proba_score(dataset, estimator, cv, scoreFunction = mean_squared_error):
	
	score = 0

	for year in range(LAST_DATA_YEAR - cv + 1, LAST_DATA_YEAR + 1):

		print("fold {}".format(year - LAST_DATA_YEAR + cv))
		
		testDate = date(year, 1, 1)

		trainSet = dataset[dataset["date"] < testDate.toordinal()]
		testSet = dataset[dataset["date"] >= testDate.toordinal()]

		trainSet = trainSet.drop("date", axis = 1)
		trainSet.replace([np.inf, -np.inf], np.nan)
		trainSet = trainSet.dropna()
		testSet = testSet.drop("date", axis = 1)
		testSet.replace([np.inf, -np.inf], np.nan)
		testSet = testSet.dropna()

		trainX = trainSet.drop("outcome", axis = 1)
		trainY = trainSet["outcome"]

		testX = testSet.drop("outcome", axis = 1)
		testY = testSet["outcome"]

		#trainX.to_csv("trainX.csv", header = True, index = False)
		#trainY.to_csv("trainY.csv", header = True, index = False)

		estimator.fit(trainX, trainY)
		proba = estimator.predict_proba(testX)
		predictY = proba[:, np.where(estimator.classes_ == 1)[0][0]]

		score += scoreFunction(testY, predictY)

	score /= cv

	return score

class RandomizedSearchCV():

	def __init__(self, estimator, param_distribution, n_iter = 10, cv = 3, cvFunction = cross_val_score):

		self.estimator = estimator
		self.param_distribution = param_distribution
		self.n_iter = n_iter
		self.cv = cv
		self.best_param_ = None
		self.best_score_ = math.inf
		self.cvFunction = cvFunction


	def fit(self, dataset, filename):

		self.best_param_ = None
		self.best_score_ = math.inf

		file = open(filename, "w")

		for iteration in range(self.n_iter):

			print("iteration {}".format(iteration))

			testedParam = {name : random.choice(distribution) for name, distribution in self.param_distribution.items()}
			self.estimator.set_params(**testedParam)

			score = self.cvFunction(dataset, self.estimator, self.cv)

			for key, value in testedParam.items():
				file.write("{} = {}\n".format(key, value))

			file.write("score = {}\n \n".format(score))

			if score < self.best_score_:
				self.best_score_ = score
				self.best_param_ = testedParam

		for key, value in self.best_param_.items():
				file.write("best {} = {}\n".format(key, value))

		file.write("best score = {}\n \n".format(self.best_score_))

		trainX = dataset.drop("outcome", axis = 1)
		trainY = dataset["outcome"]

		print("final fit")

		self.estimator.set_params(**self.best_param_)
		self.estimator.fit(trainX, trainY)

		file.close()		

	def predict(self, x):
		return self.estimator.predict(x)


def _my_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


class GridSearchCV():

	def __init__(self, estimator, param_grid, cv = 3, cvFunction = cross_val_score):

		self.estimator = estimator
		self.param_grid = param_grid
		self.cv = cv
		self.best_param_ = None
		self.best_score_ = math.inf
		self.cvFunction = cvFunction

	def fit(self, dataset, filename):

		self.best_param_ = None
		self.best_score_ = math.inf

		file = open(filename, "w")

		iteration = 0

		for testedParam in _my_product(self.param_grid):

			print("iteration {}".format(iteration))
			iteration += 1

			self.estimator.set_params(**testedParam)

			score = self.cvFunction(dataset, self.estimator, self.cv)

			for key, value in testedParam.items():
				file.write("{} = {}\n".format(key, value))

			file.write("score = {}\n".format(score))

			if score < self.best_score_:
				self.best_score_ = score
				self.best_param_ = testedParam

		for key, value in self.best_param_.items():
				file.write("best {} = {}\n".format(key, value))

		file.write("best score = {}\n".format(self.best_score_))

		trainX = dataset.drop("outcome", axis = 1)
		trainY = dataset["outcome"]

		self.estimator.set_params(best_param_)
		self.estimator.fit(trainX, trainY)

		file.close()

	def predict(self, x):
		return self.estimator.predict(x)
