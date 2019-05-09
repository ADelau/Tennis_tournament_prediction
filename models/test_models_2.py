import pandas as pd
import numpy as np
from data_exploration import load_from_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from scipy.stats import levene
from scipy.stats import bartlett
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import TensorBoard
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

def generate_dataset():
	df = load_from_csv("week_data_0997.csv").dropna()

	#df = df.loc[(df["level"] == "G") | (df["level"] == "M")]

	# Keep avg_stats + infos
	cols = ['outcome', 'age_1', 'rank_1', 'rank_points_1', 'age_2', 'rank_2', 'rank_points_2' ] + [col for col in df if col.startswith('common')]

	target_df = df[cols]
	print("Before drop NA: {}".format(target_df.shape))
	target_df = target_df.replace([np.inf, -np.inf], 0)
	target_df = target_df.dropna()
	print("After drop NA: {}".format(target_df.shape))


	# In[5]:


	# Added feature: Completeness 
	target_df["completeness_1"] = target_df.common_serve_win_1 * target_df.common_return_win_1
	target_df["completeness_2"] = target_df.common_serve_win_2 * target_df.common_return_win_2

	# Added feature: Serve Advantage 
	target_df["serve_adv_1"] = target_df.common_serve_win_1 - target_df.common_return_win_2
	target_df["serve_adv_2"] = target_df.common_serve_win_2 - target_df.common_return_win_1

	# Differences between features
	cols_1 = [col for col in target_df if col.endswith('1')] # Player 1
	cols_2 = [col for col in target_df if col.endswith('2')] # Player 2

	cols = [col[0:-2] for col in cols_1] # Final cols


	diff = target_df[cols_1].values - target_df[cols_2].values

	final_df = pd.DataFrame(diff, columns=cols)


	# In[6]:


	X, y = final_df, target_df["outcome"]

	return X,y

def generate_year_06_G_dataset():
	from sklearn.model_selection import train_test_split

	df = load_from_csv("year_data_06.csv").dropna()
	df = df.replace([np.inf, -np.inf], 0).dropna()

	df.drop("date", axis = 1, inplace = True)

	train, test = train_test_split(df, test_size = 0.2, shuffle = False)

	test = test.loc[(test["level"] == "G")]

	train.drop("level", axis = 1, inplace = True)
	test.drop("level", axis = 1, inplace = True)

	X_train = train.drop("outcome", axis = 1)
	y_train = train["outcome"]

	X_test = test.drop("outcome", axis = 1)
	y_test = test["outcome"]

	return X_train, X_test, y_train, y_test

def generate_year_06_dataset():
	df = load_from_csv("year_data_06.csv").dropna()
	df = df.replace([np.inf, -np.inf], 0).dropna()
	df = df.loc[(df["level"] == "G")]
	df.drop("date", axis = 1, inplace = True)
	df.drop("level", axis = 1, inplace = True)
	X = df.drop("outcome", axis = 1)
	y = df["outcome"]

	X_train, X_test, y_train, y_test = split_dataset(X, y)
	return X_train, X_test, y_train, y_test

def split_dataset(X, y):
	# In[8]:

	# Split train/test sets
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

	return X_train, X_test, y_train, y_test

def lda_dataset(X, y):
	X_train, X_test, y_train, y_test = split_dataset(X, y)
	lda = LinearDiscriminantAnalysis(n_components = 5)
	lda.fit(X_train, y_train)
	X_train = lda.transform(X_train)
	X_test = lda.transform(X_test)
   
	return X_train, X_test, y_train, y_test

def pca_dataset(X, y):
	X_train, X_test, y_train, y_test = split_dataset(X, y)
	pca = PCA(n_components = 3, whiten = True)
	pca.fit(X)
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)

	return X_train, X_test, y_train, y_test

def load_lda_set():
	X, y = generate_dataset()
	X_train, X_test, y_train, y_test = lda_dataset(X, y)

	return X_train, X_test, y_train, y_test

def load_pca_set():
	X, y = generate_dataset()
	X_train, X_test, y_train, y_test = pca_dataset(X, y)

	return X_train, X_test, y_train, y_test

def load_regular_set():
	X, y = generate_dataset()
	X_train, X_test, y_train, y_test = split_dataset(X, y)

	return X_train, X_test, y_train, y_test

def logistic_regression(X_train, X_test, y_train, y_test):
	clf = LogisticRegression(solver='lbfgs', penalty='l2', C=100000).fit(X_train, y_train.values.ravel())
	score = clf.score(X_test, y_test.values.ravel())

	print("Accuracy Logistic Regression full stats : {}".format(score))

def test_logistic_regression_year_06():
	X_train, X_test, y_train, y_test = generate_year_06_G_dataset()
	
	logistic_regression(X_train, X_test, y_train, y_test)

def test_logistic_regression_regular():
	X_train, X_test, y_train, y_test = load_regular_set()
	
	logistic_regression(X_train, X_test, y_train, y_test)

def test_logistic_regression_pca():
	X_train, X_test, y_train, y_test = load_pca_set()
	
	logistic_regression(X_train, X_test, y_train, y_test)

def test_random_forest_regular():
	X_train, X_test, y_train, y_test = load_regular_set()
	
	random_forest(X_train, X_test, y_train, y_test)

def test_random_forest_pca():
	X_train, X_test, y_train, y_test = load_pca_set()
	
	random_forest(X_train, X_test, y_train, y_test)

def test_random_forest_year_06():
	X_train, X_test, y_train, y_test = generate_year_06_dataset()
	
	random_forest(X_train, X_test, y_train, y_test)

def random_forest(X_train, X_test, y_train, y_test):
	
	rf = RandomForestClassifier(n_estimators=1000, max_depth=10).fit(X_train, y_train.values.ravel())
	score = rf.score(X_test, y_test.values.ravel())

	print("Accuracy Random Forests: {}".format(score))

def test_base_line_year_06():
	from sklearn.metrics import accuracy_score
	X_train, X_test, y_train, y_test = generate_year_06_G_dataset()

	# Always predict the better ranked player as winner
	y_pred = np.where(X_test.rank_points > 0, 1, 0)

	score = accuracy_score(y_test.values.ravel(), y_pred)
	print("Accuracy Baseline Model rank-based : {}".format(score))

def test_knn_year_06():
	from sklearn.neighbors import KNeighborsClassifier

	X_train, X_test, y_train, y_test = generate_year_06_G_dataset()
	classifier = KNeighborsClassifier(n_neighbors = 50)

	classifier.fit(X_train, y_train)
	score = classifier.score(X_test, y_test)
	print("KNN score = {}".format(score))


def neural_network_1(X_train, X_test, y_train, y_test):
	def create_model():
		DROP_OUT_RATE = 0.1

		model = Sequential()
		model.add(Dense(units=4, activation='relu', input_dim=X_train.shape[1]))
		model.add(Dense(units=4, activation='relu'))
		model.add(Dense(units=1))
	
		opt = keras.optimizers.Adam(lr = 0.001)
		model.compile(loss = keras.losses.mean_squared_error, optimizer = opt, metrics = ["accuracy"])

		return model

	estimator = KerasClassifier(build_fn=create_model, epochs=1000, batch_size=128)

	history = estimator.fit(X_train, y_train, validation_data = (X_test, y_test), verbose = 2)

	print(history.history.keys())

	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	plt.savefig("history.png")

	y_pred = estimator.predict(X_test)

	score = accuracy_score(y_test, y_pred)

	print("Accuracy neural network = {}".format(score))

def neural_network_2(X_train, X_test, y_train, y_test):
	def create_model():
		DROP_OUT_RATE = 0.1

		model = Sequential()
		model.add(Dense(units=50, activation='relu', input_dim=X_train.shape[1]))
		#model.add(Dropout(DROP_OUT_RATE))
		model.add(Dense(units=50, activation='relu'))
		#model.add(Dropout(DROP_OUT_RATE))
		model.add(Dense(units=50, activation='relu'))
		#model.add(Dropout(DROP_OUT_RATE))
		model.add(Dense(units=50, activation='relu'))
		#model.add(Dropout(DROP_OUT_RATE))
		model.add(Dense(units=50, activation='relu'))
		#model.add(Dropout(DROP_OUT_RATE))
		model.add(Dense(units=50, activation='relu'))
		#model.add(Dropout(DROP_OUT_RATE))
		model.add(Dense(units=50, activation='relu'))
		#model.add(Dropout(DROP_OUT_RATE))
		model.add(Dense(units=50, activation='relu'))
		#model.add(Dropout(DROP_OUT_RATE))
		model.add(Dense(units=50, activation='relu'))
		#model.add(Dropout(DROP_OUT_RATE))
		model.add(Dense(units=50, activation='relu'))
		#model.add(Dropout(DROP_OUT_RATE))
		model.add(Dense(units=50, activation='relu'))
		#model.add(Dropout(DROP_OUT_RATE))
		model.add(Dense(units=1, activation = 'sigmoid'))

		opt = keras.optimizers.Adam(lr = 0.001)
		model.compile(loss = keras.losses.mean_squared_error, optimizer = opt, metrics = ["accuracy"])

		return model

	estimator = KerasClassifier(build_fn=create_model, epochs=100, batch_size=128)

	history = estimator.fit(X_train, y_train, validation_data = (X_test, y_test), verbose = 2)

	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	plt.savefig("history.png")

	y_pred = estimator.predict(X_test)

	score = accuracy_score(y_test, y_pred)

	print("Accuracy neural network = {}".format(score))

def test_neural_network_regular():
	X_train, X_test, y_train, y_test = load_regular_set()
	
	neural_network_1(X_train, X_test, y_train, y_test)

def test_neural_network_1_year_06():
	X_train, X_test, y_train, y_test = generate_year_06_dataset()
	
	neural_network_1(X_train, X_test, y_train, y_test)

def test_neural_network_2_year_06():
	X_train, X_test, y_train, y_test = generate_year_06_G_dataset()
	
	neural_network_2(X_train, X_test, y_train, y_test)

def test_lda_year_06():
	X_train, X_test, y_train, y_test = generate_year_06_G_dataset()
	lda = LinearDiscriminantAnalysis(solver = "lsqr", shrinkage = "auto")

	lda.fit(X_train, y_train)
	score = lda.score(X_test, y_test)

	print("Accuracy score lda = {0:.3f}".format(score))

def test_adaboost_pca():
	X_train, X_test, y_train, y_test = load_pca_set()
	
	clf = AdaBoostClassifier(n_estimators=100).fit(X_train, y_train.values.ravel())
	score = clf.score(X_test, y_test.values.ravel())

	print("Accuracy AdaBoost: {}".format(score))

def gradient_boosting_pca():
	X_train, X_test, y_train, y_test = load_pca_set()

	# Gradient Boosting

	learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
	for learning_rate in learning_rates:
	    gb = GradientBoostingClassifier(n_estimators=100, learning_rate = learning_rate, max_features=16, max_depth = None, random_state = 0)
	    gb.fit(X_train, y_train)
	    print("Learning rate: ", learning_rate)
	    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))
	    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))
	    print()

def test_lda():
	X_train, X_test, y_train, y_test = load_regular_set()
	lda = LinearDiscriminantAnalysis()

	lda.fit(X_train, y_train)
	score = lda.score(X_test, y_test)

	print("Accuracy score lda = {0:.3f}".format(score))

def test_lda_normal():
	X_train, X_test, y_train, y_test = load_regular_set()
	X_train = normalize(X_train)
	X_test = normalize(X_test)
	lda = LinearDiscriminantAnalysis()

	lda.fit(X_train, y_train)
	score = lda.score(X_test, y_test)

	print("Accuracy score lda = {0:.3f}".format(score))

def check_homoscedasticy():
	X, _, y, _ = generate_year_06_dataset()

	statistic, p_value = levene(*list(X.T.to_numpy()))

	print("levene: statistic = {}, p_value = {}".format(statistic, p_value))

	statistic, p_value = bartlett(*list(X.T.to_numpy()))

	print("bartlett: statistic = {}, p_value = {}".format(statistic, p_value))

if __name__ == "__main__":
	#test_neural_network_1_year_06()
	#test_neural_network_2_year_06()
	#test_lda_year_06()
	#test_logistic_regression_year_06()
	#test_random_forest_year_06()
	#test_base_line_year_06()
	#test_knn_year_06()
	check_homoscedasticy()