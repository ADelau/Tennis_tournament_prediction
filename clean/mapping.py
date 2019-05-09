import pandas as pd
from data import load_from_csv
import numpy as np

def map(df, dict):
	df["name"] = df["name"].astype("int64", copy = False)
	df["name"] = df["name"].map(dict)
	return df

def load_dic(fileName):
	df = load_from_csv(fileName)
	dic = {}

	for index, row in df.iterrows():
		dic[row["winner_id"]] = row["winner_name"]
		dic[row["loser_id"]] = row["loser_name"]

	for key, value in dic.items():
		splitted = value.split()
		name = splitted[0]
		splitted[0] = name[0] + "."
		dic[key] = " ".join(splitted)

	return dic

def get_last_name(name):
	names = name.split()
	return names[-1]

def load_reached_dic(fileName):
	df = load_from_csv(fileName)
	dic = {}

	df["last_name"] = df["name"].apply(get_last_name)
	
	for index, row in df.iterrows():
		print("row = {}".format(row))
		dic[row["last_name"]] = str(row["reached"])

	return dic

def replace_nan(position):
	if str(position) == "nan":
		return "absent"
	else:
		return position

def merge_reached(df, reachedDic):
	last_names = df["name"].apply(get_last_name)
	reached = last_names.map(reachedDic)
	print(reached)
	reached = reached.apply(replace_nan)
	reached = reached.rename("reached")

	return pd.concat((df, reached), axis = 1)

def make_mapped_file(loadFileName, targetFileName, dicFileName = None, reachedFileName = None):
	targetDf = load_from_csv(loadFileName)
	
	if reachedFileName:
		reachedDic = load_reached_dic(reachedFileName)
		targetDf = merge_reached(targetDf, reachedDic)

	if dicFileName:
		dic = load_dic(dicFileName)
		for key, value in dic.items():
			print("key = {}, value = {}".format(key, value))

		targetDf = map(targetDf, dic)
	
	targetDf.to_csv(targetFileName, index = False, float_format = "%.4f")

def create_reached(loadFileName, dicFileName, targetFileName):
	resultDf = load_from_csv(loadFileName)
	ids = set(resultDf["name"].astype("int64"))
	dic = load_dic(dicFileName)
	df = pd.DataFrame(columns = ("name", "id", "reached"))
	for id, name in dic.items():
		if id in ids:
			df.loc[len(df)] = (name, id, "")

	df = df.sort_values(by = "name")
	df.to_csv(targetFileName, index = False)

if __name__ == "__main__":
	make_mapped_file("results_unknown_2018.csv", "replaced_results_unknown_2018.csv", reachedFileName = "reached_2018_saved.csv")
	#create_reached("results_2018_saved.csv", "2018.csv", "reached_2018.csv")