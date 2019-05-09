import pandas as pd
from data import load_from_csv

def convert_percentage(proba):
	return str(round(proba * 100), 3) + "%"

FILES_TO_TRANSFORM = ["Results/result_RG2017_probas.csv",
					  "Results/result_RG2018_probas.csv",
					  "Results/result_RG2019.csv",
					  "replaced_results_2018_saved"
					  ]

for file in FILES_TO_TRANSFORM:
	dataset = load_from_csv(file + ".csv")
	dataset["1"] = dataset["1"].apply(convert_percentage)
	dataset["2"] = dataset["2"].apply(convert_percentage)
	dataset["4"] = dataset["4"].apply(convert_percentage)
	dataset["8"] = dataset["8"].apply(convert_percentage)
	dataset["16"] = dataset["16"].apply(convert_percentage)
	dataset["32"] = dataset["32"].apply(convert_percentage)
	dataset["64"] = dataset["64"].apply(convert_percentage)
	dataset["128"] = dataset["128"].apply(convert_percentage)

	dataset.to_csv(file + "_percentage.csv", index = False)