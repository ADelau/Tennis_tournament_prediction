import pandas as pd
import os
from datetime import date, timedelta
import numpy as np

STAT_COL_NAMES = ["tourney_order",
				  "match_id",
				  "match_stats_url_suffix",
				  "match_time",
				  "match_duration",
				  "winner_aces",
				  "winner_double_faults",
				  "winner_first_serves_in",
				  "winner_first_serves_total",
				  "winner_first_serve_points_won",
				  "winner_first_serve_points_total",
				  "winner_second_serve_points_won",
				  "winner_second_serve_points_total",
				  "winner_break_points_saved",
				  "winner_break_points_serve_total",
				  "winner_service_points_won",
				  "winner_service_points_total",
				  "winner_first_serve_return_won",
				  "winner_first_serve_return_total",
				  "winner_second_serve_return_won",
				  "winner_second_serve_return_total",
				  "winner_break_points_converted",
				  "winner_break_points_return_total",
				  "winner_service_games_played",
				  "winner_return_games_played",
				  "winner_return_points_won",
				  "winner_return_points_total",
				  "winner_total_points_won",
				  "winner_total_points_total",
				  "loser_aces",
				  "loser_double_faults",
				  "loser_first_serves_in",
				  "loser_first_serves_total",
				  "loser_first_serve_points_won",
				  "loser_first_serve_points_total",
				  "loser_second_serve_points_won",
				  "loser_second_serve_points_total",
				  "loser_break_points_saved",
				  "loser_break_points_serve_total",
				  "loser_service_points_won",
				  "loser_service_points_total",
				  "loser_first_serve_return_won",
				  "loser_first_serve_return_total",
				  "loser_second_serve_return_won",
				  "loser_second_serve_return_total",
				  "loser_break_points_converted",
				  "loser_break_points_return_total",
				  "loser_service_games_played",
				  "loser_return_games_played",
				  "loser_return_points_won",
				  "loser_return_points_total",
				  "loser_total_points_won",
				  "loser_total_points_total"]

SCORE_COL_NAMES = ["tourney_year_id",
				   "tourney_order",
				   "tourney_slug",
				   "tourney_url_suffix",
				   "tourney_round_name",
				   "round_order",
				   "match_order",
				   "winner_name",
				   "winner_player_id",
				   "winner_slug",
				   "loser_name",
				   "loser_player_id",
				   "loser_slug",
				   "winner_seed",
				   "loser_seed",
				   "match_score_tiebreaks",
				   "winner_sets_won",
				   "loser_sets_won",
				   "winner_games_won",
				   "loser_games_won",
				   "winner_tiebreaks_won",
				   "loser_tiebreaks_won",
				   "match_id",
				   "match_stats_url_suffix"]

RANKING_COL_NAMES = ["week_year",
					 "week_month",
					 "week_day",
					 "rank_number",
					 "points",
					 "player_name",
					 "player_age",
					 "move",
					 "direction"]

PLAYER_COL_NAMES = ["player_id",
					"player_slug",
					"first_name",
					"last_name",
					"player_url",
					"flag_code",
					"residence",
					"birthplace",
					"birthdate",
					"birth_year",
					"birth_month",
					"birth_day",
					"turned_pro",
					"weight_lbs",
					"weight_kg",
					"height_ft",
					"height_inches",
					"height_cm",
					"handedness",
					"backhand"]

TOURNAMENT_COL_NAMES = ["tourney_year",
						"tourney_order",
						"tourney_name",
						"tourney_id",
						"tourney_slug",
						"tourney_location",
						"tourney_dates",
						"tourney_month",
						"tourney_day",
						"tourney_singles_draw",
						"tourney_doubles_draw",
						"tourney_conditions",
						"tourney_surface",
						"tourney_fin_commit",
						"tourney_url_suffix",
						"singles_winner_name",
						"singles_winner_url",
						"singles_winner_player_slug",
						"singles_winner_player_id",
						"doubles_winner_1_name",
						"doubles_winner_1_url",
						"doubles_winner_1_player_slug",
						"doubles_winner_1_player_id",
						"doubles_winner_2_name",
						"doubles_winner_2_url",
						"doubles_winner_2_player_slug",
						"doubles_winner_2_player_id",
						"tourney_year_id"]

COL_RENAME = {"tourney_surface" : "surface",
			  "tourney_dates": "date",
			  "winner_name": "winner_name",
			  "handedness": "winner_hand",
			  "height_cm": "winner_ht",
			  "age": "winner_age",
			  "rank_number": "winner_rank",
			  "points": "winner_rank_points",
			  "loser_name": "loser_name",
			  "handedness_loser": "loser_hand",
			  "height_cm_loser": "loser_ht",
			  "age_loser": "loser_age",
			  "rank_number_loser": "loser_rank",
			  "points_loser": "loser_rank_points",
		      "winner_aces": "w_ace",
			  "winner_double_faults": "w_df",
			  "winner_first_serves_in": "w_1stIn",
			  "winner_first_serve_points_won": "w_1stWon",
			  "winner_second_serve_points_won": "w_2ndWon",
			  "winner_break_points_saved": "w_bpSaved",
			  "winner_break_points_serve_total": "w_bpFaced",
			  "winner_service_games_played": "w_SvGms",
			  "winner_service_points_total": "w_svpt",
			  "loser_aces": "l_ace",
			  "loser_double_faults": "l_df",
			  "loser_first_serves_in": "l_1stIn",
			  "loser_first_serve_points_won": "l_1stWon",
			  "loser_second_serve_points_won": "l_2ndWon",
			  "loser_break_points_saved": "l_bpSaved",
			  "loser_break_points_serve_total": "l_bpFaced",
			  "loser_service_games_played": "l_SvGms",
			  "loser_service_points_total": "l_svpt"
			  }

FINAL_COLS = ["surface",
			  "level",
			  "date",
			  "winner_id",
			  "winner_name",
			  "winner_hand",
			  "winner_ht",
			  "winner_age",
			  "winner_rank",
			  "winner_rank_points",
			  "loser_id",
			  "loser_name",
			  "loser_hand",
			  "loser_ht",
			  "loser_age",
			  "loser_rank",
			  "loser_rank_points",
			  "score",
			  "best_of",
			  "w_ace",
			  "w_df",
			  "w_svpt",
			  "w_1stIn",
			  "w_1stWon",
			  "w_2ndWon",
			  "w_SvGms",
			  "w_bpSaved",
			  "w_bpFaced",
			  "l_ace",
			  "l_df",
			  "l_svpt",
			  "l_1stIn",
			  "l_1stWon",
			  "l_2ndWon",
			  "l_SvGms",
			  "l_bpSaved",
			  "l_bpFaced",
			  "W1",
			  "L1",
			  "W2",
			  "L2",
			  "W3",
			  "L3",
			  "W4",
			  "L4",
			  "W5",
			  "L5",
			  "Wsets",
			  "Lsets",
			  "Comment"]

def compare_names(name1, name2):
	"""
	Compare two player names in a robust way.

	Parameters
    ----------
	name1: The first name to compare
	name2: The second name to compare
    
    Returns
    -------
    A boolean indicating whether those names are equal
	"""

	names1 = name1.split()
	names2 = name2.split()

	if names1[0][0].lower() != names2[0][0].lower():
		return False

	lastNames1 = names1[1:]
	lastNames2 = names2[1:]

	lastNames1 = [name.split("-") for name in lastNames1]
	lastNames1 = [name for sublist in lastNames1 for name in sublist]
	lastNames2 = [name.split("-") for name in lastNames2]
	lastNames2 = [name for sublist in lastNames2 for name in sublist]

	for name1 in lastNames1:
		for name2 in lastNames2:
			if name1.lower() == name2.lower():
				return True

	return False

def replace_name(name, nameIDList):
	"""
	Replace the given name by the correspond name and id in the database

	Parameters
    ----------
	name: The name to replace
	nameIDList: A list of tuples containing the name and the ID the each player 
				in the database
    
    Returns
    -------
    The tuple in nameIDList matching the name, ("", 0) if no such tuple found
	"""

	for targetName in nameIDList:
		if compare_names(name, targetName[0]):
			return targetName

	print("{} missing".format(name))
	return "", 0

def replace_names_dataset(dataset, targetDataset):
	"""
	Replace name and id column in the dataset to match those in targetDataset

	Parameters
    ----------
	dataset: The dataset in which to replace the names and id
	targetDataset: The dataset containing the target names and ids.
    
    Returns
    -------
    The transformed dataset
	"""

	nameIDSet = set()
	for i, row in targetDataset.iterrows():
		nameIDSet.add((row["winner_name"], int(row["winner_id"])))
		nameIDSet.add((row["loser_name"], int(row["loser_id"])))

	nameIDList = list(nameIDSet)

	for i, row in dataset.iterrows():
		winnerNameID = replace_name(row["winner_name"], nameIDList)
		loserNameID = replace_name(row["loser_name"], nameIDList)

		dataset.loc[i, "winner_name"] = winnerNameID[0]
		dataset.loc[i, "winner_id"] = winnerNameID[1]
		dataset.loc[i, "loser_name"] = loserNameID[0]
		dataset.loc[i, "loser_id"] = loserNameID[1]

	dataset = dataset[dataset["winner_name"] != ""]
	dataset = dataset[dataset["loser_name"] != ""]
	dataset = dataset[dataset["winner_id"] != 0]
	dataset = dataset[dataset["loser_id"] != 0]

	return dataset

def clean_string(str):
	"""
	Clean a string

	Parameters
    ----------
	str: The string to clean
    
    Returns
    -------
    The cleaned string
	"""

	str = str.replace("b'", "")
	str = str.replace("'", "")
	str = str.rstrip()
	str = str.lstrip()

	return str

def get_year_from_date(date):
	"""
	Get the year from a date string

	Parameters
    ----------
	date: The date string
    
    Returns
    -------
    The year present in the date string
	"""

	dates = date.split(".")
	return int(dates[0])

def create_ranking_dates(tournament, rankings):
	"""
	Add features in tournament dataset containing the more recent previous date
	for which we have ranking data

	Parameters
    ----------
	tournament: A dataFrame containing the tournaments
	rankings: A dataFrame containing the ranking data
    
    Returns
    -------
    The update tournament dataFrame
	"""

	dates = set()

	valueErrors = 0
	for i, row in rankings.iterrows():
		try:
			dates.add(date(int(row["week_year"]), int(row["week_month"]), 
						   int(row["week_day"])))
		except ValueError:
			valueErrors += 1

	print("{} invalid rows loading rankings".format(valueErrors))

	dates = list(dates)

	tournament["week_year"] = pd.Series(dtype = np.int64)
	tournament["week_month"] = pd.Series(dtype = np.int64)
	tournament["week_day"] = pd.Series(dtype = np.int64)

	for i, row in tournament.iterrows():
		maxDate = None
		rowDate = date(int(row["tourney_year"]), int(row["tourney_month"]), 
					   int(row["tourney_day"])) + timedelta(days = 1)

		for tmpDate in dates:
			if (maxDate is None or tmpDate > maxDate) and tmpDate < rowDate:
				maxDate = tmpDate
		
		tournament.loc[i, "week_year"] = int(maxDate.year)
		tournament.loc[i, "week_month"] = int(maxDate.month)
		tournament.loc[i, "week_day"] = int(maxDate.day)

	return tournament

def create_age(dataset):
	"""
	Create an age field in the dataset based on the torunament date and on
	the player birthDate.

	Parameters
    ----------
	dataset: The dataset to transform, must contain the tournement date and the
	player birth date.
    
    Returns
    -------
    The transformed dataset with the age of the player added.
	"""

	dataset["age"] = pd.Series(dtype = np.float64)
	dataset["age_loser"] = pd.Series(dtype = np.float64)

	for i, row in dataset.iterrows():
		dates = row["tourney_dates"].split(".")
		tourneyDate = date(int(dates[0]), int(dates[1]), int(dates[2]))

		dates = row["birthdate"].split(".")
		winnerBirthDate = date(int(dates[0]), int(dates[1]), int(dates[2]))

		dates = row["birthdate_loser"].split(".")
		loserBirthDate = date(int(dates[0]), int(dates[1]), int(dates[2]))

		winnerAge = (tourneyDate - winnerBirthDate).days/365.25
		loserAge = (tourneyDate - loserBirthDate).days/365.25

		dataset.loc[i, "age"] = winnerAge
		dataset.loc[i, "age_loser"] = loserAge

	return dataset

def modify_date_format(originalDate):
	dates = originalDate.split(".")
	year = int(dates[0])
	month = int(dates[1])
	day = int(dates[2])
	tmpDate = date(year, month, day)
	return tmpDate.strftime("%d/%m/%Y")

def fuse_files(tournamentFile, scoreFile, statsFile, playerFile, rankingFiles, baseFile, fusedFile):
	"""
	Fuse the files scrapped on the atp into a single file matching the database
	format

	Parameters
    ----------
	tournamentFile: The csv file containing the tournaments
	scoreFile: The csv file containing the scores
	statsFile: The csv file containing the statistics
	rankingFiles: A list of csv files containing the rankings
	baseFile: A csv file containing the elements already present in the dataset
	fuseFile: The name in which to save the fused dataset
	"""

	stats = pd.read_csv(statsFile, header = None, names = STAT_COL_NAMES)
	statsSize = stats.shape[0]
	
	scores = pd.read_csv(scoreFile, header = None, names = SCORE_COL_NAMES)
	scores["winner_name"] = scores["winner_name"].apply(clean_string)
	scores["loser_name"] = scores["loser_name"].apply(clean_string)

	matches = stats.merge(scores, on = "match_stats_url_suffix")
	scoreSize = matches.shape[0]
	print("{} elements lost out of {} while merging scores".format(statsSize - scoreSize, statsSize))

	players = pd.read_csv(playerFile, header = None, names = PLAYER_COL_NAMES)

	matches = matches.merge(players, left_on = "winner_player_id", right_on = "player_id", suffixes = ("", "_winner"))
	matches = matches.merge(players, left_on = "loser_player_id", right_on = "player_id", suffixes = ("", "_loser"))
	playerSize = matches.shape[0]
	print("{} elements lost out of {} while merging players".format(scoreSize - playerSize, scoreSize))

	tournament = pd.read_csv(tournamentFile, header = None, names = TOURNAMENT_COL_NAMES)
	tournament["tourney_name"] = tournament["tourney_name"].apply(clean_string)
	tournament["tourney_year"] = tournament["tourney_dates"].apply(get_year_from_date)

	rankings = pd.DataFrame(columns = RANKING_COL_NAMES)
	for rankingFile in rankingFiles:
		ranking = pd.read_csv(rankingFile, header = None, names = RANKING_COL_NAMES)
		ranking.dropna(subset = ["player_name", "week_year", "week_month", "week_day"], inplace = True)
		rankings = pd.concat((rankings, ranking), axis = 0)

	rankings.drop_duplicates(inplace = True)
	rankings["player_name"] = rankings["player_name"].apply(clean_string)
	rankings.to_csv("rankings.csv", index = False)

	tournament = create_ranking_dates(tournament, rankings)

	matches = matches.merge(tournament, on = "tourney_url_suffix")
	tournamentSize = matches.shape[0]
	print("{} elements lost out of {} while merging tournament".format(playerSize - tournamentSize, playerSize))

	matches = create_age(matches)

	matches.to_csv("matches.csv", index = False)

	matches = matches.merge(rankings, left_on = ["winner_name", "week_year", "week_month", "week_day"], 
							right_on = ["player_name", "week_year", "week_month", "week_day"],
							suffixes = ("", "_winner"))

	matches = matches.merge(rankings, left_on = ["loser_name", "tourney_year", "tourney_month", "tourney_day"], 
							right_on = ["player_name", "week_year", "week_month", "week_day"],
							suffixes = ("", "_loser"))

	rankingSize = matches.shape[0]
	print("{} elements lost out of {} while merging rankings".format(tournamentSize - rankingSize, tournamentSize))

	columnNames = matches.columns
	for name in columnNames:
		if name not in COL_RENAME.keys():
			matches.drop(name, axis = 1, inplace = True)

	matches = matches.rename(index = str, columns = COL_RENAME)
	matches["date"] = matches["date"].apply(modify_date_format)

	def fill_comment(value):
		return "Completed"

	matches["Comment"] = pd.Series(dtype = str)
	matches["Comment"] = matches["Comment"].apply(fill_comment)

	matches["level"] = pd.Series(dtype = str)
	matches["winner_id"] = pd.Series(dtype = np.int64)
	matches["loser_id"] = pd.Series(dtype = np.int64)
	matches["score"] = pd.Series(dtype = str)
	matches["best_of"] = pd.Series(dtype = np.int64)
	matches["W1"] = pd.Series(dtype = np.int64)
	matches["L1"] = pd.Series(dtype = np.int64)
	matches["W2"] = pd.Series(dtype = np.int64)
	matches["L2"] = pd.Series(dtype = np.int64)
	matches["W3"] = pd.Series(dtype = np.int64)
	matches["L3"] = pd.Series(dtype = np.int64)
	matches["W4"] = pd.Series(dtype = np.int64)
	matches["L4"] = pd.Series(dtype = np.int64)
	matches["W5"] = pd.Series(dtype = np.int64)
	matches["L5"] = pd.Series(dtype = np.int64)
	matches["Wsets"] = pd.Series(dtype = np.int64)
	matches["Lsets"] = pd.Series(dtype = np.int64)

	matches = matches[FINAL_COLS]

	targetDataset = pd.read_csv(BASE_FILE)
	matches = replace_names_dataset(matches, targetDataset)

	renameSize = matches.shape[0]
	print("{} elements lost out of {} while renaming".format(rankingSize - renameSize, rankingSize))
	print("{} elements lost out of {} in total".format(statsSize - renameSize, statsSize))

	targetDataset = pd.concat((targetDataset, matches), axis = 0)
	
	targetDataset.to_csv(fusedFile, index = False)
	targetDataset.drop_duplicates(inplace = True)

if __name__ == "__main__":
	SCORE_FILE = "csv/match_scores_2019-2019.csv"
	STATS_FILE = "csv/match_stats_2019_0.csv"
	PLAYER_FILE = "csv/player_overviews_UNINDEXED.csv"
	TOURNAMENT_FILE = "csv/tournaments_2019-2019.csv"
	BASE_FILE = "csv/raw_2001_2018.csv"
	FUSED_FILE = "csv/match_2019.csv"

	path = "csv/"
	files = os.listdir(path)
	rankingFiles = [path + file for file in files if "rankings" in file]

	fuse_files(TOURNAMENT_FILE, SCORE_FILE, STATS_FILE, PLAYER_FILE, rankingFiles, BASE_FILE, FUSED_FILE)

