import unittest
from fuse_match_score_stats import *

class UnitTestChannelCoding(unittest.TestCase):
	SCORES_FILE = "testFiles/scores.csv"
	STATS_FILE = "testFiles/stats.csv"
	PLAYERS_FILE = "testFiles/players.csv"
	TOURNAMENTS_FILE = "testFiles/tournaments.csv"
	RANKING_FILES = ["testFiles/rankings_1.csv",
					 "testFiles/rankings_2.csv"]

	TEST_FILE = "testFiles/target_file.csv"

	fuse_files(TOURNAMENTS_FILE, SCORES_FILE, STATS_FILE, PLAYERS_FILE, RANKING_FILES, TEST_FILE)

if __name__ == "__main__":
	unittest.main(verbosity = 2)