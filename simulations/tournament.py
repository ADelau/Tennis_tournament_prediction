from data import load_from_csv, load_dataset, load_train_set, encode
import pandas as pd 
import math
import random
import math
import numpy as np

PLAYER_NOT_PARTICIPATING = -1

NB_PLAYERS = 128
TOT_NB_ROUNDS = int(math.log(NB_PLAYERS, 2)) + 1
random.seed(None)

def load_player_matrix(fileName):
	"""
	Load the player matrix

	Parameters
    ----------
    fileName: The name of the file containing the player matrix
    
    Returns
    -------
    The player matrix, a dictionary of dictionary of player names containg the
    probability that the first index player wins, the rank of the first index
    player, the rank of the second index player, the atp points of the first
    index player and the atp points of the second index player.
	"""

	playerMatrixDF = load_from_csv(fileName)
	playerMatrix = {}

	for index, row in playerMatrixDF.iterrows():
		playerMatrix[(row["name_1"], row["name_2"])] = (row["proba_1"], 
														  row["rank_1"], 
														  row["rank_2"],
														  row["rank_points_1"],
														  row["rank_points_2"])

	return playerMatrix

def load_players(playerMatrix):
	"""
	Load the player list from a player matrix

	Parameters
    ----------
    playerMatrix: The player matrix
    
    Returns
    -------
    A list of players Objects.
	"""

	playersSet = set()

	for key in playerMatrix.keys():
		for playerName in key:
			if playerName not in playersSet:
				playersSet.add(playerName)

	players = []
	id = 0

	for playerName in playersSet:
		for playerName2 in playersSet:

			try:
				atpRank = playerMatrix[(playerName, playerName2)][1]
				atpPoints = playerMatrix[(playerName, playerName2)][3]
				players.append(Player(id, playerName, atpRank, atpPoints))
				id += 1
				break
			
			except Exception as e:
				print(e)

	return players

class Player():
	def __init__(self, id, name, atpRank, atpPoints):
		self.id = id
		self.name = name
		self.atpRank = atpRank
		self.atpPoints = atpPoints
		self.counterRoundReached = [0]*TOT_NB_ROUNDS # index  nbRound
		self.counterPlayed = 0
		self.lastEncounteredRank = None
		self.qualified = False

	def update(self, roundReached):
		"""
		Update all round < roundReached as reached and the others as not 
		reached.

		Parameters
	    ----------
	    roundReached: The round reached at a simulation
		"""

		if roundReached == PLAYER_NOT_PARTICIPATING:
			return

		self.counterPlayed += 1
		for i in range(roundReached + 1):
			self.counterRoundReached[i] += 1

	def proba_round_reached(self):
		"""
		Get the probabilities of the player to reach each round.

	    Returns
	    -------
	    A list of probabilities, the first index correspond to round 128, second
	    to round 64, ...
		"""

		return [x/self.counterPlayed for x in self.counterRoundReached]

	def __lt__(self, other):
		return self.atpRank < other.atpRank

	def __repr__(self):
		return str(self.name)

def encountered(playerA, playerB):
	"""
	Update the players to indicate their last encoutered each other.

	Parameters
    ----------
    playerA: The first player
    playerB: The second player
	"""

	playerA.lastEncounteredRank = playerB.atpRank
	playerB.lastEncounteredRank = playerA.atpRank

class Match():
	def __init__(self, playerMatrix, playerA = None, playerB = None):
		self.playerMatrix = playerMatrix
		self.playerA = playerA
		self.playerB = playerB

	def update_players(self, playerA, playerB):
		"""
		Set the player participating to the match

		Parameters
	    ----------
	    playerA: The first player
	    playerB: The second player
		"""

		self.playerA = playerA
		self.playerB = playerB

	def winner(self):
		"""
		Draw a winner of the match according to winning probability

		Returns
	    -------
	    The player that won the match
		"""

		probaA = self._get_proba()

		if random.uniform(0, 1) <= probaA:
			return self.playerA
		return self.playerB

	def _get_proba(self):
		"""
		Compute the probability of playerA winning against playerB

		Returns
	    -------
	    The probability fo playerA winning against playerB
		"""

		return self.playerMatrix[(self.playerA.name, self.playerB.name)][0]

	def __repr__(self):
		return("({}, {})".format(self.playerA.name, self.playerB.name))

class Round():
	def __init__(self, roundNb, players, matches = None, prevRounds = []):
		self.roundNb = roundNb
		self.players = players
		self.matches = matches
		self.prevRounds = prevRounds

	def next_round(self, matches = None, debug = False):
		"""
		Return a round resulting from a simulation of the current one.

		Parameters
	    ----------
	    matches: The matches of the next round, can only be None if it is the 
	    		 last round
	    
	    Returns
	    -------
	    A Round object corresponding to the resulting next round
		"""

		nextPlayers = self.simulate(debug)
		nextRound = Round(self.roundNb + 1, nextPlayers, matches, 
						  self.prevRounds + [self])

		return nextRound

	def simulate(self, debug = False):
		"""
		construct a matching from current players, simulate matches and returns 
		a list of players that passed the round.

		Parameters
	    ----------
		debug: Boolean indicating of it must be executed in debug mode.
	    
	    Returns
	    -------
	    A list of players that passed the round simulation.
		"""
		
		# roundNb starts at 0, this is for the 2 first rounds
		if self.roundNb == 0:
			self._initialisation_matching()
		
		elif self.roundNb == 1:
			self._early_matching()

		elif self.roundNb == TOT_NB_ROUNDS - 2:
			self._last_round_matching()

		else:
			self._end_matching()

		if debug:
			print("players:")
			for player in self.players:
				print(player.name)

			print("matches:")
			for match in self.matches:
				print(match)

		nextPlayers = [match.winner() for match in self.matches]

		return nextPlayers

	def _initialisation_matching(self):
		"""
		Assign the players to the matches according to the rules of the first
		round.
		"""

		# Order players, better ranked first
		orderedPlayers = sorted(self.players)
		
		firstPlayers = random.sample(orderedPlayers[:32], 
									 len(orderedPlayers[:32]))
		otherPlayers = random.sample(orderedPlayers[32:], 
									 len(orderedPlayers[32:]))

		for evenMatch, A, B in zip(self.matches[0::2], firstPlayers, 
								   otherPlayers[:32]):
			evenMatch.playerA = A
			evenMatch.playerB = B

			encountered(A, B)

		for oddMatch, A, B in zip(self.matches[1::2], otherPlayers[32:64], 
								  otherPlayers[64:]):
			oddMatch.playerA = A
			oddMatch.playerB = B

			encountered(A, B)

	def _early_matching(self):
		"""
		Assign the players to the matches according to the rules of the second
		round, simply propagate the tree.
		"""
		
		# Player are ordered as previous matches
		for match, playerA, playerB in zip(self.matches, self.players[0::2], 
										   self.players[1::2]):

			match.playerA = playerA
			match.playerB = playerB

			encountered(playerA, playerB)

	def _end_matching(self):
		"""
		Assign the players to the matches according to the rules of the last
		rounds.
		"""
		
		# Sort according to atp rank
		orderedPlayers = sorted(self.players)

		# Take half better ranked
		firstPlayers = orderedPlayers[:int(len(orderedPlayers)/2)]

		# First part = the one who encountered the strongest in the half best
		# Second part = the one who encountered the weakest in the half best
		orderedFirstPlayers = sorted(firstPlayers, 
									 key = lambda x: x.lastEncounteredRank)
		
		firstPart = orderedFirstPlayers[:int(len(orderedFirstPlayers)/2)]
		secondPart = orderedFirstPlayers[int(len(orderedFirstPlayers)/2):]

		# Third part = the half best in the half worst		
		thirdPart = orderedPlayers[int(len(orderedPlayers)/2): 
								   int(3*len(orderedPlayers)/4)]

		# Fourth part is the half worst in the half worst
		fourthPart = orderedPlayers[int(3*len(orderedPlayers)/4):]

		# Match the one in first half that encountered strongest against 
		# weakest in second half
		sampleFirstPart = random.sample(firstPart, len(firstPart))
		samplesecondPart = random.sample(secondPart, len(secondPart))
		samplethirdPart = random.sample(thirdPart, len(thirdPart))
		samplefourthPart = random.sample(fourthPart, len(fourthPart))

		for match, playerA, playerB in zip(
				self.matches[:int(len(self.matches)/2)], sampleFirstPart, 
				samplefourthPart):

			match.playerA = playerA
			match.playerB = playerB

			encountered(playerA, playerB)

		# Match the one in first half that encountered weakest against strongest 
		# in second half
		for match, playerA, playerB in zip(
				self.matches[int(len(self.matches)/2):], samplesecondPart, 
				samplethirdPart):
		
			match.playerA = playerA
			match.playerB = playerB

			encountered(playerA, playerB)

	def _last_round_matching(self):
		"""
		Assign the players to the matches according to the rules of the last
		round.
		"""

		self.matches[0].playerA = self.players[0]
		self.matches[0].playerB = self.players[1]

class Tournament():
	def __init__(self, playerMatrixFile, playerMatrix = None, players = None, totPlayers = None):
		"""
		Constructor

		Parameters
	    ----------
		playerMatrixFile: The name of the file containing the playerMatrix
		playerMarix: The playerMatrix, if passed playerMaxtrixFile won't be used
		player: The players participating to the tournament, is not passed, the
				player matrix must be of size 128 and the players will be the 
				ones in the player matrix.
		"""
		
		if playerMatrix is None:
			self.playerMatrix = load_player_matrix(playerMatrixFile)
		else:
			self.playerMatrix = playerMatrix

		if players is None:
			self.players = load_players(self.playerMatrix)
		else:
			self.players = players

		if totPlayers is None:
			self.totPlayers = len(self.players)
		else:
			self.totPlayers = totPlayers

		self.matches = self._create_matches()

	def _create_matches(self):
		"""
		Create the matches of the tournament

	    Returns
	    -------
	    A list of lists of matches. The first index is the round number, the
	    second index is the match number inside the round.
		"""

		matches = []

		# Last round  contains no matches, just a single player
		for round in range(1, TOT_NB_ROUNDS):
			nbMatchs = int(NB_PLAYERS / (2 ** round))
			matches.append([])
			for _ in range(nbMatchs):
				matches[round - 1].append(Match(self.playerMatrix))

		return matches

	def rollout(self, debug = False):
		"""
		Perform a simulation of the tournament and update the results.

		Parameters
	    ----------
		debug: Boolean indicating of it must be executed in debug mode.
		"""

		results = self._simulate_tournament(debug)
		for player in self.players:
			player.update(results[player.id])

	def _simulate_tournament(self, debug = False):
		"""
		Perform a simulation of the tournament

		Parameters
	    ----------
		debug: Boolean indicating of it must be executed in debug mode.
	    
	    Returns
	    -------
	    An array of round reached, the index is the internal id of the player.
		"""
		
		rounds = [Round(0, self.players, self.matches[0])]

		for roundNb in range(1, TOT_NB_ROUNDS - 1):
			rounds.append(rounds[-1].next_round(self.matches[roundNb], debug))

		rounds.append(rounds[-1].next_round(None, debug))

		results = np.full((self.totPlayers,), PLAYER_NOT_PARTICIPATING, 
						  dtype = np.intc)

		for round in rounds:
			for player in round.players:
				results[player.id] = round.roundNb

		return results

	def monte_carlo(self, nbSimulations, debug = False):
		"""
		Perform several monte carlo simulations of the tournament.

		Parameters
	    ----------
	    nbSimulations: The number of simulations to perform.
		debug: Boolean indicating of it must be executed in debug mode.
		"""

		PRINT_EVERY = 1000
		printed = 0
		for simulation in range(nbSimulations):
			if simulation//PRINT_EVERY > printed:
				print("iteration {}".format(simulation))
				printed += 1

			self.rollout(debug)

	def print_results(self):
		"""
		Print the results of the simulations
		"""

		for player in self.players:
			print("name = {}, result = {}".format(player.name, 
												  player.proba_round_reached()))

	def results_to_csv(self, fileName):
		"""
		Write the results in a csv file.

		Parameters
	    ----------
		fileName: the name of the file without the extension.
		"""

		fileName = fileName + ".csv"
		df = pd.DataFrame(columns = ("name", "128", "64", "32", "16", "8", "4", 
									 "2", "1"))
		
		for player in self.players:
			print([round(item, 5) for item in player.proba_round_reached()])
			df.loc[len(df)] = [player.name] + [round(item, 4) for item in 
											   player.proba_round_reached()]

		df = df.sort_values(by = ["1", "2", "4", "8", "16", "32", "64", "128"], 
							ascending = False)

		df.to_csv(fileName, index = False, float_format = "%.4f")


if __name__ == "__main__":
	tournament = Tournament("probas.csv")
	for player in tournament.players:
		print("id = {}, name = {}, atpRank = {}".format(player.id, player.name, 
														player.atpRank))

	tournament.monte_carlo(1, True)
	tournament.print_results()
	
	tournament.monte_carlo(100000)
	tournament.results_to_csv("results_2018")	