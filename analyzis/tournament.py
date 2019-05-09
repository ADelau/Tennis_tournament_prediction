from data import load_from_csv, load_dataset, load_train_set, encode
import pandas as pd 
import math
import random

NB_PLAYERS = 128
TOT_NB_ROUNDS = int(math.log(NB_PLAYERS, 2)) + 1
random.seed(None)

class Player():
	def __init__(self, id, name, atpRank):
		self.id = id
		self.name = name
		self.atpRank = atpRank
		self.counterRoundReached = [0]*TOT_NB_ROUNDS # index  nbRound
		self.counterPlayed = 0
		self.lastEncounteredRank = None

	def update(self, roundReached):
		"""
		update all round < roundReached as reached and the others as not reached.
		"""

		self.counterPlayed += 1
		for i in range(roundReached + 1):
			self.counterRoundReached[i] += 1

	def proba_round_reached(self):
		"""
		return list containing proba of reaching the round of the index
		"""

		return [x/self.counterPlayed for x in self.counterRoundReached]

	def __lt__(self, other):
		return self.atpRank < other.atpRank

	def __repr__(self):
		return str(self.name)

def encountered(playerA, playerB):
	playerA.lastEncounteredRank = playerB.atpRank
	playerB.lastEncounteredRank = playerA.atpRank

class Match():
	def __init__(self, playerMatrix, playerA = None, playerB = None):
		self.playerMatrix = playerMatrix
		self.playerA = playerA
		self.playerB = playerB

	def update_players(self, playerA, playerB):
		self.playerA = playerA
		self.playerB = playerB

	def winner(self):
		probaA = self.self.playerMatrix[(self.playerA.name, self.playerB.name)]

		if random.uniform(0, 1) <= probaA:
			return self.playerA
		return self.playerB

	def _contruct_input_vector(self):
		"""
		generate the vector of features used to predict the outcome
		"""

		return self.playerMatrix[(self.playerA.name, self.playerB.name)]

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
		return a round resulting from the current one, matches are the matches of the next round 
		can only be None if last round
		"""

		nextPlayers = self.simulate(debug)
		nextRound = Round(self.roundNb + 1, nextPlayers, matches, self.prevRounds + [self])

		return nextRound

	def simulate(self, debug = False):
		"""
		construct a tree from players, simulated matches and returns a list of players that passed
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
		Updates matches accord to the rules of the first round. 32 best one player out of 4.
		"""
		# Order players, better ranked first
		orderedPlayers = sorted(self.players)
		
		firstPlayers = random.sample(orderedPlayers[:32], len(orderedPlayers[:32]))
		otherPlayers = random.sample(orderedPlayers[32:], len(orderedPlayers[32:]))

		for evenMatch, A, B in zip(self.matches[0::2], firstPlayers, otherPlayers[:32]):
			evenMatch.playerA = A
			evenMatch.playerB = B

			encountered(A, B)

		for oddMatch, A, B in zip(self.matches[1::2], otherPlayers[32:64], otherPlayers[64:]):
			oddMatch.playerA = A
			oddMatch.playerB = B

			encountered(A, B)

	def _early_matching(self):
		"""
		Update matches according to rules of the second round, simply propagate the tree
		"""
		
		# Player are ordered as previous matches
		for match, playerA, playerB in zip(self.matches, self.players[0::2], self.players[1::2]):

			match.playerA = playerA
			match.playerB = playerB

			encountered(playerA, playerB)

	def _end_matching(self):
		"""
		Updates matches according to rules of the last rounds
		"""
		
		# Sort according to atp rank
		orderedPlayers = sorted(self.players)

		# Take half better ranked
		firstPlayers = orderedPlayers[:int(len(orderedPlayers)/2)]

		# First part = the one who encountered the strongest in the half best
		# Second part = the one who encountered the weakest in the half best
		orderedFirstPlayers = sorted(firstPlayers, key = lambda x: x.lastEncounteredRank)
		
		firstPart = orderedFirstPlayers[:int(len(orderedFirstPlayers)/2)]
		secondPart = orderedFirstPlayers[int(len(orderedFirstPlayers)/2):]

		# Third part = the half best in the half worst		
		thirdPart = orderedPlayers[int(len(orderedPlayers)/2): int(3*len(orderedPlayers)/4)]

		# Fourth part is the half worst in the half worst
		fourthPart = orderedPlayers[int(3*len(orderedPlayers)/4):]

		# Match the one in first half that encountered strongest against weakest in second half
		sampleFirstPart = random.sample(firstPart, len(firstPart))
		samplesecondPart = random.sample(secondPart, len(secondPart))
		samplethirdPart = random.sample(thirdPart, len(thirdPart))
		samplefourthPart = random.sample(fourthPart, len(fourthPart))

		for match, playerA, playerB in zip(self.matches[:int(len(self.matches)/2)], sampleFirstPart, samplefourthPart):
			match.playerA = playerA
			match.playerB = playerB

			encountered(playerA, playerB)

		# Match the one in first half that encountered weakest against strongest in second half
		for match, playerA, playerB in zip(self.matches[int(len(self.matches)/2):], samplesecondPart, samplethirdPart):
			match.playerA = playerA
			match.playerB = playerB

			encountered(playerA, playerB)

	def _last_round_matching(self):
		"""
		Perform matching for the last round
		"""

		self.matches[0].playerA = self.players[0]
		self.matches[0].playerB = self.players[1]

class Tournament():
	def __init__(self, playerMatrixFile):
		"""
		model is the model used to predict
		"""

		self.playerMatrix = self._load_player_matrix(playerMatrixFile)
		self.players = self._load_players(self.playerMatrix)
		self.matches = self._load_matches()

	def _load_player_matrix(self, fileName):
		playerMatrixDF = load_from_csv(fileName)
		playerMatrix = {}

		for index, row in playerMatrixDF.iterrows():
			playerMatrix[(row["name_1"], row["name_2"])] = (row["rank_1"], row["rank_2"])

		return playerMatrix

	def _load_players(self, playerMatrix):
		"""
		Return a list of Players
		"""
		
		playersSet = set()

		for key in playerMatrix.keys():
			for playerName in key:
				if playerName not in playersSet:
					playersSet.add(playerName)

		players = []
		id = 0
		for playerName in playersSet:
			atpRank = playerMatrix[(playerName, playerName)][0]
			players.append(Player(id, playerName, atpRank))
			id += 1

		return players

	def _load_matches(self):
		"""
		Return a list of list of Matches, first index = round number
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
		results = self._simulate_tournament(debug)
		for player in self.players:
			player.update(results[player.id])

	def _simulate_tournament(self, debug = False):
		"""
		return a list with round reached, index = player ID
		"""
		
		rounds = [Round(0, self.players, self.matches[0])]

		for roundNb in range(1, TOT_NB_ROUNDS - 1):
			rounds.append(rounds[-1].next_round(self.matches[roundNb], debug))

		rounds.append(rounds[-1].next_round(None, debug))

		results = [0] * len(self.players)

		for round in rounds:
			for player in round.players:
				results[player.id] = round.roundNb

		return results

	def monte_carlo(self, nbSimulations, debug = False):
		for _ in range(nbSimulations):
			self.rollout(debug)

		for player in self.players:
			print("name = {}, result = {}".format(player.name, player.proba_round_reached()))

class TestEstimator():
	def __init__(self):
		pass

	def predict(self, ranks):
		rank1 = ranks[0]
		rank2 = ranks[1]
		# Proba que 1 gagne = rank2 / (rank1 + rank2) donc si rank 2 bas proba que 1 gagne faible.
		return rank2 / (rank1 + rank2)

if __name__ == "__main__":
	model = TestEstimator()
	tournament = Tournament("testMatrix.csv", None, model)
	for player in tournament.players:
		print("id = {}, name = {}, atpRank = {}".format(player.id, player.name, player.atpRank))

	print("matches:")
	for round in tournament.matches:
		for match in round:
			print("features = {}".format(match.features))

	tournament.monte_carlo(1, True)

	tournament.monte_carlo(500)