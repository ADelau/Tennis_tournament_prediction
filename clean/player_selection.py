from tournament import *
import copy
import random
import numpy as np
import os

PROBA_INJURY = 0.1
random.seed(None)

class UnknownTournament():
	"""
	Class representing a tournament with unknown participating players
	"""

	def __init__(self, playerMatrixFile, nbSimulationTournament):
		"""
		Constructor

		Parameters
	    ----------
		playerMatrixFile: The name of the file containing the player matrix
		nbSimulationTournament: The number of simulations to perform for a
								given set of player.

	    Returns
	    -------
	    An array of round reached, the index is the internal id of the player.
		"""

		self.playerMatrix = load_player_matrix(playerMatrixFile)
		self.players = load_players(self.playerMatrix)
		self.players.sort()
		self.nbSimulationTournament = nbSimulationTournament

	def _select_players(self):
		"""
		Select randomly according to rules players participating to a 
		tournament. It sets the qualifiedPlayers field with the players 
		participating.
		"""

		self.qualifiedPlayers = []

		self._select_injured()
		self._select_rank_qualified()
		self._select_qualification_qualified()
		self._select_wildcard_qualified()

	def _select_injured(self):
		"""
		Randomly make players injured. Put the others in the eligiblePlayers 
		list field.
		"""

		self.eligiblePlayers = []

		nbEligible = 0

		for player in self.players:
			if random.uniform(0, 1) >= PROBA_INJURY:
				self.eligiblePlayers.append(player)
				nbEligible += 1

		if nbEligible < 128:
			unqualified = [player for player in self.players if player not in 
						   set(self.eligiblePlayers)]
			print(unqualified)
			qualified = random.choices(unqualified, k = 128 - nbEligible)

			self.eligiblePlayers += qualified

	def _select_rank_qualified(self):
		"""
		make qualifications based on the atp rank. The qualified players are
		put in the qualified players list field.
		"""

		self.qualifiedPlayers += self.eligiblePlayers[:104]
		self.eligiblePlayers = self.eligiblePlayers[104:]

	def _select_qualification_qualified(self):
		"""
		make qualifications from qualification tournaments. The qualified 
		players are put in the qualified players list field.
		"""

		points = [player.atpPoints for player in self.eligiblePlayers]
		sumPoints = sum(points)
		probas = [point / sumPoints for point in points]

		toQualify = list(np.random.choice(np.array(self.eligiblePlayers), p = probas, 
								   	  	   size = 16, replace = False))
		
		for player in toQualify:
			player.qualified = True

		self.qualifiedPlayers += toQualify

		self.eligiblePlayers = [player for player in self.eligiblePlayers if not 
								player.qualified]

		for player in toQualify:
			player.qualified = False

	def _select_wildcard_qualified(self):
		"""
		make wildcard qualifications. The qualified players are put in the 
		qualified players list field.
		"""

		self.qualifiedPlayers += list(np.random.choice(np.array(self.eligiblePlayers), size = 8, replace = False))

	def _rollout(self, debug = False):
		"""
		Perform a set of tournament simulations for a set of players in entry.

		Parameters
	    ----------
		debug: Indicate if the simulations must be done in debug mode.
		"""

		self._select_players()

		tournament = Tournament("", playerMatrix = self.playerMatrix, 
								players = self.qualifiedPlayers,
								totPlayers = len(self.players))

		tournament.monte_carlo(self.nbSimulationTournament, debug)

	def monte_carlo(self, nbSimulations, debug = False):
		"""
		Perform simulation for several set of players.

		Parameters
	    ----------
		nbSimulations: The number of set of players that should be simulated
		debug: Indicate if the simulations must be done in debug mode.
		"""

		for simulation in range(nbSimulations):
			print("Tournament {}".format(simulation))
			self._rollout(debug)

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
	"""
	tournament = UnknownTournament("Probas/RG2019_probas.csv", 100)
	
	tournament.monte_carlo(100)
	tournament.results_to_csv("Results/result_RG2019.csv")
	"""
	
	PROBA_PATH = "Probas/"
	probaFiles = os.listdir(PROBA_PATH)

	for proba in probaFiles:
		print(proba)
		tournament = UnknownTournament(PROBA_PATH + proba, 100)
		tournament.monte_carlo(100)
		tournament.results_to_csv("Results/result_" + proba)