#! /usr/bin/env python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
from ATPMatchScraper import LineType
from ATPMatchScraper import detect_type
import time

class FailedMatch(Exception):
	pass

class Statistic:

	def __init__(self, label, winningPlayer, loosingPlayer):
		self.label = label
		self.winningPlayer = winningPlayer
		self.loosingPlayer = loosingPlayer

	def write(self, file):
		file.write(self.label + ": " + self.winningPlayer + " " + self.loosingPlayer + "\n")


class Match:

	def __init__(self, winningPlayer, loosingPlayer, scoreInfo, durationInfo):
		self.winningPlayer = winningPlayer
		self.loosingPlayer = loosingPlayer
		self.scoreInfo = scoreInfo
		self.durationInfo = durationInfo
		self.stats = []

	def add_stat(self, stat):
		self.stats.append(stat)

	def write(self, file):
		newMatch = "New match:"
		for i in range(len(newMatch)):
			file.write("-")
		file.write("\n")
		file.write(newMatch + "\n")
		for i in range(len(newMatch)):
			file.write("-")
		file.write("\n\n")

		file.write("winning player: " + self.winningPlayer + "\n")
		file.write("loosing player: " + self.loosingPlayer + "\n")
		file.write("scores: " + self.scoreInfo + "\n")
		file.write("duration: " + self.durationInfo + "\n")
		file.write("\n")

		for stat in self.stats:
			stat.write(file)

		file.write("\n")


def clean_string(string):
	string = string.replace(".", " ")
	string = string.replace("\n", " ")
	string = string.replace("\r", " ")
	string = string.rstrip(" ")
	string = string.lstrip(" ")

	return string

def get_parent(item, nb_up):
	for i in range(nb_up):
		item = item.parent

	return item

if __name__ == "__main__":

	TIME_OUT = 5
	PAGE_LINK = "https://www.atpworldtour.com"

	matchs = open("testMatchs.txt", "r")
	scores = open("testScore.txt", "w")

	timeouts = 0
	failedMatchs = 0

	start = time.time()
	getTime = 0

	for line in matchs:

		try:

			type = detect_type(line)
			line = clean_string(line)

			if type is LineType.LINK:

				try:
					getStart = time.time()
					pageResponse = requests.get(PAGE_LINK + line, timeout = TIME_OUT)
					getEnd = time.time()
					getTime += (getEnd - getStart)

				except requests.Timeout:
					timeouts += 1
					continue

				pageContent = BeautifulSoup(pageResponse.content, "html.parser")

				players = pageContent.find_all(class_ = "scoring-player-name")
				if not players:
					raise FailedMatch()

				playersName = [item.string for item in players]
				playersName = [clean_string(item) for item in playersName]

				if get_parent(players[0], 5).find(class_ = "won-game" is None):
					loosingPlayer = playersName[0]
					winningPlayer = playersName[1]

				else:
					loosingPlayer = playersName[1]
					winningPlayer = playersName[0]

				matchInfo = pageContent.find(class_ = "match-info-row")
				if matchInfo is None:
					raise FailedMatch()

				matchInfo = matchInfo.find_all("td")
				if not matchInfo or len(matchInfo) != 2:
					raise FailedMatch()

				scoreInfo = matchInfo[0]
				durationInfo = matchInfo[1]

				scoreInfo = scoreInfo.string
				scoreInfo = ''.join(char for char in scoreInfo if not char.isalpha())
				scoreInfo = clean_string(scoreInfo)

				durationInfo = durationInfo.string
				durationInfo = clean_string(durationInfo)

				match = Match(winningPlayer, loosingPlayer, scoreInfo, durationInfo)

				for stat in pageContent.find_all(class_ = "match-stats-row percent-on"):
					leftState = stat.find(class_ = "match-stats-number-left")
					if leftState is None:
						raise FailedMatch()

					testLeftState = leftState.find('a')
					if testLeftState is None:
						testLeftState = leftState.find('span')
						if testLeftState is None:
							raise FailedMatch()

					leftState = testLeftState.string
					leftState = clean_string(leftState)

					label = stat.find(class_ = "match-stats-label")
					if label is None:
						raise FailedMatch()

					label = label.string
					label = clean_string(label)

					rightState = stat.find(class_ = "match-stats-number-right")
					if rightState is None:
						raise FailedMatch()

					testRightState = rightState.find('a')
					if testRightState is None:
						testRightState = rightState.find('span')
						if testRightState is None:
							raise FailedMatch()

					rightState = testRightState.string
					rightState = clean_string(rightState)

					match.add_stat(Statistic(label, leftState, rightState))

				match.write(scores)

			elif type is LineType.YEAR:
				print(line)
				scores.write("\n" + line + "\n")

		except FailedMatch:
			failedMatchs += 1
			continue

	end = time.time()
	matchs.close()
	print("time to get pages = {}".format(getTime))
	print("time elapsed = {} \n".format(end - start))
	print(str(timeouts) + " timeouts and " + str(failedMatchs) + " failed matchs were raised \n")