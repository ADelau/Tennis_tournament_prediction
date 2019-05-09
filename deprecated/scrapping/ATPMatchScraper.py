#! /usr/bin/env python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
from enum import Enum
from enum import auto

class LineType(Enum):
	YEAR = auto()
	EMPTY = auto()
	LINK = auto()

def detect_type(readString):
	if not readString:
		return LineType.EMPTY

	if(readString[0] is '/'): 
		return LineType.LINK

	else:
		return LineType.YEAR

if __name__ == "__main__":

	TIME_OUT = 5
	PAGE_LINK = "https://www.atpworldtour.com"
	MATCH_TYPE = {"single" : "?matchType=singles", "double" : "results?matchType=doubles"}

	matchs = open("testMatchs.txt", "w")
	tournaments = open("testTournaments.txt", "r")

	timeouts = 0

	for line in tournaments:

		type = detect_type(line)

		line = line.rstrip('\n')

		if type is LineType.LINK:

			try:
				pageResponse = requests.get(PAGE_LINK + line + MATCH_TYPE["single"], timeout = TIME_OUT)

			except requests.Timeout:
				timeouts += 1
				continue

			pageContent = BeautifulSoup(pageResponse.content, "html.parser")

			for match in pageContent.find_all(class_ = " "):
				matchs.write(match.get("href") + "\n")

		elif type is LineType.YEAR:
			print(line)
			matchs.write("\n" + line + "\n")

	print(str(timeouts) + " timeouts were raised")

	matchs.close()