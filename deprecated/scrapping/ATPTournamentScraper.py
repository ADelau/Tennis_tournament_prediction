#! /usr/bin/env python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests

TIME_OUT = 5

FIRST_YEAR = 1915
LAST_YEAR = 2018

PAGE_LINK = "https://www.atpworldtour.com/en/scores/results-archive"

timeouts = 0

file = open("ATPTournaments.txt", "w")

for year in range(FIRST_YEAR, LAST_YEAR+1):

	print(year)

	link = PAGE_LINK + "?year=" + str(year)

	try:
		pageResponse = requests.get(link, timeout = TIME_OUT)

	except requests.Timeout:
		timeouts += 1
		continue

	pageContent = BeautifulSoup(pageResponse.content, "html.parser")

	file.write(str(year) + ":\n\n")

	for link in pageContent.find_all(class_ = "button-border"):
		file.write(link.get("href") + "\n")

	file.write("\n")

print(str(timeouts) + " timeouts were raised")

file.close()