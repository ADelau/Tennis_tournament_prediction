#! /usr/bin/env python
# -*- coding: utf-8 -*-

import csv

def sameKey(atpRow, tennisDataRow):
	#check tournament
	if atpRow[1] != tennisDataRow[1] and atpRow[1] != tennisDataRow[2]:
		return False

	#check winner
	atpWinnerNames = atpRow[10].split()
	tennisDataWinner = tennisDataRow[9].split()[0]
	found = False
	
	for i in range(len(atpWinnerNames)):
		if atpWinnerNames[i] == tennisDataWinner:
			found = True
			break

	if not found:
		return False

	#check looser
	atpWinnerNames = atpRow[20].split()
	tennisDataWinner = tennisDataRow[10].split()[0]
	found = False
	
	for i in range(len(atpWinnerNames)):
		if atpWinnerNames[i] == tennisDataWinner:
			found = True
			break

	if not found:
		return False

	return True

def fuse(atpRow, tennisRow, rowsToAdd, rowsToReplace):

	newRow = []

	if tennisRow is None:
		newRow = atpRow
		for i in rowsToAdd:
			newRow.append("")

	else:
		for i in range(len(atpRow)):
			replaced = False
			replace = None
			for j in range(len(rowsToReplace)):
				if i == rowsToReplace[j][0]:
					replace = rowsToReplace[j][1]
					replaced = True
			
			if replaced:
				newRow.append(tennisRow[replace])

			else:
				newRow.append(atpRow[i])

		for i in rowsToAdd:
			newRow.append(tennisRow[i]) 

	return newRow


ATPBEGIN = 1968
ATPEND = 2018

atpFiles = {}

for year in range(ATPBEGIN, ATPEND + 1):
	atpFiles[year] = csv.reader(open("Jeff_Sackmann/atp_matches_" + str(year) + ".csv", mode = 'r'))

TENNISDATABEGIN = 2001
TENNISDATAEND = 2018

tennisDataFiles = {}

for year in range(TENNISDATABEGIN, TENNISDATAEND + 1):
	tennisDataFiles[year] = csv.reader(open("Tennis_data/" + str(year) + ".csv", mode = 'r'))

for year in range(min(TENNISDATABEGIN, ATPBEGIN), max(TENNISDATAEND, ATPEND) + 1):
	print(str(year) + "\n")
	file = csv.writer(open("fusion1/" + str(year) + ".csv", mode = 'w'))

	if not year in tennisDataFiles.keys():
		for row in atpFiles[year]:
			file.writerow(row)

	elif not year in atpFiles.keys():
		for row in tennisDataFiles[year]:
			file.writerow(row)

	else:
		tennisDataReader = tennisDataFiles[year]
		atpReader = atpFiles[year]

		rowsToAdd = (0, 4, 5, 13, 14, 15, 16, 17, 18, 19, 20 ,21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33) #row in tennis to append to atp
		rowsToReplace = ((5, 3),) #(row in atp, row in tennis) replace row in atp by row in tennis

		tennisDataContent = []

		for row in tennisDataReader:
			tennisDataContent.append(row)

		first = True
		for row in atpReader:
			#write names
			if first:
				file.writerow(fuse(row, tennisDataContent[0], rowsToAdd, rowsToReplace))
				first = False
			else:
				found = False
				correspondingRow = None
				for tennisRow in tennisDataContent[1:]:
					if sameKey(row, tennisRow):
						correspondingRow = tennisRow
						found = True
				
				if(found):
					file.writerow(fuse(row, correspondingRow, rowsToAdd, rowsToReplace))

				else:
					file.writerow(fuse(row, None, rowsToAdd, rowsToReplace))