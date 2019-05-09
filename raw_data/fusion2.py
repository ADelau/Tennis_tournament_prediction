#! /usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from enum import Enum
from datetime import date

MONTH_TO_INT = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12
}


class IllFormated(Exception):
    pass


class Format(Enum):
    PBP = 1  # formatted as in point by point files
    ATP = 2  # formatted as in the atp files
    TENNIS = 3  # formatted as in tennis data files


class MyDate():

    def __init__(self, strDate):

        testSplit1 = strDate.split("/")
        testSplit2 = strDate.split(" ")
        try:

            if len(testSplit1) == 3:  # formatted as in tennis data files
                self.format = Format.TENNIS
                year = int(testSplit1[0])
                month = int(testSplit1[1])
                day = int(testSplit1[2])

            elif len(testSplit2) == 3:  # formatted as in point by point files
                self.format = Format.PBP
                year = 2000 + int(testSplit2[2])
                day = int(testSplit2[0])
                month = MONTH_TO_INT[testSplit2[1]]

            else:  # formatted as in atp files
                self.format = Format.ATP
                year = int(strDate[:4])
                month = int(strDate[4:6])
                day = int(strDate[6:8])
        except:
            raise IllFormated()

        self.dateObj = date(year, month, day)

    def distance_to(self, compareDate):
        delta = compareDate.dateObj - self.dateObj
        return abs(delta.days)

    def get_str(self):
        return "{}/{}/{}".format(self.dateObj.year, self.dateObj.month, self.dateObj.day)


class PBPMatch():

    def __init__(self, row, dateColumn, playerAColumn, playerBColumn):
        self.row = row
        self.date = MyDate(row[dateColumn])

        tmp = row[playerAColumn].split(" ")
        if len(tmp) < 2:
            raise IllFormated()
        self.playerA = tmp[1]

        tmp = row[playerBColumn].split(" ")
        if len(tmp) < 2:
            raise IllFormated()
        self.playerB = tmp[1]


class PBPFilesGest():

    def __init__(self):

        self.matchs = {}

    def add(self, filename):

        reader = csv.reader(open(filename, mode='r'))
        First = True

        dateColumn = 0
        playerAColumn = 0
        playerBcolumn = 0

        for row in reader:
            if First:
                First = False
                dateColumn = row.index("date")
                playerAColumn = row.index("server1")
                playerBcolumn = row.index("server2")

            else:
                try:
                    match = PBPMatch(row, dateColumn, playerAColumn, playerBcolumn)

                    if match.date.dateObj.year in self.matchs.keys():
                        self.matchs[match.date.dateObj.year].append(match)

                    else:
                        self.matchs[match.date.dateObj.year] = [match]
                except IllFormated:
                    continue

    def is_year_present(self, year):
        return year in self.matchs.keys()

    def get_corresponding_match(self, row, dateColumn, playerAColumn, playerBcolumn):
        MAX_DELTA_DATE = 7

        rowDate = MyDate(row[dateColumn])

        if rowDate.dateObj.year not in self.matchs.keys():
            return None

        else:
            rowPlayerA = row[playerAColumn]
            rowPlayerA = rowPlayerA.split(" ")[1]
            rowPlayerB = row[playerBcolumn]
            rowPlayerB = rowPlayerB.split(" ")[1]

            for match in self.matchs[rowDate.dateObj.year]:

                if (rowPlayerA != match.playerA and rowPlayerA != match.playerB) or \
                        (rowPlayerB != match.playerA and rowPlayerB != match.playerB):
                    continue
                """
                if rowDate.format == Format.TENNIS:
                    if rowDate.dateObj != match.date:
                        continue
                else: #format = ATP
                    if rowDate.distance_to(match.date) > MAX_DELTA_DATE:
                        continue
                """
                if rowDate.distance_to(match.date) > MAX_DELTA_DATE:
                    continue

                return match

            return None


def fuse(row, pbpRow, dateColumn):
    newRow = []

    for i in range(len(row)):
        if i == dateColumn:
            tmpDate = MyDate(row[i])
            newRow.append(tmpDate.get_str())
        else:
            newRow.append(row[i])

    if pbpRow is not None:
        newRow.append(pbpRow[5])
        newRow.append(pbpRow[6])

        currRow = 8
        curr = pbpRow[currRow]
        while curr[0] == "A" or curr[0] == "D" or curr[0] == "S" or curr[0] == "R":
            newRow.append(curr)
            currRow += 1
            curr = pbpRow[currRow]

    return newRow


if __name__ == "__main__":

    PBPFiles = PBPFilesGest()

    PBPFiles.add("pointbypoint/pbp_matches_atp_main_archive.csv")
    PBPFiles.add("pointbypoint/pbp_matches_atp_main_current.csv")
    PBPFiles.add("pointbypoint/pbp_matches_atp_qual_archive.csv")
    PBPFiles.add("pointbypoint/pbp_matches_atp_qual_current.csv")
    PBPFiles.add("pointbypoint/pbp_matches_ch_main_archive.csv")
    PBPFiles.add("pointbypoint/pbp_matches_ch_main_current.csv")
    PBPFiles.add("pointbypoint/pbp_matches_ch_qual_archive.csv")
    PBPFiles.add("pointbypoint/pbp_matches_ch_qual_current.csv")
    PBPFiles.add("pointbypoint/pbp_matches_fu_main_archive.csv")
    PBPFiles.add("pointbypoint/pbp_matches_fu_main_current.csv")
    PBPFiles.add("pointbypoint/pbp_matches_fu_qual_current.csv")
    PBPFiles.add("pointbypoint/pbp_matches_itf_main_archive.csv")
    PBPFiles.add("pointbypoint/pbp_matches_itf_main_current.csv")
    PBPFiles.add("pointbypoint/pbp_matches_itf_qual_current.csv")
    PBPFiles.add("pointbypoint/pbp_matches_wta_main_archive.csv")
    PBPFiles.add("pointbypoint/pbp_matches_wta_main_current.csv")
    PBPFiles.add("pointbypoint/pbp_matches_wta_qual_archive.csv")
    PBPFiles.add("pointbypoint/pbp_matches_wta_qual_current.csv")

    YEAR_BEGIN = 1968
    YEAR_END = 2018

    dateColumn = None
    playerAColumn = None
    playerBcolumn = None

    for year in range(YEAR_BEGIN, YEAR_END + 1):
        print(str(year) + "\n")
        reader = csv.reader(open("fusion1/" + str(year) + ".csv", mode='r'))
        writer = csv.writer(open("fusion2/" + str(year) + ".csv", mode='w'))

        if not PBPFiles.is_year_present(year):
            first = True
            for row in reader:
                if first:
                    try:
                        dateColumn = row.index("Date")
                    except:
                        dateColumn = row.index("tourney_date")
                    row[dateColumn] = "Date"
                    writer.writerow(row)
                    first = False

                else:
                    try:
                        writer.writerow(fuse(row, None, dateColumn))
                    except IllFormated:
                        continue

        else:
            first = True
            i = 0
            for row in reader:

                if first:
                    try:
                        dateColumn = row.index("Date")
                    except:
                        dateColumn = row.index("tourney_date")

                    playerAColumn = row.index("winner_name")
                    playerBcolumn = row.index("loser_name")

                    newRow = row
                    newRow.append("server1")
                    newRow.append("server2")
                    newRow.append("pbp")
                    newRow[dateColumn] = "Date"
                    writer.writerow(newRow)

                    first = False

                else:
                    match = PBPFiles.get_corresponding_match(row, dateColumn, playerAColumn, playerBcolumn)
                    if match is not None:
                        writer.writerow(fuse(row, match.row, dateColumn))
                    else:
                        writer.writerow(fuse(row, None, dateColumn))
