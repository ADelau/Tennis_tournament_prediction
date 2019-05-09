import sys
from functions import html_parse_tree, xpath_parse, regex_strip_array, read_csv, array2csv
import os

try:
    os.mkdir("csv")
except:
    pass

start_index = int(sys.argv[1])
end_index = int(sys.argv[2])
max_rank = int(sys.argv[3])

csv_file = 'weeks.csv'

weeks_list = []
read_csv(weeks_list, csv_file)

print("")
print("Collecting weekly rankings data from " + str(len(weeks_list)) + " weeks...")

print("")
print("Index    Week")
print("-----    ----")

for h in range(start_index, end_index + 1):
    week = weeks_list[h][0]
    dateList = week.split("-")
    yearRanking = dateList[0]
    monthRanking = dateList[1]
    dayRanking = dateList[2]

    week_url = "http://www.atpworldtour.com/en/rankings/singles?rankDate=" + week + "&rankRange=1-3000"

    week_tree = html_parse_tree(week_url)

    rank_xpath = "//table[@class='mega-table']/tbody/tr/td[@class='rank-cell']/text()"
    rank_parsed = xpath_parse(week_tree, rank_xpath)
    rank_cleaned = regex_strip_array(rank_parsed)

    player_name_xpath = "//table[@class='mega-table']/tbody/tr/td[@class='player-cell']/a/@data-ga-label"
    player_name_parsed = xpath_parse(week_tree, player_name_xpath)
    player_name_cleaned = regex_strip_array(player_name_parsed)

    move_xpath = "//table[@class='mega-table']/tbody/tr/td[@class='move-cell']/div[@class='move-text']/text()"
    move_parsed = xpath_parse(week_tree, move_xpath)
    move_cleaned = regex_strip_array(move_parsed)

    age_xpath = "//table[@class='mega-table']/tbody/tr/td[@class='age-cell']/text()"
    age_parsed = xpath_parse(week_tree, age_xpath)
    age_cleaned = regex_strip_array(age_parsed)

    points_xpath = "//table[@class='mega-table']/tbody/tr/td[@class='points-cell']/a/text()"
    points_parsed = xpath_parse(week_tree, points_xpath)

    rankings = []
    for i in range(0, max_rank):
        try:
            rank_text = rank_cleaned[i]
        except IndexError:
            break

        rank_number = rank_text.replace('T', '')

        player_name = player_name_cleaned[i]

        move = move_cleaned[i]
        move_up_xpath = "//table[@class='mega-table']/tbody/tr[" + \
            str(i + 1) + "]/td[@class='move-cell']/div[@class='move-up']"
        move_up_parsed = xpath_parse(week_tree, move_up_xpath)
        move_down_xpath = "//table[@class='mega-table']/tbody/tr[" + \
            str(i + 1) + "]/td[@class='move-cell']/div[@class='move-down']"
        move_down_parsed = xpath_parse(week_tree, move_down_xpath)
        if len(move_up_parsed) > 0:
            move_direction = 'up'
        elif len(move_down_parsed) > 0:
            move_direction = 'down'
        else:
            move_direction = ''

        age = age_cleaned[i]
        points = int(points_parsed[i].replace(',', ''))

        data = [yearRanking, monthRanking, dayRanking, rank_number, points, player_name, age, move, move_direction]
        rankings.append(data)

        filename = "csv/" + 'rankings_' + str(h) + '_' + week
        array2csv(rankings, filename)

    print(str(h) + "        " + week)
