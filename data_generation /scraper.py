from functions import *
import pandas as pd


def scrape_rank_weeks():
    """
        Scrapes the ATP Website for ranking weeks.
        
    """
    weeks_url = "http://www.atpworldtour.com/en/rankings/singles"
    weeks_tree = html_parse_tree(weeks_url)
    weeks_xpath = "//ul[@data-value = 'rankDate']/li/@data-value"
    weeks_parsed = xpath_parse(weeks_tree, weeks_xpath)
    weeks_cleaned = regex_strip_array(weeks_parsed)
    weeks_list = [week for week in weeks_cleaned]

    return weeks_list


def scrape_ranking_at_week(week, max_rank):
    """
        Scrapes the ATP Website for a precise weekly ranking.
        
        Args:
        ----- 
        :week: (str) Date of the week,
        :max_rank: (int) Max rank to be scrapped.
        
        Returns:
        --------
        :rankings_df: DataFrame containing the ranking for the specified week. 
                      (cols: rank, rank_points, name, age)
    """ 

    week_url = "http://www.atpworldtour.com/en/rankings/singles?rankDate=" + week + "&rankRange=1-3000"
    week_tree = html_parse_tree(week_url)

    rank_xpath = "//table[@class='mega-table']/tbody/tr/td[@class='rank-cell']/text()"
    rank_parsed = xpath_parse(week_tree, rank_xpath)
    rank_cleaned = regex_strip_array(rank_parsed)

    player_name_xpath = "//table[@class='mega-table']/tbody/tr/td[@class='player-cell']/a/@data-ga-label"
    player_name_parsed = xpath_parse(week_tree, player_name_xpath)
    player_name_cleaned = regex_strip_array(player_name_parsed)

    age_xpath = "//table[@class='mega-table']/tbody/tr/td[@class='age-cell']/text()"
    age_parsed = xpath_parse(week_tree, age_xpath)
    age_cleaned = regex_strip_array(age_parsed)

    points_xpath = "//table[@class='mega-table']/tbody/tr/td[@class='points-cell']/a/text()"
    points_parsed = xpath_parse(week_tree, points_xpath)

    rankings = []
    for i in range(0, max_rank):
        rank_text = rank_cleaned[i]
        rank_number = rank_text.replace('T', '')

        player_name = player_name_cleaned[i]
        age = age_cleaned[i]
        points = int(points_parsed[i].replace(',', ''))

        data = [rank_number, points, player_name, age]
        rankings.append(data)

    rankings_df = pd.DataFrame(rankings, index=None, columns=[
                               'rank', 'rank_points', 'name', 'age'])

    return rankings_df
