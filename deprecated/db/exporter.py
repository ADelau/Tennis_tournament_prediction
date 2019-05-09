import os
import csv

sources_2001_2018 = [
    os.getcwd() + '/bigdata/data/fusion1/{0}.csv'.format(i) for i in range(2001, 2019)]

sources2_2001_2018 = [
    os.getcwd() + '/bigdata/data/fusion2/{0}.csv'.format(i) for i in range(2001, 2019)]

destination_bookmakers_2001_2018 = os.getcwd(
) + '/bigdata/db/bookmakers_2001_2018.csv'
destination_odds_2001_2018 = os.getcwd() + '/bigdata/db/odds_2001_2018.csv'
destination_sets_2001_2018 = os.getcwd() + '/bigdata/db/sets_2001_2018.csv'
destination_games_2001_2018 = os.getcwd() + '/bigdata/db/games_2001_2018.csv'
destination_points_2001_2018 = os.getcwd() + '/bigdata/db/points_2001_2018.csv'

players_2001_2018 = {
    'sources': sources_2001_2018,
    'destination': os.getcwd() + '/bigdata/db/players_2001_2018.csv',
    'source_colnames': [['loser_id', 'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc'],
                        ['winner_id', 'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc']],
    'dest_colnames': ['id', 'name', 'hand', 'height', 'country'],
    'unique_on': [['loser_id'], ['winner_id']]}

tournaments_2001_2018 = {
    'sources': sources_2001_2018,
    'destination': os.getcwd() + '/bigdata/db/tournaments_2001_2018.csv',
    'source_colnames': [
        ['tourney_id', 'tourney_name', 'surface', 'Court', 'draw_size', 'Series']],
    'dest_colnames': ['id', 'name', 'surface', 'draw_size', 'court', 'level', ],
    'unique_on': [['tourney_id']]}

matches_2001_2018 = {
    'sources': sources_2001_2018,
    'destination': os.getcwd() + '/bigdata/db/matches_2001_2018.csv',
    'source_colnames': [['tourney_id', 'match_num', 'round', 'Date', 'winner_id', 'loser_id']],
    'dest_colnames': ['tournament', 'n', 'round', 'date', 'player_1', 'player_2'],
    'unique_on': [['tourney_id', 'match_num']]}


players_in_tournaments_2001_2018 = {
    'sources': sources_2001_2018,
    'destination': os.getcwd() + '/bigdata/db/players_in_tournaments_2001_2018.csv',
    'source_colnames': [['tourney_id', 'loser_id', 'loser_age', 'loser_rank', 'loser_rank_points', 'loser_seed', 'loser_entry'],
                        ['tourney_id', 'winner_id', 'winner_age', 'winner_rank', 'winner_rank_points', 'winner_seed', 'winner_entry']],
    'dest_colnames': ['tournament', 'player', 'age', 'atp_rank', 'atp_points', 'seed', 'entry'],
    'unique_on': [['tourney_id', 'loser_id'], ['tourney_id', 'winner_id']]}


matches_stats_2001_2018 = {
    'sources': sources_2001_2018,
    'destination': os.getcwd() + '/bigdata/db/matches_stats_2001_2018.csv',
    'source_colnames': [['tourney_id', 'match_num', 'loser_id', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon',  'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'],
                        ['tourney_id', 'match_num', 'winner_id', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon',  'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced']],
    'dest_colnames': ['tournament', 'match_n', 'player', 'ace', 'df', 'svpt', 'first_in', 'first_won', 'second_won', 'sv_gms', 'bp_saved', 'bp_faced'],
    'unique_on': [['tourney_id', 'match_num', 'loser_id'], ['tourney_id', 'match_num', 'winner_id']]}


def get_odds_from_files(sources, destination, headers=False):
    with open(destination, 'w') as csvdest:
        writer = csv.DictWriter(csvdest, fieldnames=[
                                'tournament', 'match', 'bookmaker', 'odd_1', 'odd_2'])
        if headers:
            writer.writeheader()
        for source in sources:
            with open(source, 'rt') as csvsource:
                reader = csv.DictReader(csvsource)

                for row in reader:
                    for book in ['CB', 'GB' 'IW', 'SB', 'B365', 'EX', 'LB']:
                        try:
                            book_labels = [book + i for i in ['W', 'L']]
                            data_read = {d: row[s] for s, d in zip(['tourney_id', 'match_num'] + book_labels,
                                                                   ['tournament', 'match', 'odd_1', 'odd_2'])}
                            data_read['bookmaker'] = book
                            if not data_read['odd_1'] or not data_read['odd_2']:
                                continue
                            writer.writerow(data_read)
                        except KeyError:  # Key does not exist
                            continue


def get_bookmakers_from_files(sources, destination, headers=False):
    uniques = set()
    id = 0

    with open(destination, 'w') as csvdest:
        writer = csv.DictWriter(csvdest, fieldnames=['id', 'name'])
        if headers:
            writer.writeheader()
        for source in sources:
            with open(source, 'rt') as csvsource:
                reader = csv.DictReader(csvsource)
                first = reader.fieldnames.index('Comment') + 1
                bookies = reader.fieldnames[first:]
                for book in bookies[::2]:  # skip Name_L
                    name = book[:-1]  # get rid of the L/W
                    if name not in uniques:
                        uniques.add(name)
                        writer.writerow({'id': id, 'name': name})
                        id += 1


def get_sets_from_files(sources, destination, headers=False):
    with open(destination, 'w') as csvdest:
        writer = csv.DictWriter(csvdest, fieldnames=['tournament', 'match_n', 'n', 'score1', 'score2'])
        if headers:
            writer.writeheader()
        for source in sources:
            with open(source, 'rt') as csvsource:
                reader = csv.DictReader(csvsource)

                for row in reader:
                    for n in range(1, 6):
                        Wn = 'W{0}'.format(n)
                        Ln = 'L{0}'.format(n)
                        if not row[Wn]:
                            break
                        writer.writerow({'tournament': row['tourney_id'], 'match_n': row['match_num'], 'n': n, 'score1': row[Wn], 'score2': row[Ln]})


def get_games_from_files(sources, destination, headers=False):

    with open(destination, 'w') as csvdest:
        writer = csv.DictWriter(csvdest, fieldnames=['tournament', 'match_n', 'set_n', 'n', 'server', 'tie_break'])
        if headers:
            writer.writeheader()
        for source in sources:
            with open(source, 'rt') as csvsource:
                reader = csv.DictReader(csvsource)
                try:
                    for row in reader:
                        if not row['pbp']:
                            continue
                        servers = [row['server1'], row['server2']]
                        sets = row['pbp'].split('.')
                        games = list(map(lambda s: s.split(';'), sets))
                        g_n = 0
                        for s, s_n in zip(games, range(0, len(games))):     # set_n
                            if s == ['']:
                                continue
                            for g, n in zip(s, range(0, len(s))):           # game_n within the set
                                tie_break = (g[1] == "/")
                                writer.writerow({'tournament': row['tourney_id'], 'match_n': row['match_num'], 'set_n': s_n,
                                                 'n': n, 'server': servers[g_n%2], 'tie_break': tie_break})
                                g_n += 1                                    # total game_n
                except KeyError:  # Key does not exist
                    continue


def get_points_from_files(sources, destination, headers=False):

    with open(destination, 'w') as csvdest:
        writer = csv.DictWriter(csvdest, fieldnames=['tournament', 'match_n', 'set_n', 'game_n', 'n', 'outcome'])
        if headers:
            writer.writeheader()
        for source in sources:
            with open(source, 'rt') as csvsource:
                reader = csv.DictReader(csvsource)
                try:
                    for row in reader:
                        if not row['pbp']:
                            continue
                        sets = row['pbp'].split('.')
                        games = list(map(lambda s: s.split(';'), sets))
                        for s, s_n in zip(games, range(0, len(games))):     # set_n
                            if s == ['']:
                                continue
                            for g, g_n in zip(s, range(0, len(s))):           # game_n within the set
                                p_n = 0
                                for p in g:
                                    if (p == "/"):
                                        continue
                                    writer.writerow({'tournament': row['tourney_id'], 'match_n': row['match_num'], 'set_n': s_n,
                                                    'game_n': g_n, 'n': p_n, 'outcome': p})
                                    p_n += 1

                except KeyError:  # Key does not exist
                    continue


def file_to_table(document_dict, headers=False):
    uniques = set()

    with open(document_dict['destination'], 'w') as csvdest:
        writer = csv.DictWriter(
            csvdest, fieldnames=document_dict['dest_colnames'])
        if headers:
            writer.writeheader()
        for source in document_dict['sources']:
            with open(source, 'rt') as csvsource:
                reader = csv.DictReader(csvsource)

                for row in reader:
                    for s_c, u_o in zip(document_dict['source_colnames'], document_dict['unique_on']):
                        key = tuple(row[u] for u in u_o)
                        if key not in uniques:
                            uniques.add(key)
                            read = {d: row[s] for s, d in zip(
                                s_c, document_dict['dest_colnames'])}
                            writer.writerow(read)




if __name__ == '__main__':
    # Exports to tables
    file_to_table(players_in_tournaments_2001_2018)
    file_to_table(tournaments_2001_2018)
    file_to_table(players_2001_2018)
    file_to_table(matches_2001_2018)
    file_to_table(matches_stats_2001_2018)
    get_bookmakers_from_files(
        sources_2001_2018, destination=destination_bookmakers_2001_2018)
    get_odds_from_files(sources=sources_2001_2018,
                       destination=destination_odds_2001_2018)
    get_sets_from_files(sources_2001_2018, destination_sets_2001_2018)
    get_games_from_files(sources2_2001_2018, destination_games_2001_2018)
    get_points_from_files(sources2_2001_2018, destination_points_2001_2018)
