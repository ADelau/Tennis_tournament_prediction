import sqlite3
import pandas as pd
import numpy as np
from sqlite3 import Error


class DBManager:

    @classmethod
    def create_connection(cls, db_file):
        """ create a database connection to the SQLite database
            specified by the db_file
        :param db_file: database file
        :return: Connection object or None
        """
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print(e)

        return None

    @classmethod
    def select_match_between(cls, date1, date2):
        return "select * from matches where date > '{0}' and date < '{1}' ORDER BY date;".format(date1, date2)

    @classmethod
    def select_player(cls, id):
        return "select * from players where id = {0};".format(id)

    @classmethod
    def select_tournament(cls, id):
        return "select * from tournaments where id = '{0}';".format(id)

    @classmethod
    def select_odds(cls, tournament, match):
        return "select * from odds where tournament = '{0}' and match_n = {1};".format(tournament, match)

    @classmethod
    def select_pl_in_t(cls, tournament, player):
        return "select * from players_in_tournaments where tournament = '{0}' and player = {1};".format(tournament,
                                                                                                        player)

    @classmethod
    def select_matches_stats(cls, player, date):
        return "select matches.tournament, match_n, player, ace, df, svpt, first_in, first_won, second_won, sv_gms, bp_saved, bp_faced from " \
               "matches JOIN matches_stats " \
               "ON matches.tournament = matches_stats.tournament and  matches.n = matches_stats.match_n " \
               "WHERE date < '{0}' and player = {1};".format(date, player)

    @classmethod
    def select_matches_player(cls, player, date):
        return "select * from matches where (player_1 = {0} or player_2 = {0}) and date < '{1}' ORDER BY date;".format(
            player, date)


class TrainingSetBuilder:
    matches_labels = ["date", "round"]

    stats_labels_ = ['ace', 'df', 'svpt', 'first_in', 'first_won', 'second_won', 'sv_gms', 'bp_saved', 'bp_faced']

    tourn_labels = ["surface", "court", "draw_size", "level"]

    player_in_tour_labels = ["age", "atp_ranking", "atp_points"]

    players_labels = ["hand", "height"]

    odds_labels = ["odd_avg_1", "odd_avg_2", "odd_max_1", "odd_max_2"]

    ratio_labels = ["ratio" + "_" + suffix for suffix in ['avg', 'prev', 'common']]

    def __init__(self, db_file):
        self.labels = self.compute_labels()
        self.db_file = db_file

    def build(self, date1, date2, save=True):
        conn = DBManager.create_connection(self.db_file)
        with conn:
            matches_df = pd.read_sql_query(DBManager.select_match_between(date1, date2), conn)  # get all the matches

            entries = []
            for i, m in matches_df.iterrows():
                tour = self.get_tournament_infos(m, conn)
                player1, player2 = self.get_players(m, conn)
                avgodds, maxodds = self.get_odds(m, conn)
                pt1, pt2 = self.get_players_in_tournaments(m, conn)
                avg, prev, common = self.get_player_matches(m, conn)

                stats_avg_1, stats_avg_2, ratio_avg_1, ratio_avg_2 = avg
                stats_prev_1, stats_prev_2, ratio_prev_1, ratio_prev_2 = prev
                stats_common_1, stats_common_2, ratio_common_1, ratio_common_2 = common

                entries.append([m['date'], m[
                    'round']] + tour + pt1 + player1 + ratio_avg_1 + ratio_prev_1 + ratio_common_1 + stats_avg_1 + stats_prev_1 + stats_common_1 + pt2 + player2 + stats_avg_2 + stats_prev_2 + stats_common_2 + ratio_avg_2 + ratio_prev_2 + ratio_common_2 + maxodds + avgodds)

            return pd.DataFrame(entries, columns=self.labels)

    def get_tournament_infos(self, m, conn):
        df = pd.read_sql_query(DBManager.select_tournament(m['tournament']), conn)
        tour = df[self.tourn_labels].iloc[0]
        return tour.tolist()

    def get_players(self, m, conn):
        df_1 = pd.read_sql_query(DBManager.select_player(m['player_1']), conn)
        p_1 = df_1[self.players_labels].iloc[0]
        df_2 = pd.read_sql_query(DBManager.select_player(m['player_2']), conn)
        p_2 = df_2[self.players_labels].iloc[0]

        return p_1.tolist(), p_2.tolist()

    def get_odds(self, m, conn, default=2):
        df = pd.read_sql_query(DBManager.select_odds(m['tournament'], m['n']), conn)
        if df.empty:  # no odd for the match
            return [default, default], [default, default]  # return default

        try:
            odds = df[['odd_1', 'odd_2']].apply(lambda x: x.str.replace(',', '.').astype(float))
        except AttributeError:
            odds = df[['odd_1', 'odd_2']]

        avg = odds.mean(axis=0).tolist()
        max = odds.max(axis=0).tolist()

        return avg, max

    def get_players_in_tournaments(self, m, conn):
        df_1 = pd.read_sql_query(DBManager.select_pl_in_t(m['tournament'], m['player_1']), conn)
        p_1 = df_1[self.player_in_tour_labels].iloc[0]
        df_2 = pd.read_sql_query(DBManager.select_pl_in_t(m['tournament'], m['player_2']), conn)
        p_2 = df_2[self.player_in_tour_labels].iloc[0]

        return p_1.tolist(), p_2.tolist()

    def get_player_matches(self, m, conn):

        def average(m, dfm_1, dfm_2, dfms_1, dfms_2):
            stats_avg_1 = dfms_1[self.stats_labels_].mean(axis=0).tolist()
            stats_avg_2 = dfms_2[self.stats_labels_].mean(axis=0).tolist()

            ratio_avg_1 = 0 if len(dfm_1) == 0 else sum(dfm_1['player_1'] == m['player_1']) / len(dfm_1)
            ratio_avg_2 = 0 if len(dfm_2) == 0 else sum(dfm_1['player_1'] == m['player_2']) / len(dfm_2)

            return stats_avg_1, stats_avg_2, [ratio_avg_1], [ratio_avg_2]

        def prev(m, dfm_1, dfm_2, dfms_1, dfms_2):
            dfm_1_loss = dfm_1[dfm_1.player_1 == m['player_2']]
            dfm_1_win = dfm_1[dfm_1.player_2 == m['player_2']]
            dfm_1_all = pd.concat([dfm_1_loss, dfm_1_win])

            dfms_1['match_n'] = dfms_1['match_n'].astype(int)
            dfms_2['match_n'] = dfms_2['match_n'].astype(int)
            dfm_1_all['n'] = dfm_1_all['n'].astype(int)

            dfms_1_prev = pd.merge(dfm_1_all[['tournament', 'n']], dfms_1, how='left', left_on=['tournament', 'n'],
                                   right_on=['tournament', 'match_n'])
            dfms_2_prev = pd.merge(dfm_1_all[['tournament', 'n']], dfms_2, how='left', left_on=['tournament', 'n'],
                                   right_on=['tournament', 'match_n'])

            stats_prev_1 = dfms_1_prev[self.stats_labels_].mean(axis=0).tolist()
            stats_prev_2 = dfms_2_prev[self.stats_labels_].mean(axis=0).tolist()

            ratio_prev_1 = 0 if len(dfm_1_all) == 0 else len(dfm_1_win) / len(dfm_1_all)
            ratio_prev_2 = 0 if len(dfm_1_all) == 0 else 1 - ratio_prev_1

            return stats_prev_1, stats_prev_2, [ratio_prev_1], [ratio_prev_2]

        def common(m, dfm_1, dfm_2, dfms_1, dfms_2):
            # remove direct oppositions
            dfm_1 = dfm_1[(dfm_1.player_1 != m['player_2']) & (dfm_1.player_2 != m['player_2'])]
            dfm_2 = dfm_2[(dfm_2.player_1 != m['player_1']) & (dfm_2.player_2 != m['player_1'])]

            o_11 = tuple(list(dfm_1['player_1']))
            o_12 = tuple(list(dfm_1['player_2']))
            o_21 = tuple(list(dfm_2['player_1']))
            o_22 = tuple(list(dfm_2['player_2']))

            opp1 = set(o_11 + o_12)
            opp2 = set(o_21 + o_22)

            common = set.intersection(opp1, opp2)

            join_1 = pd.merge(dfm_1, dfms_1, how='left', left_on=['tournament', 'n'],
                              right_on=['tournament', 'match_n'])

            join_2 = pd.merge(dfm_2, dfms_2, how='left', left_on=['tournament', 'n'],
                              right_on=['tournament', 'match_n'])

            stats_1 = pd.DataFrame(columns=self.stats_labels_)
            stats_2 = pd.DataFrame(columns=self.stats_labels_)

            loss_1 = 0
            win_1 = 0

            loss_2 = 0
            win_2 = 0
            for adv in common:
                loss_1 += len(dfm_1[(dfm_1.player_1 == adv)])
                win_1 += len(dfm_1[(dfm_1.player_2 == adv)])

                loss_2 += len(dfm_2[(dfm_2.player_1 == adv)])
                win_2 += len(dfm_2[(dfm_2.player_2 == adv)])

                stats_1 = pd.concat(
                    [stats_1, join_1[(join_1.player_1 == adv) | (join_1.player_2 == adv)][self.stats_labels_]])
                stats_2 = pd.concat(
                    [stats_2, join_2[(join_2.player_1 == adv) | (join_2.player_2 == adv)][self.stats_labels_]])

            stats_1 = stats_1.apply(pd.to_numeric)
            stats_2 = stats_2.apply(pd.to_numeric)

            if stats_1.empty:
                stats_1 = [0 for i in self.stats_labels_]
            else:
                stats_1 = stats_1.values.mean(axis=0).tolist()
            if stats_2.empty:
                stats_2 = [0 for i in self.stats_labels_]
            else:
                stats_2 = stats_2.values.mean(axis=0).tolist()

            ratio_1 = 0 if (win_1 + loss_1) == 0 else win_1 / (win_1 + loss_1)
            ratio_2 = 0 if (win_2 + loss_2) == 0 else win_2 / (win_2 + loss_2)

            return stats_1, stats_2, [ratio_1], [ratio_2]

        dfm_1 = pd.read_sql_query(DBManager.select_matches_player(m['player_1'], m['date']), conn)
        dfm_2 = pd.read_sql_query(DBManager.select_matches_player(m['player_2'], m['date']), conn)

        dfms_1 = pd.read_sql_query(DBManager.select_matches_stats(m['player_1'], m['date']), conn)
        dfms_2 = pd.read_sql_query(DBManager.select_matches_stats(m['player_2'], m['date']), conn)

        _dfms_1 = dfms_1.copy()
        _dfms_2 = dfms_2.copy()

        return average(m, dfm_1, dfm_2, dfms_1, dfms_2), prev(m, dfm_1, dfm_2, _dfms_1, _dfms_2), common(m, dfm_1,
                                                                                                         dfm_2, dfms_1,
                                                                                                         dfms_2)

    def compute_labels(self):

        stats_labels = [l + '_' + suffix for suffix in ['avg', 'prev', 'common'] for l in self.stats_labels_]
        labels = self.matches_labels + self.tourn_labels + [f + '_' + p for p in ['1', '2'] for f in
                                                            self.player_in_tour_labels + self.players_labels
                                                            + self.ratio_labels + stats_labels] + self.odds_labels

        return labels


if __name__ == '__main__':
    t = TrainingSetBuilder("tennis_db")

    for i in range(2008, 2011):
        print("Building for year {0}...".format(i))
        t.build("{0}/1/1".format(i), "{0}/12/31".format(i)).to_csv('dataset_{0}.csv'.format(i), header=True, index=False)
        print("Building ended")
