import numpy as np
import pandas as pd
import os
import datetime
import multiprocessing
import functools
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Surfaces correlation
corr = np.array([[1.0, 0.7015425398108537, 0.3623050295375809, 0.4170855617635379],
                 [0.7015425398108537, 1.0, 0.5328161870767436, 0.5960299654435783],
                 [0.3623050295375809, 0.5328161870767436, 1.0, 0.22536731067055044],
                 [0.4170855617635379, 0.5960299654435783, 0.22536731067055044, 1.0]])


class DatasetMaker:
    """
        Class that enables the generation and the preparation
        of a dataset based on raw historical data.
    """
    DEFAULT_CONFIG = {"stats_type": [
        'common', 'avg'], "discount": [("week", 0.995)], "surface": corr}

    FULL_CONFIG = {"stats_type": [
        'common', 'avg'], "discount": [("week", 0.991), ("week", 0.993), ("week", 0.995), ("week", 0.997), ("week", 0.999), ("year", 0.5), ("year", 0.6), ("year", 0.7), ("year", 0.8),
                                       ("year", 0.9), ("year", 1)], "surface": corr}

    DEFAULT_STATS_TYPE = ['common']
    DEFAULT_DISCOUNT = [('year', 1), ]
    SURFACES = ['Grass', 'Hard', 'Clay', 'Carpet']
    WRITE_PACE = 100  # Write every 100 matches

    # Important to keep the order
    ODDS = ['B365_1', 'EX_1', 'LB_1', 'CB_1', 'GB_1',  'IW_1', 'SB_1',
            'SB_2', 'IW_2', 'GB_2', 'CB_2', 'LB_2', 'EX_2', 'B365_2']

    AS_SUCH = ['level', 'date',
               'winner_id', 'winner_name', 'winner_ht', 'winner_age',
               'winner_rank', 'winner_rank_points',
               'loser_id', 'loser_name', 'loser_ht', 'loser_age',
               'loser_rank', 'loser_rank_points'] + ['w_B365', 'w_EX', 'w_LB', 'w_CB', 'w_GB',  'w_IW', 'w_SB', 'l_SB', 'l_IW', 'l_GB',  'l_CB', 'l_LB', 'l_EX', 'l_B365']
    STATS_WINNER = [
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
        'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt',
        'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'discount']

    STATS_LOSER = [
        'l_ace', 'l_df', 'l_svpt',
        'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
        'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'discount']

    STATS_FINAL = [
        'self_ace', 'self_df', 'self_svpt',
        'self_1stIn', 'self_1stWon', 'self_2ndWon', 'self_SvGms', 'self_bpSaved', 'self_bpFaced',
        'opp_ace', 'opp_df', 'opp_svpt',
        'opp_1stIn', 'opp_1stWon', 'opp_2ndWon', 'opp_SvGms', 'opp_bpSaved', 'opp_bpFaced', 'discount']

    HEADER_AS_SUCH = ['level', 'date',
                      'id_1', 'name_1', 'ht_1', 'age_1',
                      'rank_1', 'rank_points_1',
                      'id_2', 'name_2', 'ht_2', 'age_2',
                      'rank_2', 'rank_points_2'] + ['B365_1', 'EX_1', 'LB_1', 'CB_1', 'GB_1',  'IW_1', 'SB_1', 'SB_2', 'IW_2', 'GB_2',  'CB_2', 'LB_2', 'EX_2', 'B365_2']

    HEADER_STATS_GENERATION = ['ratio', 'ace', 'df', '1st_in',
                               '1st_win', '2nd_win', 'serve_win', 'break_saved', 'break_lost', 'return_1st_win', 'return_2nd_win', 'ace_faced', 'break_win', 'points_win']

    NO_DIFF_COLS = ['id', 'name'] + ['B365', 'EX', 'LB', 'CB', 'GB',  'IW', 'SB']

    def __init__(self, full_df, config, debug=False):
        """
         DatasetMaker constructor.
         Args:
         ----
            :full_df: DataFrame containing the raw matches.
            :config: Dictionnary containing the configuration for the dataset generation
                    * stats_type (default:DEFAULT_STATS_TYPE): list of stats type ('h2h', 'common', 'avg')
                    * discount: list of tuples (default: DEFAULT_DISCOUNT)
                                    * first element in {'year', 'week'},
                                    * second element discount factor
                    * surface: numpy array containing correlation matrix, if not provided
                                the correlation will be set to one for every pair.
                    * as_such: list of str (default: AS_SUCH)
                                        Columns that have to be kept as such.
            :debug: bool (default:False)
        """
        self._debug_mode = debug
        self._full_df = self._clean(full_df)
        self._parse_config(config)

    def generate(self, target_df, output):
        """
        Generates and saves a dataset based on raw historical
        data according the configuration of the DatasetMaker instance.
        Args:
        -----
            :target_df: DataFrame containing the desired entries.
                        It has to be composed, at least, of the following
                        columns: 'date', 'surface', 'winner_id', 'loser_id'
            :output:    str, prefix of the file that will contain
                            the generated datsets.
        """
        self._debug("It will take a certain time... \nHave to generate {0} matches!".format(
            target_df.shape[0]))
        self._generate(target_df, output)
        self._debug("Already finished...")

    def prepare(self, target_df, output, duplicate=True):
        """
        Prepares a dataset generated by the 'generate' function.
        Args:
        -----
            :target_df: DataFrame containing the generated dataset
            :output: str, prefix of the file that will contain
                            the prepared datsets.
            :duplicate: bool, if the matches have to be duplicated (p1-p2 and p2-p1)
        """
        # Prepare
        prepared = self._prepare(target_df, duplicate)
        prepared.to_csv(output, index=False)

    def _debug(self, msg):
        if self._debug_mode:
            print("[DEBUG msg] {}".format(msg))

    def _clean(self, df):
        """
        Cleans a DataFrame containing raw historical data:
            * Missing surface,
            * Not finished games,
            * Clean date format.
        Args:
        ----
            df: DataFrame to be cleaned
        """
        self._debug('Before Cleaning: {} entries'.format(df.shape[0]))
        # remove entries with missing surfaces
        self._debug("*Remove missing surfaces.")
        df = df[df.surface != 'None']
        df = df[~df.surface.isnull()]
        # remove not finished matches
        self._debug("*Remove not finished games.")
        df = df[~df.Comment.isin(['Walkover', 'Retired'])]
        df = df[df.score.str.contains("RET") == False]
        df = df[df.score.str.contains("W/O") == False]

        self._debug("*Clean dates format + Sorting.")
        df['date'] = pd.to_datetime(df['date'], format="%Y/%m/%d")

        self._debug('After Cleaning: {} entries'.format(df.shape[0]))

        return df.sort_values(by='date')

    def _parse_config(self, config):
        """
            Parses a configuration dictionnary.
        """
        self._stats_type = config.get("stats_type", self.DEFAULT_STATS_TYPE)
        self._discount = config.get("discount", self.DEFAULT_DISCOUNT)
        self._surface_struct(config.get("surface", np.ones(
            (len(self.SURFACES), len(self.SURFACES)))))
        self._as_such = config.get("as_such", self.AS_SUCH)
        self._header_generation = self._final_header_generation()

        self._debug('Config: \n\t{0}\n\t{1}\n\t{2}\n\t{3}\n\t{4}'.format(
            self._stats_type, self._discount, self._surface, self._as_such, self._header_generation))

    def _surface_struct(self, corr):
        """
            Creates the surface correlation structure.
        """
        self._surface = pd.DataFrame(corr, index=self.SURFACES, columns=self.SURFACES)

    def _generate(self, target_df, output):

        containers = [[] for _ in self._discount]

        for i, match in target_df.iterrows():
            date = match['date']
            surface = match['surface']
            winner = match['winner_id']
            loser = match['loser_id']

            as_such = match[self._as_such].tolist()

            stats = self._compute_stats(winner, loser, surface, date)

            for c, cont in enumerate(containers):  # append to containers
                cont.append(as_such + stats[c])

            if i % self.WRITE_PACE == 0:
                self._write_to_file(containers, output)
                self._debug("[{}/{}]".format(i, target_df.shape[0]))

        self._write_to_file(containers, output)  # write the end

    def _prepare(self, target_df, duplicate):

        for type in self._stats_type:
            # Completeness
            target_df[type + "_completeness_1"] = target_df[type + "_serve_win_1"] * \
                (target_df[type + "_return_1st_win_1"] + target_df[type + "_return_2nd_win_1"])
            target_df[type + "_completeness_2"] = target_df[type + "_serve_win_2"] * \
                (target_df[type + "_return_1st_win_2"] + target_df[type + "_return_2nd_win_2"])

            # Feature: First Serve Advantage
            target_df[type + "_serve1_adv_1"] = target_df[type + "_1st_win_1"] - \
                target_df[type + "_return_1st_win_2"]
            target_df[type + "_serve1_adv_2"] = target_df[type + "_1st_win_2"] - \
                target_df[type + "_return_1st_win_1"]

            # Feature: Second Serve Advantage
            target_df[type + "_serve2_adv_1"] = target_df[type + "_2nd_win_1"] - \
                target_df[type + "_return_2nd_win_2"]
            target_df[type + "_serve2_adv_2"] = target_df[type + "_2nd_win_2"] - \
                target_df[type + "_return_2nd_win_1"]

        # Player 1
        cols_1 = [col for col in target_df if col.endswith(
            '1') and col[0:-2] not in self.NO_DIFF_COLS]
        # Player 2
        cols_2 = [col for col in target_df if col.endswith(
            '2') and col[0:-2] not in self.NO_DIFF_COLS]

        # Final columns
        cols = [col[0:-2] for col in cols_1]  # Remove number and underscore

        if duplicate:
            diff_1 = target_df[cols_1].values - target_df[cols_2].values
            diff_2 = target_df[cols_2].values - target_df[cols_1].values

            final = pd.DataFrame(np.concatenate([diff_1, diff_2]), columns=cols)
            # outcome
            final["outcome"] = -1
            final.loc[0:diff_1.shape[0], "outcome"] = 1
            final.loc[diff_1.shape[0]:final.shape[0], "outcome"] = 0

            # names
            final["name_1"] = np.concatenate([target_df.name_1, target_df.name_2])
            final["name_2"] = np.concatenate([target_df.name_2, target_df.name_1])

            # date
            final["date"] = np.concatenate([target_df.date, target_df.date])

            # level
            final["level"] = np.concatenate([target_df.level, target_df.level])

            # odds
            odds = np.concatenate([target_df[self.ODDS].values,
                                   target_df[list(reversed(self.ODDS))].values])
            for i, odd in enumerate(self.ODDS):
                final[odd] = odds[:, i]

        if not duplicate:
            diff_1 = target_df[cols_1].values - target_df[cols_2].values

            final = pd.DataFrame(np.concatenate([diff_1]), columns=cols)
            # outcome
            final["outcome"] = 1

            # names
            final["name_1"] = np.concatenate([target_df.name_1])
            final["name_2"] = np.concatenate([target_df.name_2])

            # date
            final["date"] = np.concatenate([target_df.date])

            # level
            final["level"] = np.concatenate([target_df.level])

            # odds
            odds = np.concatenate([target_df[self.ODDS].values])
            for i, odd in enumerate(self.ODDS):
                final[odd] = odds[:, i]

        return final.sort_index()

    def _compute_stats(self, winner_id, loser_id, surface, date):

        stats = [[] for _ in self._discount]

        for t in self._stats_type:
            method_name = '_' + t
            try:
                method = getattr(self, method_name)
            except AttributeError:
                raise NotImplementedError("Class `{}` does not implement `{}`".format(
                    self.__class__.__name__, method_name))

            _stats_1, _stats_2 = method(winner_id, loser_id, surface, date)

            stats_1 = list(map(self._process_total_stats, _stats_1))
            stats_2 = list(map(self._process_total_stats, _stats_2))
            for i, (s1, s2) in enumerate(zip(stats_1, stats_2)):
                stats[i].extend(s1)
                stats[i].extend(s2)

        return stats

    def _h2h(self, player1, player2, surface, date):
        matches_player1_as_winner = self._full_df[(self._full_df.date < date) & (
            self._full_df.winner_id == player1) & (self._full_df.loser_id == player2)]

        matches_player2_as_winner = self._full_df[(self._full_df.date < date) & (
            self._full_df.winner_id == player2) & (self._full_df.loser_id == player1)]
        discounted_p1 = self._compute_discount(matches_player1_as_winner, date, surface)
        discounted_p2 = self._compute_discount(matches_player2_as_winner, date, surface)

        stats1 = []
        stats2 = []

        for d1, d2 in zip(discounted_p1, discounted_p2):
            s1, s2 = self._concat_and_sum_stats(d1, d2, h2h=True)
            stats1.append(s1)
            stats2.append(s2)

        return stats1, stats2

    def _common(self, player1, player2, surface, date):

        opp1_as_winner = self._full_df[(self._full_df.date < date) & (
            self._full_df.winner_id == player1) & (self._full_df.loser_id != player2)]
        opp1_as_loser = self._full_df[(self._full_df.date < date) & (
            self._full_df.loser_id == player1) & (self._full_df.winner_id != player2)]
        opp2_as_winner = self._full_df[(self._full_df.date < date) & (
            self._full_df.winner_id == player2) & (self._full_df.loser_id != player1)]
        opp2_as_loser = self._full_df[(self._full_df.date < date) & (
            self._full_df.loser_id == player2) & (self._full_df.winner_id != player1)]

        commons = pd.unique(np.concatenate(
            [opp1_as_winner.loser_id.values, opp1_as_loser.winner_id.values, opp2_as_winner.loser_id.values, opp2_as_loser.winner_id.values]))

        com_opp1_as_winner = opp1_as_winner[opp1_as_winner.loser_id.isin(commons)]
        com_opp1_as_loser = opp1_as_loser[opp1_as_loser.winner_id.isin(commons)]
        com_opp2_as_winner = opp2_as_winner[opp2_as_winner.loser_id.isin(commons)]
        com_opp2_as_loser = opp2_as_loser[opp2_as_loser.winner_id.isin(commons)]

        discounted_wins_player1 = self._compute_discount(com_opp1_as_winner, date, surface)
        discounted_loses_player1 = self._compute_discount(com_opp1_as_loser, date,  surface)

        stats1 = []
        for d1, d2 in zip(discounted_wins_player1, discounted_loses_player1):
            s1 = self._concat_and_sum_stats(d1, d2, h2h=False)
            stats1.append(s1)

        discounted_wins_player2 = self._compute_discount(com_opp2_as_winner, date, surface)
        discounted_loses_player2 = self._compute_discount(com_opp2_as_loser, date, surface)

        stats2 = []
        for d1, d2 in zip(discounted_wins_player2, discounted_loses_player2):
            s2 = self._concat_and_sum_stats(d1, d2, h2h=False)
            stats2.append(s2)

        return stats1, stats2

    def _avg(self, player1, player2, surface, date):

        matches_player1_as_winner = self._full_df[(
            self._full_df.date < date) & (self._full_df.winner_id == player1)]
        matches_player1_as_loser = self._full_df[(
            self._full_df.date < date) & (self._full_df.loser_id == player1)]
        matches_player2_as_winner = self._full_df[(
            self._full_df.date < date) & (self._full_df.winner_id == player2)]
        matches_player2_as_loser = self._full_df[(
            self._full_df.date < date) & (self._full_df.loser_id == player2)]

        discounted_wins_player1 = self._compute_discount(matches_player1_as_winner, date, surface)
        discounted_loses_player1 = self._compute_discount(matches_player1_as_loser, date, surface)

        stats1 = []
        for d1, d2 in zip(discounted_wins_player1, discounted_loses_player1):
            s1 = self._concat_and_sum_stats(d1, d2, h2h=False)
            stats1.append(s1)

        discounted_wins_player2 = self._compute_discount(matches_player2_as_winner, date, surface)
        discounted_loses_player2 = self._compute_discount(matches_player2_as_loser, date, surface)

        stats2 = []
        for d1, d2 in zip(discounted_wins_player2, discounted_loses_player2):
            s2 = self._concat_and_sum_stats(d1, d2, h2h=False)
            stats2.append(s2)

        return stats1, stats2

    def _compute_discount(self, df, reference_date, surface):
        """
        Returns list dataframes with time and surface discounted stats for each discount factor.
        Args:
        ----
            :reference_date: date
                Date with respect to which the discount has to be computed
            :surface: str
                Surface with respect to which the discount has to be computed

        Returns:
        --------
            :dfs: list of DataFrame
                List of discounted dataframe
        """

        surface_discount = df["surface"].apply(lambda s: self._surface.loc[surface, s])

        dfs = []
        for type, factor in self._discount:

            if type == "year":
                delta = reference_date.year - df["date"].dt.year
            elif type == "week":
                delta = (reference_date - df["date"]) // np.timedelta64(1, 'W')
            else:
                delta = 1

            _df = df.copy()
            _df["discount"] = (factor**(delta)) * surface_discount
            dfs.append(_df)
        return dfs

    def _concat_and_sum_stats(self, df1, df2, h2h=False):
        """
        Args:
        -----
        df1: player1 as winner
        df2: player1 as loser
        h2h: True if dfs only contain h2h oppositions
        """
        def _weight(df, col):
            """
            Avoids na in the weighting
            """
            w = sum(df[~col.isnull()].discount)
            w = 1 if w == 0 else w
            return col.multiply(df.discount).divide(w)

        win1 = df1.discount.sum()
        lose1 = df2.discount.sum()

        ratio1 = win1 / (win1 + lose1) if (win1 + lose1) != 0 else 1

        stats1_as_winner = df1[self.STATS_WINNER]
        stats1_as_loser = df2[self.STATS_LOSER]
        stats1 = pd.DataFrame(np.concatenate(
            [stats1_as_winner.values, stats1_as_loser.values]), columns=self.STATS_FINAL)

        stats1 = stats1.apply(lambda c: _weight(stats1, c)).sum()

        stats1['ratio'] = ratio1

        if h2h:
            stats2_as_loser = df1[self.STATS_LOSER]
            stats2_as_winner = df2[self.STATS_WINNER]

            ratio2 = lose1 / (win1 + lose1) if (win1 + lose1) != 0 else 1

            stats2 = pd.DataFrame(np.concatenate(
                [stats2_as_winner.values, stats2_as_loser.values]), columns=self.STATS_FINAL)
            stats2 = stats2.apply(lambda c: _weight(stats2, c)).sum()

            stats2['ratio'] = ratio2

            return stats1, stats2
        else:
            return stats1

    def _process_total_stats(self, entry):
        """
        Computes meaningful stats from the raw ones.

        Arg:
        ----
            entry: DataFrame entry containing the following columns
                'ratio'

                'self_ace', 'self_df', 'self_svpt',
                'self_1stIn', 'self_1stWon', 'self_2ndWon',
                'self_SvGms', 'self_bpSaved', 'self_bpFaced',

                'opp_ace', 'opp_df', 'opp_svpt',
                'opp_1stIn', 'opp_1stWon', 'opp_2ndWon',
                'opp_SvGms', 'opp_bpSaved', 'opp_bpFaced'
        Return:
        -------
            stats: list containing the following stats
                    ratio,              % winning ratio
                    ace,                % ace
                    df,                 % df
                    1st_in,             % first serve in
                    1st_win,            % first serve point won
                    2nd_win,            % second serve point won
                    serve_win           % serve win
                    break_saved,        % break points saved
                    break_lost,         % break points conceeded
                    return_1st_win      % return win first serve
                    return_2nd_win      % return win second serve
                    ace_faced           % ace faced
                    break_win           % break win
                    points_win          % points win
        """
        # Self stats
        ratio = entry['ratio']
        ace = entry['self_ace']/entry['self_svpt']
        df = entry['self_df']/entry['self_svpt']
        serve_win = (entry['self_1stWon'] + entry['self_2ndWon'])/entry['self_svpt']
        _1st_in = entry['self_1stIn']/entry['self_svpt']
        _1st_win = entry['self_1stWon']/entry['self_1stIn']
        _2nd_win = entry['self_2ndWon']/(entry['self_svpt'] - entry['self_1stIn'])

        break_saved = entry['self_bpSaved']/entry['self_bpFaced']
        break_lost = (entry['self_bpFaced']-entry['self_bpSaved'])/entry['self_SvGms']

        # With opponent
        return_1st_win = 1 - ((entry['opp_1stWon'])/entry['opp_1stIn'])
        return_2nd_win = 1 - ((entry['opp_2ndWon'])/(entry['opp_svpt'] - entry['opp_1stIn']))
        ace_faced = entry['opp_ace']/entry['opp_svpt']
        break_win = (entry['opp_bpFaced'] - entry['opp_bpSaved'])/entry['opp_svpt']
        points_win = (entry['self_1stWon'] + entry['self_2ndWon'] + entry['opp_svpt'] -
                      entry['opp_1stWon'] - entry['opp_2ndWon'])/(entry['opp_svpt'] + entry['self_svpt'])

        processed_stats = [ratio, ace, df, _1st_in,
                           _1st_win, _2nd_win, serve_win, break_saved, break_lost, return_1st_win, return_2nd_win, ace_faced, break_win, points_win]

        return processed_stats

    def _final_header_generation(self):
        """
        Returns the final stats header
        """
        return self.HEADER_AS_SUCH + [t + '_' + s + '_' + p for t in self._stats_type for p in ['1', '2'] for s in self.HEADER_STATS_GENERATION]

    def _write_to_file(self, containers, output):

        for i, c in enumerate(containers):
            my_df = pd.DataFrame(c)
            fname = '{0}{1}.csv'.format(output, i)

            if not os.path.isfile(fname):  # check if exists
                my_df.to_csv(fname, index=False, header=self._header_generation)
            else:
                my_df.to_csv(fname, mode='a', index=False, header=False)

            c.clear()


if __name__ == '__main__':
    # DB
    f = 'raw/raw_2019.csv'
    db = pd.read_csv(f, index_col=None, header=0, low_memory=False)
    db['date'] = pd.to_datetime(db['date'], format="%d/%M/%Y")

    dm = DatasetMaker(db, DatasetMaker.FULL_CONFIG, debug=True)

    # generation
    start_date = '1/1/2005'
    end_date = '31/12/2019'
    start = pd.to_datetime(start_date, format="%d/%m/%Y")
    end = pd.to_datetime(end_date, format="%d/%m/%Y")

    target_df = db[((db.date >= start) & (db.date < end))]

    n_cores = multiprocessing.cpu_count()
    fname_generation = "final-datasets/dataset"

    print("[1/2] Generate datasets...")
    print("(On {} cores)".format(n_cores))

    df_split = np.array_split(target_df, n_cores)  # Split the dataset
    pool = multiprocessing.Pool(n_cores)
    generate = functools.partial(dm.generate, output=fname_generation)
    pool.map(generate, df_split)
    print("Generated.")

    print("[2/2] Prepare datasets")
    fname_preparation = "final-datasets/prepared"
    for i, _ in enumerate(dm._discount):
        generated_df = pd.read_csv("{0}{1}.csv".format(
            fname_generation, i), index_col=None, header=0, low_memory=False)
        dm.prepare(generated_df, "{0}{1}.csv".format(fname_preparation, i))
    print("Prepared.")
