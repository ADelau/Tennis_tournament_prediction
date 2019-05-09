import multiprocessing
import sys
import os
import pandas as pd
import numpy as np
import datetime
import functools


"""
* Constants
"""
DIRNAME = 'prepocessed/'

surfaces = ['Grass', 'Hard', 'Clay', 'Carpet']

surf = {s: i for i, s in enumerate(surfaces)}

# Obtained by running the compute_surface_correlation function
corr = np.array([[1.0, 0.7015425398108537, 0.3623050295375809, 0.4170855617635379],
                 [0.7015425398108537, 1.0, 0.5328161870767436, 0.5960299654435783],
                 [0.3623050295375809, 0.5328161870767436, 1.0, 0.22536731067055044],
                 [0.4170855617635379, 0.5960299654435783, 0.22536731067055044, 1.0]])

to_be_kept_as_such = ['level', 'date',
                      'winner_id', 'winner_name', 'winner_ht', 'winner_age',
                      'winner_rank', 'winner_rank_points',
                      'loser_id', 'loser_name', 'loser_ht', 'loser_age', 'loser_rank',
                      'loser_rank_points']

stats_winner = [
    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
    'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt',
    'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'discount']

stats_loser = [
    'l_ace', 'l_df', 'l_svpt',
    'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
    'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'discount']

stats = [
    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
    'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt',
    'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']

stats_final = [
    'self_ace', 'self_df', 'self_svpt',
    'self_1stIn', 'self_1stWon', 'self_2ndWon', 'self_SvGms', 'self_bpSaved', 'self_bpFaced',
    'opp_ace', 'opp_df', 'opp_svpt',
    'opp_1stIn', 'opp_1stWon', 'opp_2ndWon', 'opp_SvGms', 'opp_bpSaved', 'opp_bpFaced', 'discount']


stats_header = ['ratio', 'ace', 'df', '1st_in',
                '1st_win', '2nd_win', 'serve_win', 'break_saved', 'break_lost', 'return_1st_win', 'return_2nd_win', 'ace_faced', 'break_win', 'points_win']


discount_factors = {"week": [0.997, 0.995, 0.993, 0.991], "year": [1, 0.9, 0.8, 0.7, 0.6, 0.5]}


def discount(df, reference_date, discount_type, discount_factors, surface):
    """
    Returns list dataframes with time and surface discounted stats for each discount factor.
    Args:
    ----
        :df: DataFrame
            Initial DataFrame
        :reference_date: date
            Date with respect to which the discount has to be computed
        :discount_type: str ("week", "year")
            Type of discount
        :discount_factors: list of float
            List of discount factors
        :surface: str
            Surface with respect to which the discount has to be computed

    Returns:
    --------
        :dfs: list of DataFrame
            List of discounted dataframe
    """
    surface_discount = df["surface"].apply(lambda s: corr[surf[s], surf[surface]])

    if discount_type == "year":
        delta = reference_date.year - df["date"].dt.year
    elif discount_type == "week":
        delta = (reference_date - df["date"]) // np.timedelta64(1, 'W')
    else:
        delta = 1

    dfs = []
    for f in discount_factors:
        _df = df.copy()
        _df["discount"] = (f**(delta)) * surface_discount
        dfs.append(_df)
    return dfs


def concat_and_sum_stats(df1, df2, h2h=False):
    """
    Args:
    -----
    df1: player1 as winner
    df2: player1 as loser
    h2h: True if dfs only contain h2h oppositions
    """
    def weight(df, col):
        """
        Avoids na in the weighting
        """
        w = sum(df[~col.isnull()].discount)
        w = 1 if w == 0 else w
        return col.multiply(df.discount).divide(w)

    win1 = df1.discount.sum()
    lose1 = df2.discount.sum()

    ratio1 = win1 / (win1 + lose1) if (win1 + lose1) != 0 else 1

    stats1_as_winner = df1[stats_winner]
    stats1_as_loser = df2[stats_loser]
    stats1 = pd.DataFrame(np.concatenate(
        [stats1_as_winner.values, stats1_as_loser.values]), columns=stats_final)

    stats1 = stats1.apply(lambda c: weight(stats1, c)).sum()

    stats1['ratio'] = ratio1

    if h2h:
        stats2_as_loser = df1[stats_loser]
        stats2_as_winner = df2[stats_winner]

        ratio2 = lose1 / (win1 + lose1) if (win1 + lose1) != 0 else 1

        stats2 = pd.DataFrame(np.concatenate(
            [stats2_as_winner.values, stats2_as_loser.values]), columns=stats_final)
        stats2 = stats2.apply(lambda c: weight(stats2, c)).sum()

        stats2['ratio'] = ratio2

        return stats1, stats2
    else:
        return stats1


def process_total_stats(entry):
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


def compute_stats(df, player1, player2, surface, date, discount_type, discount_factors, types):
    """
    Computes the stats of two players, on given surface, at a given date
    based on their previous matches.

    Args:
    -----
        :df: DataFrame
            containing the matches database
        :player1: int
            id of the first player
        :player2: int
            id of the second player
        :surface: str
            surface on which the match will be played
        :discount_type: str ('week', 'year')
            time frequency for time discounting
        :discount_factors: list of float
            discount_factors
        :**kwargs: ('h2h'=True/False, 'common'=True/False, 'avg'=True/False)

    Return:
    -------
        :stats: list of lists (1 per discount factor)
            They contain, in order, the following stats:
                ace, df, 1st_in, 1st_win, 2nd_win, serve_win, break_saved, break_lost,
                return_1st_win, return_2nd_win, ace_faced, break_win
                points_win
    """
    stats = [[] for _ in discount_factors]

    for func, value in types.items():
        if value:
            _stats_1, _stats_2 = globals()[func](df, player1, player2, date,
                                                 surface, discount_type, discount_factors)
            stats_1 = list(map(process_total_stats, _stats_1))
            stats_2 = list(map(process_total_stats, _stats_2))
            for i, (s1, s2) in enumerate(zip(stats_1, stats_2)):
                stats[i].extend(s1)
                stats[i].extend(s2)

    return stats


def h2h(df, player1, player2, date, surface, discount_type, discount_factors):
    matches_player1_as_winner = df[(df.date < date) & (
        df.winner_id == player1) & (df.loser_id == player2)]

    matches_player2_as_winner = df[(df.date < date) & (
        df.winner_id == player2) & (df.loser_id == player1)]
    discounted_p1 = discount(matches_player1_as_winner, date, discount_type,
                             discount_factors, surface)
    discounted_p2 = discount(matches_player2_as_winner, date, discount_type,
                             discount_factors, surface)

    stats1 = []
    stats2 = []

    for d1, d2 in zip(discounted_p1, discounted_p2):
        s1, s2 = concat_and_sum_stats(d1, d2, h2h=True)
        stats1.append(s1)
        stats2.append(s2)

    return stats1, stats2


def common(df, player1, player2, date, surface, discount_type, discount_factors):
    opp1_as_winner = df[(df.date < date) & (
        df.winner_id == player1) & (df.loser_id != player2)]
    opp1_as_loser = df[(df.date < date) & (
        df.loser_id == player1) & (df.winner_id != player2)]
    opp2_as_winner = df[(df.date < date) & (
        df.winner_id == player2) & (df.loser_id != player1)]
    opp2_as_loser = df[(df.date < date) & (
        df.loser_id == player2) & (df.winner_id != player1)]

    commons = np.unique(np.concatenate(
        [opp1_as_winner.loser_id.values, opp1_as_loser.winner_id.values, opp2_as_winner.loser_id.values, opp2_as_loser.winner_id.values]))

    com_opp1_as_winner = opp1_as_winner[opp1_as_winner.loser_id.isin(commons)]
    com_opp1_as_loser = opp1_as_loser[opp1_as_loser.winner_id.isin(commons)]
    com_opp2_as_winner = opp2_as_winner[opp2_as_winner.loser_id.isin(commons)]
    com_opp2_as_loser = opp2_as_loser[opp2_as_loser.winner_id.isin(commons)]

    discounted_wins_player1 = discount(
        com_opp1_as_winner, date, discount_type, discount_factors, surface)
    discounted_loses_player1 = discount(
        com_opp1_as_loser, date, discount_type, discount_factors, surface)

    stats1 = []
    for d1, d2 in zip(discounted_wins_player1, discounted_loses_player1):
        s1 = concat_and_sum_stats(d1, d2, h2h=False)
        stats1.append(s1)

    discounted_wins_player2 = discount(
        com_opp2_as_winner, date, discount_type, discount_factors, surface)
    discounted_loses_player2 = discount(
        com_opp2_as_loser, date, discount_type, discount_factors, surface)

    stats2 = []
    for d1, d2 in zip(discounted_wins_player2, discounted_loses_player2):
        s2 = concat_and_sum_stats(d1, d2, h2h=False)
        stats2.append(s2)

    return stats1, stats2


def avg(df, player1, player2, date, surface, discount_type, discount_factors):

    matches_player1_as_winner = df[(df.date < date) & (df.winner_id == player1)]
    matches_player1_as_loser = df[(df.date < date) & (df.loser_id == player1)]
    matches_player2_as_winner = df[(df.date < date) & (df.winner_id == player2)]
    matches_player2_as_loser = df[(df.date < date) & (df.loser_id == player2)]

    discounted_wins_player1 = discount(matches_player1_as_winner, date,
                                       discount_type, discount_factors, surface)
    discounted_loses_player1 = discount(
        matches_player1_as_loser, date, discount_type, discount_typediscount_factors, surface)

    stats1 = []
    for d1, d2 in zip(discounted_wins_player1, discounted_loses_player1):
        s1 = concat_and_sum_stats(d1, d2, h2h=False)
        stats1.append(s1)

    discounted_wins_player2 = discount(matches_player2_as_winner, date,
                                       discount_type, discount_factors, surface)
    discounted_loses_player2 = discount(
        matches_player2_as_loser, date, discount_type, discount_factors, surface)

    stats2 = []
    for d1, d2 in zip(discounted_wins_player2, discounted_loses_player2):
        s2 = concat_and_sum_stats(d1, d2, h2h=False)
        stats2.append(s2)

    return stats1, stats2


def clean_df(df):
    # remove entries with missing surfaces
    print("*Remove missing surfaces.")
    df = df[df.surface != 'None']
    df = df[~df.surface.isnull()]
    # remove not finished matches
    print("*Remove not finished games.")
    df = df[~df.Comment.isin(['Walkover', 'Retired'])]
    df = df[df.score.str.contains("RET") == False]
    df = df[df.score.str.contains("W/O") == False]

    print("*Clean dates format.")
    df['date'] = pd.to_datetime(df['date'], format="%Y/%m/%d")
    df['discount'] = np.nan

    return df


def _generate_dataset(target_df, full_df,
                      discount_type,
                      discount_factors,
                      types,
                      output, final_header):

    containers = [[] for d in discount_factors]

    for i, match in target_df.iterrows():
        date = match['date']
        surface = match['surface']
        winner = match['winner_id']
        loser = match['loser_id']

        as_such = match[to_be_kept_as_such].tolist()

        stats = compute_stats(full_df, winner, loser, surface, date,
                              discount_type, discount_factors, types)
        # save
        for c, cont in enumerate(containers):
            cont.append(as_such + stats[c])

        if i % 100 == 0:
            for c, d in zip(containers, discount_factors):
                my_df = pd.DataFrame(c)
                fname = '{0}{1}.csv'.format(output, d)

                if not os.path.isfile(fname):
                    my_df.to_csv(fname, index=False, header=final_header)
                else:
                    my_df.to_csv(fname, mode='a', index=False, header=False)

                c.clear()

    # write the end
    for c, d in zip(containers, discount_factors):
        my_df = pd.DataFrame(c)
        fname = '{0}{1}.csv'.format(output, d)

        if not os.path.isfile(fname):
            my_df.to_csv(fname, index=False, header=final_header)
        else:
            my_df.to_csv(fname, mode='a', index=False, header=False)

        c.clear()


def get_final_header(types):
    """
    Returns the final stats header
    """
    tt = []
    for key, value in types.items():
        if value:
            tt.append('{}_'.format(key))
    return to_be_kept_as_such + [p + t + s for t in tt for p in ['winner_', 'loser_'] for s in stats_header]


if __name__ == '__main__':
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    discount_type = sys.argv[3]
    output = sys.argv[4]  # Output file

    types = {'common': True, 'h2h': True}

    final_header = get_final_header(types)

    # Read File
    f = 'raw/raw_2001_2018.csv'
    df = pd.read_csv(f, index_col=None, header=0, low_memory=False)

    # First Cleaning
    print('Before Cleaning: {} entries'.format(df.shape[0]))
    clean_df = clean_df(df)
    print('After Cleaning: {} entries'.format(clean_df.shape[0]))

    # ---------Process---------------------
    # Select relevant matches
    start = datetime.datetime.strptime('{}'.format(start_date), '%Y/%m/%d').date()
    end = datetime.datetime.strptime('{}'.format(end_date), '%Y/%m/%d').date()
    target_df = clean_df[(clean_df.date >= str(start)) & (
        clean_df.date < str(end))].sort_values(by='date')
    print("It will take a certain time... \nHave to generate {0} matches!".format(
        target_df.shape[0]))

    # Multiprocessing setup
    n_cores = 3
    df_split = np.array_split(target_df, 10)  # Split the dataset
    pool = multiprocessing.Pool(n_cores)

    generate = functools.partial(_generate_dataset, full_df=clean_df,
                                 discount_type=discount_type,
                                 discount_factors=discount_factors[discount_type],
                                 types=types,
                                 output=DIRNAME + output, final_header=final_header)

    pool.map(generate, df_split)

    print("Finished...")
