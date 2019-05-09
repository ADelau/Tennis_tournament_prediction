raw_stats = [
    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
    'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt',
    'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']

stats_final = [
    'self_ace', 'self_df', 'self_svpt',
    'self_1stIn', 'self_1stWon', 'self_2ndWon', 'self_SvGms', 'self_bpSaved', 'self_bpFaced',
    'opp_ace', 'opp_df', 'opp_svpt',
    'opp_1stIn', 'opp_1stWon', 'opp_2ndWon', 'opp_SvGms', 'opp_bpSaved', 'opp_bpFaced', 'discount']


def compute_stats(df, player1, player2, surface, date, discount_type, discount_factors, **kwargs):
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

    for func, value in kwargs.items():
        if value:
            _stats_1, _stats_2 = func(df, player1, player2, date,
                                      surface, discount_type, discount_factors)
            stats_1 = list(map(process_total_stats, _stats_1))
            stats_2 = list(map(process_total_stats, _stats_2))
            for i, (s1, s2) in enumerate(zip(stats_1, stats_2)):
                stats[i].extend(s1)
                stats[i].extend(s2)
