import functools
import multiprocessing
import argparse
from scraper import *
from DatasetMaker import *
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


HEADER_AS_SUCH = ['level', 'date',
                  'id_1', 'name_1', 'ht_1', 'age_1',
                  'rank_1', 'rank_points_1',
                  'id_2', 'name_2', 'ht_2', 'age_2',
                  'rank_2', 'rank_points_2']

HEADER = ['rank', 'rank_points', 'name', 'age', 'move', 'move_direction']

HEADER_MERGE = ['rank', 'rank_points', 'name', 'age', 'id', 'ht']


class TournamentMaker:

    COLS = ['surface', 'level', 'date',
            'winner_id', 'winner_name', 'winner_ht', 'winner_age',
            'winner_rank', 'winner_rank_points',
            'loser_id', 'loser_name', 'loser_ht', 'loser_age', 'loser_rank',
            'loser_rank_points']

    def __init__(self, db, debug=False):
        self.db = db.sort_values(by='date')
        self.debug_mode = debug

    def generate_tournament(self, date, surface, n_players=256):
        """
            Generates a tournament at the given date, on the given surface
                and for a the <n_players> first ranked players.
            Args:
            -----
                :date: date (datetime or str) %Y-%m-%d
                :surface: str (in ['Clay', 'Hard', 'Grass', 'Carpet'])
                :n_players: int (default:256)
        """
        self.date = date
        self.surface = surface
        players = self._get_players(date, n_players)
        oppositions = self._generate_oppositions(players)

        return oppositions

    def _get_players(self, date, n_players):

        weeks = scrape_rank_weeks()  # Get all weeks
        self._debug("Nb Weeks scraped: {}".format(len(weeks)))
        for week in weeks:
            if week <= date:
                closest_prior_week = week  # Closest week prior to the tournament
                break

        players = scrape_ranking_at_week(closest_prior_week, n_players)  # Rankings at that week
        self._debug("Nb Rankings scraped: {}".format(len(players)))
        self._debug("Rankings scraped: \n{}".format(players))

        return players

    def _generate_oppositions(self, players):
        """
            Args:
            -----
                players: (DataFrame) containing  a list of players with ranking, and
                                            rank_points and age.
                NB: height and id are inferred from the database.
        """
        inferred = self._infer_from_db(players)
        self._debug("Inferred :\n {}".format(inferred))

        n_oppositions = (inferred.values.shape[0])*(inferred.values.shape[0] - 1)//2
        self._debug("Nb possible oppositions :\n {}".format(n_oppositions))

        oppositions_df = pd.DataFrame(
            np.nan, index=np.arange(n_oppositions), columns=self.COLS)

        oppositions_df.date = pd.to_datetime(self.date)
        oppositions_df.surface = self.surface

        k = 0
        for i in range(len(players)):
            for j in range(i):
                self._debug('[{}/{}] Oppositions Generation'.format(k+1, n_oppositions))
                oppositions_df.ix[k, ['winner_' + h for h in HEADER_MERGE] + ['loser_' + h for h in HEADER_MERGE]
                                  ] = players.ix[i, HEADER_MERGE].tolist() + players.ix[j, HEADER_MERGE].tolist()
                k += 1

        self._debug("Tournament Generated: \n{}".format(oppositions_df))

        return oppositions_df

    def _infer_from_db(self, players, to_infer=['ht', 'id']):

        loser_labels = ['loser_' + t for t in ['name'] + to_infer]
        winner_labels = ['winner_' + t for t in ['name'] + to_infer]

        for col in to_infer:
            players[col] = np.NaN

        for i, p in players.iterrows():
            try:
                rows_as_loser = self.db['loser_name'].str.contains(
                    p["name"], regex=False, case=False)
                df = self.db[rows_as_loser][loser_labels]
                players.ix[i, ['name'] + to_infer] = df.iloc[-1, :].tolist()
            except:
                try:
                    rows_as_winner = self.db['winner_name'].str.contains(
                        p["name"], regex=False, case=False)
                    df = self.db[rows_as_winner][winner_labels]
                    players.ix[i, ['name'] + to_infer] = df.iloc[-1, :].tolist()
                except:
                    print("Error: {} Not Found".format(p["name"]))
        return players

    def _debug(self, msg):
        if self.debug_mode:
            print("DEBUG: {}".format(msg))


if __name__ == '__main__':

    db_file = 'raw/raw_2001_2018.csv'
    db = pd.read_csv(db_file, index_col=None, header=0, low_memory=False)
    db['date'] = pd.to_datetime(db['date'], format="%d/%m/%Y")

    parser = argparse.ArgumentParser(description='Generate dataset for tournament.')
    parser.add_argument("-s", "--surface", type=str,
                        help="Surface", required=True)
    parser.add_argument("-d", "--date", type=str,
                        help="Date %Y-%m-%d", required=True)
    parser.add_argument("-p", "--players", type=int,
                        help="Nb players", required=True)
    args = parser.parse_args()
    surface = args.surface
    date = args.date
    n_players = args.players

    tm = TournamentMaker(db=db)

    # Generate oppositions
    print("[1/3] Generate Oppositions...")
    tournament_df = tm.generate_tournament(date, surface, n_players)
    print("Generated.")

    # Generation and Preparation
    dm = DatasetMaker(db, DatasetMaker.DEFAULT_CONFIG)
    n_cores = multiprocessing.cpu_count()
    # Generate
    print("[2/3] Generate dataset...")
    print("(On {} cores)".format(n_cores))
    fname_generation = "raw_tournament_{}_{}_{}".format(surface, n_players, date)
    df_split = np.array_split(tournament_df, 10)  # Split the dataset
    pool = multiprocessing.Pool(n_cores)
    generate = functools.partial(
        dm.generate, output=fname_generation)
    pool.map(generate, df_split)
    print("Generated.")

    # Prepare
    print("[3/3] Prepare generated dataset...")
    fname_preparation = "prepared_tournament_{}_{}_{}".format(surface, n_players, date)
    generated_df = pd.read_csv(fname_generation+"0.csv", index_col=None, header=0, low_memory=False)
    dm.prepare(generated_df, fname_preparation+'.csv')
    print("Prepared.")
