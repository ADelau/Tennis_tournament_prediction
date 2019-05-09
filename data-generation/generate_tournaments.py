from TournamentMaker import *
from sklearn.linear_model import LogisticRegression

n_players = 256

f = 'raw/raw_2019.csv'
tournaments = {'AO2017': ('2017-01-11', 'Hard'),
               'RG2017': ('2017-05-22', 'Clay'),
               'WB2017': ('2017-06-26', 'Grass'),
               'US2017': ('2017-08-22', 'Hard'),
               'AO2018': ('2018-01-10', 'Hard'),
               'RG2018': ('2018-05-21', 'Clay'),
               'WB2018': ('2018-06-25', 'Grass'),
               'US2018': ('2018-08-21', 'Hard'),
               'RG2019': ('2018-05-26', 'Clay'),
               }


def generate_tournaments(db_file=f, tournaments=tournaments, n_players=n_players):
    # Setup
    db = pd.read_csv(db_file, index_col=None, header=0, low_memory=False)
    db['date'] = pd.to_datetime(db['date'], format="%d/%m/%Y")
    tm = TournamentMaker(db=db)
    dm = DatasetMaker(db, DatasetMaker.DEFAULT_CONFIG)
    n_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n_cores)

    for t, (date, surface) in tournaments.items():
        print("Generating: {}".format(t))
        print("[1/3] Generate Oppositions...")
        tournament_df = tm.generate_tournament(date, surface, n_players)
        print("Generated.")

        # Generate
        print("[2/3] Generate dataset...")
        print("(On {} cores)".format(n_cores))
        fname_generation = "generation/{}_{}_{}_{}".format(t, surface, n_players, date)
        df_split = np.array_split(tournament_df, 10)  # Split the dataset
        generate = functools.partial(
            dm.generate, output=fname_generation)
        pool.map(generate, df_split)
        print("Generated.")

        # Prepare
        print("[3/3] Prepare generated dataset...")
        fname_preparation = "preparation/prepared_{}_{}_{}_{}".format(t, surface, n_players, date)
        generated_df = pd.read_csv(fname_generation+"0.csv",
                                   index_col=None, header=0, low_memory=False)
        dm.prepare(generated_df, fname_preparation+'.csv')
        print("Prepared.")


def predict_tournaments():
    def get_training_data(df, date=None):
        if date is not None:
            df = df[df.date < date]
        # X, y
        to_remove = ['outcome', 'name_1', 'name_2', 'date', 'level'] + ['B365_1', 'EX_1', 'LB_1',
                                                                        'CB_1', 'GB_1',  'IW_1', 'SB_1', 'SB_2', 'IW_2', 'GB_2', 'CB_2', 'LB_2', 'EX_2', 'B365_2']
        X_cols = [col for col in df if col not in to_remove]

        return df[X_cols], df["outcome"]

    def get_test_data(df):
        return get_training_data(df)

    def construct_proba_matrix(df, pred, infos, save_name="probas"):
        _names = ['name_1', 'name_2']
        __names = ['name_2', 'name_1']
        _probas = ['proba_1', 'proba_2']
        _ranks = ['rank_1', 'rank_2', 'rank_points_1', 'rank_points_2']

        m = pd.DataFrame(np.nan, index=np.arange(len(df)), columns=_names + _probas + _ranks)
        m[_names] = df[__names]
        m.proba_1 = pred[:, 0]
        m.proba_2 = pred[:, 1]

        for i, p in infos.iterrows():
            fullname = p["name"]
            split_name = fullname.replace("-", " ").split()
            first_name = split_name[0]
            last_name = split_name[-1]

            # Player 1
            rows_1 = (m['name_1'].str.contains(last_name, regex=True, case=False) &
                      m['name_1'].str.contains(first_name, regex=True, case=False))
            m.loc[rows_1, 'rank_1'] = p['rank']
            m.loc[rows_1, 'rank_points_1'] = p['rank_points']

            # Player 2
            rows_2 = (m['name_2'].str.contains(last_name, regex=True, case=False) &
                      m['name_2'].str.contains(first_name, regex=True, case=False))
            m.loc[rows_2, 'rank_2'] = p['rank']
            m.loc[rows_2, 'rank_points_2'] = p['rank_points']

        m.to_csv(save_name+".csv", index=False)
        return m

    df = pd.read_csv("prepared/prepared.csv", index_col=None,
                     header=0, low_memory=False).sort_values(by='date')
    df['date'] = pd.to_datetime(df['date'], format="%Y/%m/%d")
    df = df.replace([np.inf, -np.inf], 0).drop(['ht'], axis=1).dropna()

    for tournament, values in tournaments.items():
        date = values[0]
        surface = values[1]
        # Training
        X_train, y_train = get_training_data(df, date)
        print(X_train.columns)
        lr = LogisticRegression(penalty='l1', max_iter=1000).fit(X_train, y_train)
        # Testing
        fname = 'preparation/prepared_' + tournament + '_' + surface + '_256_' + date + '.csv'
        frankings = 'rankings/rankings_256_' + date + '.csv'

        test_df = pd.read_csv(fname, index_col=None, header=0, low_memory=False).replace(
            [np.inf, -np.inf, np.NaN], 0).drop(['ht'], axis=1)
        X_test, _ = get_test_data(test_df)
        print(X_test.columns)

        probas = lr.predict_proba(X_test)
        rankings = pd.read_csv(frankings, index_col=None, header=0, low_memory=False)
        construct_proba_matrix(test_df, probas, rankings, save_name=tournament + "_probas")


if __name__ == '__main__':
    # generate_tournaments()
    predict_tournaments()
