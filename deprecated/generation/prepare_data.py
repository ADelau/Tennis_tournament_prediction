import pandas as pd
import numpy as np

DIRNAME_RAW = ""
#DIRNAME_RAW = "prepocessed/raw/"
DIRNAME_CONCAT = "prepocessed/concatenated/"

files = ["year_data_05.csv", "year_data_06.csv", "year_data_07.csv",
         "year_data_08.csv", "year_data_09.csv", "year_data_1.csv",
         "week_data_0991.csv", "week_data_0993.csv",
         "week_data_0995.csv", "week_data_0997.csv"]

#files = ['tournament_06.csv']

raw_datasets = [DIRNAME_RAW + f for f in files]
concat_datasets = [DIRNAME_CONCAT + f for f in files]

#to_drop = ['winner_hand', 'winner_name', 'loser_hand', 'loser_name']
to_drop = []

to_swap = ['name', 'id', 'ht', 'age', 'rank', 'rank_points',

           'h2h_ratio', 'h2h_ace', 'h2h_df', 'h2h_1st_in',
           'h2h_1st_win', 'h2h_2nd_win', 'h2h_serve_win',
           'h2h_break_saved', 'h2h_break_lost', 'h2h_return_1st_win',
           'h2h_return_2nd_win', 'h2h_ace_faced', 'h2h_break_win', 'h2h_points_win',

           'common_ratio', 'common_ace', 'common_df', 'common_1st_in',
           'common_1st_win', 'common_2nd_win', 'common_serve_win',
           'common_break_saved', 'common_break_lost', 'common_return_1st_win',
           'common_return_2nd_win', 'common_ace_faced', 'common_break_win', 'common_points_win']

to_keep = ['winner_rank', 'loser_rank']


def prepare_df(df):
    def _diff_df(df):
        for col in to_swap:
            col_winner = "winner_" + col
            col_loser = "loser_" + col

            df[col_winner] = df[col_winner] - df[col_loser]
            df[col_loser] = -df[col_winner]
        return df

    print(df.columns)
    winner_df1 = pd.DataFrame(df[["winner_" + col for col in to_swap]
                                 ].values, columns=[col + "_1" for col in to_swap])
    loser_df1 = pd.DataFrame(df[["loser_" + col for col in to_swap]].values,
                             columns=[col + "_2" for col in to_swap])

    df1 = pd.concat([winner_df1, loser_df1], axis=1)
    df1["outcome"] = 1
    df1["date"] = df["date"]
    df1["level"] = df["level"]

    winner_df2 = pd.DataFrame(df[["winner_" + col for col in to_swap]
                                 ].values, columns=[col + "_2" for col in to_swap])
    loser_df2 = pd.DataFrame(df[["loser_" + col for col in to_swap]].values,
                             columns=[col + "_1" for col in to_swap])

    df2 = pd.concat([loser_df2, winner_df2], axis=1)
    df2["outcome"] = 0
    df2["date"] = df["date"]
    df2["level"] = df["level"]

    prepared_df = pd.concat([df1, df2], axis=0, ignore_index=True).fillna(value=0)

    prepared_df["completeness_1"] = prepared_df.common_serve_win_1 * \
        (prepared_df.common_return_1st_win_1 + prepared_df.common_return_2nd_win_1)
    prepared_df["completeness_2"] = prepared_df.common_serve_win_2 * \
        (prepared_df.common_return_1st_win_2 + prepared_df.common_return_2nd_win_2)

    # Added feature: First Serve Advantage
    prepared_df["serve1_adv_1"] = prepared_df.common_1st_win_1 - prepared_df.common_return_1st_win_2
    prepared_df["serve1_adv_2"] = prepared_df.common_1st_win_2 - prepared_df.common_return_1st_win_1

    # Added feature: Second Serve Advantage
    prepared_df["serve2_adv_1"] = prepared_df.common_2nd_win_1 - prepared_df.common_return_2nd_win_2
    prepared_df["serve2_adv_2"] = prepared_df.common_2nd_win_2 - prepared_df.common_return_2nd_win_1

    # Differences between features
    # Player 1
    cols_1 = [col for col in prepared_df if col.endswith('1') and col != 'id_1' and col != 'name_1']
    # Player 2
    cols_2 = [col for col in prepared_df if col.endswith('2') and col != 'id_2' and col != 'name_2']

    cols = [col[0:-2] for col in cols_1]  # Final cols

    diff = prepared_df[cols_1].values - prepared_df[cols_2].values

    final = pd.DataFrame(diff, columns=cols)
    final["date"] = prepared_df.date
    final["level"] = prepared_df.level
    final["outcome"] = prepared_df.outcome
    final["name_1"] = prepared_df.name_1
    final["name_2"] = prepared_df.name_2
    final["rank_1"] = prepared_df.rank_1
    final["rank_2"] = prepared_df.rank_2

    return final.sort_values(by='date')


def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter, encoding="latin_1")


if __name__ == '__main__':

    for fr, to in zip(raw_datasets, concat_datasets):
        _df = load_from_csv(fr).drop(columns=to_drop, axis=1)
        df = prepare_df(_df)
        df.to_csv(to, index=False)
