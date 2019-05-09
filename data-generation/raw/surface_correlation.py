import sys
import pandas as pd
import numpy as np

surfaces = ['Grass', 'Hard', 'Clay', 'Carpet']


def compute_surfaces_correlation(source, destination):
    """
     Computes the correlation between the different surfaces.
     It is done by computing the correlation between the win ratio
     of the players on each surface.
     Args:
     ----
        :source: str
            Source filename containing the raw dataset with the matches.
        :destination: str
            Destination filename that will contain the correlation matrix.
    """
    df = pd.read_csv(source, index_col=None, header=0, low_memory=False)

    min_support = 10

    winners = df['winner_id'].unique()
    losers = df['loser_id'].unique()

    players = np.unique(np.concatenate((winners, losers)))
    dic = {s: {p: np.nan for p in players} for s in surfaces}

    for p in players:
        for s in surfaces:
            wins = df[(df.winner_id == p) & (df.surface == s)].shape[0]
            loses = df[(df.loser_id == p) & (df.surface == s)].shape[0]
            if (wins + loses >= min_support):
                dic[s][p] = wins/(wins+loses)

    _df = pd.DataFrame(dic)

    corr = _df.dropna().corr().values
    corr[0:3, 0:3] = _df.drop('Carpet', axis=1).dropna().corr().values

    corr_df = pd.DataFrame(data=corr,
                           index=surfaces,
                           columns=surfaces)
    corr_df.to_csv(destination, index=False)

    return df


if __name__ == '__main__':
    source = str(sys.argv[1])
    dest = str(sys.argv[2])

    compute_surfaces_correlation(source=source, destination=dest)
