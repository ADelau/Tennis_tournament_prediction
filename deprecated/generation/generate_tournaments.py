import multiprocessing
import functools
import pandas as pd
import numpy as np
from DatasetMaker import *

discount_type = "year"
discount_factors = {"year": [0.6]}
types = {"common": True}
level = 'G'
surface = 'Clay'

cols = ['surface', 'level', 'date',
        'winner_id', 'winner_name', 'winner_ht', 'winner_age',
        'winner_rank', 'winner_rank_points',
        'loser_id', 'loser_name', 'loser_ht', 'loser_age', 'loser_rank',
        'loser_rank_points']


FILE = 'raw/2018.csv'
FULL_FILE = 'raw/raw_2001_2018.csv'


df = pd.read_csv(FILE, index_col=None, header=0, low_memory=False)

# Get matches from RG 2018
matches = df[(df.tourney_id == '2018-520')]

# Date
date = matches.Date.iloc[0]

# players
players = pd.unique(np.concatenate([matches.winner_id.values, matches.loser_id.values]))
players_names = pd.unique(np.concatenate([matches.winner_name.values, matches.loser_name.values]))
# create df
n_possible = len(players)*(len(players)-1)/2
tournament_df = pd.DataFrame(np.nan, index=np.arange(n_possible), columns=cols)
tournament_df.level = level
tournament_df.date = pd.to_datetime(date)
tournament_df.surface = surface

ages = {}
hts = {}
ranks = {}
rank_points = {}

for i, m in matches.iterrows():
    winner = m["winner_id"]
    loser = m["loser_id"]

    ages[winner] = m["winner_age"]
    hts[winner] = m["winner_ht"]
    ranks[winner] = m["winner_rank"]
    rank_points[winner] = m["winner_rank_points"]

    ages[loser] = m["loser_age"]
    hts[loser] = m["loser_ht"]
    ranks[loser] = m["loser_rank"]
    rank_points[loser] = m["loser_rank_points"]

k = 0
for i in range(len(players)):
    for j in range(i):
        print('[{}/{}]'.format(k, n_possible))
        tournament_df.ix[k, ['winner_id', 'loser_id', 'winner_name', 'loser_name', 'winner_ht', 'loser_ht', 'winner_age', 'loser_age',
                             'winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points']] = [players[i], players[j], players_names[i], players_names[j], hts[players[i]], hts[players[j]], ages[players[i]], ages[players[j]], ranks[players[i]], ranks[players[j]], rank_points[players[i]],  rank_points[players[j]]]

        k += 1

tournament_df.to_csv("tournament.csv", index=False, header=cols)


final_header = get_final_header(types=types)
full_df = pd.read_csv(FULL_FILE, index_col=None, header=0, low_memory=False)
full_df = clean_df(full_df)

# Multiprocessing setup
n_cores = 4
df_split = np.array_split(tournament_df, 4)  # Split the dataset
pool = multiprocessing.Pool(n_cores)

generate = functools.partial(_generate_dataset, full_df=full_df,
                             discount_type=discount_type,
                             discount_factors=discount_factors[discount_type],
                             types=types,
                             output="tournament", final_header=final_header)

pool.map(generate, df_split)

print("Finished...")
