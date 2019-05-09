import argparse
from csv_utils import concat_csv

DIRNAME = "raw/"

cols_to_be_kept = ['tourney_level', 'surface', 'Date', 'winner_id',
                   'winner_name', 'winner_hand', 'winner_ht', 'winner_age',
                   'winner_rank', 'winner_rank_points', 'loser_id',
                   'loser_name', 'loser_hand', 'loser_ht', 'loser_age', 'loser_rank',
                   'loser_rank_points', 'score', 'best_of',
                   'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
                   'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt',
                   'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
                   'W1', 'L1', 'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5', 'Wsets', 'Lsets', 'Comment']
to_rename = {'Date': 'date', 'tourney_level': 'level'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concatenate years files.')
    parser.add_argument("-s", "--startyear", type=int,
                        help="The Start Year", required=True)
    parser.add_argument("-e", "--endyear", type=int,
                        help="The End Year", required=True)
    args = parser.parse_args()
    start_year = args.startyear
    end_year = args.endyear

    files = [DIRNAME + '{}.csv'.format(y) for y in range(start_year, end_year + 1)]
    dest_file = DIRNAME + 'raw_{0}_{1}.csv'.format(start_year, end_year)

    concat_csv(files, dest_file, cols_to_be_kept, to_rename=to_rename, sort_key='date')
