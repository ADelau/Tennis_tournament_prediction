SELECT  tttt.date as date, tttt.round as round,  tttt.name as opponent, tttt.surface as surface, tttt.court as court, (goff.age - tttt.age)/(goff.age/2 + tttt.age/2) as age_diff, (goff.atp_ranking - o_rank)/(goff.atp_ranking/2 + o_rank/2) as rank_diff, (goff.atp_points - o_points)/(goff.atp_points/2 + o_points/2) as points_diff, tttt.hand, (AVG(ace) - o_ace)/(AVG(ace)/2 + o_ace/2) as ace_diff, (AVG(df) - o_df)/(AVG(df)/2 + o_df/2) as df_diff, (AVG(svpt) - o_svpt)/(AVG(svpt)/2 + o_svpt/2) as svpt_diff, (AVG(first_in) - o_first_in)/(AVG(first_in)/2 + o_first_in/2) as first_in_diff, (AVG(first_won) - o_first_won)/(AVG(first_won)/2 + o_first_won/2) as first_won_diff, (AVG(second_won) - o_second_won)/(AVG(second_won)/2 + o_second_won/2) as second_won_diff, (AVG(sv_gms) - o_sv_gms)/(AVG(sv_gms)/2 + o_sv_gms/2) as sv_gms_diff, (AVG(bp_saved) - o_bp_saved)/(AVG(bp_saved)/2 + o_bp_saved/2) as bp_saved_diff, (AVG(bp_faced) - o_bp_faced)/(AVG(bp_faced)/2 + o_bp_faced/2) as bp_faced_diff, 0 as outcome
FROM
  (SELECT ttt.id, n, round, ttt.date, player_1, surface, court, age, atp_ranking as o_rank, atp_points as o_points, name, hand, height, odd, AVG(ace) as o_ace, AVG(df) as o_df, AVG(svpt) as o_svpt, AVG(first_in) as o_first_in, AVG(first_won) as o_first_won, AVG(second_won) as o_second_won, AVG(sv_gms) as o_sv_gms, AVG(bp_saved) as o_bp_saved, AVG(bp_faced) as o_bp_faced
    FROM
      (SELECT tt.id, n, round, date, player_1, surface, court, age, atp_ranking, atp_points, name, hand, height, AVG(odd_2) as odd FROM
        (SELECT t.id, n, round, date, player_1, surface, court, age, atp_ranking, atp_points, name, hand, height FROM
          (SELECT id, n, round, date, player_1, surface, court, age, atp_ranking, atp_points
                FROM
                    ( SELECT *
                      FROM
                        ( SELECT tournament, n, round, date, player_1 from matches
                          WHERE  player_2
                          IN (SELECT id
                              FROM   players
                              WHERE  name = 'David Goffin')) as ms
                    JOIN tournaments
                    ON tournaments.id = ms.tournament) as mst

                    JOIN players_in_tournaments
                    ON players_in_tournaments.tournament = mst.tournament  AND players_in_tournaments.player = mst.player_1) as t

                    JOIN players
                    ON t.player_1 = players.id) as tt
        JOIN odds
        ON tt.id = odds.tournament AND tt.n = odds.match_n
        GROUP BY tt.id, n) as ttt
      JOIN
        (SELECT date, matches.tournament, match_n, player, ace, df, svpt, first_in, first_won, second_won, sv_gms, bp_saved, bp_faced
        FROM matches_stats
        JOIN matches
        ON matches_stats.tournament = matches.tournament AND matches_stats.match_n = matches.n) as ms_date

      ON ms_date.player = ttt.player_1 AND ms_date.date < ttt.date
      GROUP BY ttt.id, n) as tttt
   JOIN
      (SELECT *
      FROM
        (SELECT *
          FROM matches
          JOIN matches_stats
          ON matches.tournament = matches_stats.tournament AND matches.n = matches_stats.match_n AND matches_stats.player = 105676) as gof
      JOIN players_in_tournaments
      ON gof.tournament = players_in_tournaments.tournament AND gof.player = players_in_tournaments.player) as goff
  ON tttt.date > goff.date
GROUP BY tttt.id, tttt.n
UNION
SELECT  tttt.date as date, tttt.round as round,  tttt.name as opponent, tttt.surface as surface, tttt.court as court, (goff.age - tttt.age) as age_diff, (goff.atp_ranking - o_rank) as rank_diff, (goff.atp_points - o_points) as points_diff, tttt.hand, tttt.height, odd, (AVG(ace) - o_ace) as ace_diff, (AVG(df) - o_df) as df_diff, (AVG(svpt) - o_svpt) as svpt_diff, (AVG(first_in) - o_first_in) as first_in_diff, (AVG(first_won) - o_first_won) as first_won_diff, (AVG(second_won) - o_second_won) as second_won_diff, (AVG(sv_gms) - o_sv_gms) as sv_gms_diff, (AVG(bp_saved) - o_bp_saved) as bp_saved_diff, (AVG(bp_faced) - o_bp_faced) as bp_faced_diff, 1 as outcome
FROM
(SELECT ttt.id, n, round, ttt.date, player_2, surface, court, age, atp_ranking as o_rank, atp_points as o_points, name, hand, height, odd, AVG(ace) as o_ace, AVG(df) as o_df, AVG(svpt) as o_svpt, AVG(first_in) as o_first_in, AVG(first_won) as o_first_won, AVG(second_won) as o_second_won, AVG(sv_gms) as o_sv_gms, AVG(bp_saved) as o_bp_saved, AVG(bp_faced) as o_bp_faced
    FROM
      (SELECT tt.id, n, round, date, player_2, surface, court, age, atp_ranking, atp_points, name, hand, height, AVG(odd_2) as odd FROM
        (SELECT t.id, n, round, date, player_2, surface, court, age, atp_ranking, atp_points, name, hand, height FROM
          (SELECT id, n, round, date, player_2, surface, court, age, atp_ranking, atp_points
                FROM
                    ( SELECT *
                      FROM
                        ( SELECT tournament, n, round, date, player_2 from matches
                          WHERE  player_1
                          IN (SELECT id
                              FROM   players
                              WHERE  name = 'David Goffin')) as ms
                    JOIN tournaments
                    ON tournaments.id = ms.tournament) as mst

                    JOIN players_in_tournaments
                    ON players_in_tournaments.tournament = mst.tournament  AND players_in_tournaments.player = mst.player_2) as t

                    JOIN players
                    ON t.player_2 = players.id) as tt
        JOIN odds
        ON tt.id = odds.tournament AND tt.n = odds.match_n
        GROUP BY tt.id, n) as ttt
      JOIN
        (SELECT date, matches.tournament, match_n, player, ace, df, svpt, first_in, first_won, second_won, sv_gms, bp_saved, bp_faced
        FROM matches_stats
        JOIN matches
        ON matches_stats.tournament = matches.tournament AND matches_stats.match_n = matches.n) as ms_date

      ON ms_date.player = ttt.player_2 AND ms_date.date < ttt.date
      GROUP BY ttt.id, n) as tttt
   JOIN
      (SELECT *
      FROM
        (SELECT *
          FROM matches
          JOIN matches_stats
          ON matches.tournament = matches_stats.tournament AND matches.n = matches_stats.match_n AND matches_stats.player = 105676) as gof
      JOIN players_in_tournaments
      ON gof.tournament = players_in_tournaments.tournament AND gof.player = players_in_tournaments.player) as goff
  ON tttt.date > goff.date
GROUP BY tttt.id, tttt.n;


SELECT  tttt.date as date, tttt.round as round,  tttt.name as opponent, tttt.surface as surface, tttt.court as court, (goff.age - tttt.age)/(goff.age/2 + tttt.age/2) as age_diff, (goff.atp_ranking - o_rank)* 1.0/(goff.atp_ranking/2 + o_rank/2) as rank_diff, (goff.atp_points - o_points)*1.0/(goff.atp_points/2 + o_points/2) as points_diff, tttt.hand, (AVG(ace) - o_ace)/(AVG(ace)/2 + o_ace/2) as ace_diff, (AVG(df) - o_df)/(AVG(df)/2 + o_df/2) as df_diff, (AVG(svpt) - o_svpt)/(AVG(svpt)/2 + o_svpt/2) as svpt_diff, (AVG(first_in) - o_first_in)/(AVG(first_in)/2 + o_first_in/2) as first_in_diff, (AVG(first_won) - o_first_won)/(AVG(first_won)/2 + o_first_won/2) as first_won_diff, (AVG(second_won) - o_second_won)/(AVG(second_won)/2 + o_second_won/2) as second_won_diff, (AVG(sv_gms) - o_sv_gms)/(AVG(sv_gms)/2 + o_sv_gms/2) as sv_gms_diff, (AVG(bp_saved) - o_bp_saved)/(AVG(bp_saved)/2 + o_bp_saved/2) as bp_saved_diff, (AVG(bp_faced) - o_bp_faced)/(AVG(bp_faced)/2 + o_bp_faced/2) as bp_faced_diff, 0 as outcome
FROM
  (SELECT ttt.id, n, round, ttt.date, player_1, surface, court, age, atp_ranking as o_rank, atp_points as o_points, name, hand, height, odd, AVG(ace) as o_ace, AVG(df) as o_df, AVG(svpt) as o_svpt, AVG(first_in) as o_first_in, AVG(first_won) as o_first_won, AVG(second_won) as o_second_won, AVG(sv_gms) as o_sv_gms, AVG(bp_saved) as o_bp_saved, AVG(bp_faced) as o_bp_faced
    FROM
      (SELECT tt.id, n, round, date, player_1, surface, court, age, atp_ranking, atp_points, name, hand, height, AVG(odd_2) as odd FROM
        (SELECT t.id, n, round, date, player_1, surface, court, age, atp_ranking, atp_points, name, hand, height FROM
          (SELECT id, n, round, date, player_1, surface, court, age, atp_ranking, atp_points
                FROM
                    ( SELECT *
                      FROM
                        ( SELECT tournament, n, round, date, player_1 from matches
                          WHERE  player_2
                          IN (SELECT id
                              FROM   players
                              WHERE  name = 'David Goffin')) as ms
                    JOIN tournaments
                    ON tournaments.id = ms.tournament) as mst

                    JOIN players_in_tournaments
                    ON players_in_tournaments.tournament = mst.tournament  AND players_in_tournaments.player = mst.player_1) as t

                    JOIN players
                    ON t.player_1 = players.id) as tt
        JOIN odds
        ON tt.id = odds.tournament AND tt.n = odds.match_n
        GROUP BY tt.id, n) as ttt
      JOIN
        (SELECT date, matches.tournament, match_n, player, ace, df, svpt, first_in, first_won, second_won, sv_gms, bp_saved, bp_faced
        FROM matches_stats
        JOIN matches
        ON matches_stats.tournament = matches.tournament AND matches_stats.match_n = matches.n) as ms_date

      ON ms_date.player = ttt.player_1 AND ms_date.date < ttt.date
      GROUP BY ttt.id, n) as tttt
   JOIN
      (SELECT *
      FROM
        (SELECT *
          FROM matches
          JOIN matches_stats
          ON matches.tournament = matches_stats.tournament AND matches.n = matches_stats.match_n AND matches_stats.player = 105676) as gof
      JOIN players_in_tournaments
      ON gof.tournament = players_in_tournaments.tournament AND gof.player = players_in_tournaments.player) as goff
  ON tttt.date > goff.date
GROUP BY tttt.id, tttt.n
UNION
SELECT  tttt.date as date, tttt.round as round,  tttt.name as opponent, tttt.surface as surface, tttt.court as court, (goff.age - tttt.age)/(goff.age/2 + tttt.age/2) as age_diff, (goff.atp_ranking - o_rank)*1.0/(goff.atp_ranking/2 + o_rank/2) as rank_diff, (goff.atp_points - o_points)*1.0/(goff.atp_points/2 + o_points/2) as points_diff, tttt.hand, (AVG(ace) - o_ace)/(AVG(ace)/2 + o_ace/2) as ace_diff, (AVG(df) - o_df)/(AVG(df)/2 + o_df/2) as df_diff, (AVG(svpt) - o_svpt)/(AVG(svpt)/2 + o_svpt/2) as svpt_diff, (AVG(first_in) - o_first_in)/(AVG(first_in)/2 + o_first_in/2) as first_in_diff, (AVG(first_won) - o_first_won)/(AVG(first_won)/2 + o_first_won/2) as first_won_diff, (AVG(second_won) - o_second_won)/(AVG(second_won)/2 + o_second_won/2) as second_won_diff, (AVG(sv_gms) - o_sv_gms)/(AVG(sv_gms)/2 + o_sv_gms/2) as sv_gms_diff, (AVG(bp_saved) - o_bp_saved)/(AVG(bp_saved)/2 + o_bp_saved/2) as bp_saved_diff, (AVG(bp_faced) - o_bp_faced)/(AVG(bp_faced)/2 + o_bp_faced/2) as bp_faced_diff, 1 as outcome
FROM
(SELECT ttt.id, n, round, ttt.date, player_2, surface, court, age, atp_ranking as o_rank, atp_points as o_points, name, hand, height, odd, AVG(ace) as o_ace, AVG(df) as o_df, AVG(svpt) as o_svpt, AVG(first_in) as o_first_in, AVG(first_won) as o_first_won, AVG(second_won) as o_second_won, AVG(sv_gms) as o_sv_gms, AVG(bp_saved) as o_bp_saved, AVG(bp_faced) as o_bp_faced
    FROM
      (SELECT tt.id, n, round, date, player_2, surface, court, age, atp_ranking, atp_points, name, hand, height, AVG(odd_2) as odd FROM
        (SELECT t.id, n, round, date, player_2, surface, court, age, atp_ranking, atp_points, name, hand, height FROM
          (SELECT id, n, round, date, player_2, surface, court, age, atp_ranking, atp_points
                FROM
                    ( SELECT *
                      FROM
                        ( SELECT tournament, n, round, date, player_2 from matches
                          WHERE  player_1
                          IN (SELECT id
                              FROM   players
                              WHERE  name = 'David Goffin')) as ms
                    JOIN tournaments
                    ON tournaments.id = ms.tournament) as mst

                    JOIN players_in_tournaments
                    ON players_in_tournaments.tournament = mst.tournament  AND players_in_tournaments.player = mst.player_2) as t

                    JOIN players
                    ON t.player_2 = players.id) as tt
        JOIN odds
        ON tt.id = odds.tournament AND tt.n = odds.match_n
        GROUP BY tt.id, n) as ttt
      JOIN
        (SELECT date, matches.tournament, match_n, player, ace, df, svpt, first_in, first_won, second_won, sv_gms, bp_saved, bp_faced
        FROM matches_stats
        JOIN matches
        ON matches_stats.tournament = matches.tournament AND matches_stats.match_n = matches.n) as ms_date

      ON ms_date.player = ttt.player_2 AND ms_date.date < ttt.date
      GROUP BY ttt.id, n) as tttt
   JOIN
      (SELECT *
      FROM
        (SELECT *
          FROM matches
          JOIN matches_stats
          ON matches.tournament = matches_stats.tournament AND matches.n = matches_stats.match_n AND matches_stats.player = 105676) as gof
      JOIN players_in_tournaments
      ON gof.tournament = players_in_tournaments.tournament AND gof.player = players_in_tournaments.player) as goff
  ON tttt.date > goff.date
GROUP BY tttt.id, tttt.n;

