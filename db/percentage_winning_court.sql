--percentage winning on the same court 
/* @court : SELECT DISTINCT court FROM tounaments 
@player_id : SELECT DISTINCT id FROM players  */
SELECT COUNT (n)
FROM matches
INNER JOIN tournaments ON matches.tournament=tournaments.id
WHERE tournaments.court = "Outdoor" AND matches.player_1 = 104925;

SELECT COUNT (n)
FROM matches
INNER JOIN tournaments ON matches.tournament=tournaments.id
WHERE tournaments.court = "Outdoor" AND (matches.player_1 = 104925 OR matches.player_2 = 104925);