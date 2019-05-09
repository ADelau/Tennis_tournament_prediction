--X last match of 1 player
SELECT *
FROM 
(SELECT * 
FROM matches 
WHERE player_1=200282 OR player_2=200282
ORDER BY date DESC)
LIMIT 2;