--last match of 1 player
SELECT MAX(date), * 
FROM 
(SELECT *
FROM matches
WHERE player_1=103292 OR player_2=103292);
