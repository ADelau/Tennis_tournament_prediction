--percentage winning opponent right/left handed 
/* @hand : SELECT DISTINCT hand FROM players */ 

SELECT COUNT (n)
FROM matches
WHERE player_1=104925 AND EXISTS
(SELECT * 
FROM players 
WHERE id=player_2 and hand="L");


SELECT COUNT (n) 
FROM matches
WHERE player_2=104925 AND EXISTS
(SELECT * 
FROM players 
WHERE id=player_1 and hand="L");