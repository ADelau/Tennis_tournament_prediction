--previous matches between 2 opponents
SELECT * 
FROM matches 
WHERE (player_1= 104745 AND player_2 = 104925) OR (player_1=104925  AND player_2 = 104745);