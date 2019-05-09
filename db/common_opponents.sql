-- commun opponents
DROP VIEW IF EXISTS [opponents A];
CREATE VIEW [opponents A] AS 
SELECT DISTINCT player
FROM 
(SELECT DISTINCT player_2 AS player
FROM matches 
WHERE player_1 = 104745

UNION

SELECT DISTINCT player_1 AS player
FROM matches 
WHERE player_2 = 104745);

DROP VIEW IF EXISTS [opponents B];
CREATE VIEW [opponents B] AS 
SELECT DISTINCT player
FROM 
(SELECT DISTINCT player_2 AS player
FROM matches 
WHERE player_1 = 104925

UNION

SELECT DISTINCT player_1 AS player 
FROM matches 
WHERE player_2 = 104925);

DROP VIEW IF EXISTS [common opponents A B];
CREATE VIEW [common opponents A B] AS
SELECT * 
FROM matches
WHERE player_1 = 104745 AND player_2 IN(SELECT player AS player_2 FROM [opponents B])

UNION

SELECT * 
FROM matches 
WHERE player_2=104745 AND player_1 IN(SELECT player AS player_1 FROM [opponents B])

UNION

SELECT * 
FROM matches 
WHERE player_1=104925 AND player_2 IN(SELECT player AS player_2 FROM [opponents A])

UNION

SELECT * 
FROM matches 
WHERE player_2=104925 AND player_1 IN(SELECT player AS player_1 FROM [opponents A]);

SELECT * 
FROM [common opponents A B];